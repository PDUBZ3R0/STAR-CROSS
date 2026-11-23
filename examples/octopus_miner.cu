// octopus_miner.cu
// Standalone CUDA implementation of the Conflux Octopus PoW hashing pipeline
// Includes: Full Keccak-f1600 (Keccak-256 & Keccak-512), Light Cache generation,
// GPU DAG generation kernel, Warp-cooperative Octopus search kernel, and Host launcher.
//
// Compile with:
//   nvcc -O3 -arch=sm_75 octopus_miner.cu -o octopus_miner

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OCTOPUS_LANES       32
#define OCTOPUS_REGS        16
#define OCTOPUS_CNT_DAG     64
#define CACHE_ROUNDS        3
#define DAG_PARENTS         256

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

typedef struct {
    uint32_t words[8];
} hash256_t;

// ============================================================================
// 1. REAL KECCAK-F1600 IMPLEMENTATION (Host & Device)
// ============================================================================

__host__ __device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64 - n));
}

__constant__ uint64_t d_keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const uint64_t h_keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__host__ __device__ void keccak_f1600_core(uint64_t st[25], const uint64_t rc[24]) {
    const uint32_t rho[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
        27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
    };
    const uint32_t pi[24] = {
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
    };

    for (int round = 0; round < 24; round++) {
        uint64_t bc[5];
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }

        uint64_t t = st[1];
        for (int i = 0; i < 24; i++) {
            uint32_t j = pi[i];
            uint64_t tmp = st[j];
            st[j] = rotl64(t, rho[i]);
            t = tmp;
        }

        for (int j = 0; j < 25; j += 5) {
            uint64_t tmp[5];
            for (int i = 0; i < 5; i++) tmp[i] = st[j + i];
            for (int i = 0; i < 5; i++) {
                st[j + i] = tmp[i] ^ ((~tmp[(i + 1) % 5]) & tmp[(i + 2) % 5]);
            }
        }

        st[0] ^= rc[round];
    }
}

__device__ __forceinline__ void d_keccak_f1600(uint64_t st[25]) {
    keccak_f1600_core(st, d_keccak_rc);
}

__host__ inline void h_keccak_f1600(uint64_t st[25]) {
    keccak_f1600_core(st, h_keccak_rc);
}

// Host Keccak-512 (rate = 72 bytes = 9 uint64_t words, output = 64 bytes)
void h_keccak512(const uint8_t* input, size_t len, uint8_t output[64]) {
    uint64_t st[25] = {0};
    const size_t rate = 72;

    while (len >= rate) {
        for (size_t i = 0; i < rate / 8; i++) {
            uint64_t w;
            memcpy(&w, input + i * 8, 8);
            st[i] ^= w;
        }
        h_keccak_f1600(st);
        input += rate;
        len -= rate;
    }

    uint8_t pad[72] = {0};
    if (len > 0) memcpy(pad, input, len);
    pad[len] ^= 0x01;
    pad[rate - 1] ^= 0x80;

    for (size_t i = 0; i < rate / 8; i++) {
        uint64_t w;
        memcpy(&w, pad + i * 8, 8);
        st[i] ^= w;
    }
    h_keccak_f1600(st);
    memcpy(output, st, 64);
}

// Device Keccak-512 over a fixed 64-byte input (used in DAG item derivation)
__device__ void d_keccak512_64B(const uint64_t in_words[8], uint64_t out_words[8]) {
    uint64_t st[25] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        st[i] = in_words[i];
    }
    // Keccak padding for 64-byte message with 72-byte rate:
    // byte 64 gets 0x01, byte 71 gets 0x80 -> st[8] = 0x8000000000000001ULL
    st[8] = 0x8000000000000001ULL;
    d_keccak_f1600(st);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_words[i] = st[i];
    }
}

// Device Keccak-256 over a fixed 96-byte input (64-byte header/seed + 32-byte digest)
__device__ void d_keccak256_96B(const uint64_t in_words[12], uint64_t out_words[4]) {
    uint64_t st[25] = {0};
    // Rate for Keccak-256 is 136 bytes (17 uint64_t words)
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        st[i] = in_words[i];
    }
    // byte 96 gets 0x01, byte 135 gets 0x80
    st[12] = 0x0000000000000001ULL;
    st[16] = 0x8000000000000000ULL;
    d_keccak_f1600(st);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out_words[i] = st[i];
    }
}

__host__ __device__ __forceinline__ uint32_t fnv1a(uint32_t h, uint32_t d) {
    return (h ^ d) * 0x01000193U;
}

// ============================================================================
// 2. HOST LIGHT CACHE GENERATOR (Mersenne-like FNV Mixing)
// ============================================================================

void generate_light_cache(uint8_t* cache, uint32_t num_nodes, const uint8_t seed[32]) {
    // Node 0 = Keccak512(seed)
    h_keccak512(seed, 32, cache);

    // Sequential chain: Node[i] = Keccak512(Node[i - 1])
    for (uint32_t i = 1; i < num_nodes; i++) {
        h_keccak512(cache + (size_t)(i - 1) * 64, 64, cache + (size_t)i * 64);
    }

    // RandMemoHash rounds
    uint32_t* nodes32 = (uint32_t*)cache;
    for (int round = 0; round < CACHE_ROUNDS; round++) {
        for (uint32_t i = 0; i < num_nodes; i++) {
            uint32_t* cur = nodes32 + i * 16;
            uint32_t v = cur[0] % num_nodes;
            uint32_t prev_idx = (i == 0) ? (num_nodes - 1) : (i - 1);
            const uint32_t* prev = nodes32 + prev_idx * 16;
            const uint32_t* target = nodes32 + v * 16;

            uint8_t xored[64];
            uint32_t* xored32 = (uint32_t*)xored;
            for (int w = 0; w < 16; w++) {
                xored32[w] = prev[w] ^ target[w];
            }
            h_keccak512(xored, 64, (uint8_t*)cur);
        }
    }
}

// ============================================================================
// 3. GPU DAG GENERATION KERNEL (Produces 128-byte DAG Elements from Light Cache)
// ============================================================================

__global__ void generate_dag_kernel(
    const uint4* __restrict__ g_cache,   // Light cache in VRAM (each node = 4 x uint4 = 64B)
    uint4* __restrict__ g_dag,           // Full DAG in VRAM (each item = 8 x uint4 = 128B)
    uint32_t start_item,
    uint32_t total_items,
    uint32_t cache_nodes
) {
    uint32_t item_idx = blockIdx.x * blockDim.x + threadIdx.x + start_item;
    if (item_idx >= total_items) return;

    // Each 128-byte DAG item consists of two 64-byte halves
    for (uint32_t half = 0; half < 2; half++) {
        uint32_t node_index = item_idx * 2 + half;
        uint32_t init_idx = node_index % cache_nodes;

        uint64_t mix64[8];
        const uint4* init_ptr = g_cache + init_idx * 4;
        uint4 c0 = init_ptr[0];
        uint4 c1 = init_ptr[1];
        uint4 c2 = init_ptr[2];
        uint4 c3 = init_ptr[3];

        uint32_t* mix32 = (uint32_t*)mix64;
        mix32[0]  = c0.x ^ node_index; mix32[1]  = c0.y; mix32[2]  = c0.z; mix32[3]  = c0.w;
        mix32[4]  = c1.x;              mix32[5]  = c1.y; mix32[6]  = c1.z; mix32[7]  = c1.w;
        mix32[8]  = c2.x;              mix32[9]  = c2.y; mix32[10] = c2.z; mix32[11] = c2.w;
        mix32[12] = c3.x;              mix32[13] = c3.y; mix32[14] = c3.z; mix32[15] = c3.w;

        d_keccak512_64B(mix64, mix64);

        for (uint32_t p = 0; p < DAG_PARENTS; p++) {
            uint32_t parent_idx = fnv1a(node_index ^ p, mix32[p & 15]) % cache_nodes;
            const uint4* p_ptr = g_cache + parent_idx * 4;
            uint4 p0 = p_ptr[0];
            uint4 p1 = p_ptr[1];
            uint4 p2 = p_ptr[2];
            uint4 p3 = p_ptr[3];

            mix32[0]  = fnv1a(mix32[0],  p0.x); mix32[1]  = fnv1a(mix32[1],  p0.y);
            mix32[2]  = fnv1a(mix32[2],  p0.z); mix32[3]  = fnv1a(mix32[3],  p0.w);
            mix32[4]  = fnv1a(mix32[4],  p1.x); mix32[5]  = fnv1a(mix32[5],  p1.y);
            mix32[6]  = fnv1a(mix32[6],  p1.z); mix32[7]  = fnv1a(mix32[7],  p1.w);
            mix32[8]  = fnv1a(mix32[8],  p2.x); mix32[9]  = fnv1a(mix32[9],  p2.y);
            mix32[10] = fnv1a(mix32[10], p2.z); mix32[11] = fnv1a(mix32[11], p2.w);
            mix32[12] = fnv1a(mix32[12], p3.x); mix32[13] = fnv1a(mix32[13], p3.y);
            mix32[14] = fnv1a(mix32[14], p3.z); mix32[15] = fnv1a(mix32[15], p3.w);
        }

        d_keccak512_64B(mix64, mix64);

        uint4* dst = g_dag + (size_t)item_idx * 8 + half * 4;
        dst[0] = make_uint4(mix32[0],  mix32[1],  mix32[2],  mix32[3]);
        dst[1] = make_uint4(mix32[4],  mix32[5],  mix32[6],  mix32[7]);
        dst[2] = make_uint4(mix32[8],  mix32[9],  mix32[10], mix32[11]);
        dst[3] = make_uint4(mix32[12], mix32[13], mix32[14], mix32[15]);
    }
}

// ============================================================================
// 4. OCTOPUS SEARCH KERNEL (Real Keccak-512 Init + Warp Mixing + Keccak-256 Out)
// ============================================================================

__global__ void octopus_search_kernel(
    const uint4* __restrict__ g_dag,     // Full DAG in VRAM (128B items = 8 x uint4)
    uint32_t* __restrict__ g_output,     // [0] = count, [1..16] = winning nonce offsets
    const hash256_t header_hash,
    const uint64_t start_nonce,
    const uint64_t target,
    const uint32_t dag_items             // Number of 128-byte items (grouped in 4-item 512B chunks)
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & (OCTOPUS_LANES - 1);
    const uint32_t warp_id = tid / OCTOPUS_LANES;
    const uint64_t nonce = start_nonce + warp_id;

    // 1. Real Keccak-512 Seed Initialization: Keccak512(header_hash (32B) || nonce (8B))
    uint64_t init_seed[8] = {0};
    uint64_t st[25] = {0};

    st[0] = ((uint64_t)header_hash.words[1] << 32) | header_hash.words[0];
    st[1] = ((uint64_t)header_hash.words[3] << 32) | header_hash.words[2];
    st[2] = ((uint64_t)header_hash.words[5] << 32) | header_hash.words[4];
    st[3] = ((uint64_t)header_hash.words[7] << 32) | header_hash.words[6];
    st[4] = nonce;
    // Keccak-512 padding for 40-byte input (rate = 72 bytes):
    // byte 40 gets 0x01, byte 71 gets 0x80
    st[5] = 0x0000000000000001ULL;
    st[8] = 0x8000000000000000ULL;
    d_keccak_f1600(st);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        init_seed[i] = st[i];
    }

    // Expand the 64-byte Keccak-512 seed across the 32 warp lanes (16 registers per lane)
    const uint32_t* seed32 = (const uint32_t*)init_seed;
    uint32_t mix[OCTOPUS_REGS];
    uint32_t lane_state = fnv1a(seed32[lane_id & 15], lane_id);

    #pragma unroll
    for (int r = 0; r < OCTOPUS_REGS; r++) {
        lane_state = fnv1a(lane_state, seed32[(r + lane_id) & 15]);
        mix[r] = lane_state;
    }

    // Number of 512-byte warp-aligned blocks in the DAG
    const uint32_t dag_chunks = dag_items / 4;

    // 2. Main Memory-Hard DAG Mixing Loop (64 iterations)
    #pragma unroll 1
    for (uint32_t i = 0; i < OCTOPUS_CNT_DAG; i++) {
        uint32_t src_lane = i % OCTOPUS_LANES;
        uint32_t chunk_idx = __shfl_sync(0xFFFFFFFF, mix[i % OCTOPUS_REGS], src_lane);
        chunk_idx %= dag_chunks;

        // 32 lanes * 16 bytes (uint4) = 512 bytes (4 x 128-byte DAG items) per warp fetch
        uint4 dag_data = g_dag[(size_t)chunk_idx * OCTOPUS_LANES + lane_id];

        mix[0] = fnv1a(mix[0], dag_data.x);
        mix[1] = fnv1a(mix[1], dag_data.y);
        mix[2] = fnv1a(mix[2], dag_data.z);
        mix[3] = fnv1a(mix[3], dag_data.w);

        #pragma unroll
        for (int r = 4; r < OCTOPUS_REGS; r++) {
            uint32_t peer = __shfl_xor_sync(0xFFFFFFFF, mix[r - 4], (r & 31));
            mix[r] = fnv1a(mix[r], peer ^ __funnelshift_l(mix[r - 1], mix[r - 2], 13));
        }
    }

    // 3. Compress 16 registers per lane into a single 32-bit word per lane
    uint32_t lane_digest = 0x811c9dc5U;
    #pragma unroll
    for (int r = 0; r < OCTOPUS_REGS; r++) {
        lane_digest = fnv1a(lane_digest, mix[r]);
    }

    // Gather 8 words across lanes 0..7 to form the 32-byte compressed mix digest
    uint32_t mix_digest[8];
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint32_t val = __shfl_sync(0xFFFFFFFF, lane_digest, w * 4);
        mix_digest[w] = val;
    }

    // 4. Final Keccak-256(init_seed (64B) || mix_digest (32B)) and Target Comparison
    if (lane_id == 0) {
        uint64_t final_in[12];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            final_in[i] = init_seed[i];
        }
        final_in[8]  = ((uint64_t)mix_digest[1] << 32) | mix_digest[0];
        final_in[9]  = ((uint64_t)mix_digest[3] << 32) | mix_digest[2];
        final_in[10] = ((uint64_t)mix_digest[5] << 32) | mix_digest[4];
        final_in[11] = ((uint64_t)mix_digest[7] << 32) | mix_digest[6];

        uint64_t final_hash[4];
        d_keccak256_96B(final_in, final_hash);

        // Compare top 64 bits (big-endian target convention via byte swap)
        uint64_t result_hi = __byte_perm((uint32_t)(final_hash[0] >> 32), 0, 0x0123);
        uint64_t result_lo = __byte_perm((uint32_t)(final_hash[0] & 0xFFFFFFFFULL), 0, 0x0123);
        uint64_t score = (result_lo << 32) | result_hi;

        if (score <= target) {
            uint32_t slot = atomicAdd(&g_output[0], 1);
            if (slot < 16) {
                g_output[slot + 1] = (uint32_t)(nonce - start_nonce);
            }
        }
    }
}

// ============================================================================
// 5. STANDALONE HOST LAUNCHER
// ============================================================================

int main(void) {
    // Use compact test sizes so this runs quickly on any GPU;
    // Production Conflux Epoch 0 uses ~16 MB light cache and ~4 GB DAG.
    const uint32_t cache_nodes = 16384;          // 1 MB Light Cache (16384 * 64B)
    const uint32_t dag_items   = 262144;         // 32 MB DAG (262144 * 128B)

    const size_t cache_bytes = (size_t)cache_nodes * 64;
    const size_t dag_bytes   = (size_t)dag_items * 128;

    uint8_t epoch_seed[32] = {
        0x43, 0x6f, 0x6e, 0x66, 0x6c, 0x75, 0x78, 0x20,
        0x4f, 0x63, 0x74, 0x6f, 0x70, 0x75, 0x73, 0x20,
        0x45, 0x70, 0x6f, 0x63, 0x68, 0x20, 0x53, 0x65,
        0x65, 0x64, 0x20, 0x30, 0x30, 0x30, 0x30, 0x31
    };

    printf("[1/4] Generating %zu KB Light Cache on Host...\n", cache_bytes / 1024);
    uint8_t* h_cache = (uint8_t*)malloc(cache_bytes);
    if (!h_cache) {
        fprintf(stderr, "Failed to allocate host cache memory\n");
        return EXIT_FAILURE;
    }
    generate_light_cache(h_cache, cache_nodes, epoch_seed);

    uint4* d_cache = NULL;
    uint4* d_dag   = NULL;
    uint32_t* d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_dag, dag_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, 17 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache, cache_bytes, cudaMemcpyHostToDevice));
    free(h_cache);

    printf("[2/4] Building %zu MB DAG on GPU...\n", dag_bytes / (1024 * 1024));
    const uint32_t dag_threads = 256;
    const uint32_t dag_blocks  = (dag_items + dag_threads - 1) / dag_threads;
    generate_dag_kernel<<<dag_blocks, dag_threads>>>(d_cache, d_dag, 0, dag_items, cache_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_cache));

    // Set up sample block header and an easy target to verify nonce discovery
    hash256_t header_hash = {{
        0x12345678U, 0x9abcdef0U, 0x0fedcba9U, 0x87654321U,
        0xdeadbeefU, 0xcafebabeU, 0x01020304U, 0x05060708U
    }};
    uint64_t start_nonce = 0x0000000100000000ULL;
    uint64_t target      = 0x0000FFFFFFFFFFFFULL; // ~1 in 65,536 difficulty for quick test

    const uint32_t search_threads = 256;
    const uint32_t warps_per_block = search_threads / OCTOPUS_LANES;
    const uint32_t total_nonces = 131072; // 131,072 warps = 4,194,304 CUDA threads
    const uint32_t search_blocks = (total_nonces + warps_per_block - 1) / warps_per_block;

    CUDA_CHECK(cudaMemset(d_output, 0, 17 * sizeof(uint32_t)));

    printf("[3/4] Launching Octopus Search Kernel (%u nonces)...\n", total_nonces);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    octopus_search_kernel<<<search_blocks, search_threads>>>(
        d_dag, d_output, header_hash, start_nonce, target, dag_items
    );
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    uint32_t h_output[17] = {0};
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 17 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double hashrate_mhs = (total_nonces / (ms / 1000.0)) / 1e6;
    printf("[4/4] Done in %.2f ms (%.2f MH/s). Found %u valid nonces:\n",
           ms, hashrate_mhs, h_output[0]);

    uint32_t reported = (h_output[0] < 16) ? h_output[0] : 16;
    for (uint32_t i = 0; i < reported; i++) {
        printf("  -> Nonce: 0x%016llx (offset +%u)\n",
               (unsigned long long)(start_nonce + h_output[i + 1]), h_output[i + 1]);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_dag));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
