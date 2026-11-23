// 64-bit hashing with a __constant__ table passed to a helper (needs the
// __constant address-space fixup) -- GLSL works, WGSL will refuse u64.
#include <stdint.h>
#include <string.h>

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

__device__ inline uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

__device__ uint64_t mix(const uint64_t* table, int round, uint64_t v) {
    return rotl64(v ^ table[round % 24], 7);
}

__global__ void hash_kernel(const uint64_t* in, uint64_t* out, size_t count) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t state[4];
    memset(state, 0, sizeof(state));
    state[0] = in[i];
    for (int r = 0; r < 24; ++r)
        state[r & 3] = mix(RC, r, state[(r + 1) & 3]);
    out[i] = state[0] ^ state[1] ^ state[2] ^ state[3];
}

void run(const uint64_t* d_in, uint64_t* d_out, size_t n) {
    hash_kernel<<<(unsigned)((n + 63) / 64), 64>>>(d_in, d_out, n);
}
