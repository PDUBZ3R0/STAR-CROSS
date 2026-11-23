// Block reduction with static + dynamic shared memory and a global atomic.
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_safe_add(float a, float b) { return a + b; }

__global__ void block_sum(const float* in, float* out, unsigned int n) {
    __shared__ float partial[128];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float v = 0.0f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v = warp_safe_add(v, in[i + blockDim.x]);
    partial[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, partial[0]);
}

// histogram using dynamically sized shared memory
__global__ void histogram(const unsigned char* data, unsigned int* bins, int n, int nbins) {
    extern __shared__ unsigned int local_bins[];
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) local_bins[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local_bins[data[i] % nbins], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) atomicAdd(&bins[b], local_bins[b]);
}

void launch(const float* in, float* out, unsigned int n, const unsigned char* d, unsigned int* bins) {
    block_sum<<<(n + 255) / 256, 128>>>(in, out, n);
    histogram<<<64, 256, 64 * sizeof(unsigned int)>>>(d, bins, (int)n, 64);
}
