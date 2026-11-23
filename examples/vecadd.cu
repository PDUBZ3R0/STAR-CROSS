// Classic vector add: the "hello world" for the pipeline.
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK 256

__global__ void vecadd(const float* __restrict__ a, const float* __restrict__ b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    vecadd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, c, n);
    cudaDeviceSynchronize();
    printf("done\n");
    return 0;
}
