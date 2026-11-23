// C++ features: structs with __device__ methods, references, templates.
#include <cuda_runtime.h>
#include <vector>

struct Particle {
    float3 pos;
    float3 vel;
    __host__ __device__ void step(float dt) {
        pos.x += vel.x * dt; pos.y += vel.y * dt; pos.z += vel.z * dt;
    }
    void print_host() const;   // host only, dropped
};

template <typename T>
__device__ T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

__device__ void bounce(Particle& p, float lo, float hi) {
    if (p.pos.y < lo || p.pos.y > hi) p.vel.y = -p.vel.y;
    p.pos.y = clampv(p.pos.y, lo, hi);
}

__global__ void simulate(Particle* ps, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle p = ps[i];
    p.step(dt);
    bounce(p, 0.0f, 100.0f);
    ps[i] = p;
}

void host_side(std::vector<Particle>& v) {
    Particle* d = nullptr;
    simulate<<<dim3((unsigned)v.size() / 128 + 1), dim3(128)>>>(d, (int)v.size(), 0.016f);
}
