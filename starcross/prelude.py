"""OpenCL-side compatibility prelude emitted at the top of every converted file.

It maps CUDA builtins, intrinsics, atomics and libm-style names onto OpenCL C /
C++ for OpenCL equivalents so the converted kernel bodies can stay close to the
original source.
"""

PRELUDE = r"""
/* ===================== starcross CUDA compatibility prelude ===================== */
#define __STARCROSS__ 1
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 700   /* device-side compilation; picks CUDA's device code paths */
#endif

#ifdef __cplusplus
#define SC_OVL
#else
#define SC_OVL __attribute__((overloadable))
#endif
#define SC_FN static inline __attribute__((always_inline))

/* ---- fixed width integers (OpenCL has no <stdint.h>) ---- */
typedef char           int8_t;
typedef uchar          uint8_t;
typedef short          int16_t;
typedef ushort         uint16_t;
typedef int            int32_t;
typedef uint           uint32_t;
typedef long           int64_t;
typedef ulong          uint64_t;
#define INT8_MIN   (-128)
#define INT8_MAX   127
#define UINT8_MAX  255
#define INT16_MIN  (-32768)
#define INT16_MAX  32767
#define UINT16_MAX 65535
#define INT32_MIN  (-2147483647 - 1)
#define INT32_MAX  2147483647
#define UINT32_MAX 0xffffffffU
#define INT64_MIN  (-9223372036854775807L - 1)
#define INT64_MAX  9223372036854775807L
#define UINT64_MAX 0xffffffffffffffffUL

/* ---- CUDA vector type aliases / constructors ---- */
typedef long2  longlong2;
typedef long3  longlong3;
typedef long4  longlong4;
typedef ulong2 ulonglong2;
typedef ulong3 ulonglong3;
typedef ulong4 ulonglong4;
#define SC_MAKE(T) \
  SC_FN T##2 make_##T##2(T a, T b) { return (T##2)(a, b); } \
  SC_FN T##3 make_##T##3(T a, T b, T c) { return (T##3)(a, b, c); } \
  SC_FN T##4 make_##T##4(T a, T b, T c, T d) { return (T##4)(a, b, c, d); }
SC_MAKE(char) SC_MAKE(uchar) SC_MAKE(short) SC_MAKE(ushort)
SC_MAKE(int) SC_MAKE(uint) SC_MAKE(long) SC_MAKE(ulong) SC_MAKE(float)
#define make_longlong2  make_long2
#define make_longlong4  make_long4
#define make_ulonglong2 make_ulong2
#define make_ulonglong4 make_ulong4

/* ---- thread indexing (bare uses; ".x/.y/.z" uses are rewritten inline) ---- */
#define threadIdx ((uint3)((uint)get_local_id(0),  (uint)get_local_id(1),  (uint)get_local_id(2)))
#define blockIdx  ((uint3)((uint)get_group_id(0),  (uint)get_group_id(1),  (uint)get_group_id(2)))
#define blockDim  ((uint3)((uint)get_local_size(0),(uint)get_local_size(1),(uint)get_local_size(2)))
#define gridDim   ((uint3)((uint)get_num_groups(0),(uint)get_num_groups(1),(uint)get_num_groups(2)))
#define warpSize  32   /* assumption: real subgroup size is device dependent */

/* ---- synchronisation ---- */
#define __syncthreads()        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define __threadfence()        mem_fence(CLK_GLOBAL_MEM_FENCE)
#define __threadfence_block()  mem_fence(CLK_LOCAL_MEM_FENCE)
#define __threadfence_system() mem_fence(CLK_GLOBAL_MEM_FENCE)
#if defined(__opencl_c_work_group_collective_functions)
#define __syncthreads_count(p) work_group_reduce_add((int)((p) != 0))
#define __syncthreads_and(p)   work_group_all((p) != 0)
#define __syncthreads_or(p)    work_group_any((p) != 0)
#endif
#if defined(__opencl_c_subgroups) || defined(cl_khr_subgroups)
#define __syncwarp(...)        sub_group_barrier(CLK_LOCAL_MEM_FENCE)
#define __all_sync(m, p)       sub_group_all((p) != 0)
#define __any_sync(m, p)       sub_group_any((p) != 0)
#define __laneid()             ((uint)get_sub_group_local_id())
#endif
#if defined(cl_khr_subgroup_shuffle)
#define __shfl_sync(m, v, l, ...)      sub_group_shuffle((v), (l))
#define __shfl_xor_sync(m, v, l, ...)  sub_group_shuffle_xor((v), (l))
#endif
#if defined(cl_khr_subgroup_shuffle_relative)
#define __shfl_up_sync(m, v, d, ...)   sub_group_shuffle_up((v), (d))
#define __shfl_down_sync(m, v, d, ...) sub_group_shuffle_down((v), (d))
#endif
#if defined(cl_khr_subgroup_ballot)
#define __ballot_sync(m, p)    (sub_group_ballot((p) != 0).x)  /* lanes 0..31 only */
#endif

/* ---- misc ---- */
#define __ldg(p)       (*(p))
#define __trap()       ((void)0)
#define __brkpt()      ((void)0)
#ifndef assert
#define assert(x)      ((void)0)
#endif

/* ---- byte memory helpers (no libc on the device) ---- */
SC_FN SC_OVL void *sc_memcpy(void *d, const void *s, size_t n) {
  uchar *dd = (uchar *)d; const uchar *ss = (const uchar *)s;
  for (size_t i = 0; i < n; ++i) dd[i] = ss[i];
  return d;
}
SC_FN SC_OVL void *sc_memcpy(void *d, __constant void *s, size_t n) {
  uchar *dd = (uchar *)d; __constant uchar *ss = (__constant uchar *)s;
  for (size_t i = 0; i < n; ++i) dd[i] = ss[i];
  return d;
}
SC_FN void *sc_memset(void *d, int v, size_t n) {
  uchar *dd = (uchar *)d;
  for (size_t i = 0; i < n; ++i) dd[i] = (uchar)v;
  return d;
}
SC_FN SC_OVL int sc_memcmp(const void *a, const void *b, size_t n) {
  const uchar *x = (const uchar *)a, *y = (const uchar *)b;
  for (size_t i = 0; i < n; ++i) if (x[i] != y[i]) return (int)x[i] - (int)y[i];
  return 0;
}

/* ---- integer intrinsics ---- */
SC_FN int  __popc(uint x)    { return (int)popcount(x); }
SC_FN int  __popcll(ulong x) { return (int)popcount(x); }
SC_FN int  __clz(int x)      { return (int)clz(x); }
SC_FN int  __clzll(long x)   { return (int)clz(x); }
SC_FN int  __ffs(int x)      { return x == 0 ? 0 : (int)popcount((uint)((x & -x) - 1)) + 1; }
SC_FN int  __ffsll(long x)   { return x == 0 ? 0 : (int)popcount((ulong)((x & -x) - 1)) + 1; }
SC_FN uint __brev(uint x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}
SC_FN ulong __brevll(ulong x) { return ((ulong)__brev((uint)x) << 32) | (ulong)__brev((uint)(x >> 32)); }
SC_FN uint __byte_perm(uint x, uint y, uint s) {
  ulong v = ((ulong)y << 32) | (ulong)x;
  uint r = 0;
  for (int i = 0; i < 4; ++i) r |= (uint)((v >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
  return r;
}
SC_FN uint __funnelshift_l(uint lo, uint hi, uint sh)  { return (uint)(((((ulong)hi << 32) | lo) << (sh & 31u)) >> 32); }
SC_FN uint __funnelshift_lc(uint lo, uint hi, uint sh) { sh = min(sh, 32u); return (uint)(((((ulong)hi << 32) | lo) << sh) >> 32); }
SC_FN uint __funnelshift_r(uint lo, uint hi, uint sh)  { return (uint)((((ulong)hi << 32) | lo) >> (sh & 31u)); }
SC_FN uint __funnelshift_rc(uint lo, uint hi, uint sh) { sh = min(sh, 32u); return (uint)((((ulong)hi << 32) | lo) >> sh); }
SC_FN uint  __umulhi(uint a, uint b)     { return mul_hi(a, b); }
SC_FN int   __mulhi(int a, int b)        { return mul_hi(a, b); }
SC_FN ulong __umul64hi(ulong a, ulong b) { return mul_hi(a, b); }
SC_FN long  __mul64hi(long a, long b)    { return mul_hi(a, b); }
SC_FN int   __mul24(int a, int b)        { return mul24(a, b); }
SC_FN uint  __umul24(uint a, uint b)     { return mul24(a, b); }
SC_FN uint  __sad(int x, int y, uint z)  { return z + abs_diff(x, y); }
SC_FN uint  __usad(uint x, uint y, uint z) { return z + abs_diff(x, y); }
SC_FN int   __hadd(int a, int b)         { return hadd(a, b); }
SC_FN int   __rhadd(int a, int b)        { return rhadd(a, b); }
SC_FN uint  __uhadd(uint a, uint b)      { return hadd(a, b); }
SC_FN uint  __urhadd(uint a, uint b)     { return rhadd(a, b); }
#define __float_as_uint(x) as_uint(x)
#define __uint_as_float(x) as_float(x)
#define __float_as_int(x)  as_int(x)
#define __int_as_float(x)  as_float(x)
#if defined(cl_khr_fp64) || defined(__opencl_c_fp64)
#define __double_as_longlong(x) as_long(x)
#define __longlong_as_double(x) as_double(x)
#endif
#define llabs(x) abs(x)
#define llmin(a, b) min((long)(a), (long)(b))
#define llmax(a, b) max((long)(a), (long)(b))
#define umin(a, b)  min((uint)(a), (uint)(b))
#define umax(a, b)  max((uint)(a), (uint)(b))

/* ---- float math: CUDA "f" names -> OpenCL overloaded builtins ---- */
#define SC_F1(cu, cl) SC_FN float cu(float x) { return cl(x); }
#define SC_F2(cu, cl) SC_FN float cu(float x, float y) { return cl(x, y); }
SC_F1(sqrtf, sqrt) SC_F1(cbrtf, cbrt) SC_F1(rsqrtf, rsqrt) SC_F1(expf, exp) SC_F1(exp2f, exp2)
SC_F1(exp10f, exp10) SC_F1(expm1f, expm1) SC_F1(logf, log) SC_F1(log2f, log2) SC_F1(log10f, log10)
SC_F1(log1pf, log1p) SC_F1(sinf, sin) SC_F1(cosf, cos) SC_F1(tanf, tan) SC_F1(asinf, asin)
SC_F1(acosf, acos) SC_F1(atanf, atan) SC_F1(sinhf, sinh) SC_F1(coshf, cosh) SC_F1(tanhf, tanh)
SC_F1(asinhf, asinh) SC_F1(acoshf, acosh) SC_F1(atanhf, atanh) SC_F1(floorf, floor) SC_F1(ceilf, ceil)
SC_F1(roundf, round) SC_F1(truncf, trunc) SC_F1(fabsf, fabs) SC_F1(rintf, rint) SC_F1(erff, erf)
SC_F1(erfcf, erfc) SC_F1(tgammaf, tgamma) SC_F1(lgammaf, lgamma) SC_F1(logbf, logb)
SC_F1(sinpif, sinpi) SC_F1(cospif, cospi) SC_F1(nearbyintf, rint)
SC_F2(powf, pow) SC_F2(atan2f, atan2) SC_F2(fminf, fmin) SC_F2(fmaxf, fmax) SC_F2(fmodf, fmod)
SC_F2(hypotf, hypot) SC_F2(copysignf, copysign) SC_F2(remainderf, remainder)
SC_F2(nextafterf, nextafter) SC_F2(fdimf, fdim)
SC_FN float fmaf(float a, float b, float c) { return fma(a, b, c); }
SC_FN float ldexpf(float x, int e) { return ldexp(x, e); }
SC_FN float rcbrtf(float x) { return 1.0f / cbrt(x); }
SC_FN void  sincosf(float x, float *s, float *c) { *s = sin(x); *c = cos(x); }
SC_FN void  sincospif(float x, float *s, float *c) { *s = sinpi(x); *c = cospi(x); }
SC_FN float __saturatef(float x) { return clamp(x, 0.0f, 1.0f); }
SC_FN float __fdividef(float a, float b) { return native_divide(a, b); }
SC_FN float __expf(float x)   { return native_exp(x); }
SC_FN float __exp10f(float x) { return native_exp10(x); }
SC_FN float __logf(float x)   { return native_log(x); }
SC_FN float __log2f(float x)  { return native_log2(x); }
SC_FN float __log10f(float x) { return native_log10(x); }
SC_FN float __sinf(float x)   { return native_sin(x); }
SC_FN float __cosf(float x)   { return native_cos(x); }
SC_FN float __tanf(float x)   { return native_tan(x); }
SC_FN float __powf(float x, float y) { return native_powr(x, y); }
SC_FN float __frcp_rn(float x)   { return 1.0f / x; }
SC_FN float __fsqrt_rn(float x)  { return sqrt(x); }
SC_FN float __frsqrt_rn(float x) { return rsqrt(x); }
SC_FN float __fadd_rn(float a, float b) { return a + b; }
SC_FN float __fmul_rn(float a, float b) { return a * b; }
SC_FN float __fmaf_rn(float a, float b, float c) { return fma(a, b, c); }

/* ---- atomics (32-bit int/uint, plus float add via CAS) ---- */
#define SC_ATOM_AS(AS, T)                                                                      \
  SC_FN SC_OVL T atomicAdd(volatile AS T *p, T v)  { return atomic_add(p, v); }              \
  SC_FN SC_OVL T atomicSub(volatile AS T *p, T v)  { return atomic_sub(p, v); }              \
  SC_FN SC_OVL T atomicExch(volatile AS T *p, T v) { return atomic_xchg(p, v); }             \
  SC_FN SC_OVL T atomicMin(volatile AS T *p, T v)  { return atomic_min(p, v); }              \
  SC_FN SC_OVL T atomicMax(volatile AS T *p, T v)  { return atomic_max(p, v); }              \
  SC_FN SC_OVL T atomicAnd(volatile AS T *p, T v)  { return atomic_and(p, v); }              \
  SC_FN SC_OVL T atomicOr(volatile AS T *p, T v)   { return atomic_or(p, v); }               \
  SC_FN SC_OVL T atomicXor(volatile AS T *p, T v)  { return atomic_xor(p, v); }              \
  SC_FN SC_OVL T atomicCAS(volatile AS T *p, T c, T v) { return atomic_cmpxchg(p, c, v); }
SC_ATOM_AS(__global, int)  SC_ATOM_AS(__global, uint)
SC_ATOM_AS(__local, int)   SC_ATOM_AS(__local, uint)
#define SC_ATOM_INCDEC(AS)                                                                     \
  SC_FN SC_OVL uint atomicInc(volatile AS uint *p, uint lim) {                                 \
    uint old = *p, assumed;                                                                    \
    do { assumed = old; old = atomic_cmpxchg(p, assumed, (assumed >= lim) ? 0u : assumed + 1u); } \
    while (old != assumed);                                                                    \
    return old;                                                                                \
  }                                                                                            \
  SC_FN SC_OVL uint atomicDec(volatile AS uint *p, uint lim) {                                 \
    uint old = *p, assumed;                                                                    \
    do { assumed = old;                                                                        \
         old = atomic_cmpxchg(p, assumed, (assumed == 0u || assumed > lim) ? lim : assumed - 1u); } \
    while (old != assumed);                                                                    \
    return old;                                                                                \
  }                                                                                            \
  SC_FN SC_OVL float atomicAdd(volatile AS float *p, float v) {                                \
    volatile AS uint *u = (volatile AS uint *)p;                                               \
    uint old = *u, assumed;                                                                    \
    do { assumed = old; old = atomic_cmpxchg(u, assumed, as_uint(as_float(assumed) + v)); }    \
    while (old != assumed);                                                                    \
    return as_float(old);                                                                      \
  }                                                                                            \
  SC_FN SC_OVL float atomicExch(volatile AS float *p, float v) { return atomic_xchg(p, v); }
SC_ATOM_INCDEC(__global) SC_ATOM_INCDEC(__local)
/* NOTE: in OpenCL C (not C++) atomics need a pointer that is visibly __global or __local. */
/* =================================================================================== */
"""
