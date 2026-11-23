# STARCROSS

Convert CUDA kernels to **OpenCL**, then to **GLSL** compute shaders and (where possible) **WGSL**.

```
 .cu ──cu2cl──▶ .cl / .clcpp ──clspv──▶ Vulkan SPIR-V ──spirv-cross──▶ GLSL (.comp, one per kernel)
                                                        └──naga/tint───▶ WGSL
```

- **cu2cl** (this repo, pure Python) rewrites CUDA device code into OpenCL C 3.0, or C++ for OpenCL 2021
  when the source uses C++ features. It drops host code and writes a manifest describing every kernel's
  arguments and every `<<<grid, block>>>` launch it found.
- **clspv** (Google) compiles OpenCL to *Vulkan-style* (Shader model) SPIR-V. That's the SPIR-V
  SPIRV-Cross and naga can read. chipStar produces OpenCL-style (Kernel model) SPIR-V, which they can't.
- **spirv-cross** writes GLSL. **naga** (or tint) writes WGSL.

You can enter the pipeline at either end: pass a `.cu` to convert everything, or a hand-written or
hand-fixed `.cl` to run only the second half.

## Quick start

```bash
# uses podman if installed, else docker (override with STARCROSS_ENGINE=docker).
# The image (ghcr.io/pdubz3r0/starcross-toolchain) is pulled on first use. To build it yourself
# instead (compiles clspv + LLVM: slow, once):
./bin/starcross-container build

# full pipeline (the current directory is mounted at /work: use relative paths)
./bin/starcross-container convert examples/reduce.cu
#   -> reduce.starcross/reduce.cl, reduce.manifest.json, reduce.spv,
#      glsl/block_sum.comp, glsl/histogram.comp, reduce.wgsl, reduce.backend.json

# just CUDA -> OpenCL (needs only python3; clang optional but recommended)
./bin/starcross cu2cl examples/keccak.cu -o out/

# just OpenCL -> GLSL/WGSL
./bin/starcross cl2x out/keccak.cl --to glsl

# what's installed?
./bin/starcross doctor
```

Common options:

| option | what it does |
|---|---|
| `-k NAME` | convert only this kernel (repeatable). Everything it doesn't reach is dropped |
| `--to cl,spv,glsl,wgsl` | choose outputs (`convert` default: `cl,glsl,wgsl`) |
| `--std c\|cpp` | force OpenCL C or C++ for OpenCL (default: auto-detect) |
| `--wgsize K=256,1,1` | fix a kernel's work-group size (otherwise inferred from literal launches) |
| `--glsl-version 310 --es` | GLSL ES output |
| `--vulkan-glsl` | Vulkan-flavoured GLSL (keeps descriptor sets) |
| `--clspv-arg ARG` | pass extra flags to clspv, e.g. `--clspv-arg=-pod-pushconstant` |
| `--line-directives` | emit `#line` so clspv errors point at the `.cu` lines |
| `--force` | keep going after conversion errors |
| `-v` | show every external command |

## What cu2cl handles

- **Pruning:** local `#include "..."` files are inlined and system/CUDA headers dropped. Only the kernels
  you asked for, plus everything they reach (functions, structs, typedefs, `__constant__` tables, macros),
  is emitted. Host functions and host-only struct methods are removed.
- **Qualifiers:** `__global__`→`__kernel`, `__shared__`→`__local`, `__constant__`→`__constant`;
  `__device__`/`__host__` removed; `__forceinline__`, `__align__`, `__restrict__` and `__launch_bounds__`
  translated or removed.
- **Kernel signatures:** pointer args become `__global`. `size_t`/`bool` args (not legal in OpenCL)
  become `ulong`/`int`. `extern __shared__ T x[]` becomes an extra `__local T *x` argument.
- **Builtins:** `threadIdx/blockIdx/blockDim/gridDim`, `__syncthreads`, fences, `memcpy/memset`,
  CUDA math (`sqrtf`, `__expf`, ...), integer intrinsics (`__popc`, `__byte_perm`, `__funnelshift_*`,
  `__brev`, `__umulhi`, ...), 32-bit atomics (including CUDA's `atomicInc`/`atomicDec` wrap semantics and
  float `atomicAdd` via CAS), and warp shuffles/votes mapped to subgroups. These live in a prelude at
  the top of the output.
- **Array parameters:** `T a[16]` helper parameters become pointers. OpenCL would otherwise make them
  `__private`-only.
- **`__constant` address-space fixups:** CUDA code passes `__constant__` tables to helpers that take
  plain pointers, but OpenCL forbids converting `__constant` to generic pointers. cu2cl compiles the
  output with clang, then clones the offending helper with a `__constant` parameter, or qualifies the
  local pointer. It repeats until clang is happy.
- **Work-group size:** if every launch of a kernel uses a literal block size (`256`, `BLOCK`,
  `dim3(16,16)`), the kernel gets `reqd_work_group_size`. That turns into a fixed `local_size_x` in GLSL
  and `@workgroup_size` in WGSL.
- **Language choice:** templates, references, classes, methods, overloads or bare struct tags select
  **C++ for OpenCL** (`.clcpp`). Otherwise it emits plain **OpenCL C** (`.cl`).

## What it refuses (with a pointer to the line)

| CUDA feature | why / what to do |
|---|---|
| kernel launches inside kernels (dynamic parallelism) | no equivalent. Flatten the recursion into host-side launches |
| device `malloc`/`free`/`new` | allocate on the host and pass a buffer argument |
| inline PTX `asm()` | rewrite in C |
| textures / surfaces | port to buffers or images by hand |
| cooperative groups, `__activemask`, `__match_*` | no equivalent |
| mutable `__device__` globals | clspv/Vulkan have no program-scope globals: pass a buffer (or `--device-globals constant` if read-only) |
| references / pointer-to-pointer kernel args, templated kernels | not legal OpenCL entry points |

It also warns about things that compile but behave differently: `float3` is 16 bytes in
OpenCL/GLSL (12 in CUDA), warps vs subgroups (subgroup size isn't always 32), and device `printf`.

## WGSL caveats

WGSL has **no 64-bit integers, no 8/16-bit integers and no doubles**. After clspv runs, starcross
reads the SPIR-V capabilities and tells you exactly which one blocks WGSL (`*.backend.json` has the full
list). Hash and crypto kernels such as BLAKE3, Argon2 and Keccak are full of `uint64_t`. GLSL handles
them via `GL_ARB_gpu_shader_int64`, but for WGSL the 64-bit arithmetic has to be rewritten as
`vec2<u32>` by hand.
`-int8=0` is passed to clspv automatically when WGSL is requested.

## Outputs

| file | contents |
|---|---|
| `NAME.cl` / `NAME.clcpp` | converted OpenCL, with `// from file.cu:LINE` markers |
| `NAME.manifest.json` | kernels (params, kinds, inferred work-group size), host launch sites (grid/block/shared/args), features used, diagnostics |
| `NAME.spv` | Vulkan SPIR-V from clspv |
| `NAME.backend.json` | capabilities, entry points, **per-kernel descriptor bindings** (from clspv's reflection), warnings |
| `NAME.reflect.json` | spirv-cross reflection |
| `glsl/KERNEL.comp` | one GLSL compute shader per kernel |
| `NAME.wgsl` | WGSL module (all kernels as entry points) |

The manifest and backend reports together give you what you need to write the Vulkan, WebGPU or GL
host side: which buffer goes in which binding, the POD argument layout, and the original launch
dimensions.

## Example: RinHash

```bash
starcross convert ../RinHash-cuda/rinhash.cu -k rinhash_cuda_kernel
```

It converts all ~1600 lines of BLAKE3, Argon2d and SHA3 device code to C++ for OpenCL. It
auto-fixes the `__constant` table issues (`g_IV`, `blake2b_sigma`), then stops on one real problem:
`rinhash_cuda_kernel` calls `malloc` for the Argon2 memory. Change the kernel to take that buffer as an
argument and re-run. GLSL should then work with `GL_ARB_gpu_shader_int64`. WGSL won't, because of
`uint64_t`.

## Development

```bash
python3 -m unittest discover -s tests -v     # needs clang for the full checks
```

Layout: `starcross/scan.py` (source splitting/masking), `cu2cl.py` (the converter), `prelude.py`
(CUDA→OpenCL compatibility header), `asfix.py` (clang-driven address-space fixes), `backend.py`
(clspv / spirv-cross / naga), `cli.py`.

The original chipStar-based script is still here as `starcross.sh` (built by `Containerfile`).
