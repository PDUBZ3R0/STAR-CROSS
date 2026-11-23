# CLAUDE.md — STAR-CROSS

## What this is
`starcross` converts CUDA kernels to OpenCL, then to GLSL compute shaders and (where possible) WGSL:

```
.cu ──cu2cl (this repo, Python)──▶ .cl / .clcpp ──clspv──▶ Vulkan SPIR-V ──spirv-cross──▶ GLSL
                                                                       └──naga / tint───▶ WGSL
```

Entry points: `./bin/starcross convert|cu2cl|cl2x|doctor`. See README.md for user docs.

## Why this architecture (don't re-litigate without reason)
- The project started as chipStar (CUDA → SPIR-V) + SPIRV-Cross. That **can't work**. chipStar emits
  OpenCL-style *Kernel* SPIR-V (`OpCapability Kernel`, physical pointers). SPIRV-Cross, naga and tint
  only accept Vulkan-style *Shader* SPIR-V. The old path still lives in `__re_cyc_le__/~Containerfile` + `starcross.sh` (gitignored)
  as a reference runner (useful for checking results on the Intel GPU via Level Zero/OpenCL).
- clspv's native, best-tested input is OpenCL C. So we go source-to-source CUDA → OpenCL first,
  then let clspv produce Shader SPIR-V.
- Slang was the other option (hand-port once, emit GLSL+WGSL). It was rejected for the *tool*, but it's
  still the pragmatic choice for porting one specific project by hand.

## Layout
- `starcross/scan.py`: comment/string masking, include inlining (with line provenance), top-level
  declaration splitter. Deliberately not a real parser.
- `starcross/cu2cl.py`: the converter. Classifies chunks (kernel/device/host/record/var/macro),
  keeps only what the selected kernels reach, then rewrites qualifiers, builtins and kernel signatures.
  Also writes the manifest (kernel params, host `<<<>>>` launch sites, features) and infers
  `reqd_work_group_size` from literal block sizes.
- `starcross/prelude.py`: CUDA→OpenCL compatibility header emitted at the top of every output
  (stdint typedefs, thread-index macros, intrinsics, math, atomics, subgroup mappings, sc_memcpy).
- `starcross/asfix.py`: clang-driven fixups. CUDA passes `__constant__` tables to plain-pointer helpers,
  which is illegal in OpenCL (`__constant` can't convert to `__generic`). We compile the output with
  clang, parse the errors, and clone the helper with a `__constant` parameter or qualify the local
  pointer, repeating until clean. Handles both the C++ ("no matching function") and the C
  ("changes address space") diagnostics.
- `starcross/backend.py`: clspv → spirv-val → spirv-dis (capabilities, plus clspv
  NonSemantic.ClspvReflection → descriptor bindings) → spirv-cross (one `.comp` per entry point) →
  naga/tint. Tool paths can be overridden with `STARCROSS_<TOOL>` env vars (e.g. `STARCROSS_CLSPV`).
- `starcross/cli.py`: argparse CLI. `tests/test_cu2cl.py`: unit tests. `examples/`: test inputs.
- `bin/starcross-container`: runs the image with podman or docker (whichever is installed;
  `STARCROSS_ENGINE` overrides). Image defaults to `ghcr.io/pdubz3r0/starcross-toolchain:latest`
  (`STARCROSS_IMAGE` overrides), pulled if missing. `build` builds and tags it, `shell` opens bash, anything else goes to `starcross`.
  Under docker it passes `--user $(id -u):$(id -g)` so outputs aren't root-owned.
- `Containerfile`: Ubuntu 24.04 multi-stage image that builds clspv (from source, with its
  own LLVM: slow), SPIRV-Cross and naga-cli, plus python3/clang/spirv-tools/glslang at runtime.

## Status (as of 2026-11-22)
**Tested:**
- The toolchain image (`localhost/starcross-toolchain`) is built and `starcross doctor` passes in it
  (everything but tint).
- cu2cl on all 4 examples. Output passes `clang -fsyntax-only` in both OpenCL C 3.0 and
  C++ for OpenCL 2021 modes.
- `python3 -m unittest discover -s tests` passes (6 tests; needs clang for the full checks).
- The backend plumbing (spirv-val, dis, reflect, spirv-cross, naga, reports), using a fake clspv
  that copied in naga-generated SPIR-V.

**NOT yet tested: do this first.**
- **clspv has never actually been run on our output.** The flags in `backend.compile_cl` were written
  from clspv's source (`lib/Option.cpp`, `include/clspv/FeatureMacro.h`) but never executed:
  - C mode: `-cl-std=CL3.0 --enable-feature-macros=<CL30_FEATURES> -inline-entry-points -cl-kernel-arg-info`
  - C++ mode: `-cl-std=CLC++2021 -inline-entry-points -cl-kernel-arg-info`
  - plus `-int8=0` when WGSL is requested (verify this doesn't break `uchar` buffers, e.g. reduce.cu's histogram)
- The clspv reflection parser (`parse_clspv_reflection`) hasn't been run against real clspv output.

Suggested order: build the toolchain image → `starcross doctor` → `convert examples/vecadd.cu -v`
→ reduce.cu → keccak.cu (GLSL should work; WGSL is expected to fail on Int64) → particles.cu (C++ mode).

## Known gaps / ideas
- `asfix` only fixes `__constant` → generic for function arguments and local pointer initializers, not
  assignments (`p = TABLE;`) or returns.
- In OpenCL C mode, atomics need a visibly `__global`/`__local` pointer (C++ mode is fine).
- Mixed-type `min`/`max` calls that CUDA accepts can be ambiguous in OpenCL. The tool can't fix these
  without type info; clang reports them.
- WGSL has no i64/u64/i8/i16/f64. A future pass could emulate u64 as vec2<u32>.
- Kernel params are re-joined onto fewer lines, so "near file:line" origin mapping can be off by one.

## Real-world test case: RinHash
`github.com/Rin-coin/RinHash-cuda` (BLAKE3 + Argon2d + SHA3). Clone it next to this repo:
`starcross convert ../RinHash-cuda/rinhash.cu -k rinhash_cuda_kernel`
- Converts ~1600 lines to C++ for OpenCL. asfix fixes `g_IV` / `blake2b_sigma`.
- Blocks on one real issue: `rinhash_cuda_kernel` does device `malloc` for the Argon2 memory
  (m_cost=64 → 64 KiB per hash). Fix: pass that memory in as a kernel buffer arg.
- The upstream `compute` kernel in `blake3_device.cuh` uses dynamic parallelism. It isn't reached by
  `rinhash_cuda_kernel` (which uses `light_hash_device`), so select the kernel with `-k`. An iterative
  replacement (`compress_leaves` + `merge_level`) was written during the chipStar experiments.

## Conventions
- Pure Python 3 stdlib in the tool itself (no pip deps). External tools are invoked as subprocesses.
- Keep the generated OpenCL readable: textual rewrites for qualifiers/indexing, prelude for the rest,
  `// from file:line` provenance markers.
- New CUDA constructs: add a mapping to the prelude if it's a 1:1 builtin, a rewrite in
  `Converter.rewrite` if it's syntax, or a check in `scan_device_code` if it's unsupported. Add a test.

## Working with Paul
- Experienced engineer (Node.js, Python, Linux); skip basics.
- Projects live in `~/Development`.
- Nudge him to commit to GitHub at sensible checkpoints.
