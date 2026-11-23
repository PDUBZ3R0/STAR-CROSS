"""OpenCL (.cl / .clcpp) -> Vulkan SPIR-V (clspv) -> GLSL (spirv-cross) / WGSL (naga or tint)."""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

# Feature macros we ask clspv to enable for OpenCL C 3.0 input. With CL3.0, any
# feature not listed is disabled, so this is the "CUDA-like" baseline.
CL30_FEATURES = [
    "__opencl_c_generic_address_space",
    "__opencl_c_int64",
    "__opencl_c_subgroups",
    "__opencl_c_atomic_order_acq_rel",
    "__opencl_c_atomic_order_seq_cst",
    "__opencl_c_atomic_scope_device",
    "__opencl_c_work_group_collective_functions",
]

# SPIR-V capabilities that WGSL has no equivalent for.
WGSL_BLOCKERS = {
    "Int64": "64-bit integers (WGSL has no i64/u64; emulate with vec2<u32>)",
    "Int16": "16-bit integers",
    "Int8": "8-bit integers",
    "Float64": "double precision floats",
    "Int64Atomics": "64-bit atomics",
    "StorageBuffer8BitAccess": "8-bit values in storage buffers",
    "UniformAndStorageBuffer8BitAccess": "8-bit values in buffers",
    "StorageBuffer16BitAccess": "16-bit values in storage buffers",
    "VariablePointers": "variable pointers",
    "VariablePointersStorageBuffer": "variable pointers into storage buffers",
    "PhysicalStorageBufferAddresses": "raw buffer device addresses",
}
GLSL_NOTES = {
    "Int64": "GL_ARB_gpu_shader_int64 / GL_EXT_shader_explicit_arithmetic_types_int64",
    "Int8": "GL_EXT_shader_explicit_arithmetic_types_int8",
    "Int16": "GL_EXT_shader_explicit_arithmetic_types_int16",
    "Float64": "GL_ARB_gpu_shader_fp64",
    "StorageBuffer8BitAccess": "GL_EXT_shader_8bit_storage",
    "VariablePointers": "Vulkan-only (VK_KHR_variable_pointers); not expressible in desktop GL GLSL",
    "PhysicalStorageBufferAddresses": "GL_EXT_buffer_reference (Vulkan GLSL only)",
}

TOOL_NAMES = ["clspv", "spirv-val", "spirv-dis", "spirv-cross", "naga", "tint", "glslangValidator", "clang"]


class ToolError(RuntimeError):
    pass


def tool(name):
    env = os.environ.get("STARCROSS_" + name.upper().replace("-", "_"))
    if env:
        return env
    return shutil.which(name)


def tool_version(name):
    p = tool(name)
    if not p:
        return None
    for flag in ("--version", "-version", "-v"):
        try:
            r = subprocess.run([p, flag], capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        out = (r.stdout or r.stderr).strip().splitlines()
        if out and r.returncode == 0:
            return out[0][:100]
    return "found"


def run(cmd, log, check=True):
    log(" $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise ToolError(f"{os.path.basename(cmd[0])} failed (exit {r.returncode}):\n{(r.stderr or r.stdout).strip()}")
    return r


@dataclass
class BackendOptions:
    targets: set = field(default_factory=lambda: {"glsl", "wgsl"})
    std: str = None                 # c | cpp ; default from file extension
    glsl_version: int = 450
    glsl_es: bool = False
    vulkan_glsl: bool = False
    wgsl_backend: str = "auto"      # auto | naga | tint
    clspv_args: list = field(default_factory=list)
    spv_target_env: str = "vulkan1.0"
    wgsl_friendly: bool = True      # pass clspv flags that avoid WGSL-hostile capabilities
    force: bool = False


def parse_spirv(dis):
    caps = re.findall(r"^\s*OpCapability (\w+)", dis, re.M)
    exts = re.findall(r'^\s*OpExtension "([^"]+)"', dis, re.M)
    entries = re.findall(r'OpEntryPoint GLCompute %\w+ "([^"]+)"', dis)
    return caps, exts, entries


def parse_clspv_reflection(dis):
    """Extract kernel argument -> descriptor binding info from NonSemantic.ClspvReflection."""
    consts = {}
    for m in re.finditer(r"^\s*(%\w+) = OpConstant %\w+ (-?\d+)", dis, re.M):
        consts[m.group(1)] = int(m.group(2))
    strings = {m.group(1): m.group(2) for m in re.finditer(r'^\s*(%\w+) = OpString "([^"]*)"', dis, re.M)}
    kernels, args = {}, {}
    for m in re.finditer(r"^\s*(%\w+) = OpExtInst %\w+ %\w+ (\w+) (.*)$", dis, re.M):
        res, op, rest = m.group(1), m.group(2), m.group(3).split()
        if op == "Kernel" and len(rest) >= 2:
            kernels[res] = {"name": strings.get(rest[1], rest[1]), "args": [], "workgroup_size": None}
        elif op == "ArgumentInfo" and rest:
            args[res] = strings.get(rest[0], rest[0])
    for m in re.finditer(r"^\s*(?:%\w+ = )?OpExtInst %\w+ %\w+ (Argument\w+|PropertyRequiredWorkgroupSize|SpecConstantWorkgroupSize|PushConstant\w+) (.*)$", dis, re.M):
        op, rest = m.group(1), m.group(2).split()
        if op == "PropertyRequiredWorkgroupSize" and rest and rest[0] in kernels:
            kernels[rest[0]]["workgroup_size"] = [consts.get(x, x) for x in rest[1:4]]
            continue
        if not op.startswith("Argument") or op == "ArgumentInfo" or not rest or rest[0] not in kernels:
            continue
        vals = [consts.get(x, x) for x in rest[1:]]
        entry = {"kind": op.replace("Argument", ""), "ordinal": vals[0] if vals else None}
        names = ["descriptor_set", "binding", "offset", "size"] if "Pod" in op or "PointerPushConstant" in op else \
            ["descriptor_set", "binding"]
        if "PushConstant" in op:
            names = ["offset", "size"]
        if "Workgroup" in op:  # __local pointer arg
            names = ["spec_id", "element_size"]
        for i, nm in enumerate(names):
            if i + 1 < len(vals) and isinstance(vals[i + 1], int):
                entry[nm] = vals[i + 1]
        info = [x for x in rest if x in args]
        if info:
            entry["name"] = args[info[0]]
        kernels[rest[0]]["args"].append(entry)
    for k in kernels.values():
        k["args"].sort(key=lambda a: (a.get("ordinal") if isinstance(a.get("ordinal"), int) else 0))
    return list(kernels.values())


def compile_cl(cl_path, outdir, opts: BackendOptions, log=print):
    """Run the whole backend. Returns a report dict."""
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(cl_path))[0]
    std = opts.std or ("cpp" if cl_path.endswith((".clcpp", ".clpp", ".cppcl")) else "c")
    report = {"input": cl_path, "std": std, "outputs": {}, "warnings": [], "errors": []}

    clspv = tool("clspv")
    if not clspv:
        raise ToolError("clspv not found on PATH (set STARCROSS_CLSPV to override). Run `starcross doctor`.")
    spv = os.path.join(outdir, stem + ".spv")
    cmd = [clspv]
    if std == "cpp":
        cmd += ["-cl-std=CLC++2021"]
    else:
        cmd += ["-cl-std=CL3.0", "--enable-feature-macros=" + ",".join(CL30_FEATURES)]
    cmd += ["-inline-entry-points", "-cl-kernel-arg-info"]
    if opts.wgsl_friendly and "wgsl" in opts.targets:
        cmd += ["-int8=0"]
    cmd += list(opts.clspv_args) + [cl_path, "-o", spv]
    run(cmd, log)
    report["outputs"]["spv"] = spv

    if tool("spirv-val"):
        r = run([tool("spirv-val"), "--target-env", opts.spv_target_env, spv], log, check=False)
        if r.returncode != 0:
            report["errors"].append("spirv-val: " + (r.stderr or r.stdout).strip())
            if not opts.force:
                raise ToolError("spirv-val rejected clspv's output:\n" + (r.stderr or r.stdout).strip())

    caps, exts, entries, kernels = [], [], [], []
    if tool("spirv-dis"):
        dis = run([tool("spirv-dis"), "--raw-id", spv], log).stdout
        caps, exts, entries = parse_spirv(dis)
        try:
            kernels = parse_clspv_reflection(dis)
        except Exception as e:  # reflection is best effort
            report["warnings"].append(f"could not parse clspv reflection: {e}")
    report.update(capabilities=caps, extensions=exts, entry_points=entries, kernels=kernels)

    if tool("spirv-cross"):
        r = run([tool("spirv-cross"), spv, "--reflect"], log, check=False)
        if r.returncode == 0:
            path = os.path.join(outdir, stem + ".reflect.json")
            with open(path, "w") as f:
                f.write(r.stdout)
            report["outputs"]["reflect"] = path
            if not entries:
                try:
                    entries = [e["name"] for e in json.loads(r.stdout).get("entryPoints", [])]
                    report["entry_points"] = entries
                except ValueError:
                    pass

    if "glsl" in opts.targets:
        sc = tool("spirv-cross")
        if not sc:
            raise ToolError("spirv-cross not found")
        gdir = os.path.join(outdir, "glsl")
        os.makedirs(gdir, exist_ok=True)
        report["outputs"]["glsl"] = []
        for c in caps:
            if c in GLSL_NOTES:
                report["warnings"].append(f"GLSL needs {GLSL_NOTES[c]} (capability {c})")
        for e in entries or [None]:
            out = os.path.join(gdir, (e or stem) + ".comp")
            cmd = [sc, spv, "--version", str(opts.glsl_version), "--output", out]
            if opts.glsl_es:
                cmd.append("--es")
            if opts.vulkan_glsl:
                cmd.append("--vulkan-semantics")
            if e:
                cmd += ["--entry", e, "--stage", "comp"]
            r = run(cmd, log, check=False)
            if r.returncode != 0:
                report["errors"].append(f"spirv-cross ({e}): {(r.stderr or r.stdout).strip()}")
                continue
            report["outputs"]["glsl"].append(out)
            gv = tool("glslangValidator")
            if gv:
                vcmd = [gv, "-S", "comp", out] + (["-V", "-o", os.devnull] if opts.vulkan_glsl else [])
                vr = run(vcmd, log, check=False)
                if vr.returncode != 0:
                    report["warnings"].append(f"glslangValidator rejected {os.path.basename(out)}: "
                                              + (vr.stdout or vr.stderr).strip()[-600:])

    if "wgsl" in opts.targets:
        blockers = [f"{c}: {WGSL_BLOCKERS[c]}" for c in caps if c in WGSL_BLOCKERS]
        for b in blockers:
            report["warnings"].append("WGSL: module uses " + b)
        backend = opts.wgsl_backend
        if backend == "auto":
            backend = "naga" if tool("naga") else ("tint" if tool("tint") else None)
        if not backend or not tool(backend):
            report["errors"].append("no WGSL backend found (install naga-cli: `cargo install naga-cli`, or tint)")
        else:
            out = os.path.join(outdir, stem + ".wgsl")
            if backend == "naga":
                cmd = [tool("naga"), spv, out]
            else:
                cmd = [tool("tint"), "--format", "wgsl", "-o", out, spv]
            r = run(cmd, log, check=False)
            if r.returncode != 0 or not os.path.exists(out):
                msg = (r.stderr or r.stdout).strip()
                report["errors"].append(f"{backend}: {msg[-1500:]}")
                if blockers:
                    report["errors"].append("WGSL conversion most likely failed because of: "
                                            + "; ".join(b.split(":")[0] for b in blockers))
            else:
                report["outputs"]["wgsl"] = out

    with open(os.path.join(outdir, stem + ".backend.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def doctor(out=print):
    ok = True
    out("starcross toolchain:")
    for name in TOOL_NAMES:
        p = tool(name)
        v = tool_version(name) if p else None
        need = {"clspv": "required for glsl/wgsl", "spirv-cross": "required for glsl",
                "naga": "wgsl (or tint)", "tint": "wgsl (or naga)", "clang": "cu2cl checks + __constant fixups",
                "spirv-val": "validation", "spirv-dis": "capability report + bindings",
                "glslangValidator": "optional GLSL check"}[name]
        mark = "ok " if p else "-- "
        out(f"  {mark} {name:<17} {p or 'not found':<45} {need}")
        if not p and name in ("clspv", "spirv-cross"):
            ok = False
    return ok
