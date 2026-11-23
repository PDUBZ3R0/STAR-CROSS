"""starcross command line interface."""

import argparse
import json
import os
import re
import sys

from . import __version__
from . import asfix, backend, cu2cl
from .diag import Diagnostics

CU_EXT = (".cu", ".cuh")
CL_EXT = (".cl", ".clcpp")


def _log(quiet):
    def log(msg):
        if not quiet:
            print(msg, file=sys.stderr)
    return log


def _parse_wgsize(items):
    out = {}
    for it in items or []:
        name, _, dims = it.partition("=")
        vals = [int(x) for x in dims.replace("x", ",").split(",") if x.strip()]
        if not name or not vals:
            raise SystemExit(f"bad --wgsize '{it}', expected KERNEL=X[,Y[,Z]]")
        out[name] = tuple((vals + [1, 1])[:3])
    return out


def origin_of(text, line):
    """Map an output line back to the CUDA source using the nearest '// from file:N' marker."""
    lines = text.split("\n")
    for i in range(min(line, len(lines)) - 1, -1, -1):
        m = re.match(r"\s*// from (.+):(\d+)\s*$", lines[i])
        if m:
            return m.group(1), int(m.group(2)) + (line - 1 - i - 1)
    return None


def run_cu2cl(args, out_path, log):
    diags = Diagnostics()
    opts = cu2cl.Options(
        kernels=args.kernel or None, std=args.std, include_dirs=args.include,
        device_globals=args.device_globals, line_directives=args.line_directives,
        wgsize=_parse_wgsize(args.wgsize), auto_wgsize=not args.no_auto_wgsize)
    res = cu2cl.convert(args.input, opts, diags)
    text = res.text
    ext = ".clcpp" if res.std == "cpp" else ".cl"
    if out_path is None:
        out_path = os.path.splitext(os.path.basename(args.input))[0] + ext
    elif os.path.isdir(out_path) or out_path.endswith(os.sep):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, os.path.splitext(os.path.basename(args.input))[0] + ext)
    elif not out_path.endswith(CL_EXT):
        out_path += ext
    elif out_path.endswith(".cl") and res.std == "cpp":
        diags.note(f"output is C++ for OpenCL; writing {out_path[:-3]}.clcpp instead of {out_path}")
        out_path = out_path[:-3] + ".clcpp"

    clang = asfix.find_clang()
    remaining = None
    if clang and not args.no_check:
        if not args.no_asfix:
            text, applied, remaining = asfix.fix_address_spaces(text, res.std, clang)
            for a in applied:
                diags.note("address-space fix: " + a)
        else:
            _, remaining = asfix.run_clang(text, res.std, clang)
        if remaining:
            for d in remaining:
                if d["level"] == "error":
                    org = origin_of(text, d["line"])
                    where = f" (near {org[0]}:{org[1]})" if org else ""
                    diags.error(f"clang: {d['msg']}{where}", os.path.basename(out_path), d["line"])
    elif not clang and not args.no_check:
        diags.note("clang not found: skipped syntax check and __constant address-space fixups")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    manifest = dict(res.manifest)
    manifest["output"] = out_path
    manifest["diagnostics"] = diags.to_list()
    mpath = os.path.splitext(out_path)[0] + ".manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    diags.dump()
    log(f"wrote {out_path}  ({res.std == 'cpp' and 'C++ for OpenCL' or 'OpenCL C'}, "
        f"{len(manifest['kernels'])} kernel(s))")
    log(f"wrote {mpath}")
    return out_path, diags, manifest


def run_backend(args, cl_path, outdir, log):
    targets = set(t.strip() for t in args.to.split(",")) - {"cl"}
    if not targets:
        return None
    opts = backend.BackendOptions(
        targets=targets | {"spv"}, glsl_version=args.glsl_version, glsl_es=args.es,
        vulkan_glsl=args.vulkan_glsl, wgsl_backend=args.wgsl_backend, clspv_args=args.clspv_arg or [],
        wgsl_friendly=not args.no_wgsl_friendly, force=args.force)
    if args.std in ("c", "cpp") and cl_path.endswith(CL_EXT) and args.input.endswith(CL_EXT):
        opts.std = args.std
    rep = backend.compile_cl(cl_path, outdir, opts, log=log if args.verbose else (lambda m: None))
    for w in rep["warnings"]:
        print("warning: " + w, file=sys.stderr)
    for e in rep["errors"]:
        print("error: " + e, file=sys.stderr)
    for k, v in rep["outputs"].items():
        for p in (v if isinstance(v, list) else [v]):
            log(f"wrote {p}")
    return rep


def add_common(p, backend_opts=True, cu_opts=True):
    if cu_opts:
        g = p.add_argument_group("CUDA -> OpenCL")
        g.add_argument("-I", "--include", action="append", default=[], help="extra include directory")
        g.add_argument("-k", "--kernel", action="append", help="only convert this kernel (repeatable)")
        g.add_argument("--device-globals", choices=["global", "constant"], default="global",
                       help="how to emit mutable __device__ variables (default: global)")
        g.add_argument("--line-directives", action="store_true",
                       help="emit #line so clspv errors point at the .cu source")
        g.add_argument("--wgsize", action="append", metavar="KERNEL=X,Y,Z",
                       help="force reqd_work_group_size for a kernel")
        g.add_argument("--no-auto-wgsize", action="store_true",
                       help="don't infer work-group size from literal <<<grid, block>>> launches")
        g.add_argument("--no-asfix", action="store_true", help="skip clang-assisted __constant fixups")
        g.add_argument("--no-check", action="store_true", help="skip the clang syntax check")
    p.add_argument("--std", choices=["auto", "c", "cpp"], default="auto",
                   help="OpenCL C 3.0 or C++ for OpenCL 2021 (default: auto)")
    if backend_opts:
        g = p.add_argument_group("OpenCL -> GLSL / WGSL")
        g.add_argument("--glsl-version", type=int, default=450)
        g.add_argument("--es", action="store_true", help="emit GLSL ES (use with --glsl-version 310)")
        g.add_argument("--vulkan-glsl", action="store_true", help="emit Vulkan-flavoured GLSL")
        g.add_argument("--wgsl-backend", choices=["auto", "naga", "tint"], default="auto")
        g.add_argument("--clspv-arg", action="append", metavar="ARG", help="extra argument passed to clspv")
        g.add_argument("--no-wgsl-friendly", action="store_true",
                       help="don't pass -int8=0 to clspv when WGSL is requested")
    p.add_argument("--force", action="store_true", help="keep going after conversion errors")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="print the tool commands being run")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="starcross", description="CUDA -> OpenCL -> GLSL / WGSL converter")
    ap.add_argument("--version", action="version", version=f"starcross {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="convert .cu or .cl to any of: cl, spv, glsl, wgsl")
    p.add_argument("input")
    p.add_argument("-o", "--outdir", default=None, help="output directory (default: <name>.starcross/)")
    p.add_argument("--to", default="cl,glsl,wgsl", help="comma list of cl,spv,glsl,wgsl (default: cl,glsl,wgsl)")
    add_common(p)

    p = sub.add_parser("cu2cl", help="CUDA -> OpenCL only")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None, help="output .cl/.clcpp file or directory")
    add_common(p, backend_opts=False)

    p = sub.add_parser("cl2x", help="OpenCL -> SPIR-V -> GLSL / WGSL")
    p.add_argument("input")
    p.add_argument("-o", "--outdir", default=None)
    p.add_argument("--to", default="glsl,wgsl")
    add_common(p, cu_opts=False)

    sub.add_parser("doctor", help="check which external tools are installed")

    args = ap.parse_args(argv)
    if args.cmd == "doctor":
        return 0 if backend.doctor() else 1

    log = _log(args.quiet)
    if not os.path.isfile(args.input):
        print(f"error: {args.input}: no such file", file=sys.stderr)
        return 2

    is_cu = args.input.endswith(CU_EXT)
    is_cl = args.input.endswith(CL_EXT)
    if not (is_cu or is_cl):
        print("error: input must be .cu/.cuh or .cl/.clcpp", file=sys.stderr)
        return 2

    try:
        if args.cmd == "cu2cl":
            if not is_cu:
                print("error: cu2cl expects a .cu file", file=sys.stderr)
                return 2
            _, diags, _ = run_cu2cl(args, args.output, log)
            return 1 if diags.errors else 0

        outdir = args.outdir or (os.path.splitext(os.path.basename(args.input))[0] + ".starcross")
        if args.cmd == "cl2x" or is_cl:
            if is_cu:
                print("error: cl2x expects a .cl/.clcpp file (use `convert` for .cu)", file=sys.stderr)
                return 2
            rep = run_backend(args, args.input, outdir, log)
            return 1 if rep and rep["errors"] else 0

        cl_path, diags, _ = run_cu2cl(args, outdir + os.sep, log)
        if diags.errors and not args.force:
            print(f"\nstopping: {len(diags.errors)} conversion error(s) above. Fix the CUDA source "
                  f"(or hand-edit {cl_path} and run `starcross cl2x {cl_path}`), or pass --force.",
                  file=sys.stderr)
            return 1
        rep = run_backend(args, cl_path, outdir, log)
        return 1 if (rep and rep["errors"]) else 0
    except backend.ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
