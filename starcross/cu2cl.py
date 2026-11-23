"""CUDA (.cu) -> OpenCL C / C++ for OpenCL source-to-source conversion.

Pipeline:
  1. inline local includes, drop system/CUDA headers
  2. split the translation unit into top-level declarations
  3. classify each one (kernel / device fn / host fn / record / variable / macro)
  4. keep only what the selected kernels can reach (host code is pruned)
  5. rewrite CUDA qualifiers, builtins and kernel signatures for OpenCL
  6. emit prelude + converted code + a JSON manifest describing kernels/launches
"""

import os
import re
from dataclasses import dataclass, field

from . import scan
from .diag import Diagnostics
from .prelude import PRELUDE

KERNEL_AS = ("__global", "__local", "__constant", "__private", "global", "local", "constant", "private")


@dataclass
class Options:
    kernels: list = None            # restrict to these kernel names (default: all)
    std: str = "auto"               # auto | c | cpp
    include_dirs: list = field(default_factory=list)
    device_globals: str = "global"  # global | constant  (how to emit mutable __device__ vars)
    provenance: bool = True         # "// from file:line" comments
    line_directives: bool = False   # emit #line so later compilers report CUDA locations
    wgsize: dict = field(default_factory=dict)  # kernel -> (x, y, z) reqd_work_group_size override
    auto_wgsize: bool = True        # infer reqd_work_group_size from literal <<<grid, block>>> launches


@dataclass
class Result:
    text: str
    std: str                  # "c" or "cpp"
    manifest: dict
    diags: Diagnostics
    line_map: list            # output line index -> (file, line) or None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

ATTR_STRIP_RE = re.compile(r"__launch_bounds__\s*\([^()]*\)|__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)|__align__\s*\([^()]*\)")
TEMPLATE_PREFIX_RE = re.compile(r"^\s*template\s*<")
QUALIFIER_WORDS = {"__global__", "__device__", "__host__", "__forceinline__", "__noinline__", "static",
                   "inline", "extern", "constexpr", "__inline__", "virtual", "explicit"}

BUILTIN_IDX = {"threadIdx": "get_local_id", "blockIdx": "get_group_id",
               "blockDim": "get_local_size", "gridDim": "get_num_groups"}
AXIS = {"x": 0, "y": 1, "z": 2}

CPP_HINT_RE = re.compile(
    r"\btemplate\s*<|\bclass\b|\bnamespace\b|::|\bnullptr\b|\bconstexpr\b|\boperator\b|\bthis\b|"
    r"\bstatic_cast\b|\breinterpret_cast\b|\bconst_cast\b|\bauto\b|\benum\s+class\b|\bnew\b|\bdelete\b")


def _strip_template_prefix(h):
    m = TEMPLATE_PREFIX_RE.match(h)
    if not m:
        return h, False
    depth, i = 0, m.end() - 1
    while i < len(h):
        if h[i] == "<":
            depth += 1
        elif h[i] == ">":
            depth -= 1
            if depth == 0:
                return h[i + 1:], True
        i += 1
    return h, True


def _func_name(masked_header):
    """Return (name, qualified, paren_index, qualifier_words, is_template)."""
    h, is_tmpl = _strip_template_prefix(masked_header)
    offset = len(masked_header) - len(h)
    clean = ATTR_STRIP_RE.sub(lambda mm: " " * len(mm.group(0)), h)
    # first '(' preceded by an identifier that is not a keyword-ish qualifier
    for mm in re.finditer(r"([~\w:]+)\s*\(", clean):
        name = mm.group(1)
        if name in ("__attribute__", "__declspec", "decltype", "sizeof", "alignas"):
            continue
        quals = set(re.findall(r"\b\w+\b", h[:mm.start()]))
        paren = offset + clean.index("(", mm.start())
        short = name.split("::")[-1]
        return short, name, paren, quals, is_tmpl
    return None, None, -1, set(re.findall(r"\b\w+\b", h)), is_tmpl


def _declarator_names(masked_decl):
    body = masked_decl.rstrip().rstrip(";")
    body, _ = _strip_template_prefix(body)
    body = ATTR_STRIP_RE.sub(" ", body)
    names = set()
    fp = re.search(r"\(\s*\*\s*(\w+)\s*\)", body)
    if fp:
        return {fp.group(1)}
    for part in scan.top_level_commas(body):
        part = part.split("=", 1)[0]
        part = re.sub(r"\[[^\]]*\]", " ", part)
        ids = re.findall(r"[A-Za-z_]\w*", part)
        if ids:
            names.add(ids[-1])
    return names


def _qualifier_words(masked):
    return set(re.findall(r"\b\w+\b", masked))


def _record_members(c):
    """Split a record body into members; yield (member_chunk, masked, is_host_only_method, access_label)."""
    m = scan.mask(c.text)
    close = m.rfind("}")
    inner = c.text[c.body_open + 1:close]
    inner_m = m[c.body_open + 1:close]
    line0 = c.line + c.text[:c.body_open + 1].count("\n")
    for mem in scan.split_top(inner, line0, inner_m):
        mm = scan.mask(mem.text)
        access = re.match(r"\s*((?:public|private|protected)\s*:)", mm)
        is_func = mem.kind == "func" or bool(
            re.search(r"[\w~]\s*\(", mm) and not re.search(r"\(\s*\*", mm)
            and "=" not in mm.split("(")[0] and not re.match(r"\s*(?:typedef|using|static_assert)\b", mm))
        host_only = False
        if is_func:
            words = _qualifier_words(mm.split("(")[0])
            host_only = not (words & {"__device__", "constexpr"}) and not re.search(r"=\s*(?:default|delete)\s*;", mm)
        yield mem, mm, is_func, host_only, (access.group(1) if access else None)


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------

class Converter:
    def __init__(self, path, opts: Options, diags: Diagnostics):
        self.path = path
        self.opts = opts
        self.d = diags
        self.src = scan.load_source(path, opts.include_dirs, diags)
        self.chunks = []
        self.by_name = {}
        self.features = set()
        self.kernels = []
        self.launches = []
        self.dyn_shared = {}   # name -> type for top-level extern __shared__ arrays

    # ---- location helpers -------------------------------------------------
    def loc(self, chunk, offset=0):
        line = chunk.line + chunk.text[:offset].count("\n")
        return self.src.origin(line)

    def diag(self, level, msg, chunk, offset=0, hint=""):
        f, l = self.loc(chunk, offset)
        self.d.add(level, msg, f, l, hint)

    # ---- classification ---------------------------------------------------
    def flatten(self, chunks, ns=None):
        out = []
        for c in chunks:
            if c.kind == "container":
                if re.match(r'^\s*extern\s*"', c.text):
                    out.extend(self.flatten(c.children, ns))
                else:
                    c.children = self.flatten(c.children, c)
                    c.role = "namespace"
                    out.append(c)
                    for ch in c.children:
                        ch.ns = c
            else:
                out.append(c)
        return out

    def classify(self, c):
        m = scan.mask(c.text)
        c.masked = m
        if c.kind == "pp":
            c.role = "pp"
            mm = re.match(r"\s*#\s*define\s+(\w+)", m)
            if mm:
                c.names = {mm.group(1)}
                c.refs = scan.idents(m[mm.end():])
            return
        if c.kind == "namespace" or c.role == "namespace":
            for ch in c.children:
                self.classify(ch)
            c.names, c.refs = set(), set()
            return
        c.refs = scan.idents(m)
        if c.kind == "func":
            name, qual, _, quals, is_tmpl = _func_name(c.header)
            c.names = {name} if name else set()
            c.qualified = qual
            c.is_template = is_tmpl
            if "__global__" in quals:
                c.role = "kernel"
            elif "__device__" in quals or "constexpr" in quals:
                c.role = "device"
            else:
                c.role = "host"
            if qual and "::" in qual:
                c.refs |= set(qual.split("::")[:-1])
            return
        if c.kind == "record":
            c.role = "shared"
            if c.body_open >= 0:
                c.refs = scan.idents(m[:c.body_open])
                for mem, mm, is_func, host_only, _ in _record_members(c):
                    if not host_only:
                        c.refs |= scan.idents(mm)
            tag = re.search(r"\b(?:struct|class|union|enum)\s+(?:class\s+)?(?:__align__\s*\([^)]*\)\s*|__attribute__\s*\(\([^)]*\)\)\s*)*(\w+)", m)
            if tag:
                c.names.add(tag.group(1))
            if c.body_open >= 0:
                close = m.rfind("}")
                tail = m[close + 1:]
                c.names |= set(re.findall(r"[A-Za-z_]\w*", tail))
                if re.search(r"\benum\b", c.header):
                    body = m[c.body_open + 1:close]
                    c.names |= {p.split("=")[0].strip() for p in body.split(",") if p.strip()}
            c.names.discard("")
            return
        # plain declarations
        h = m.strip()
        words = _qualifier_words(h)
        if re.match(r"\s*using\s+namespace\b", h):
            c.role = "host"
            return
        if re.match(r"\s*typedef\b", h):
            c.role = "shared"
            c.names = _declarator_names(re.sub(r"^\s*typedef\b", "", h))
            return
        mm = re.match(r"\s*(?:template\s*<.*?>\s*)?using\s+(\w+)\s*=", h, re.S)
        if mm:
            c.role = "shared"
            c.names = {mm.group(1)}
            return
        name, qual, paren, quals, is_tmpl = _func_name(h)
        is_proto = name is not None and "=" not in h[:paren] and not re.search(r"\(\s*\*", h) \
            and not re.match(r"\s*(?:static\s+)?(?:const\s+)?\w[\w\s\*]*\b\w+\s*\[", h)
        if is_proto:
            c.names = {name}
            c.kind = "proto"
            c.role = "kernel-proto" if "__global__" in quals else (
                "device" if ("__device__" in quals or "constexpr" in quals) else "host")
            return
        c.names = _declarator_names(h)
        if "extern" in words and "__shared__" in words:
            c.role = "dynshared"
            mm = re.search(r"__shared__\s+(?:__align__\s*\([^)]*\)\s*)?([\w\s\*:<>]+?)\s+(\w+)\s*\[\s*\]", c.text)
            if mm:
                self.dyn_shared[mm.group(2)] = mm.group(1).strip()
            return
        if "__constant__" in words:
            c.role = "constant"
        elif "__device__" in words or "__managed__" in words:
            c.role = "constant" if ("const" in words or "constexpr" in words) else "devglobal"
        elif "__shared__" in words:
            c.role = "shared-var"
        elif "const" in words or "constexpr" in words:
            c.role = "hostconst"
        else:
            c.role = "hostvar"

    def index(self, chunks):
        for c in chunks:
            if c.role == "namespace":
                self.index(c.children)
                continue
            for n in c.names:
                self.by_name.setdefault(n, []).append(c)

    def all_chunks(self, chunks=None):
        for c in (self.chunks if chunks is None else chunks):
            if c.role == "namespace":
                yield from self.all_chunks(c.children)
            else:
                yield c

    # ---- reachability -----------------------------------------------------
    def reach(self):
        kernels = [c for c in self.all_chunks() if c.role == "kernel"]
        if self.opts.kernels:
            wanted = set(self.opts.kernels)
            found = {next(iter(k.names)) for k in kernels if k.names}
            for missing in sorted(wanted - found):
                self.d.error(f"kernel '{missing}' not found (available: {', '.join(sorted(found)) or 'none'})")
            kernels = [k for k in kernels if k.names & wanted]
        if not kernels:
            self.d.error("no __global__ kernels to convert")
        work = list(kernels)
        for k in kernels:
            k.keep = True
        host_hits = {}
        while work:
            c = work.pop()
            for ident in c.refs:
                defs = self.by_name.get(ident)
                if not defs:
                    continue
                usable = [x for x in defs if x.role not in ("host", "hostvar")]
                if not usable:
                    if any(x.role == "host" for x in defs) and re.search(rf"\b{re.escape(ident)}\s*\(", c.masked):
                        host_hits.setdefault(ident, c)
                    continue
                for x in usable:
                    if x.role == "kernel" and x is not c and x not in kernels:
                        continue  # referencing another kernel (a launch) is diagnosed separately
                    if not x.keep:
                        x.keep = True
                        work.append(x)
        for ident, c in host_hits.items():
            if c.role in ("kernel", "device", "shared") and c.keep:
                self.diag("note", f"'{ident}' is called from device code but only a host definition exists "
                          "(assuming it's an OpenCL builtin or a local name)", c, c.masked.find(ident),
                          hint=f"if it's yours, mark '{ident}' __device__ in the CUDA source")
        self.selected = kernels

    # ---- rewriting --------------------------------------------------------
    def rewrite(self, text):
        text = re.sub(r'\bextern\s*"C"\s*', "", text)

        def code(t):
            t = re.sub(r"\b__global__\b", "__kernel", t)
            t = re.sub(r"\b__(?:device|host|managed)__\s*", "", t)
            t = re.sub(r"\b__shared__\b", "__local", t)
            t = re.sub(r"\b__constant__\b", "__constant", t)
            t = re.sub(r"\b__forceinline__\b", "__attribute__((always_inline))", t)
            t = re.sub(r"\b__inline__\b", "inline", t)
            t = re.sub(r"\b__noinline__\b", "__attribute__((noinline))", t)
            t = re.sub(r"\b__align__\s*\(\s*([^()]+?)\s*\)", r"__attribute__((aligned(\1)))", t)
            t = re.sub(r"\b__launch_bounds__\s*\([^()]*\)\s*", "", t)
            t = re.sub(r"\b__restrict__\b", "__restrict", t)
            t = re.sub(r"\b(threadIdx|blockIdx|blockDim|gridDim)\s*\.\s*([xyz])\b",
                       lambda m: f"((uint){BUILTIN_IDX[m.group(1)]}({AXIS[m.group(2)]}))", t)
            t = re.sub(r"\b(memcpy|memset|memcmp)\s*\(", r"sc_\1(", t)
            t = re.sub(r"\bunsigned\s+long\s+long(?:\s+int)?\b", "ulong", t)
            t = re.sub(r"\b(?:signed\s+)?long\s+long(?:\s+int)?\b", "long", t)
            return t

        return scan.map_code(text, code)

    def scan_device_code(self, c):
        m = c.masked
        checks = [
            (r"<<<", "error", "kernel launch inside device code (dynamic parallelism) is not supported",
             "launch the kernels from the host, one level per launch (see README: flattening recursion)"),
            (r"\b(?:malloc|free)\s*\(", "error", "device-side malloc/free is not supported",
             "allocate the memory on the host and pass it in as a kernel buffer argument"),
            (r"\bnew\b|\bdelete\b", "error", "device-side new/delete is not supported", ""),
            (r"\bcuda[A-Z]\w*\s*\(", "error", "CUDA runtime API call in device code", ""),
            (r"\basm\b|\b__asm__\b", "error", "inline PTX assembly cannot be converted",
             "rewrite the asm block in plain C"),
            (r"\btex[123]D\w*\s*\(|\btexture\s*<|\bsurf[123]D\w*\s*\(|\btex1Dfetch\b", "error",
             "CUDA textures/surfaces are not converted", "port to image types or plain buffers by hand"),
            (r"\bcooperative_groups\b|\bcg::", "error", "cooperative groups are not supported", ""),
            (r"\b__activemask\s*\(|\b__match_\w+\s*\(|\b__reduce_\w+_sync\s*\(", "error",
             "warp intrinsic has no OpenCL equivalent", ""),
            (r"\b__shfl(?:_up|_down|_xor)?\s*\(|\b__ballot\s*\(|\b__all\s*\(|\b__any\s*\(", "error",
             "legacy (non-_sync) warp intrinsic", "use the *_sync variant"),
            (r"\bclock(?:64)?\s*\(|\b__nanosleep\s*\(|\b__globaltimer\b", "error",
             "device timers are not available", ""),
            (r"\bstd::", "error", "C++ standard library used in device code", ""),
            (r"\bprintf\s*\(", "warning", "device printf: clspv support is limited; it won't exist in GLSL/WGSL", ""),
            (r"\b__shfl\w*_sync\b|\b__ballot_sync\b|\b__syncwarp\b|\bwarpSize\b", "warning",
             "warp intrinsics map to OpenCL subgroups; subgroup size is device dependent (not always 32)", ""),
        ]
        for pat, level, msg, hint in checks:
            mm = re.search(pat, m)
            if mm:
                self.diag(level, msg, c, mm.start(), hint)
        if re.search(r"\b__shared__\b", m) and c.role != "kernel":
            mm = re.search(r"\b__shared__\b", m)
            self.diag("error", "__shared__ variable outside kernel scope; OpenCL requires __local variables "
                      "to be declared in the kernel body", c, mm.start(),
                      "declare it in the kernel and pass a __local pointer to the helper")
        if re.search(r"\b(?:u?int64_t|u?long|size_t|ptrdiff_t|long\s+long|uint64|u64|i64)\b", m):
            self.features.add("int64")
        if re.search(r"\b(?:u?int8_t|u?char|u8|i8|uint8)\b", m):
            self.features.add("int8")
        if re.search(r"\b(?:u?int16_t|u?short|half|__half)\b", m):
            self.features.add("int16")
        if re.search(r"\bdouble\b", m):
            self.features.add("fp64")
        v3 = re.search(r"\b(?:float|double|u?int|u?char|u?short|u?long|longlong|ulonglong)3\b", m)
        if v3 and (c.kind == "record" or c.role == "kernel"):
            self.diag("warning", f"'{v3.group(0)}' is 16 bytes in OpenCL/GLSL but 12 bytes in CUDA; "
                      "buffer layouts using it will not match the host", c, v3.start(),
                      "use float4 / an explicit {float x, y, z;} struct, or pad on the host")
        if re.search(r"\batomic\w+\s*\(", m):
            self.features.add("atomics")
        if re.search(r"\b__shfl|\b__ballot|\b__syncwarp|\b__all_sync|\b__any_sync", m):
            self.features.add("subgroups")
        if re.search(r"\bprintf\s*\(", m):
            self.features.add("printf")

    def convert_kernel(self, c):
        name = next(iter(c.names))
        _, _, paren, quals, is_tmpl = _func_name(c.header)
        if is_tmpl:
            self.diag("error", f"templated kernel '{name}' cannot be an OpenCL entry point", c, 0,
                      "add a non-template __global__ wrapper per instantiation")
        if not re.search(r"\bvoid\b", c.header[:paren]):
            self.diag("error", f"kernel '{name}' must return void", c, 0)
        close = scan.matching_paren(c.masked, paren)
        params_txt = c.text[paren + 1:close]
        body = c.text[close + 1:]
        head = c.text[:paren + 1]

        # dynamic shared memory -> extra __local pointer parameters
        extra = []
        def drop_dyn(mm):
            extra.append((mm.group(2), mm.group(1).strip()))
            return "/* starcross: extern __shared__ moved to a __local kernel argument */"
        body = re.sub(r"extern\s+__shared__\s+(?:__align__\s*\([^)]*\)\s*)?([\w\s\*:<>]+?)\s+(\w+)\s*\[\s*\]\s*;",
                      drop_dyn, body)
        for ident in c.refs:
            if ident in self.dyn_shared and ident not in {e[0] for e in extra}:
                extra.append((ident, self.dyn_shared[ident]))

        params = []
        info = []
        raw = [p for p in scan.top_level_commas(params_txt) if p.strip() and p.strip() != "void"]
        for idx, p in enumerate(raw):
            mp = scan.mask(p)
            p_clean = re.sub(r"\s*=\s*[^,]*$", "", p)  # default args
            if "=" in mp:
                self.diag("note", "default argument removed from kernel parameter", c, paren)
            words = _qualifier_words(mp)
            pname = (re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", mp.split("=")[0])) or ["arg%d" % idx])[-1]
            kind = "pod"
            if "&" in mp:
                self.diag("error", f"kernel parameter '{pname}' is a reference; OpenCL kernels take values or pointers",
                          c, paren, "pass a pointer instead")
            if mp.count("*") >= 2:
                self.diag("error", f"kernel parameter '{pname}' is a pointer-to-pointer; not allowed in OpenCL kernels",
                          c, paren, "flatten the data into one buffer and pass offsets")
            is_array = re.search(r"\[[^\]]*\]\s*$", mp.strip()) is not None
            if "*" in mp or is_array:
                kind = "buffer"
                if not (words & set(KERNEL_AS)):
                    first = len(scan.mask(p_clean)) - len(scan.mask(p_clean).lstrip())
                    p_clean = p_clean[:first] + "__global " + p_clean[first:]
                elif words & {"__local", "local"}:
                    kind = "local"
            else:
                for t, repl in (("size_t", "ulong"), ("ptrdiff_t", "long"), ("intptr_t", "long"),
                                ("uintptr_t", "ulong"), ("bool", "int")):
                    if re.search(rf"\b{t}\b", mp):
                        p_clean = re.sub(rf"\b{t}\b", repl, p_clean)
                        self.diag("note", f"kernel parameter '{pname}': {t} is not a legal OpenCL kernel "
                                  f"argument type, changed to {repl}", c, paren)
                tname = [w for w in re.findall(r"[A-Za-z_]\w*", mp) if w != pname]
                for w in tname:
                    for rec in self.by_name.get(w, []):
                        if rec.kind == "record" and "*" in scan.mask(rec.text[rec.body_open:]):
                            self.diag("warning", f"kernel parameter '{pname}' is a struct containing pointers; "
                                      "pointers inside structs are not valid across host/device in OpenCL/Vulkan",
                                      c, paren)
            params.append(p_clean.strip())
            decl = " ".join(scan.strip_comments(self.rewrite(p_clean)).split())
            info.append({"index": idx, "name": pname, "decl": decl, "kind": kind})
        for nm, ty in extra:
            params.append(f"__local {ty} *{nm}")
            info.append({"index": len(info), "name": nm, "decl": f"__local {ty} *{nm}", "kind": "local",
                         "note": "was extern __shared__; size it at launch (clspv: specialization constant)"})
            self.diag("note", f"dynamic shared memory '{nm}' became a __local pointer parameter", c, 0)

        head = re.sub(r"\bstatic\s+|\binline\s+", "", head)
        if name in getattr(self, "wgsizes", {}):
            x, y, z = self.wgsizes[name]
            head = re.sub(r"\b__global__\b", f"__global__ __attribute__((reqd_work_group_size({x}, {y}, {z})))", head, count=1)
        sep = ",\n    " if any("\n" in p for p in raw) or len(params) > 3 else ", "
        params_out = ("\n    " if sep.startswith(",\n") else "") + sep.join(params)
        text = self.rewrite(head) + self.rewrite(params_out) + ")" + self.rewrite(body)
        f, l = self.loc(c)
        entry = {"name": name, "source": f"{f}:{l}", "params": info}
        if name in getattr(self, "wgsizes", {}):
            entry["workgroup_size"] = list(self.wgsizes[name])
        self.kernels.append(entry)
        return text

    def convert_record(self, c):
        if c.body_open < 0:
            return self.rewrite(c.text)
        close = c.masked.rfind("}")
        out = []
        for mem, mm, is_func, host_only, access in _record_members(c):
            if host_only:
                if access:
                    out.append(access)
                continue  # host-only member function
            if is_func:
                self.features.add("cpp")
            if re.search(r"\bstd::|\bthrust::", mm):
                self.diag("error", "struct member uses a host library type", c, c.text.find(mem.text))
            lead = (mem.lead.strip() + "\n    ") if mem.lead.strip() else ""
            out.append(lead + mem.text.strip())
        body = "\n    ".join(out)
        text = c.text[:c.body_open + 1] + "\n    " + body + "\n" + c.text[close:]
        return self.rewrite(text)

    def convert_var(self, c):
        t = self.rewrite(c.text)
        if c.role == "hostconst":
            t = re.sub(r"^(\s*)(?:static\s+)?", r"\1__constant ", t, count=1)
            t = re.sub(r"\bconstexpr\b\s*", "const ", t)
            t = re.sub(r"\bconst\s+const\b", "const", t)
        elif c.role == "constant":
            if "__constant" not in t:
                t = "__constant " + t.lstrip()
            t = re.sub(r"\bconstexpr\b\s*", "", t)
        elif c.role == "devglobal":
            if self.opts.device_globals == "constant":
                t = "__constant " + t.lstrip()
                self.diag("warning", "mutable __device__ variable emitted as __constant (--device-globals=constant); "
                          "writes to it will no longer compile", c)
            else:
                t = "__global " + t.lstrip()
                self.diag("warning", "mutable program-scope __device__ variable: clspv/Vulkan don't support "
                          "program-scope global variables", c, 0,
                          "pass it as a kernel buffer argument, or use --device-globals=constant if it's never written")
        elif c.role == "shared-var":
            self.diag("error", "top-level __shared__ variable; OpenCL __local variables must live in a kernel", c)
        return t

    def array_params_to_pointers(self, c):
        """`T a[16]` parameters decay to __private pointers in OpenCL; make them plain (generic) pointers."""
        if c.kind not in ("func", "proto"):
            return c.text
        hdr = c.header if c.kind == "func" else c.masked
        _, _, paren, _, _ = _func_name(hdr)
        if paren < 0:
            return c.text
        close = scan.matching_paren(c.masked, paren)
        if close < 0:
            return c.text
        parts = scan.top_level_commas(c.text[paren + 1:close])
        changed = False
        out = []
        for p in parts:
            mp = scan.mask(p)
            mm = re.search(r"([A-Za-z_]\w*)\s*\[[^\]]*\]((?:\s*\[[^\]]*\])*)\s*$", mp)
            if mm and "(" not in mp:
                rest = p[mm.start(2):mm.end(2)].strip()
                name = mm.group(1)
                p = p[:mm.start()] + (f"(*{name}){rest}" if rest else f"*{name}") + p[mm.end():]
                changed = True
            out.append(p)
        if not changed:
            return c.text
        return c.text[:paren + 1] + ",".join(out) + c.text[close:]

    def convert_chunk(self, c):
        if c.role == "kernel":
            return self.convert_kernel(c)
        if c.role == "device" and c.kind in ("func", "proto"):
            return self.rewrite(self.array_params_to_pointers(c))
        if c.kind == "record":
            return self.convert_record(c)
        if c.role in ("constant", "hostconst", "devglobal", "shared-var"):
            return self.convert_var(c)
        if c.role == "pp":
            return self.convert_pp(c)
        return self.rewrite(c.text)

    def convert_pp(self, c):
        if re.match(r"\s*#\s*pragma\s+(?:once|nv_|diag_|warning)", c.masked):
            return ""
        return self.rewrite(c.text)

    # ---- language detection ----------------------------------------------
    def detect_std(self, kept):
        if self.opts.std in ("c", "cpp"):
            return self.opts.std
        if "cpp" in self.features:
            return "cpp"
        seen_funcs = {}
        record_tags = set()
        for c in kept:
            m = c.masked
            if CPP_HINT_RE.search(m):
                return "cpp"
            if c.kind == "func":
                _, _, paren, _, _ = _func_name(c.header)
                close = scan.matching_paren(m, paren)
                params = m[paren:close + 1]
                if re.search(r"[\w>\]]\s*&\s*\w+\s*[,)=]", params) or "=" in params:
                    return "cpp"
                n = next(iter(c.names), None)
                if n in seen_funcs:
                    return "cpp"  # overload
                seen_funcs[n] = c
            if c.kind == "record" and not re.match(r"\s*typedef\b", m):
                tag = re.search(r"\b(?:struct|union|enum)\s+(\w+)", m)
                if tag:
                    record_tags.add(tag.group(1))
        for c in kept:
            m = c.masked
            for tag in record_tags:
                for mm in re.finditer(rf"\b{re.escape(tag)}\b", m):
                    before = m[:mm.start()].rstrip()
                    if not re.search(r"\b(?:struct|union|enum)$", before):
                        return "cpp"
        return "c"

    # ---- host launch sites -------------------------------------------------
    def find_launches(self):
        for c in self.all_chunks():
            if c.role != "host" or c.kind != "func":
                continue
            m = c.masked
            for mm in re.finditer(r"\b(\w+)\s*(?:<[^<>;()]*>)?\s*<<<(.*?)>>>\s*\(", m, re.S):
                cfg = [x.strip() for x in scan.top_level_commas(c.text[mm.start(2):mm.end(2)])]
                open_i = m.index("(", mm.end() - 1)
                close = scan.matching_paren(m, open_i)
                args = [x.strip() for x in scan.top_level_commas(c.text[open_i + 1:close])]
                f, l = self.loc(c, mm.start())
                self.launches.append({
                    "kernel": mm.group(1), "grid": cfg[0] if cfg else "", "block": cfg[1] if len(cfg) > 1 else "",
                    "shared_bytes": cfg[2] if len(cfg) > 2 else "0", "stream": cfg[3] if len(cfg) > 3 else "",
                    "args": args, "host_function": next(iter(c.names), ""), "source": f"{f}:{l}"})

    def numeric(self, expr):
        """Evaluate a launch-config expression if it is a literal / simple macro / dim3(...)."""
        e = expr.strip()
        mm = re.fullmatch(r"dim3\s*\((.*)\)", e)
        if mm:
            vals = [self.numeric(x) for x in scan.top_level_commas(mm.group(1))]
            if all(isinstance(v, int) for v in vals):
                vals = (vals + [1, 1, 1])[:3]
                return tuple(vals)
            return None
        for _ in range(4):
            if re.fullmatch(r"\d+[uUlL]*", e):
                return int(re.sub(r"[uUlL]", "", e))
            defs = self.by_name.get(e, [])
            val = None
            for d in defs:
                dm = re.match(rf"\s*#\s*define\s+{re.escape(e)}\s+\(?\s*(\d+)[uUlL]*\s*\)?\s*$", d.text)
                vm = re.search(rf"\b{re.escape(e)}\s*=\s*(\d+)[uUlL]*\s*;", d.text) if d.role in ("hostconst", "constant") else None
                if dm or vm:
                    val = (dm or vm).group(1)
            if val is None:
                return None
            e = val
        return None

    def infer_wgsizes(self):
        sizes = {}
        for k in self.selected:
            name = next(iter(k.names))
            if name in self.opts.wgsize:
                sizes[name] = tuple(self.opts.wgsize[name])
                continue
            if not self.opts.auto_wgsize:
                continue
            blocks = {l["block"] for l in self.launches if l["kernel"] == name}
            vals = {self.numeric(b) for b in blocks}
            if len(vals) == 1:
                v = vals.pop()
                if isinstance(v, int):
                    v = (v, 1, 1)
                if v:
                    sizes[name] = v
                    self.d.note(f"kernel '{name}': fixed work-group size {v} inferred from its <<<..., {blocks.pop()}>>> "
                                "launch (override with --wgsize)")
        self.wgsizes = sizes

    # ---- driver --------------------------------------------------------------
    def run(self):
        chunks = scan.split_top(self.src.text)
        self.chunks = self.flatten(chunks)
        for c in self.chunks:
            self.classify(c)
        self.index(self.chunks)
        self.reach()
        self.find_launches()
        self.infer_wgsizes()

        kept = [c for c in self.all_chunks() if c.keep]
        for c in kept:
            if c.role in ("kernel", "device") or c.kind in ("func", "proto", "record"):
                self.scan_device_code(c)
        # out-of-line device member functions imply C++
        for c in kept:
            if c.kind == "func" and getattr(c, "qualified", "") and "::" in c.qualified:
                self.features.add("cpp")
        converted = {}
        for c in self.all_chunks():
            if c.keep or c.role == "pp":
                converted[id(c)] = self.convert_chunk(c)
        std = self.detect_std(kept)
        text, line_map = self.emit(converted, std)
        manifest = {
            "source": self.path,
            "std": std,
            "kernels": self.kernels,
            "launches": self.launches,
            "features": sorted(self.features - {"cpp"}),
            "dropped_includes": sorted(set(self.src.system_includes)),
        }
        return Result(text, std, manifest, self.d, line_map)

    def emit(self, converted, std):
        lines, lmap = [], []

        def add(block, origin=None, follow=False):
            for i, ln in enumerate(block.split("\n")):
                lines.append(ln)
                if origin and follow:
                    lmap.append((origin[0], origin[1] + i))
                else:
                    lmap.append(origin)

        hdr = (f"/* Generated by starcross cu2cl from {os.path.basename(self.path)}\n"
               f" * Language: {'C++ for OpenCL 2021 (-cl-std=CLC++2021)' if std == 'cpp' else 'OpenCL C 3.0 (-cl-std=CL3.0)'}\n"
               f" * Kernels: {', '.join(k['name'] for k in self.kernels) or '(none)'}\n */")
        add(hdr)
        add(PRELUDE.strip("\n"))
        add("")

        def walk(chunks):
            for c in chunks:
                if c.role == "namespace":
                    inner_kept = any(x.keep for x in self.all_chunks(c.children))
                    if not inner_kept:
                        continue
                    add(scan.mask(c.text[:c.body_open]).strip() and c.text[:c.body_open].strip() + " {")
                    walk(c.children)
                    add("}")
                    continue
                if id(c) not in converted:
                    continue
                out = converted[id(c)]
                if not out.strip():
                    continue
                origin = self.src.origin(c.line)
                if c.role != "pp":
                    add("")
                    if c.lead:
                        lead = "\n".join(l for l in c.lead.split("\n") if "starcross:" not in l).strip("\n")
                        if lead.strip():
                            add(lead)
                    if self.opts.line_directives:
                        add(f'#line {origin[1]} "{origin[0]}"')
                    elif self.opts.provenance:
                        add(f"// from {origin[0]}:{origin[1]}")
                add(out, origin, follow=True)

        walk(self.chunks)
        if std == "c":
            for i, ln in enumerate(lines):
                lines[i] = scan.map_code(ln, lambda t: re.sub(r"\bnullptr\b", "0", t))
        return "\n".join(lines) + "\n", lmap


def convert(path, opts=None, diags=None):
    opts = opts or Options()
    diags = diags if diags is not None else Diagnostics()
    return Converter(path, opts, diags).run()
