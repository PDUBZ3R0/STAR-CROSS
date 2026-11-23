"""Lightweight C/C++/CUDA source scanning.

This is deliberately not a real parser. It understands enough structure to:
  * inline local #include "..." files (keeping track of where each line came from),
  * blank out comments and string literals so structural scans aren't fooled,
  * split a translation unit into top-level declarations (functions, records,
    variables, preprocessor directives, namespaces / extern "C" blocks).
"""

import bisect
import os
import re
from dataclasses import dataclass, field

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\b")


# --------------------------------------------------------------------------
# Comment / literal masking
# --------------------------------------------------------------------------

def _scan(text):
    """Yield (start, end, kind) spans; kind in {'code','comment','string'}."""
    i, n = 0, len(text)
    code_start = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] in "/*":
            if code_start < i:
                yield code_start, i, "code"
            if text[i + 1] == "/":
                j = text.find("\n", i)
                j = n if j < 0 else j
            else:
                j = text.find("*/", i + 2)
                j = n if j < 0 else j + 2
            yield i, j, "comment"
            i = code_start = j
            continue
        if c in "\"'":
            # don't treat digit separators (1'000) as char literals
            if c == "'" and i > 0 and text[i - 1].isalnum() and i + 1 < n and text[i + 1].isalnum():
                i += 1
                continue
            if code_start < i:
                yield code_start, i, "code"
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 1
                elif text[j] == "\n":  # unterminated; bail at end of line
                    break
                j += 1
            j = min(j + 1, n)
            yield i, j, "string"
            i = code_start = j
            continue
        i += 1
    if code_start < n:
        yield code_start, n, "code"


def mask(text):
    """Return text with comments blanked and string contents replaced by spaces.

    Length and newlines are preserved so offsets/line numbers line up.
    """
    out = []
    for s, e, kind in _scan(text):
        chunk = text[s:e]
        if kind == "code":
            out.append(chunk)
        elif kind == "comment":
            out.append(re.sub(r"[^\n]", " ", chunk))
        else:
            q = chunk[0]
            body = re.sub(r"[^\n]", " ", chunk[1:-1]) if len(chunk) >= 2 else ""
            out.append(q + body + (chunk[-1] if len(chunk) >= 2 else ""))
    return "".join(out)


def map_code(text, fn):
    """Apply fn to code segments only (comments and literals untouched)."""
    return "".join(fn(text[s:e]) if k == "code" else text[s:e] for s, e, k in _scan(text))


def strip_comments(text):
    return "".join(text[s:e] for s, e, k in _scan(text) if k != "comment")


# --------------------------------------------------------------------------
# Include inlining
# --------------------------------------------------------------------------

@dataclass
class Source:
    text: str
    origins: list  # per line: (file, lineno)
    system_includes: list = field(default_factory=list)

    def origin(self, line_index):
        if 0 <= line_index < len(self.origins):
            return self.origins[line_index]
        return ("?", 0)


def load_source(path, include_dirs=(), diags=None):
    seen = set()
    lines, origins, sysinc = [], [], []

    def find(name, cur_dir, quoted):
        cands = ([cur_dir] if quoted else []) + list(include_dirs)
        for d in cands:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return os.path.realpath(p)
        return None

    def visit(path):
        rp = os.path.realpath(path)
        if rp in seen:
            return
        seen.add(rp)
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read().splitlines()
        cur_dir = os.path.dirname(rp)
        disp = os.path.relpath(rp) if not os.path.relpath(rp).startswith("..") else rp
        masked = mask("\n".join(src)).split("\n")
        for no, line in enumerate(src, 1):
            ml = masked[no - 1] if no - 1 < len(masked) else line
            if re.match(r"^\s*#\s*include\b", ml):
                m = INCLUDE_RE.match(line)
                if m:
                    quoted = m.group(1) == '"'
                    found = find(m.group(2), cur_dir, quoted)
                    if found:
                        lines.append(f"// starcross: begin {m.group(2)}")
                        origins.append((disp, no))
                        visit(found)
                        lines.append(f"// starcross: end {m.group(2)}")
                        origins.append((disp, no))
                    else:
                        sysinc.append(m.group(2))
                        if quoted and diags is not None and not _is_cuda_header(m.group(2)):
                            diags.note(f'include "{m.group(2)}" not found; skipped', disp, no)
                        lines.append(f"// starcross: dropped #include {m.group(1)}{m.group(2)}")
                        origins.append((disp, no))
                    continue
            if PRAGMA_ONCE_RE.match(ml):
                lines.append("")
                origins.append((disp, no))
                continue
            lines.append(line)
            origins.append((disp, no))

    visit(path)
    return Source("\n".join(lines) + "\n", origins, sysinc)


CUDA_HEADERS = {
    "cuda_runtime.h", "cuda.h", "device_launch_parameters.h", "cuda_runtime_api.h",
    "device_functions.h", "math_functions.h", "cuda_fp16.h", "cuda_bf16.h",
    "vector_types.h", "vector_functions.h", "helper_cuda.h", "helper_math.h",
    "cooperative_groups.h", "curand_kernel.h", "sm_20_atomic_functions.h",
}


def _is_cuda_header(name):
    return os.path.basename(name) in CUDA_HEADERS


# --------------------------------------------------------------------------
# Top-level splitting
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    line: int          # 0-based line index into the combined source
    kind: str          # pp | func | record | decl | container | empty
    header: str = ""   # masked text before the first top-level '{' (or whole decl)
    children: list = field(default_factory=list)
    body_open: int = -1  # offset of the first top-level '{' within text

    # filled in by analysis
    names: set = field(default_factory=set)
    refs: set = field(default_factory=set)
    role: str = ""     # kernel | device | host | shared | constant | global | hostvar | pp
    keep: bool = False
    out: str = ""
    lead: str = ""     # comments immediately preceding the chunk


RECORD_RE = re.compile(r"^\s*(?:template\s*<[^{]*?>\s*)?(?:typedef\s+)?(?:struct|class|union|enum)\b")
CONTAINER_NS_RE = re.compile(r"^\s*(?:inline\s+)?namespace\b[\w\s:]*$")
CONTAINER_EXTERN_RE = re.compile(r'^\s*extern\s*"\s*"\s*$')
FUNC_TAIL_RE = re.compile(
    r"\)\s*(?:const|noexcept(?:\s*\([^)]*\))?|override|final|mutable|volatile|&|&&|"
    r"->\s*[\w:<>,\s\*&]+?|__attribute__\s*\(\(.*?\)\))*\s*$",
    re.S,
)


def _line_index(newlines, pos):
    return bisect.bisect_right(newlines, pos - 1)


def classify_header(hdr, has_body):
    h = hdr.strip()
    if not h:
        return "empty"
    if has_body:
        if CONTAINER_NS_RE.match(h) or CONTAINER_EXTERN_RE.match(h):
            return "container"
        if RECORD_RE.match(h) and not FUNC_TAIL_RE.search(h):
            return "record"
        if h.endswith("=") or h.endswith("= "):
            return "decl"
        if "(" in h and (FUNC_TAIL_RE.search(h) or re.search(r"\)\s*:\s*[\s\S]*\)\s*$", h)):
            return "func"
        if RECORD_RE.match(h):
            return "record"
        return "decl"
    if RECORD_RE.match(h):
        return "record"
    return "decl"


def split_top(text, line0=0, masked=None):
    """Split text into top-level Chunks. line0 is the line index of text[0]."""
    m = masked if masked is not None else mask(text)
    newlines = [i for i, ch in enumerate(m) if ch == "\n"]
    chunks = []
    n = len(m)
    i = 0
    depth = 0
    start = None
    brace = -1
    prev_end = 0

    def lead_text(s):
        between = text[prev_end:s]
        return between.strip("\n") if between.strip() else ""

    def emit(s, e, kind=None):
        nonlocal start, brace, prev_end
        t = text[s:e]
        mt = m[s:e]
        bo = brace - s if brace >= 0 else -1
        hdr = mt[:bo] if bo >= 0 else mt
        k = kind or classify_header(hdr if bo >= 0 else hdr.rstrip().rstrip(";"), bo >= 0)
        if k != "empty" or t.strip(" \t\n;"):
            if k == "empty":
                k = "decl"
            ch = Chunk(t, line0 + _line_index(newlines, s), k, hdr, body_open=bo, lead=lead_text(s))
            if k == "container":
                close = mt.rfind("}")
                inner_s = bo + 1
                ch.children = split_top(t[inner_s:close], ch.line + t[:inner_s].count("\n"),
                                        mt[inner_s:close])
            chunks.append(ch)
        prev_end = e
        start = None
        brace = -1

    while i < n:
        c = m[i]
        if start is None:
            if c.isspace() or c == ";":
                i += 1
                continue
            if c == "#":
                j = i
                while True:
                    k = m.find("\n", j)
                    if k < 0:
                        k = n
                        break
                    if m[k - 1] == "\\":
                        j = k + 1
                        continue
                    break
                chunks.append(Chunk(text[i:k], line0 + _line_index(newlines, i), "pp", m[i:k],
                                    lead=lead_text(i)))
                prev_end = k
                i = k
                continue
            start = i
        if c in "([{":
            if c == "{" and depth == 0 and brace < 0:
                brace = i
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                depth = 0
            if c == "}" and depth == 0:
                hdr = m[start:brace] if brace >= 0 else ""
                kind = classify_header(hdr, True)
                if kind in ("func", "container"):
                    emit(start, i + 1, kind)
                    i += 1
                    continue
        elif c == ";" and depth == 0:
            emit(start, i + 1)
            i += 1
            continue
        elif c == "#" and depth == 0 and start is not None:
            # directive in the middle of a declaration header: keep it inside the chunk
            pass
        i += 1
    if start is not None:
        emit(start, n)
    return chunks


IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def idents(masked_text):
    return set(IDENT_RE.findall(masked_text))


def top_level_commas(s):
    """Split s on commas not nested in (), [] or {}."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip() or parts:
        parts.append("".join(cur))
    return parts


def matching_paren(s, open_idx):
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1
