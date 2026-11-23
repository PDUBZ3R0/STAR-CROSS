"""Clang-assisted address-space fixups for converted OpenCL code.

CUDA code freely passes __constant__ lookup tables to helpers that take plain
pointers. In OpenCL, unqualified pointers are __generic, and __constant memory
can never be converted to __generic. Rather than guess, we compile the output
with clang, read the diagnostics, and patch exactly what it complains about:

  * "no matching function ... no known conversion from '__constant T' ... for
     Nth argument"  -> clone that function with parameter N qualified __constant
  * "cannot initialize a variable of type 'T __generic*' with ... '__constant'"
     -> qualify that local pointer declaration __constant

Repeat until clang stops reporting such errors (clones can cascade into callees).
"""

import os
import re
import shutil
import subprocess
import tempfile

from . import scan

DIAG_RE = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+): (?P<level>error|warning|note): (?P<msg>.*)$")
ORDINAL_RE = re.compile(r"for (\d+)(?:st|nd|rd|th) argument")


def find_clang():
    for name in ("clang", "clang-22", "clang-21", "clang-20", "clang-19", "clang-18", "clang-17"):
        p = shutil.which(name)
        if p:
            return p
    return None


def clang_args(std):
    if std == "cpp":
        return ["-x", "clcpp", "-cl-std=clc++2021"]
    return ["-x", "cl", "-cl-std=CL3.0"]


def run_clang(text, std, clang, extra=()):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "input.clcpp" if std == "cpp" else "input.cl")
        with open(path, "w") as f:
            f.write(text)
        cmd = [clang, *clang_args(std), "-Xclang", "-finclude-default-header", "-fsyntax-only",
               "-fno-color-diagnostics", "-fno-caret-diagnostics", "-ferror-limit=0", *extra, path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    diags = []
    for line in proc.stderr.splitlines():
        m = DIAG_RE.match(line)
        if m and os.path.basename(m.group("file")).startswith("input."):
            diags.append({"line": int(m.group("line")), "col": int(m.group("col")),
                          "level": m.group("level"), "msg": m.group("msg")})
        elif m:
            diags.append({"line": 0, "col": 0, "level": m.group("level"), "msg": m.group("msg")})
    return proc.returncode, diags


def _offset(lines, line, col):
    return sum(len(l) + 1 for l in lines[:line - 1]) + col - 1


def _param_site(text, line, col):
    """Given the location of a parameter inside a definition, return (name_line, name_col, argn)."""
    lines = text.split("\n")
    off = _offset(lines, line, col)
    m = scan.mask(text)
    depth, i = 0, off
    while i > 0:
        i -= 1
        if m[i] == ")":
            depth += 1
        elif m[i] == "(":
            if depth == 0:
                break
            depth -= 1
    open_i = i
    argn = len(scan.top_level_commas(m[open_i + 1:off] + "x"))
    j = open_i
    while j > 0 and m[j - 1] in " \t\n":
        j -= 1
    k = j
    while k > 0 and (m[k - 1].isalnum() or m[k - 1] in "_:"):
        k -= 1
    before = text[:k]
    return before.count("\n") + 1, k - (before.rfind("\n") + 1) + 1, argn


def _clone_function(text, def_line, def_col, argn, std):
    lines = text.split("\n")
    name_off = _offset(lines, def_line, def_col)
    m = scan.mask(text)
    # walk back to the start of the declaration (previous ';', '}' or blank/comment line at depth 0)
    start = name_off
    while start > 0:
        ch = m[start - 1]
        if ch in ";}":
            break
        start -= 1
    while start < name_off and text[start] in " \t\n":
        start += 1
    if text[start:].startswith("// from"):
        start = text.index("\n", start) + 1
    paren = m.find("(", name_off)
    close = scan.matching_paren(m, paren)
    brace = m.find("{", close)
    semi = m.find(";", close)
    if paren < 0 or close < 0 or brace < 0 or (0 <= semi < brace):
        return None
    depth, end = 0, -1
    for i in range(brace, len(m)):
        if m[i] == "{":
            depth += 1
        elif m[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    params = scan.top_level_commas(text[paren + 1:close])
    if argn - 1 >= len(params):
        return None
    p = params[argn - 1]
    if "__constant" in p:
        return None
    first = len(p) - len(p.lstrip())
    params[argn - 1] = p[:first] + "__constant " + p[first:]
    header = text[start:paren + 1]
    ovl = ""
    if std == "c" and "overloadable" not in header:
        ovl = "__attribute__((overloadable)) "
    clone = (ovl + header + ",".join(params) + text[close:end])
    clone_sig = re.sub(r"\s+", " ", ovl + header + ",".join(params) + ")")
    if clone_sig in re.sub(r"\s+", " ", text):
        return None  # already cloned
    original = text[start:end]
    new_original = (ovl + original) if ovl else original
    replacement = new_original + "\n\n// starcross: overload taking __constant memory for argument %d\n" % argn + clone
    return text[:start] + replacement + text[end:], start, end, replacement


def _qualify_decl(text, line, col):
    lines = text.split("\n")
    off = _offset(lines, line, col)
    m = scan.mask(text)
    start = off
    while start > 0 and m[start - 1] not in ";{}\n":
        start -= 1
    # skip back over a preceding line break inside the statement
    while start < off and text[start] in " \t":
        start += 1
    stmt = text[start:off]
    if "__constant" in stmt:
        return None
    return text[:start] + "__constant " + text[start:], start


def fix_address_spaces(text, std, clang=None, max_rounds=12, log=None):
    """Return (text, applied_fix_descriptions, remaining_clang_diags)."""
    clang = clang or find_clang()
    if not clang:
        return text, [], None
    applied = []
    diags = []
    for _ in range(max_rounds):
        rc, diags = run_clang(text, std, clang)
        todo = []
        for i, d in enumerate(diags):
            if d["level"] != "error":
                continue
            msg = d["msg"]
            if msg.startswith("no matching function for call to"):
                for n in diags[i + 1:]:
                    if n["level"] != "note":
                        break
                    if "__constant" in n["msg"] and "no known conversion" in n["msg"]:
                        o = ORDINAL_RE.search(n["msg"])
                        if o:
                            todo.append(("clone", n["line"], n["col"], int(o.group(1)), d["line"]))
                            break
            elif "changes address space of pointer" in msg and msg.startswith("passing '__constant"):
                for n in diags[i + 1:]:
                    if n["level"] != "note":
                        break
                    if "passing argument to parameter" in n["msg"]:
                        nl, nc, argn = _param_site(text, n["line"], n["col"])
                        todo.append(("clone", nl, nc, argn, d["line"]))
                        break
            elif ("cannot initialize a variable of type" in msg and "__constant" in msg.split(" with ")[-1]) or \
                    ("initializing" in msg and "__constant" in msg and "discards qualifiers" not in msg):
                todo.append(("decl", d["line"], d["col"], 0, d["line"]))
        if not todo:
            break
        # apply bottom-up so earlier line numbers stay valid; one fix per function per round
        seen = set()
        changed = False
        for kind, line, col, argn, at in sorted(set(todo), key=lambda t: (-t[1], -t[2])):
            key = (kind, line)
            if key in seen:
                continue
            seen.add(key)
            if kind == "clone":
                r = _clone_function(text, line, col, argn, std)
                if r:
                    text = r[0]
                    applied.append(f"cloned function at output line {line} with __constant parameter {argn} "
                                   f"(called from line {at})")
                    changed = True
            else:
                r = _qualify_decl(text, line, col)
                if r:
                    text = r[0]
                    applied.append(f"qualified pointer declaration at output line {line} as __constant")
                    changed = True
        if not changed:
            break
    rc, diags = run_clang(text, std, clang)
    return text, applied, diags
