"""Tests for the CUDA -> OpenCL stage. Run: python3 -m unittest discover -s tests"""

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from starcross import asfix, cu2cl  # noqa: E402
from starcross.diag import Diagnostics  # noqa: E402

EX = os.path.join(ROOT, "examples")
CLANG = asfix.find_clang()


def conv(name, **kw):
    d = Diagnostics()
    r = cu2cl.convert(os.path.join(EX, name), cu2cl.Options(**kw), d)
    text = r.text
    remaining = None
    if CLANG:
        text, _, remaining = asfix.fix_address_spaces(text, r.std, CLANG)
    return r, text, d, remaining


def clang_errors(remaining):
    return [x for x in (remaining or []) if x["level"] == "error"]


class Cu2ClTests(unittest.TestCase):
    def test_vecadd_is_plain_opencl_c(self):
        r, text, d, rem = conv("vecadd.cu")
        self.assertEqual(r.std, "c")
        self.assertEqual([k["name"] for k in r.manifest["kernels"]], ["vecadd"])
        self.assertIn("__kernel", text)
        self.assertIn("__global const float* __restrict a", text.replace("\n    ", " "))
        self.assertIn("get_local_id(0)", text)
        self.assertNotIn("int main", text)            # host code pruned
        self.assertNotIn("cudaMalloc", text)
        self.assertIn("reqd_work_group_size(256, 1, 1)", text)   # from <<<..., BLOCK>>>
        self.assertFalse(d.errors)
        self.assertFalse(clang_errors(rem), rem)

    def test_reduce_shared_memory_and_atomics(self):
        r, text, d, rem = conv("reduce.cu")
        self.assertIn("__local float partial[128];", text)
        self.assertIn("__local unsigned int *local_bins", text)  # extern __shared__ -> param
        hist = next(k for k in r.manifest["kernels"] if k["name"] == "histogram")
        self.assertEqual(hist["params"][-1]["kind"], "local")
        launches = {l["kernel"]: l for l in r.manifest["launches"]}
        self.assertEqual(launches["histogram"]["shared_bytes"], "64 * sizeof(unsigned int)")
        self.assertFalse(clang_errors(rem), rem)

    def test_constant_table_gets_overload(self):
        r, text, d, rem = conv("keccak.cu")
        self.assertIn("__constant uint64_t RC[24]", text)
        self.assertRegex(text, r"mix\(__constant const uint64_t\* table")
        self.assertIn("ulong count", text)        # size_t kernel arg
        self.assertIn("int64", r.manifest["features"])
        if CLANG:
            self.assertFalse(clang_errors(rem), rem)

    def test_cpp_features_select_cpp_for_opencl(self):
        r, text, d, rem = conv("particles.cu")
        self.assertEqual(r.std, "cpp")
        self.assertIn("void step(float dt)", text)
        self.assertNotIn("print_host", text)       # host-only member dropped
        self.assertIn("template <typename T>", text)
        self.assertTrue(any("float3" in x.message for x in d.items))
        self.assertFalse(clang_errors(rem), rem)

    def test_kernel_selection(self):
        r, text, d, rem = conv("reduce.cu", kernels=["histogram"])
        self.assertEqual([k["name"] for k in r.manifest["kernels"]], ["histogram"])
        self.assertNotRegex(text, r"void block_sum\(")
        self.assertNotIn("warp_safe_add", text)    # only reachable from block_sum

    def test_unsupported_constructs_are_reported(self):
        src = os.path.join(EX, "_bad.cu")
        with open(src, "w") as f:
            f.write("""
__global__ void child(int* p) { p[0] = 1; }
__global__ void parent(int* p) {
    int* tmp = (int*)malloc(16);
    child<<<1, 1>>>(p);
    asm("mov.u32 %0, 1;" : "=r"(p[1]));
}
""")
        try:
            d = Diagnostics()
            cu2cl.convert(src, cu2cl.Options(kernels=["parent"]), d)
            msgs = " | ".join(x.message for x in d.errors)
            self.assertIn("malloc", msgs)
            self.assertIn("dynamic parallelism", msgs)
            self.assertIn("PTX", msgs)
        finally:
            os.remove(src)


if __name__ == "__main__":
    unittest.main()
