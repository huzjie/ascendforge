import unittest

from ascendforge.core.tensor import randn, ones, matmul
from ascendforge.ops.deepgemm import gemm, grouped_gemm
from ascendforge.ops.tilekernels import rmsnorm, swiglu
from ascendforge.ops.flashmla import mla_forward, compress_kv, MlaContext
from ascendforge.ops.deepselect import minhash_dedup, quality_score, select_documents


class TestOps(unittest.TestCase):
    def test_gemm_shape(self):
        a = randn((4, 6), "g_a")
        b = randn((6, 3), "g_b")
        c = gemm(a, b)
        self.assertEqual(c.shape, (4, 3))

    def test_grouped_gemm(self):
        inputs = [randn((2, 4), f"i{i}") for i in range(3)]
        weights = [randn((4, 2), f"w{i}") for i in range(3)]
        outs = grouped_gemm(inputs, weights)
        self.assertEqual(len(outs), 3)
        self.assertEqual(outs[0].shape, (2, 2))

    def test_swiglu(self):
        x = ones((2, 2))
        g = ones((2, 2))
        out = swiglu(x, g)
        self.assertEqual(out.shape, (2, 2))

    def test_mla(self):
        t = 8
        h = randn((t, 16), "h")
        q = randn((t, 16), "q")
        w_down = randn((16, 8), "wd")
        w_k = randn((8, 16), "wk")
        w_v = randn((8, 16), "wv")
        c = compress_kv(h, w_down)
        out = mla_forward(q, c, w_k, w_v, MlaContext(kv_lora_rank=8))
        self.assertEqual(out.shape, (t, 16))

    def test_dedup(self):
        docs = ["alpha beta gamma delta epsilon zeta", "alpha beta gamma delta epsilon zeta",
                "completely different sentence here ok"]
        kept = select_documents(docs, threshold=0.9, quality_min=0.0, curriculum=False)
        self.assertLessEqual(len(kept), len(docs))

    def test_quality_in_range(self):
        s = quality_score("this is a fairly normal sentence with enough words to score")
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
