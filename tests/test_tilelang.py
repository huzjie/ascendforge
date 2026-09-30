import unittest

from ascendforge.tilelang import compile_source, Simulator, TileProgram, Op, OpKind, Tile
from ascendforge.core.tensor import ones, matmul


class TestTileLang(unittest.TestCase):
    def test_compile_source(self):
        src = "program k() { tile A : fp16[4,4] @ GM\n gemm(in=A,B, out=C) }"
        result = compile_source(src)
        self.assertIn("ascend_c", result)
        self.assertIn("k()", result["ascend_c"])

    def test_simulator_gemm(self):
        prog = TileProgram("k", [Tile("A", (2, 2)), Tile("B", (2, 2)), Tile("C", (2, 2))])
        prog.body = [Op(OpKind.GEMM, ["A", "B"], ["C"])]
        sim = Simulator()
        sim.feed("A", ones((2, 2)))
        sim.feed("B", ones((2, 2)))
        out = sim.run(prog)
        self.assertEqual(out["C"].shape, (2, 2))
        self.assertAlmostEqual(out["C"].data[0], 2.0)


if __name__ == "__main__":
    unittest.main()
