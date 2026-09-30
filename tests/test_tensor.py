import unittest

from ascendforge.core.tensor import Tensor, zeros, ones, matmul, add, mul, relu, softmax, rmsnorm, gelu


class TestTensor(unittest.TestCase):
    def test_matmul(self):
        a = ones((2, 3))
        b = ones((3, 4))
        c = matmul(a, b)
        self.assertEqual(c.shape, (2, 4))
        self.assertAlmostEqual(c.data[0], 3.0)

    def test_add_mul(self):
        a = ones((2, 2))
        b = ones((2, 2))
        self.assertAlmostEqual(add(a, b).data[0], 2.0)
        self.assertAlmostEqual(mul(a, b).data[0], 1.0)

    def test_relu(self):
        t = Tensor([-1.0, 2.0], (2,))
        self.assertEqual(relu(t).data, [0.0, 2.0])

    def test_softmax_sums_to_one(self):
        t = Tensor([1.0, 2.0, 3.0], (1, 3))
        self.assertAlmostEqual(sum(softmax(t).data), 1.0, places=5)

    def test_rmsnorm(self):
        t = ones((4,))
        self.assertAlmostEqual(rmsnorm(t).data[0], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
