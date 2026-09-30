import unittest

from ascendforge.moe.moe_layer import MoELayer
from ascendforge.moe.router import TopKRouter
from ascendforge.moe.load_balance import aux_loss
from ascendforge.core.tensor import randn


class TestMoE(unittest.TestCase):
    def test_router_topk(self):
        r = TopKRouter(num_experts=8, top_k=2)
        topk, weights = r.route(0.5)
        self.assertEqual(len(topk), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_moe_layer_shape(self):
        layer = MoELayer(hidden=64, num_experts=4, top_k=2, expert_hidden=128)
        x = randn((8, 64), "moe_x")
        out = layer.forward(x)
        self.assertEqual(out.shape, (8, 64))

    def test_aux_loss(self):
        self.assertGreaterEqual(aux_loss([4, 4, 4, 4], 16), 0.0)


if __name__ == "__main__":
    unittest.main()
