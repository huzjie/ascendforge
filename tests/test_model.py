import unittest

from ascendforge.model.transformer import AscendTransformer
from ascendforge.model.model_config import ModelConfig
from ascendforge.model.mla import MultiHeadLatentAttention


class TestModel(unittest.TestCase):
    def test_transformer_logits_shape(self):
        cfg = ModelConfig(num_layers=2, hidden_size=64, num_heads=4,
                          num_experts=4, top_k=2, expert_hidden=128, vocab_size=1000)
        model = AscendTransformer(cfg)
        logits = model.forward([1, 2, 3])
        self.assertEqual(logits.shape, (3, 1000))

    def test_mla_cache(self):
        cfg = ModelConfig(hidden_size=64, kv_lora_rank=16, num_heads=4)
        mla = MultiHeadLatentAttention(cfg)
        from ascendforge.core.tensor import randn
        h = randn((8, 64), "h")
        out = mla.forward(h)
        self.assertEqual(out.shape, (8, 64))


if __name__ == "__main__":
    unittest.main()
