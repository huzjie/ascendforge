"""AscendTransformer: a runnable MoE decoder-only transformer (reference)."""
from ..core.tensor import Tensor, matmul, zeros
from ..utils.stable import stable_vector
from .block import MoEDecoderBlock
from .model_config import ModelConfig


class AscendTransformer:
    def __init__(self, cfg: ModelConfig = None):
        self.cfg = cfg or ModelConfig()
        self.embed = self._w("embed", self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = [MoEDecoderBlock(self.cfg, layer_id=i) for i in range(self.cfg.num_layers)]
        self.lm_head = self._w("lm_head", self.cfg.hidden_size, self.cfg.vocab_size)

    def _w(self, name, r, c):
        data = stable_vector(f"model:{name}", r * c, -0.02, 0.02)
        return Tensor(data, (r, c), device="npu")

    def forward(self, input_ids, kv_cache=None):
        """input_ids: list of token ids -> logits (1, vocab)."""
        t = len(input_ids)
        emb = zeros((t, self.cfg.hidden_size), device="npu")
        for i, tid in enumerate(input_ids):
            tid = tid % self.cfg.vocab_size
            for j in range(self.cfg.hidden_size):
                emb.data[i * self.cfg.hidden_size + j] = self.embed.data[tid * self.cfg.hidden_size + j]
        h = emb
        for layer in self.layers:
            h = layer.forward(h)
        return matmul(h, self.lm_head)
