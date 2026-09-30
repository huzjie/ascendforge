"""Multi-head Latent Attention (MLA) with low-rank KV cache."""
from ..core.tensor import Tensor, matmul, zeros
from ..ops.flashmla import mla_forward, compress_kv, MlaContext
from ..utils.stable import stable_vector


class MultiHeadLatentAttention:
    def __init__(self, cfg, layer_id=0, key="mla"):
        self.cfg = cfg
        self.hidden = cfg.hidden_size
        self.kv_lora_rank = cfg.kv_lora_rank
        self.layer_id = layer_id
        self.key = key
        # low-rank compression & decompression projections
        self.w_kv_down = self._w("kv_down", self.hidden, self.kv_lora_rank)
        self.w_k = self._w("w_k", self.kv_lora_rank, self.hidden)
        self.w_v = self._w("w_v", self.kv_lora_rank, self.hidden)
        self.w_q = self._w("w_q", self.hidden, self.hidden)
        self.ctx = MlaContext(kv_lora_rank=self.kv_lora_rank,
                              sliding_window=cfg.sliding_window,
                              sparse=cfg.sparse_attention)

    def _w(self, name, r, c):
        data = stable_vector(f"{self.key}:{self.layer_id}:{name}", r * c, -0.05, 0.05)
        return Tensor(data, (r, c), device="npu")

    def forward(self, hidden_states, kv_cache=None):
        """hidden_states: (T, hidden). kv_cache: compressed cache or None."""
        t, h = hidden_states.shape
        if kv_cache is None:
            # compress hidden states to the low-rank KV cache directly
            kv_cache = compress_kv(hidden_states, self.w_kv_down)
        q = matmul(hidden_states, self.w_q)
        return mla_forward(q, kv_cache, self.w_k, self.w_v, self.ctx)
