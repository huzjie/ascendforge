"""MoE decoder block: MLA attention + MoE FFN + residuals + RMSNorm."""
from ..core.tensor import Tensor, add, mul
from ..ops.tilekernels import rmsnorm
from .mla import MultiHeadLatentAttention
from ..moe.moe_layer import MoELayer


class MoEDecoderBlock:
    def __init__(self, cfg, layer_id=0):
        self.attn = MultiHeadLatentAttention(cfg, layer_id=layer_id, key=f"block{layer_id}:attn")
        self.moe = MoELayer(hidden=cfg.hidden_size, num_experts=cfg.num_experts,
                            top_k=cfg.top_k, expert_hidden=cfg.expert_hidden,
                            key=f"block{layer_id}:moe")
        self.layer_id = layer_id

    def forward(self, x):
        # pre-norm + attention + residual
        normed = rmsnorm(x)
        attn_out = self.attn.forward(normed)
        x = add(x, attn_out)
        # pre-norm + MoE + residual
        normed = rmsnorm(x)
        moe_out = self.moe.forward(normed)
        return add(x, moe_out)
