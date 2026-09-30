"""MoE layer: router + experts + load balancing."""
from ..core.tensor import Tensor, zeros
from .router import TopKRouter
from .expert import SwiGLUExpert


class MoELayer:
    def __init__(self, hidden=512, num_experts=8, top_k=2, expert_hidden=1024, key="moe"):
        self.router = TopKRouter(num_experts, top_k, hidden, key=key + ":router")
        self.experts = [SwiGLUExpert(hidden, expert_hidden, i, key=key + ":expert")
                        for i in range(num_experts)]
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self._load = [0] * num_experts

    def forward(self, x):
        """x: (T, hidden). Route each token to top-k experts and combine."""
        t, h = x.shape
        out = zeros((t, h), x.dtype, x.device)
        for i in range(t):
            # token embedding -> use a deterministic summary (mean of row)
            emb = sum(x.data[i * h:(i + 1) * h]) / h
            topk, weights = self.router.route(emb)
            acc = [0.0] * h
            for eid, w in zip(topk, weights):
                self._load[eid] += 1
                row = Tensor(x.data[i * h:(i + 1) * h], (1, h), x.dtype, x.device)
                eout = self.experts[eid].forward(row)
                for j in range(h):
                    acc[j] += w * eout.data[j]
            for j in range(h):
                out.data[i * h + j] = acc[j]
        return out

    def load_counts(self):
        return list(self._load)
