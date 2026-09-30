"""Top-K gating router."""
import math

from ..utils.stable import stable_float


class TopKRouter:
    def __init__(self, num_experts=8, top_k=2, hidden=512, key="router"):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.key = key

    def logits(self, token_embedding):
        """Deterministic logits over experts."""
        n = self.num_experts
        return [stable_float(f"{self.key}:{i}:{token_embedding}", -1, 1) for i in range(n)]

    def route(self, token_embedding):
        """Return the top-k expert ids and their (softmax) weights."""
        logits = self.logits(token_embedding)
        topk = sorted(range(self.num_experts), key=lambda i: logits[i], reverse=True)[:self.top_k]
        mx = max(logits)
        ex = [math.exp(x - mx) for x in logits]
        s = sum(ex)
        probs = [e / s for e in ex]
        weights = [probs[i] for i in topk]
        # normalize weights
        sw = sum(weights) or 1.0
        weights = [w / sw for w in weights]
        return topk, weights
