"""AdamW optimizer (reference, deterministic)."""
import math


class AdamW:
    def __init__(self, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.betas
        for key, g in grads.items():
            p = params[key]
            self.m.setdefault(key, 0.0)
            self.v.setdefault(key, 0.0)
            self.m[key] = b1 * self.m[key] + (1 - b1) * g
            self.v[key] = b2 * self.v[key] + (1 - b2) * g * g
            m_hat = self.m[key] / (1 - b1 ** self.t)
            v_hat = self.v[key] / (1 - b2 ** self.t)
            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            update += self.lr * self.weight_decay * p
            params[key] = p - update
        return params
