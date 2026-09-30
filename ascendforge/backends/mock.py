"""Deterministic, trainable mock backend.

Models a "skill" scalar that rises monotonically during training and directly
drives prediction correctness, giving a meaningful (and reproducible) learning
signal for the Ascend operator + MoE + MLA stack.
"""
from ..utils.stable import stable_float
from .base import Backend
from . import register_backend


@register_backend("mock")
class MockBackend(Backend):
    name = "mock"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.skill = 0.5
        self.noise = 0.35
        self._train_step = 0

    def _score(self, prompt):
        """skill * correctness + noise, where correctness is a fixed latent signal."""
        correctness = stable_float(f"correct:{prompt}", 0.0, 1.0)
        return self.skill * 1.0 * correctness + self.noise * (0.5 - correctness)

    def generate(self, prompt, **kwargs):
        return self._score(prompt) > 0.5

    def train_step(self, step):
        self._train_step = step
        # monotone skill ascent toward 1.0 (delta always positive)
        delta = 0.05 * (1.0 - self.skill) + 0.001
        self.skill = min(1.0, self.skill + delta)
        # loss declines as skill grows
        loss = (1.0 - self.skill) * stable_float(f"loss:{step}", 0.5, 1.5)
        return loss

    def current_skill(self):
        return self.skill

    def evaluate(self):
        correct = 0
        n = 50
        for i in range(n):
            if self.generate(f"q{i}"):
                correct += 1
        return {"accuracy": correct / n, "skill": self.skill}
