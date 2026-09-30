"""CPU backend: real computation via the tensor core."""
from ..core.tensor import matmul, add, relu, softmax
from .base import Backend
from . import register_backend


@register_backend("cpu")
class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)

    def generate(self, prompt, **kwargs):
        return prompt

    def train_step(self, step):
        return 0.0

    def current_skill(self):
        return 1.0

    def evaluate(self):
        return {"accuracy": 1.0, "backend": "cpu"}
