"""SwiGLU expert FFN."""
from ..core.tensor import Tensor, matmul, mul, silu, zeros
from ..utils.stable import stable_vector


class SwiGLUExpert:
    def __init__(self, hidden, expert_hidden, expert_id=0, key="expert"):
        self.hidden = hidden
        self.expert_hidden = expert_hidden
        self.expert_id = expert_id
        self.key = key
        self.w_gate = self._w("gate", hidden, expert_hidden)
        self.w_up = self._w("up", hidden, expert_hidden)
        self.w_down = self._w("down", expert_hidden, hidden)

    def _w(self, name, r, c):
        data = stable_vector(f"{self.key}:{self.expert_id}:{name}", r * c, -0.1, 0.1)
        return Tensor(data, (r, c), device="npu")

    def forward(self, x):
        g = silu(matmul(x, self.w_gate))
        u = matmul(x, self.w_up)
        h = mul(g, u)
        return matmul(h, self.w_down)
