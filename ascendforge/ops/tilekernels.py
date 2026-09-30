"""TileKernels: fused vector & memory-access operators.

Regular vector compute (activation, normalization, RoPE) and memory-access ops
needed for data preparation — mirroring DeepSeek's TileKernels component.
"""
from ..core.tensor import Tensor, relu, gelu, silu, softmax, layernorm, rmsnorm, mul


def rmsnorm(t, eps=1e-6):
    from ..core.tensor import rmsnorm as _rms
    return _rms(t, eps)


def layernorm(t, eps=1e-5):
    from ..core.tensor import layernorm as _ln
    return _ln(t, eps)


def softmax(t):
    from ..core.tensor import softmax as _sm
    return _sm(t)


def rope(t, head_dim, position=0, base=10000.0):
    from ..core.tensor import rope as _rope
    return _rope(t, head_dim, position, base)


def swiglu(x, gate):
    """SiLU(x) * gate — the SwiGLU activation used in MoE experts."""
    return mul(silu(x), gate)


def fused_mlp(x, w_gate, w_up, w_down):
    """Fused SwiGLU MLP: down( silu(x @ w_gate) * (x @ w_up) )."""
    from .deepgemm import gemm
    g = silu(gemm(x, w_gate))
    u = gemm(x, w_up)
    h = mul(g, u)
    return gemm(h, w_down)
