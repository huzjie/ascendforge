"""A zero-dependency tensor with the operators needed by the framework.

Data is stored as a flat Python list of floats (plus a shape), which keeps the
reference implementation fully portable and deterministic. Performance kernels
are simulated in `ascendforge.ops`.
"""
from .dtype import DType


class Tensor:
    __slots__ = ("shape", "data", "dtype", "device")

    def __init__(self, data, shape=None, dtype=DType.fp32, device="npu"):
        self.data = list(data)
        if shape is None:
            shape = (len(self.data),)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    @staticmethod
    def _numel(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def reshape(self, *shape):
        return Tensor(self.data, shape, self.dtype, self.device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


def zeros(shape, dtype=DType.fp32, device="npu"):
    n = Tensor._numel(shape)
    return Tensor([0.0] * n, shape, dtype, device)


def ones(shape, dtype=DType.fp32, device="npu"):
    n = Tensor._numel(shape)
    return Tensor([1.0] * n, shape, dtype, device)


def randn(shape, key="randn", dtype=DType.fp32, device="npu", lo=-1.0, hi=1.0):
    from ..utils.stable import stable_vector
    n = Tensor._numel(shape)
    return Tensor(stable_vector(key, n, lo, hi), shape, dtype, device)


def _as_list(t):
    if isinstance(t, Tensor):
        return t.data
    return list(t)


def matmul(a, b):
    """2D matrix multiplication (M x K) @ (K x N)."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
    ad, bd = a.data, b.data
    out = [0.0] * (m * n)
    for i in range(m):
        arow = i * k
        for p in range(k):
            av = ad[arow + p]
            if av == 0.0:
                continue
            brow = p * n
            oi = i * n
            for j in range(n):
                out[oi + j] += av * bd[brow + j]
    return Tensor(out, (m, n), a.dtype, a.device)


def _binop(a, b, fn):
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return Tensor([fn(x, y) for x, y in zip(a.data, b.data)], a.shape, a.dtype, a.device)
    if isinstance(a, Tensor):
        return Tensor([fn(x, b) for x in a.data], a.shape, a.dtype, a.device)
    if isinstance(b, Tensor):
        return Tensor([fn(a, x) for x in b.data], b.shape, b.dtype, b.device)
    return Tensor([fn(a, b)], (1,))


def add(a, b):
    return _binop(a, b, lambda x, y: x + y)


def mul(a, b):
    return _binop(a, b, lambda x, y: x * y)


def relu(t):
    return Tensor([max(0.0, x) for x in t.data], t.shape, t.dtype, t.device)


def gelu(t):
    import math
    c = math.sqrt(2.0 / math.pi)
    return Tensor([0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x * x * x))) for x in t.data],
                  t.shape, t.dtype, t.device)


def silu(t):
    import math
    return Tensor([x / (1.0 + math.exp(-x)) for x in t.data], t.shape, t.dtype, t.device)


def softmax(t, dim=-1):
    import math
    # treat last dim as the softmax axis
    if len(t.shape) == 2:
        rows, cols = t.shape
        out = []
        for i in range(rows):
            row = t.data[i * cols:(i + 1) * cols]
            mx = max(row)
            ex = [math.exp(x - mx) for x in row]
            s = sum(ex)
            out.extend([e / s for e in ex])
        return Tensor(out, t.shape, t.dtype, t.device)
    mx = max(t.data)
    ex = [math.exp(x - mx) for x in t.data]
    s = sum(ex)
    return Tensor([e / s for e in ex], t.shape, t.dtype, t.device)


def layernorm(t, eps=1e-5):
    n = len(t.data)
    mean = sum(t.data) / n
    var = sum((x - mean) ** 2 for x in t.data) / n
    inv = 1.0 / (var + eps) ** 0.5
    return Tensor([(x - mean) * inv for x in t.data], t.shape, t.dtype, t.device)


def rmsnorm(t, eps=1e-6):
    n = len(t.data)
    ms = sum(x * x for x in t.data) / n
    inv = 1.0 / (ms + eps) ** 0.5
    return Tensor([x * inv for x in t.data], t.shape, t.dtype, t.device)


def rope(t, head_dim, position=0, base=10000.0):
    """RoPE applied along the last dim for a given position."""
    import math
    n = len(t.data)
    out = [0.0] * n
    half = head_dim // 2
    for i in range(n):
        idx = i % half
        theta = position / (base ** (2 * idx / head_dim))
        c, s = math.cos(theta), math.sin(theta)
        # even/odd pair
        j = (i // half) * head_dim + idx
        out[j] = t.data[j] * c - t.data[(j + half) % n] * s
    return Tensor(out, t.shape, t.dtype, t.device)
