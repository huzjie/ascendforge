"""FlashMLA: Multi-head Latent Attention with low-rank KV cache.

DeepSeek's MLA compresses the KV cache through a low-rank projection (kv_lora_rank),
so the per-token KV footprint shrinks dramatically. FlashMLA is the memory-efficient
attention kernel over that compressed cache, with sparse / sliding-window support.
"""
from dataclasses import dataclass

from ..core.tensor import Tensor, matmul, softmax, zeros


@dataclass
class MlaContext:
    kv_lora_rank: int = 64
    qk_rope_dim: int = 32
    sliding_window: int = 256
    sparse: bool = True
    # cached compressed KV per layer
    cache: dict = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = {}


def compress_kv(h, w_kv_down):
    """Low-rank compression: c = h @ w_kv_down -> (T, kv_lora_rank)."""
    return matmul(h, w_kv_down)


def decompress_kv(c, w_k, w_v):
    """Reconstruct K, V from the compressed cache."""
    k = matmul(c, w_k)
    v = matmul(c, w_v)
    return k, v


def _concat(a, b):
    import math
    ra, ca = a.shape
    rb, cb = b.shape
    cols = ca + cb
    data = [0.0] * (ra * cols)
    for i in range(ra):
        for j in range(ca):
            data[i * cols + j] = a.data[i * ca + j]
        for j in range(cb):
            data[i * cols + ca + j] = b.data[i * cb + j]
    return Tensor(data, (ra, cols), a.dtype, a.device)


def _transpose(t):
    m, n = t.shape
    data = [0.0] * (n * m)
    for i in range(m):
        for j in range(n):
            data[j * m + i] = t.data[i * n + j]
    return Tensor(data, (n, m), t.dtype, t.device)


def mla_forward(q, c, w_k, w_v, ctx: MlaContext = None):
    """Sparse/sliding-window attention over a compressed KV cache.

    Args:
        q: query activations (T_q, d)
        c: compressed KV cache (T_kv, kv_lora_rank)
        w_k, w_v: decompression projections
    Returns attention output (T_q, d).
    """
    ctx = ctx or MlaContext()
    k, v = decompress_kv(c, w_k, w_v)
    kt = _transpose(k)
    scores = matmul(q, kt)
    # sparse / sliding-window masking (deterministic)
    if ctx.sparse:
        scores = _mask_sliding(scores, ctx.sliding_window)
    probs = softmax(scores)
    out = matmul(probs, v)
    # cache compressed KV
    ctx.cache[id(c)] = c
    return out


def _mask_sliding(scores, window):
    rows, cols = scores.shape
    data = list(scores.data)
    for i in range(rows):
        for j in range(cols):
            # allow diagonal band of width `window`
            if abs(i - j) > window // 2:
                data[i * cols + j] = -1e9
    return Tensor(data, scores.shape, scores.dtype, scores.device)
