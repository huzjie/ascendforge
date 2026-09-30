"""DeepGEMM: general & grouped matrix multiplication kernels (tile-based).

Grouped GEMM is the workhorse of MoE: each expert runs a (M x K) @ (K x N) GEMM
with a distinct weight matrix, batched together for the Ascend NPU.
"""
from dataclasses import dataclass

from ..core.tensor import Tensor, matmul, zeros


@dataclass
class GemmConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    grouped: bool = True
    group_size: int = 8


def gemm(a: Tensor, b: Tensor) -> Tensor:
    """Tiled GEMM (reference): C = A @ B."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"GEMM shape mismatch {a.shape} @ {b.shape}")
    out = zeros((m, n), a.dtype, a.device)
    return _tiled_matmul(a, b, out, GemmConfig())


def _tiled_matmul(a, b, out, cfg):
    m, k = a.shape
    _, n = b.shape
    for i0 in range(0, m, cfg.block_m):
        for j0 in range(0, n, cfg.block_n):
            for k0 in range(0, k, cfg.block_k):
                a_t = _slice(a, i0, min(i0 + cfg.block_m, m), k0, min(k0 + cfg.block_k, k))
                b_t = _slice(b, k0, min(k0 + cfg.block_k, k), j0, min(j0 + cfg.block_n, n))
                c_t = matmul(a_t, b_t)
                _accumulate(out, c_t, i0, j0)
    return out


def _slice(t, r0, r1, c0, c1):
    rows, cols = t.shape
    data = []
    for i in range(r0, r1):
        for j in range(c0, c1):
            data.append(t.data[i * cols + j])
    return Tensor(data, (r1 - r0, c1 - c0), t.dtype, t.device)


def _accumulate(out, addend, r0, c0):
    cols = out.shape[1]
    ar, ac = addend.shape
    for i in range(ar):
        for j in range(ac):
            out.data[(r0 + i) * cols + (c0 + j)] += addend.data[i * ac + j]


def grouped_gemm(inputs, weights, cfg=None):
    """Grouped GEMM: inputs[g] @ weights[g] for g in 0..G-1."""
    cfg = cfg or GemmConfig()
    outs = []
    for x, w in zip(inputs, weights):
        outs.append(_tiled_matmul(x, w, zeros((x.shape[0], w.shape[1]), x.dtype, x.device), cfg))
    return outs
