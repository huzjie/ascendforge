"""Kernel micro-benchmarks (deterministic)."""
import time

from ..core.tensor import randn
from ..ops.deepgemm import gemm, grouped_gemm
from ..ops.flashmla import mla_forward, compress_kv, MlaContext
from ..ops.deepselect import select_documents, minhash_dedup
from ..moe.moe_layer import MoELayer
from ..comm.deepep import DeepEP


def bench_gemm(m=512, k=512, n=512, repeats=3):
    a = randn((m, k), "gemm_a")
    b = randn((k, n), "gemm_b")
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = gemm(a, b)
    dt = (time.perf_counter() - t0) / repeats
    flops = 2 * m * k * n
    return {"op": "gemm", "time_s": round(dt, 5), "flops": flops,
            "tflops": round(flops / dt / 1e12, 3), "shape": (m, k, n)}


def bench_mla(t=64, hidden=512, lora=64, repeats=3):
    h = randn((t, hidden), "mla_h")
    q = randn((t, hidden), "mla_q")
    w_down = randn((hidden, lora), "mla_down")
    w_k = randn((lora, hidden), "mla_wk")
    w_v = randn((lora, hidden), "mla_wv")
    c = compress_kv(h, w_down)
    ctx = MlaContext(kv_lora_rank=lora)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = mla_forward(q, c, w_k, w_v, ctx)
    dt = (time.perf_counter() - t0) / repeats
    # memory savings: full KV (2*T*hidden) vs compressed (T*lora)
    full = 2 * t * hidden
    comp = t * lora
    return {"op": "flashmla", "time_s": round(dt, 5),
            "kv_cache_compression_ratio": round(full / comp, 2),
            "full_kv_floats": full, "compressed_kv_floats": comp}


def bench_moe(tokens=64, hidden=128, experts=8, top_k=2, expert_hidden=256, repeats=3):
    layer = MoELayer(hidden=hidden, num_experts=experts, top_k=top_k, expert_hidden=expert_hidden)
    x = randn((tokens, hidden), "moe_x")
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = layer.forward(x)
    dt = (time.perf_counter() - t0) / repeats
    return {"op": "moe", "time_s": round(dt, 5), "tokens": tokens,
            "load": layer.load_counts()}


def bench_dedup(n=500):
    from ..data.synth import SyntheticCorpus
    docs = SyntheticCorpus(num_docs=n).generate()
    t0 = time.perf_counter()
    kept = select_documents(docs, threshold=0.85, quality_min=0.0, curriculum=False)
    dt = time.perf_counter() - t0
    return {"op": "deepselect", "time_s": round(dt, 5), "before": n,
            "after": len(kept), "removed": n - len(kept)}


def bench_comm(num_ranks=8, tokens_per_rank=1000):
    ep = DeepEP(num_ranks=num_ranks, num_experts=8)
    tokens = [i for i in range(tokens_per_rank)]
    t0 = time.perf_counter()
    routed = ep.dispatch(tokens)
    ep.combine({e: [t] for e, t in routed.items()})
    dt = time.perf_counter() - t0
    return {"op": "deepep", "time_s": round(dt, 5), "tokens": tokens_per_rank,
            "stats": ep.report()}
