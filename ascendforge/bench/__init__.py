"""Benchmark suite: kernel micro-benchmarks + end-to-end eval."""
from .kernels import bench_gemm, bench_mla, bench_moe, bench_dedup, bench_comm
from .report import render_report

__all__ = ["bench_gemm", "bench_mla", "bench_moe", "bench_dedup", "bench_comm", "render_report"]
