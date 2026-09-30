"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Micro-benchmark GEMM (the DeepGEMM reference)."""
from ascendforge.bench.kernels import bench_gemm
print(bench_gemm())
