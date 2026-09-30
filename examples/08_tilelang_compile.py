"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Compile a mini TileLang program to Ascend C pseudo-code."""
from ascendforge.tilelang import compile_source

SRC = """
program gemm_kernel(128, 128, 32) {
    tile A : fp16[128, 32] @ GM
    tile B : fp16[32, 128] @ GM
    tile C : fp16[128, 128] @ L1
    load(in=A, out=A)
    load(in=B, out=B)
    gemm(in=A,B, out=C)
    store(in=C, out=C)
}
"""
result = compile_source(SRC)
print(result["ascend_c"])
print("summary:", result["program"].summary())
