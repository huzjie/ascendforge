"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo FlashMLA: low-rank KV compression ratio."""
from ascendforge.bench.kernels import bench_mla
print(bench_mla())
