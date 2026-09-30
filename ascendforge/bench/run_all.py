"""Run the full benchmark suite and print a report."""
from .kernels import bench_gemm, bench_mla, bench_moe, bench_dedup, bench_comm
from .report import render_report


def main():
    results = [
        bench_gemm(),
        bench_mla(),
        bench_moe(),
        bench_dedup(),
        bench_comm(),
    ]
    print(render_report(results))
    return results


if __name__ == "__main__":
    main()
