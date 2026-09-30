"""CLI for the tilelang compiler: compile a .tl file and dump artifacts."""
import sys
from pathlib import Path
from .compiler import compile_source


def main():
    if len(sys.argv) < 2:
        print("usage: python -m ascendforge.tilelang <file.tl> [--no-optimize]")
        return
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    optimize = "--no-optimize" not in sys.argv
    result = compile_source(src, optimize)
    print("=== Ascend C ===")
    print(result["ascend_c"])
    print("\n=== summary ===")
    print(result["program"].summary())


if __name__ == "__main__":
    main()
