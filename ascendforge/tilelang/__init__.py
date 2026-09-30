"""TileLang: a tile-based high-level DSL compiler targeting Ascend C.

DeepSeek's TileLang wraps Ascend C low-level instructions in a high-level,
tile-oriented programming model without losing hardware performance. This module
provides a runnable reference: an IR, a scheduler, a code generator (Ascend C
pseudo-code) and a deterministic CPU simulator so tile programs run anywhere.
"""
from .ir import Tile, Loop, Op, TileProgram, OpKind
from .compiler import compile_program, compile_source
from .simulator import Simulator

__all__ = [
    "Tile", "Loop", "Op", "TileProgram", "OpKind",
    "compile_program", "compile_source", "Simulator",
]
