"""TileLang intermediate representation."""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class OpKind(Enum):
    LOAD = "load"
    STORE = "store"
    GEMM = "gemm"
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    ATTENTION = "attention"
    FUSED = "fused"


@dataclass
class Tile:
    name: str
    shape: tuple
    dtype: str = "fp16"
    memory: str = "L1"  # GM / L1 / L0C / UB


@dataclass
class Op:
    kind: OpKind
    inputs: List[str]
    outputs: List[str]
    attrs: dict = field(default_factory=dict)


@dataclass
class Loop:
    var: str
    start: int = 0
    stop: int = 1
    step: int = 1
    body: List = field(default_factory=list)


@dataclass
class TileProgram:
    name: str
    tiles: List[Tile] = field(default_factory=list)
    body: List = field(default_factory=list)
    target: str = "ascend-c"
    pipeline_stages: int = 2

    def summary(self):
        n_ops = sum(1 for node in self.body if isinstance(node, Op))
        n_loops = sum(1 for node in self.body if isinstance(node, Loop))
        return {
            "name": self.name,
            "tiles": len(self.tiles),
            "ops": n_ops,
            "loops": n_loops,
            "target": self.target,
        }
