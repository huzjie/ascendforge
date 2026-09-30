"""Device and tile abstractions mirroring Ascend NPU programming model."""
from enum import Enum


class Device(Enum):
    NPU = "npu"
    CPU = "cpu"
    MOCK = "mock"


class Tile:
    """A tile: the unit of computation on Ascend. Logical view only here."""

    def __init__(self, shape, dtype="fp32", src="GM", dst="L1"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self):
        return f"Tile(shape={self.shape}, dtype={self.dtype}, {self.src}->{self.dst})"


def tile_mem_estimate(shape, dtype="fp32"):
    per = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1}.get(dtype, 4)
    n = 1
    for s in shape:
        n *= s
    return n * per
