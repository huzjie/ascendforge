"""Tensor data types."""
from enum import Enum


class DType(Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    int8 = "int8"
    fp8 = "fp8"

    def __repr__(self):
        return self.value
