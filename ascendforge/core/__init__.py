"""Zero-dependency tensor core and device/tile abstraction."""
from .dtype import DType
from .tensor import Tensor, zeros, ones, randn, matmul, add, mul, relu, gelu, silu, softmax, layernorm, rmsnorm, rope
from .device import Device, Tile
from .registry import KernelRegistry, register_kernel, list_kernels

__all__ = [
    "DType", "Tensor", "zeros", "ones", "randn", "matmul", "add", "mul",
    "relu", "gelu", "silu", "softmax", "layernorm", "rmsnorm", "rope",
    "Device", "Tile", "KernelRegistry", "register_kernel", "list_kernels",
]
