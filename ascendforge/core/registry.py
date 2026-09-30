"""Kernel / operator registry."""
KERNELS = {}


def register_kernel(name=None):
    def deco(fn):
        n = name or fn.__name__
        KERNELS[n] = fn
        return fn
    return deco


def list_kernels():
    return sorted(KERNELS.keys())


class KernelRegistry:
    @staticmethod
    def get(name):
        return KERNELS.get(name)

    @staticmethod
    def call(name, *args, **kwargs):
        if name not in KERNELS:
            raise KeyError(f"unknown kernel: {name}")
        return KERNELS[name](*args, **kwargs)
