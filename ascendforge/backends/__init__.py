"""Backend registry: mock / cpu / ascend / openai."""
_BACKENDS = {}


def register_backend(name=None):
    def deco(fn):
        n = name or getattr(fn, "__name__", "unknown")
        _BACKENDS[n] = fn
        return fn
    return deco


def list_backends():
    return sorted(_BACKENDS.keys())


def make_backend(name, cfg=None, **kwargs):
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend: {name} (available: {list_backends()})")
    return _BACKENDS[name](cfg=cfg, **kwargs)


# import modules to trigger registration
from . import mock, cpu, ascend, openai  # noqa: E402,F401
