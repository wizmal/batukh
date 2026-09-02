"""Batukh document recognition package.

The TensorFlow and PyTorch backends are imported lazily so users only need to
install the backend they actually use.
"""

from importlib import import_module

__all__ = ("tensorflow", "torch")


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
