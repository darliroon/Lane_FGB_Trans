"""Kernel registry.

A kernel is a function

    kernel(C, origin, params, on_step=None) -> list[str]

C         chain multigraph from chains.to_chains; the kernel edits it in place
origin    (x, y) subtracted from coordinates in log lines only
params    params.Params
on_step   optional callback, see pipeline.simplify
returns   one human-readable log line per change

Register one with @register("name"). convert(), the CLI and the tests find it by name.
"""
from typing import Callable

KERNELS: dict[str, Callable] = {}


def register(name):
    def deco(fn):
        if name in KERNELS:
            raise ValueError(f"kernel {name!r} is already registered")
        KERNELS[name] = fn
        return fn
    return deco


def get_kernel(name):
    if not isinstance(name, str) or name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; available: {', '.join(sorted(KERNELS))}")
    return KERNELS[name]
