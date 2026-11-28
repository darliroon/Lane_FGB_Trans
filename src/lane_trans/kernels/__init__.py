"""The interchangeable loop-removal algorithms behind lane_trans.convert().

    v0                   connector rule, then Hausdorff fallback            kernels/v0.py
    v1, v1:weakest_tree  cut the chain that continues worst                 kernels/v1.py
    v1:shortest          cut the shortest chain of each loop
    v1:longest           cut the longest chain of each loop
    v1:hausdorff         cut the chain its alternative path explains best
    v2                   v0's decisions, computed without repeating work per candidate

To add a version, write kernels/v2.py with a function decorated by @register("v2") and import
the module below. See registry.py for the signature. convert(), the CLI and the tests pick
it up by name.
"""
from . import v0, v1, v2  # noqa: F401  (importing registers the kernels)
from .registry import KERNELS, get_kernel, register

__all__ = ["KERNELS", "get_kernel", "register"]
