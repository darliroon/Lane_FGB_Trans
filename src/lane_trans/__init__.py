"""Remove small loops from lane-centerline FlatGeobuf files.

    from lane_trans import convert
    convert("input.fgb", "output.fgb", kernel="v0")

Kernels (the algorithm behind convert) are listed in `KERNELS`.
"""
from .api import DEFAULT_KERNEL, Report, convert
from .kernels import KERNELS
from .params import Params

__all__ = ["convert", "Report", "Params", "KERNELS", "DEFAULT_KERNEL"]
