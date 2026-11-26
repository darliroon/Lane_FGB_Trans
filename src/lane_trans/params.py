"""Tunable thresholds. Passed explicitly to every rule, so nothing depends on module state."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """
    max_loop_len      m. A loop whose biconnected component is longer than this (roundabout,
                      ring road) is real and is left alone.
    connector_margin  v0 only. How much better two other chains must continue into each other
                      than into a chain, at both of its ends, for that chain to count as a connector.
    parallel_ratio    Parallel chains longer than this x the shortest are dropped instead of averaged.
    max_loop_width    m. Width (short side of the minimum rotated rectangle) above which a loop,
                      a self-loop chain or a parallel bundle is a real road, not an artefact.
                      Applies to all three rules. None = no width limit.
    """
    max_loop_len: float = 200.0
    connector_margin: float = 0.10
    parallel_ratio: float = 1.5
    max_loop_width: float | None = None

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.max_loop_len, self.connector_margin,
                                              self.parallel_ratio)):
            raise ValueError("params must be finite numbers")
        if self.max_loop_len <= 0:
            raise ValueError("max_loop_len must be > 0")
        if not 0 <= self.connector_margin <= 1:
            raise ValueError("connector_margin must be in [0, 1]")
        if self.parallel_ratio < 1:
            raise ValueError("parallel_ratio must be >= 1")
        if self.max_loop_width is not None and not (
                math.isfinite(self.max_loop_width) and self.max_loop_width > 0):
            raise ValueError("max_loop_width must be a finite number > 0, or None")
