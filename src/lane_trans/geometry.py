"""Polyline helpers. A polyline is an (n, 2) float array in projected metres."""
import numpy as np
from shapely.geometry import LineString, MultiPoint

TANGENT_SPAN = 1.5   # m; polyline length used to estimate the direction of a chain end


def length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def width(p):
    """Short side of the minimum rotated rectangle around the polyline = how wide it is.

    A straight line or a closed ring gives its lateral extent; 0 for a collinear polyline.
    """
    r = MultiPoint([tuple(q) for q in p]).minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 0.0
    c = np.asarray(r.exterior.coords)
    return float(min(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])))


def too_wide(lines, limit):
    """True if `limit` is set and the polylines together are wider than it."""
    return limit is not None and width(np.vstack(lines)) > limit


def resample(p, m):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], m)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def average(lines):
    """Point-wise mean after arc-length resampling; all lines run start -> end."""
    m = max(int(max(length(l) for l in lines) / 0.5) + 1, 3)
    avg = np.mean([resample(l, m) for l in lines], axis=0)
    avg[0], avg[-1] = lines[0][0], lines[0][-1]
    out = np.array(LineString(avg).simplify(0.05).coords)
    out[0], out[-1] = lines[0][0], lines[0][-1]
    return out


def orient(p, start):
    """p, reversed if needed so that it begins at `start`."""
    return p if np.allclose(p[0], start, rtol=0, atol=1e-6) else p[::-1]


def tangent(p, span=TANGENT_SPAN):
    """Unit direction leaving p[0]; p must already start at the end being measured."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    j = min(max(int(np.searchsorted(s, span)), 1), len(p) - 1)
    v = p[j] - p[0]
    return v / max(float(np.linalg.norm(v)), 1e-12)
