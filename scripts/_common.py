"""Shared by the figure scripts: paths, the example data, running a cut rule, drawing helpers.

Run any script as `python scripts/<name>.py` from the repository root (needs `pip install -e .[viz]`).
Figures that docs/ shows are written straight to docs/figures/ (tracked); scratch output of the
exploration scripts goes to outputs/ (git-ignored).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lane_trans.chains import chains_to_graph, to_chains  # noqa: E402
from lane_trans.graph import load_vertex_graph  # noqa: E402
from lane_trans.metrics import compare, stats  # noqa: E402
from lane_trans.params import Params  # noqa: E402
from lane_trans.pipeline import simplify  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP"]   # the figures carry Chinese labels
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
START, TARGET = ROOT / "data/start.fgb", ROOT / "data/target.fgb"
OUT = ROOT / "outputs"
FIGS = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

O = gpd.read_file(START).total_bounds[:2]     # local origin, metres
Gs, _ = load_vertex_graph(START)
Gt, _ = load_vertex_graph(TARGET)


def run(cut, params=Params()):
    """Run the shared front end + `cut` on start.fgb -> (result graph, compare() vs target, loop count)."""
    G, _ = load_vertex_graph(START)
    C = to_chains(G)
    simplify(C, O, params, cut)
    Gr = chains_to_graph(C)
    return Gr, compare(Gr, Gt, Gs), stats(Gr)["loops"]


# ------------------------------------------------------------ drawing
_pts = np.array([p for G in (Gs, Gt) for e in G.edges() for p in e]) - O
_pad = 3.0
XLIM = (_pts[:, 0].min() - _pad, _pts[:, 0].max() + _pad)
YLIM = (_pts[:, 1].min() - _pad, _pts[:, 1].max() + _pad)


def segs(G):
    return [(np.array(a) - O, np.array(b) - O) for a, b in G.edges()]


def draw(ax, G, color, lw=2.0, z=2):
    for a, b in segs(G):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round", zorder=z)


def draw_target(ax):
    draw(ax, Gt, "#c8c8c8", lw=6, z=1)


def style(ax, title, **kw):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15, **kw)
