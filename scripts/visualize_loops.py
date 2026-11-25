"""Visualize every step of a kernel run (default v0) on data/start.fgb.

Each frame: [full view | zoom before | zoom after]
  red dashed = chains removed by this step, green = chains added by this step.
Outputs outputs/loop_steps/step_XX.png (scratch), docs/figures/loop_steps.gif and loop_steps_overview.png
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from PIL import Image

from lane_trans.chains import dissolve, n_loops, snapshot, to_chains
from lane_trans.graph import key, load_vertex_graph
from lane_trans.kernels import get_kernel
from lane_trans.params import Params

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"

BASE, REMOVED, ADDED, NODE = "#8DA0CB", "#D62728", "#2CA02C", "#DD8452"


def end_nodes(lines):
    """Chain end points with degree != 2 (junctions and dangling ends)."""
    G = nx.MultiGraph()
    for p in lines:
        G.add_edge(key(p[0]), key(p[-1]))
    return [n for n in G if G.degree(n) != 2]


def draw(ax, lines, origin, removed=(), added=(), bbox=None, title=""):
    for p in lines:
        q = p - origin
        ax.plot(q[:, 0], q[:, 1], color=BASE, lw=1.2, zorder=1)
    for p in removed:
        q = p - origin
        ax.plot(q[:, 0], q[:, 1], color=REMOVED, lw=2.2, ls="--", zorder=3)
    for p in added:
        q = p - origin
        ax.plot(q[:, 0], q[:, 1], color=ADDED, lw=2.6, zorder=4)
        ax.plot(q[:, 0], q[:, 1], ".", color=ADDED, ms=3, zorder=4)
    for n in end_nodes(list(lines) + list(removed) + list(added)):
        ax.plot(n[0] - origin[0], n[1] - origin[1], "o", ms=4, color=NODE,
                mec="black", mew=0.4, zorder=5)
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal", adjustable="datalim" if bbox is None else "box")
    ax.set_title(title, fontsize=10)


def zoom_box(lines, origin, pad=4.0, min_size=15.0):
    pts = np.vstack(lines) - origin
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    # square box so the aspect ratio stays readable
    s = max(x1 - x0, y1 - y0, min_size)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def step_box(before, after, removed, added, origin):
    """Zoom box: changed chains padded by a fixed margin, at least 20 m wide."""
    return zoom_box(removed + added, origin, pad=6.0, min_size=20.0)


def legend(ax, lines):
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8,
            transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/start.fgb"))
    ap.add_argument("--out", default=str(ROOT / "outputs/loop_steps"))
    ap.add_argument("--kernel", default="v0")
    ap.add_argument("--fps", type=float, default=0.7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    Gs, _ = load_vertex_graph(args.input)
    origin = gpd.read_file(args.input).total_bounds[:2]
    C = to_chains(Gs)
    dissolve(C)
    start_lines = list(snapshot(C).values())
    loops0 = n_loops(C)

    steps = []
    get_kernel(args.kernel)(C, origin, Params(), on_step=lambda msg, b, a, n: steps.append((msg, b, a, n)))
    final_lines = list(snapshot(C).values())
    full_box = zoom_box(start_lines, origin, pad=3)

    frames = []
    # frame 0: start with every loop region boxed, plus the step list
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    draw(axes[0], start_lines, origin, bbox=full_box, title=f"start: {loops0} loops")
    for comp in nx.biconnected_components(Gs):
        if len(comp) >= 3:
            x0, y0, x1, y1 = zoom_box([np.array(list(comp))], origin, pad=1.5, min_size=0)
            axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=REMOVED, lw=1))
    draw(axes[1], final_lines, origin, bbox=full_box, title="result: 0 loops")
    legend(axes[2], ["steps:"] + [f"{i + 1:2d}. {m}" for i, (m, *_) in enumerate(steps)] +
           ["", "red box  = loop region in start",
            "red --   = chain removed in a step", "green    = chain added in a step"])
    fig.tight_layout()
    frames.append(out / "step_00.png")
    fig.savefig(frames[-1], dpi=110)
    plt.close(fig)

    for i, (msg, before, after, n_left) in enumerate(steps, 1):
        removed = [p for k, p in before.items() if k not in after]
        added = [p for k, p in after.items() if k not in before]
        box = step_box(before, after, removed, added, origin)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        draw(axes[0], list(after.values()), origin, removed, added, bbox=full_box,
             title=f"full view, {n_left} loops left")
        axes[0].add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1],
                                    fill=False, ec="black", lw=1))
        kept_before = [p for k, p in before.items() if k in after]
        draw(axes[1], kept_before, origin, removed=removed, bbox=box, title="before (zoom)")
        draw(axes[2], list(after.values()), origin, added=added, bbox=box, title="after (zoom)")
        fig.suptitle(f"step {i}/{len(steps)}: {msg}", fontsize=11, family="monospace")
        fig.tight_layout()
        frames.append(out / f"step_{i:02d}.png")
        fig.savefig(frames[-1], dpi=110)
        plt.close(fig)

    # GIF
    imgs = [Image.open(f).convert("RGB") for f in frames]
    size = imgs[0].size
    imgs = [im.resize(size) for im in imgs]
    ms = int(1000 / args.fps)
    imgs[0].save(FIGS / "loop_steps.gif", save_all=True, append_images=imgs[1:],
                 duration=[ms * 3] + [ms] * (len(imgs) - 2) + [ms * 3], loop=0)

    # overview: before / after zoom for every step
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7), squeeze=False)
    for i, (msg, before, after, _) in enumerate(steps):
        removed = [p for k, p in before.items() if k not in after]
        added = [p for k, p in after.items() if k not in before]
        box = step_box(before, after, removed, added, origin)
        draw(axes[0, i], [p for k, p in before.items() if k in after], origin,
             removed=removed, bbox=box, title=f"{i + 1}. {msg.split()[0]}")
        draw(axes[1, i], list(after.values()), origin, added=added, bbox=box)
        for ax in axes[:, i]:
            ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("before")
    axes[1, 0].set_ylabel("after")
    fig.tight_layout()
    fig.savefig(FIGS / "loop_steps_overview.png", dpi=130)
    print(f"{len(steps)} steps -> {out}")


if __name__ == "__main__":
    main()
