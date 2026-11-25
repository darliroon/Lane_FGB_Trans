"""Detailed comparison: V0 (connector) vs V1 longest-chain vs V1 weakest-tree-edge.

Re-runs the three methods with per-step tracing, then draws for each method the whole
graph plus zooms on the two loops where they disagree (A: connector cluster at x~90,
B: lanes 21/23/26 at x~55). Also prints a parameter-sensitivity sweep.

    python scripts/compare_v0_v1_detail.py        ->  docs/figures/v0_vs_v1_detail.png
"""
import re

import numpy as np
import shapely.affinity as sa
from _common import (Gs, Gt, O, FIGS, XLIM, chains_to_graph, compare, draw, draw_target, load_vertex_graph,
                     plt, simplify, START, style, to_chains)
from matplotlib.patches import Rectangle
from shapely.geometry import LineString, box

from lane_trans import geometry as geo
from lane_trans.chains import small_loops
from lane_trans.kernels import v0, v1
from lane_trans.metrics import as_geom
from lane_trans.params import Params

FIG = FIGS / "v0_vs_v1_detail.png"
TGT = sa.translate(as_geom(Gt), -O[0], -O[1])
REGIONS = {"A": box(80, 48, 100, 86), "B": box(48, 22, 62, 44)}
METHODS = [("V0 connector", v0.cut, "#1f9d55"),
           ("V1 删最长链", v1.CUTS["longest"], "#2c7be5"),
           ("V1 最弱生成树边", v1.CUTS["weakest_tree"], "#2c7be5")]


def candidates(C, params):
    """Every chain of every small loop with the scores the three methods look at."""
    H, comps = small_loops(C, params.max_loop_len)
    rows = []
    for comp in comps:
        for u, v in comp:
            k = H[u][v]["k"]
            d = C[u][v][k]
            rows.append(dict(id=id(d["pts"]), lanes=sorted(d["src"]), L=H[u][v]["w"],
                             cont=v1.continuity(C, u, v, k, d), superfl=v0.superfluous(C, u, v, k, d)))
    return rows


def run_traced(cut, params=Params()):
    """Run the shared front end + `cut`, recording every chain the cut rule deletes."""
    G, _ = load_vertex_graph(START)
    C = to_chains(G)
    cuts = []

    def traced(C, origin, params):
        cands = candidates(C, params)
        before = {id(d["pts"]): d for _, _, d in C.edges(data=True)}
        msg = cut(C, origin, params)
        if msg is None:
            return None
        after = {id(d["pts"]) for _, _, d in C.edges(data=True)}
        (gone,) = [i for i in before if i not in after]
        d = before[gone]
        haus = re.search(r"hausdorff_to_alt=([\d.]+)", msg)
        cuts.append(dict(pts=d["pts"], lanes=sorted(d["src"]), L=geo.length(d["pts"]),
                         rule=msg.split()[0].split("-")[0],
                         haus=float(haus.group(1)) if haus else None,
                         cand=next(c for c in cands if c["id"] == gone)))
        return msg

    simplify(C, O, params, traced)
    return chains_to_graph(C), cuts


def mean_dist_to_target(pts, n=60):
    ls = LineString(pts - O)
    return float(np.mean([TGT.distance(ls.interpolate(t, normalized=True)) for t in np.linspace(0, 1, n)]))


def cov(a, b, tol=0.5):
    return a.intersection(b.buffer(tol)).length / a.length if a.length else float("nan")


def local_metrics(Gr, region):
    a = sa.translate(as_geom(Gr), -O[0], -O[1]).intersection(region)
    b = TGT.intersection(region)
    return cov(a, b), cov(b, a), a.hausdorff_distance(b)


def draw_cuts(ax, cuts, label):
    for i, c in enumerate(cuts, 1):
        p = c["pts"] - O
        ax.plot(p[:, 0], p[:, 1], color="#e0301e", lw=2.2, ls=(0, (4, 2)), zorder=3)
        if label:
            m = p[len(p) // 2]
            ax.annotate(f"删{i}: 车道{'/'.join(map(str, c['lanes']))}\n{c['L']:.1f} m", m, xytext=(8, 0),
                        textcoords="offset points", fontsize=11, color="#c0200e", zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=.75, pad=1))


def margin_sweep(margins):
    """V0 outcome as connector_margin varies (the only V0 threshold)."""
    rows = []
    for m in margins:
        Gr, cuts = run_traced(v0.cut, Params(connector_margin=m))
        c = compare(Gr, Gt, Gs)
        rows.append((m, [(tuple(x["lanes"]), round(x["L"], 1)) for x in cuts],
                     c["result_within_0.5m_of_target"], c["target_within_0.5m_of_result"], c["hausdorff_m"]))
    return rows


def main():
    results = [(name, color, *run_traced(cut)) for name, cut, color in METHODS]

    fig, axs = plt.subplots(3, 3, figsize=(18, 22), gridspec_kw=dict(width_ratios=[1.0, 0.62, 0.72]))
    for row, (name, color, Gr, cuts) in zip(axs, results):
        c = compare(Gr, Gt, Gs)
        for ax, region in zip(row, [None, REGIONS["A"], REGIONS["B"]]):
            draw_target(ax)
            draw(ax, Gr, color)
            draw_cuts(ax, cuts, region is not None)
            style(ax, "")
            if region is not None:
                x0, y0, x1, y1 = region.bounds
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                cv, bk, hd = local_metrics(Gr, region)
                ax.set_title(f"{'A 连接线环' if region is REGIONS['A'] else 'B lanes 21/23/26 环'}\n"
                             f"局部覆盖 {cv:.2f} / {bk:.2f}  Hausdorff {hd:.2f} m", fontsize=13)
            else:
                ax.set_title(f"{name}（全图）\n覆盖 {c['result_within_0.5m_of_target']:.3f} / "
                             f"{c['target_within_0.5m_of_result']:.3f}  Hausdorff {c['hausdorff_m']} m", fontsize=14)
                for key, r in REGIONS.items():
                    x0, y0, x1, y1 = r.bounds
                    x1 = min(x1, XLIM[1] - 0.3)
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#777", lw=1, ls=":"))
                    ax.text(x0, y1 + 1, key, fontsize=13, color="#555")
    fig.suptitle("灰粗线 = target；彩线 = 方法结果；红色虚线 = 本方法在“环剪断”步骤删掉的链", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0)
    fig.savefig(FIG, dpi=90)
    print("saved", FIG)

    for name, _, Gr, cuts in results:
        print(f"\n=== {name}: {len(cuts)} 次环剪断")
        for i, c in enumerate(cuts, 1):
            p = c["pts"] - O
            md = mean_dist_to_target(c["pts"])
            s = c["cand"]["superfl"]
            print(f"  {i}. lanes={c['lanes']} L={c['L']:.1f}m ({p[0][0]:.0f},{p[0][1]:.0f})->({p[-1][0]:.0f},{p[-1][1]:.0f})"
                  f"  规则={c['rule']}  树连续度={c['cand']['cont']:+.2f}  冗余度=({s[0]:+.2f},{s[1]:+.2f})"
                  f"  到 target 平均距离={md:.2f} m")

    print("\n=== V0 connector_margin 敏感性")
    for m, cutlist, cv, bk, hd in margin_sweep([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
        print(f"  margin={m:.2f}  剪断={cutlist}  覆盖 {cv}/{bk}  Hausdorff {hd}")


if __name__ == "__main__":
    main()
