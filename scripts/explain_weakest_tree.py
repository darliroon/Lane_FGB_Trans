"""Step-by-step explanation of the V1 'weakest tree edge' method.

Each round: for every chain of every small loop, measure how well it continues straight
into some other chain at each of its two ends (cos of the turning angle); the chain's
score is the worse end. Delete the lowest-scoring chain, run the cheaper rules
(self-loop, parallel merge), and repeat until no small loop is left.

Outputs (outputs/):
  weakest_tree_1_score.png        how one chain's score is measured
  weakest_tree_2_rounds.png       the two cut decisions, with every candidate's score
  weakest_tree_3_pipeline.png     the whole run, step by step
  weakest_tree_4_two_answers.png  the loop at x~55 has two valid answers
"""
import numpy as np
from _common import (O, FIGS, XLIM, YLIM, Gt, START, chains_to_graph, load_vertex_graph, plt, segs,
                     to_chains)

from lane_trans import geometry as geo
from lane_trans.chains import dissolve, n_loops, small_loops
from lane_trans.kernels import v1
from lane_trans.params import Params
from lane_trans.rules import merge_parallel, remove_self_loop

PARAMS = Params()
WIN_A = (79, 101, 47, 87)   # x0, x1, y0, y1 of the connector loop
WIN_B = (46, 64, 19, 47)    # lanes 21/23/26 loop
CMAP = plt.cm.RdYlGn
KIND = {"parallel": "平行合并", "self-loop": "删自环"}


def snap(C):
    # _ref keeps the original array alive so its id() cannot be reused by a later array
    return [dict(pts=d["pts"] - O, lanes=sorted(d["src"]), mod=bool(d["mod"]), id=id(d["pts"]), _ref=d["pts"])
            for _, _, d in C.edges(data=True)]


def end_detail(C, u, v, k, d, n):
    """Score of chain (u,v,k) at its end n: the best 'straight-through' partner there.

    Same arithmetic as v1.continuity, but keeps every partner so it can be drawn.
    neg = -dot(own outgoing direction, partner outgoing direction) = cos(deviation from going straight).
    """
    dn = geo.tangent(geo.orient(d["pts"], n))
    partners = []
    for uu, vv, kk, dd in C.edges(n, keys=True, data=True):
        if (kk == k and {uu, vv} == {u, v}) or uu == vv:
            continue
        t = geo.tangent(geo.orient(dd["pts"], n))
        partners.append(dict(lanes=sorted(dd["src"]), L=geo.length(dd["pts"]), t=t,
                             neg=-float(dn @ t)))
    best = max(range(len(partners)), key=lambda i: partners[i]["neg"]) if partners else None
    return dict(node=np.array(n) - O, dn=dn, partners=partners, best=best,
                score=partners[best]["neg"] if partners else -2.0)


def loop_name(pts):
    cx, cy = pts.mean(axis=0)
    for name, (x0, x1, y0, y1) in (("A", WIN_A), ("B", WIN_B)):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return name
    return "?"


def candidates(C):
    H, comps = small_loops(C, PARAMS.max_loop_len)
    out = []
    for comp in comps:
        group = []
        for u, v in comp:
            k = H[u][v]["k"]
            d = C[u][v][k]
            e1, e2 = end_detail(C, u, v, k, d, u), end_detail(C, u, v, k, d, v)
            score = min(e1["score"], e2["score"])
            assert abs(score - v1.continuity(C, u, v, k, d)) < 1e-9
            group.append(dict(id=id(d["pts"]), pts=d["pts"] - O, lanes=sorted(d["src"]), L=H[u][v]["w"],
                              ends=(e1, e2), score=score, weak=0 if e1["score"] <= e2["score"] else 1))
        name = loop_name(np.vstack([g["pts"] for g in group]))
        for g in group:
            g["loop"] = name
        out += group
    return out


def trace():
    """Run the method; return the ordered list of events with before/after snapshots."""
    G, _ = load_vertex_graph(START)
    C = to_chains(G)
    ev = [dict(kind="start", after=snap(C), loops=n_loops(C))]
    cut = v1.CUTS["weakest_tree"]
    while True:
        dissolve(C)
        before = snap(C)
        if remove_self_loop(C, O, PARAMS):
            kind = "self-loop"
        elif merge_parallel(C, O, PARAMS):
            kind = "parallel"
        else:
            cands = candidates(C)
            if cut(C, O, PARAMS) is None:
                break
            kind = "cut"
        after = snap(C)
        e = dict(kind=kind, before=before, after=after, loops=n_loops(C))
        if kind == "cut":
            gone = ({b["id"] for b in before} - {a["id"] for a in after}).pop()
            e["cands"], e["gone"] = cands, gone
        ev.append(e)
    return ev, C


def chains_in(snapshot, win):
    x0, x1, y0, y1 = win
    return [c for c in snapshot if c["pts"][:, 0].max() >= x0 and c["pts"][:, 0].min() <= x1
            and c["pts"][:, 1].max() >= y0 and c["pts"][:, 1].min() <= y1]


def lane_txt(lanes):
    return "/".join(map(str, lanes))


def set_win(ax, win):
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def arrow(ax, n, t, color, lw, length=3.2):
    ax.annotate("", xy=n + t * length, xytext=n, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18, shrinkA=0, shrinkB=0))


def draw_end(ax, snapshot, cand, end):
    """One end of a chain: own outgoing direction, every other chain's, and the winning partner."""
    n, r = end["node"], 5.5
    for c in chains_in(snapshot, (n[0] - r, n[0] + r, n[1] - r, n[1] + r)):
        ax.plot(*c["pts"].T, color="#c4c4c4", lw=2, zorder=1)
    ax.plot(*cand["pts"].T, color="#e0301e", lw=3, alpha=.5, zorder=2)
    arrow(ax, n, end["dn"], "#e0301e", 3)
    for i, p in enumerate(end["partners"]):
        best = i == end["best"]
        arrow(ax, n, p["t"], "#1f9d55" if best else "#888888", 3 if best else 1.6)
        tip = n + p["t"] * 3.2
        ax.annotate(f"车道{lane_txt(p['lanes'])}\n-点积 = {p['neg']:+.2f}", tip, textcoords="offset points",
                    xytext=(6 if p["t"][0] > 0 else -6, 0), ha="left" if p["t"][0] > 0 else "right",
                    va="center", fontsize=11, color="#1f9d55" if best else "#666", zorder=5)
    ax.plot(*n, "ko", ms=7, zorder=5)
    dev = np.degrees(np.arccos(np.clip(end["score"], -1, 1)))
    ax.set_title(f"该端得分 = {end['score']:+.2f}（偏离直行 {dev:.0f}°）", fontsize=14)
    set_win(ax, (n[0] - r, n[0] + r, n[1] - r, n[1] + r))


def fig_score(ev):
    """Figure 1: how a single chain is scored (the 5.7 m cross-link, loop A)."""
    e = next(x for x in ev if x["kind"] == "cut")
    cand = next(c for c in e["cands"] if c["id"] == e["gone"])
    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    for c in chains_in(e["before"], WIN_A):
        ax0.plot(*c["pts"].T, color="#c4c4c4", lw=2)
    ax0.plot(*cand["pts"].T, color="#e0301e", lw=3.5)
    for end in cand["ends"]:
        ax0.plot(*end["node"], "ko", ms=8)
    ax0.set_title("被打分的链：车道36，5.7 m，横在左右两条主线之间", fontsize=14)
    set_win(ax0, WIN_A)
    order = sorted(range(2), key=lambda i: cand["ends"][i]["node"][0])  # label ends by real x position
    for j, (i, label) in enumerate(zip(order, ("左端", "右端"))):
        end = cand["ends"][i]
        ax = fig.add_subplot(gs[j + 1])
        draw_end(ax, e["before"], cand, end)
        weak = i == cand["weak"]
        ax.set_title(f"{label}：" + ax.get_title() + ("  ← 取这一端（较小）" if weak else ""), fontsize=14,
                     color="#c0200e" if weak else "black")
    fig.suptitle("第 1 步：一条链的得分 = 两端“最能直行贯通”的程度中较差的那一端\n"
                 "红箭头 = 本链离开路口的方向；其余箭头 = 路口上别的链；绿色 = 与本链最接近“直线延续”的那条，其点积取负值即得分"
                 "（1 = 正好直行，0 = 拐 90°，−1 = 掉头）", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIGS / "weakest_tree_1_score.png", dpi=90)
    plt.close(fig)


def fig_rounds(ev):
    """Figure 2: the two cut decisions with the score of every candidate."""
    cuts = [e for e in ev if e["kind"] == "cut"]
    fig, axs = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for row, (e, win, name) in zip(axs, [(cuts[0], WIN_A, "A"), (cuts[1], WIN_B, "B")]):
        cs = sorted([c for c in e["cands"] if c["loop"] == name], key=lambda c: c["score"])
        lo, hi = min(c["score"] for c in cs), max(c["score"] for c in cs)
        ax = row[0]
        for c in chains_in(e["before"], win):
            ax.plot(*c["pts"].T, color="#d8d8d8", lw=1.6, zorder=0)
        for c in cs:
            col = CMAP((c["score"] - 0.4) / 0.6)
            gone = c["id"] == e["gone"]
            ax.plot(*c["pts"].T, color="#e0301e" if gone else col, lw=4.5 if gone else 3.2,
                    ls=(0, (4, 2)) if gone else "-", zorder=3 if gone else 2)
            m = c["pts"][len(c["pts"]) // 2]
            ax.annotate(f"{c['score']:+.2f}", m, xytext=(7, 0), textcoords="offset points", fontsize=13,
                        weight="bold", color="#c0200e" if gone else "black", zorder=5,
                        bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
        ax.set_title(f"环 {name}：每条链的得分（红虚线 = 最低分，被删）", fontsize=14)
        set_win(ax, win)
        bx = row[1]
        ys = np.arange(len(cs))
        bx.barh(ys, [c["score"] for c in cs], color=["#e0301e" if c["id"] == e["gone"] else "#9ab" for c in cs])
        bx.set_yticks(ys)
        bx.set_yticklabels([f"车道{lane_txt(c['lanes'])}  {c['L']:.1f} m" for c in cs], fontsize=13)
        for y, c in zip(ys, cs):
            ends = "  /  ".join(f"{x['score']:+.2f}←车道{lane_txt(x['partners'][x['best']]['lanes'])}" for x in c["ends"])
            bx.text(0.405, y, f" 两端: {ends}", va="center", fontsize=11, color="black")
        bx.set_xlim(0.4, 1.35)
        bx.axvline(cs[0]["score"], color="#e0301e", ls=":", lw=1)
        bx.set_xlabel("得分（越低越“不顺”）", fontsize=13)
        bx.set_title(f"第 {1 if name == 'A' else 2} 次剪断（环 {name}）：删得分最低的链", fontsize=14)
    fig.suptitle("第 2 步：同一环里所有链都打分，删最低的一条", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "weakest_tree_2_rounds.png", dpi=90)
    plt.close(fig)


def frame(ax, e, prev, tgt=True):
    """Draw the state after event e; chains changed by e are highlighted."""
    if tgt:
        for a, b in segs(Gt):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#e6e6e6", lw=6, solid_capstyle="round", zorder=0)
    new = {c["id"] for c in e["after"]} - ({c["id"] for c in e["before"]} if "before" in e else set())
    for c in e["after"]:
        hot = "before" in e and c["id"] in new
        ax.plot(*c["pts"].T, color="#f08a00" if hot else "#2c7be5", lw=2.6 if hot else 1.8, zorder=2)
    if "before" in e:
        gone = {c["id"] for c in e["before"]} - {c["id"] for c in e["after"]}
        for c in e["before"]:
            if c["id"] in gone:
                ax.plot(*c["pts"].T, color="#e0301e", lw=2.4, ls=(0, (4, 2)), zorder=3)


def fig_pipeline(ev):
    """Figure 3: every event of the run, full graph, with the loop count after it."""
    n = len(ev)
    cols = 5
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 5.1 * rows))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for i, e in enumerate(ev):
        ax = axs.ravel()[i]
        ax.axis("on")
        frame(ax, e, None)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if e["kind"] == "start":
            ttl, col = f"start\n{e['loops']} 个环", "black"
        elif e["kind"] == "cut":
            gone = ({c["id"] for c in e["before"]} - {c["id"] for c in e["after"]}).pop()
            g = next(c for c in e["cands"] if c["id"] == gone)
            ttl = f"{i}. 剪断 环{g['loop']}：删车道{lane_txt(g['lanes'])}\n得分最低 {g['score']:+.2f} → {e['loops']} 个环"
            col = "#c0200e"
        else:
            ttl = f"{i}. {KIND[e['kind']]}\n→ {e['loops']} 个环"
            col = "#555"
        ax.set_title(ttl, fontsize=12, color=col)
    fig.suptitle("整个流程：每轮优先删自环 → 合并平行链 → 都没有时才“剪断”；橙 = 本步新生成的链，红虚线 = 本步删掉的链，灰粗线 = target",
                 fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.885, hspace=0.24, wspace=0.04)
    fig.savefig(FIGS / "weakest_tree_3_pipeline.png", dpi=80)
    plt.close(fig)


def fig_two_answers(ev):
    """Figure 4: loop B has two equally valid ways to be opened, they differ only in where lanes merge."""
    e = [x for x in ev if x["kind"] == "cut"][1]
    chains = {tuple(c["lanes"]): c for c in e["before"] if c["id"] in {x["id"] for x in e["cands"]}}
    c21, c23, c26 = chains[(21,)], chains[(23,)], chains[(26,)]
    stay = [c for c in e["before"] if c["id"] not in {x["id"] for x in e["cands"]}]
    variants = [("答案 1（最弱生成树边选的）：删 车道21", [c23, c26], c21, "#2c7be5"),
                ("答案 2（target / V0 选的）：删 车道23", [c21, c26], c23, "#1f9d55")]
    fig, axs = plt.subplots(1, 3, figsize=(20, 9.5), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    ax = axs[0]
    for c in chains_in(e["before"], WIN_B):
        ax.plot(*c["pts"].T, color="#c4c4c4", lw=2)
    for c, col, dx in ((c21, "#8e44ad", 10), (c23, "#d98200", -10), (c26, "#1a9aa8", 10)):
        ax.plot(*c["pts"].T, color=col, lw=3.6)
        m = c["pts"][len(c["pts"]) // 2]
        ax.annotate(f"车道{lane_txt(c['lanes'])}\n{geo.length(c['pts'] + O):.1f} m", m, xytext=(dx, 0),
                    textcoords="offset points", fontsize=13, color=col, weight="bold",
                    ha="left" if dx > 0 else "right", va="center",
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax.set_title("要解的环：三条链围成一个环\n只能留两条", fontsize=14)
    set_win(ax, WIN_B)
    for ax, (ttl, keep, drop, col) in zip(axs[1:], variants):
        for c in stay:
            ax.plot(*c["pts"].T, color="#c4c4c4", lw=2)
        for a, b in segs(Gt):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#e3e3e3", lw=9, solid_capstyle="round", zorder=0)
        for c in keep:
            ax.plot(*c["pts"].T, color=col, lw=3.4)
        ax.plot(*drop["pts"].T, color="#e0301e", lw=2.4, ls=(0, (4, 2)))
        ax.set_title(ttl, fontsize=14)
        set_win(ax, WIN_B)
    fig.suptitle("环 B 的两种合法拓扑：两个答案都是“一条主干在中途分出一条分支”，只差分叉点在 y≈27 还是 y≈42\n"
                 "灰粗线 = target 恰好选了答案 2，所以答案 1 在指标上被扣分；但这是“和哪份标注比”的问题，不是拓扑错误",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIGS / "weakest_tree_4_two_answers.png", dpi=90)
    plt.close(fig)


def main():
    ev, _ = trace()
    fig_score(ev)
    fig_rounds(ev)
    fig_pipeline(ev)
    fig_two_answers(ev)
    print("saved 4 figures to", FIGS)


if __name__ == "__main__":
    main()
