"""V0 (connector rule) vs the four V1 cut rules: one overview figure + metrics on stdout.

V0: kernels.v0.cut (connector rule, then Hausdorff fallback).
V1: same snap / self-loop / parallel-merge front end, but each loop only cuts a chain
    chosen by a loop-internal criterion (shortest, longest, Hausdorff, weakest tree edge).

    python scripts/compare_v0_v1.py        ->  docs/figures/v0_vs_v1_overview.png
"""
from _common import (Gs, Gt, O, FIGS, compare, draw, draw_target, plt, run, stats, style)

from lane_trans.kernels import v0, v1

FIG = FIGS / "v0_vs_v1_overview.png"


def metric_title(name, c, loops):
    return (f"{name}（{loops} 个环）\n0.5 m 覆盖 {c['result_within_0.5m_of_target']:.3f} / "
            f"{c['target_within_0.5m_of_result']:.3f}   Hausdorff {c['hausdorff_m']} m")


def main():
    r0 = run(v0.cut)
    r1 = [("V1-a 删最短链", run(v1.CUTS["shortest"])), ("V1-b 删最长链", run(v1.CUTS["longest"])),
          ("V1-c Hausdorff 冗余", run(v1.CUTS["hausdorff"])), ("V1-d 最弱生成树边", run(v1.CUTS["weakest_tree"]))]
    for name, (_, c, loops) in [("V0 connector", r0)] + r1:
        print(f"{name:20s} loops={loops}  {c}")

    fig, axs = plt.subplots(2, 4, figsize=(26, 15))

    draw(axs[0, 0], Gs, "#4C72B0")
    style(axs[0, 0], f"start（原始，{stats(Gs)['loops']} 个环）")
    draw(axs[0, 1], Gt, "#C44E52")
    style(axs[0, 1], f"target（人工修改，{stats(Gt)['loops']} 个环）")
    draw(axs[0, 2], r0[0], "#1f9d55")
    style(axs[0, 2], metric_title("V0 结果", r0[1], r0[2]))
    draw_target(axs[0, 3])
    draw(axs[0, 3], r0[0], "#1f9d55")
    style(axs[0, 3], "叠加：灰 = target，绿 = V0")

    for ax, (name, (Gr, c, loops)) in zip(axs[1], r1):
        draw_target(ax)
        draw(ax, Gr, "#2c7be5")
        style(ax, metric_title(name, c, loops))

    fig.suptitle("V0（connector 判据）vs V1（只按环删链的 4 种变体）  灰粗线 = target，"
                 "覆盖率为 result→target / target→result", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=1.5)
    fig.savefig(FIG, dpi=110)
    print("saved", FIG)


if __name__ == "__main__":
    main()
