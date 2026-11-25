"""v1: cut each small loop by one loop-internal criterion; no connector rule.

Four variants differ only in which chain of a small loop is deleted:

  shortest      the shortest chain
  longest       the longest chain
  hausdorff     the chain its alternative path explains best (smallest Hausdorff distance)
  weakest_tree  the chain that continues worst into its neighbours, scored at its worse end
                (see continuity). Plain "v1" is this variant.

Registered as "v1:<variant>" and "v1".
"""
from .. import geometry as geo
from ..chains import fmt, hausdorff_to_alternative, small_loops
from ..pipeline import simplify
from .registry import KERNELS, register


def continuity(C, u, v, k, d):
    """Worst end of chain (u,v,k): how straight it continues into its best partner there.

    1 = straight through, 0 = a 90 degree turn, -1 = U-turn.
    """
    vals = []
    for n in (u, v):
        dn = geo.tangent(geo.orient(d["pts"], n))
        best = -2.0
        for uu, vv, kk, dd in C.edges(n, keys=True, data=True):
            if kk == k and {uu, vv} == {u, v} or uu == vv:
                continue
            best = max(best, -float(dn @ geo.tangent(geo.orient(dd["pts"], n))))
        vals.append(best)
    return min(vals)


# A pick function chooses the chain to cut: pick(C, H, comps) -> (u, v, k, note), or None.
def _by_length(choose):
    def pick(C, H, comps):
        cands = [(H[u][v]["w"], u, v) for comp in comps for u, v in comp]
        if not cands:
            return None
        w, u, v = choose(cands)
        return u, v, H[u][v]["k"], f"L={w:.1f}"
    return pick


def _hausdorff(C, H, comps):
    best = None
    for comp in comps:
        for u, v in comp:
            k = H[u][v]["k"]
            h = hausdorff_to_alternative(C, H, u, v, k)
            if h is not None and (best is None or h < best[0]):
                best = (h, u, v, k)
    if best is None:
        return None
    h, u, v, k = best
    return u, v, k, f"hausdorff_to_alt={h:.1f}"


def _weakest_tree(C, H, comps):
    cands = [(continuity(C, u, v, H[u][v]["k"], C[u][v][H[u][v]["k"]]), u, v, H[u][v]["k"])
             for comp in comps for u, v in comp]
    if not cands:
        return None
    c, u, v, k = min(cands)
    return u, v, k, f"continuity={c:+.2f}"


PICKS = {"shortest": _by_length(min), "longest": _by_length(max),
         "hausdorff": _hausdorff, "weakest_tree": _weakest_tree}


def make_cut(variant):
    """cut(C, origin, params) -> log line or None, for one variant."""
    pick = PICKS[variant]

    def cut(C, origin, params):
        hit = pick(C, *small_loops(C, params.max_loop_len))
        if hit is None:
            return None
        u, v, k, note = hit
        lanes = sorted(C[u][v][k]["src"])
        C.remove_edge(u, v, k)
        return f"cut-{variant:<8s} {fmt(u, origin)}-{fmt(v, origin)} lanes={lanes} {note}"
    return cut


CUTS = {name: make_cut(name) for name in PICKS}


def _make_kernel(variant):
    cut = CUTS[variant]

    def run(C, origin, params, on_step=None):
        return simplify(C, origin, params, cut, on_step)
    run.__name__ = f"v1_{variant}"
    run.__doc__ = f"v1 kernel, cut rule: {variant}."
    return run


for _variant in PICKS:
    register(f"v1:{_variant}")(_make_kernel(_variant))
KERNELS["v1"] = KERNELS["v1:weakest_tree"]   # plain "v1" = weakest-tree variant
