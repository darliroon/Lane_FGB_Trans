"""v0: connector rule, Hausdorff fallback.

Every small loop is cut in one chain, picked by two rules in order:

  connector  A chain that is superfluous at BOTH of its ends: at each junction two OTHER
             chains continue into each other better (by more than params.connector_margin)
             than into this chain, so it only joins two through-lanes. The most superfluous
             such chain is deleted.
  redundant  Otherwise the chain whose alternative path through the loop is closest to it
             (smallest Hausdorff distance) is deleted.

Registered as "v0".
"""
from .. import geometry as geo
from ..chains import fmt, pick_connector, pick_redundant, small_loops
from ..pipeline import simplify
from .registry import register


def superfluous(C, u, v, k, d):
    """How unnecessary chain (u,v,k) is at each of its ends -> [at u, at v].

    At a junction two chains continue into each other when their outgoing directions are
    opposite (-dot ~ 1). For each end, compare how well the chain itself continues (best
    -dot with another incident chain) with how well two OTHER incident chains continue into
    each other. Negative values mean the junction is already covered without this chain.
    """
    out = []
    for n in (u, v):
        dn = geo.tangent(geo.orient(d["pts"], n))
        others = [geo.tangent(geo.orient(dd["pts"], n))
                  for uu, vv, kk, dd in C.edges(n, keys=True, data=True)
                  if not (kk == k and {uu, vv} == {u, v}) and uu != vv]
        own = max((-float(dn @ t) for t in others), default=-2.0)
        pair = max((-float(others[i] @ others[j]) for i in range(len(others))
                    for j in range(i + 1, len(others))), default=-2.0)
        out.append(own - pair)
    return out


def cut(C, origin, params):
    """Delete one chain of one small loop -> log line, or None if there is no small loop."""
    H, comps = small_loops(C, params.max_loop_len, params.max_loop_width)
    cands = []  # (u, v, k, superfluous_at_u, superfluous_at_v)
    for comp in comps:
        for u, v in comp:
            k = H[u][v]["k"]
            su, sv = superfluous(C, u, v, k, C[u][v][k])
            cands.append((u, v, k, su, sv))
    if not cands:
        return None

    hit = pick_connector(C, cands, params.connector_margin)
    if hit is not None:
        u, v, k, su, sv = hit
        d = C[u][v][k]
        C.remove_edge(u, v, k)
        if v < u:                 # canonical direction, so the log does not depend on edge order
            u, v, su, sv = v, u, sv, su
        return (f"connector   {fmt(u, origin)}-{fmt(v, origin)} lanes={sorted(d['src'])} "
                f"superfluous={su:.2f},{sv:.2f}")

    best = pick_redundant(C, H, cands)
    if best is None:
        return None
    h, u, v, k = best
    d = C[u][v][k]
    C.remove_edge(u, v, k)
    u, v = sorted((u, v))
    return f"redundant   {fmt(u, origin)}-{fmt(v, origin)} lanes={sorted(d['src'])} hausdorff_to_alt={h:.1f}"


@register("v0")
def run(C, origin, params, on_step=None):
    return simplify(C, origin, params, cut, on_step)
