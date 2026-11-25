"""Front-end rules shared by every kernel.

Each rule makes at most one change and returns its log line, or None if it did nothing.
"""
from . import geometry as geo
from .chains import fmt


def remove_self_loop(C, origin, params):
    """A chain that starts and ends at the same junction -> delete."""
    for u, v, k, d in C.edges(keys=True, data=True):
        if u == v and C.degree(u) > 2 and geo.length(d["pts"]) <= params.max_loop_len:
            C.remove_edge(u, v, k)
            return f"self-loop   at {fmt(u, origin)} lanes={sorted(d['src'])} L={geo.length(d['pts']):.1f}"
    return None


def merge_parallel(C, origin, params):
    """Chains with the same two end nodes -> one chain, the point-wise mean of them."""
    groups = {}
    for u, v, k, d in C.edges(keys=True, data=True):
        if u != v:
            groups.setdefault(frozenset((u, v)), []).append((u, v, k, d))
    for es in groups.values():
        if len(es) < 2:
            continue
        a = es[0][0]
        lines = [geo.orient(d["pts"], a) for _, _, _, d in es]
        lens = [geo.length(l) for l in lines]
        if sorted(lens)[0] + sorted(lens)[1] > params.max_loop_len:
            continue
        # a chain much longer than the shortest is a different route, not a duplicate
        keep = [l for l, L in zip(lines, lens) if L <= params.parallel_ratio * min(lens)]
        src = set().union(*(d["src"] for *_, d in es))
        b = es[0][1] if es[0][0] == a else es[0][0]
        for u, v, k, _ in es:
            C.remove_edge(u, v, k)
        C.add_edge(a, b, pts=geo.average(keep), src=src, mod=True)
        return (f"parallel    {fmt(a, origin)}-{fmt(b, origin)} {len(es)} chains "
                f"(averaged {len(keep)}) lanes={sorted(src)}")
    return None
