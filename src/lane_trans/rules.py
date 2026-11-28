"""Front-end rules shared by every kernel.

Each rule makes at most one change and returns its log line, or None if it did nothing.
"""
from . import geometry as geo
from .chains import chain_key, fmt


def remove_self_loop(C, origin, params):
    """A chain that starts and ends at the same junction -> delete.

    When several qualify, the first in canonical chain order goes, so the choice does not
    depend on the order the edges were inserted in.
    """
    hits = []
    for u, v, k, d in C.edges(keys=True, data=True):
        if (u == v and C.degree(u) > 2 and geo.length(d["pts"]) <= params.max_loop_len
                and not geo.too_wide([d["pts"]], params.max_loop_width)):
            hits.append((chain_key(u, v, d["pts"]), u, v, k, d))
    if not hits:
        return None
    _, u, v, k, d = min(hits)
    C.remove_edge(u, v, k)
    return f"self-loop   at {fmt(u, origin)} lanes={sorted(d['src'])} L={geo.length(d['pts']):.1f}"


def merge_parallel(C, origin, params):
    """Chains with the same two end nodes -> one chain, the point-wise mean of them."""
    groups = {}
    for u, v, k, d in C.edges(keys=True, data=True):
        if u != v:
            groups.setdefault(frozenset((u, v)), []).append((u, v, k, d))
    # every bundle that can be merged, taken in canonical node-pair order, so a run does not
    # depend on which pair the insertion order happens to put first
    for pair in sorted(groups, key=lambda g: tuple(sorted(g))):
        es = groups[pair]
        if len(es) < 2:
            continue
        msg = merge_bundle(C, es, origin, params)
        if msg is not None:
            return msg
    return None


def merge_bundle(C, es, origin, params):
    """Merge one bundle of parallel chains [(u, v, k, d)] into their point-wise mean.

    None if the bundle is not mergeable: too long, too wide, or not at least two chains.
    Shared by `merge_parallel` and kernel v2, so both merge the same bundle the same way.
    """
    if len(es) < 2:
        return None
    es = sorted(es, key=lambda e: (e[0], e[1], e[2]))
    a, b = sorted((es[0][0], es[0][1]))     # canonical direction: independent of insertion order
    lines = [geo.orient(d["pts"], a) for _, _, _, d in es]
    lens = [geo.length(l) for l in lines]
    if sorted(lens)[0] + sorted(lens)[1] > params.max_loop_len:
        return None
    if geo.too_wide(lines, params.max_loop_width):
        return None
    # a chain much longer than the shortest is a different route, not a duplicate
    keep = [l for l, L in zip(lines, lens) if L <= params.parallel_ratio * min(lens)]
    src = set().union(*(d["src"] for *_, d in es))
    for u, v, k, _ in es:
        C.remove_edge(u, v, k)
    C.add_edge(a, b, pts=geo.average(keep), src=src, mod=True)
    return (f"parallel    {fmt(a, origin)}-{fmt(b, origin)} {len(es)} chains "
            f"(averaged {len(keep)}) lanes={sorted(src)}")
