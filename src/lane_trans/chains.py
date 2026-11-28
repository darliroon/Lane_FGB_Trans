"""Chain graph: the polyline between two vertices whose degree != 2.

Loop removal works on chains, not on vertices. A chain is a nx.MultiGraph edge with
    pts  (n, 2) array, geometry of the chain
    src  set of input lane Ids it was built from
    mod  True if its geometry was rewritten (averaged)
"""
import hashlib

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from . import geometry as geo
from .graph import key


def to_chains(G):
    C = nx.MultiGraph()
    ends = {n for n in G if G.degree(n) != 2}
    seen = set()

    def walk(a, b):
        pts, src, prev, cur = [a], set(), a, b
        while True:
            seen.add(frozenset((prev, cur)))
            src |= G[prev][cur]["src"]
            pts.append(cur)
            if cur in ends:
                return pts, src
            prev, cur = cur, next(n for n in G[cur] if n != prev)

    def add_from(a):
        for b in G[a]:
            if frozenset((a, b)) not in seen:
                pts, src = walk(a, b)
                C.add_edge(pts[0], pts[-1], pts=np.array(pts), src=src, mod=False)

    for a in list(ends):
        add_from(a)
    for a, b in G.edges():  # pure rings without any end node
        if frozenset((a, b)) not in seen:
            ends.add(a)
            add_from(a)
    return C


def dissolve(C):
    """Concatenate the two chains meeting at every degree-2 node."""
    changed = True
    while changed:
        changed = False
        for n in list(C.nodes):
            if n not in C or C.degree(n) != 2:
                continue
            es = list(C.edges(n, keys=True, data=True))
            if len(es) != 2 or any(u == v for u, v, _, _ in es):
                continue
            (_, a, _, da), (_, b, _, db) = es
            pa = geo.orient(da["pts"], a)        # a -> n
            pb = geo.orient(db["pts"], n)        # n -> b
            C.remove_node(n)
            C.add_edge(a, b, pts=np.vstack([pa, pb[1:]]), src=da["src"] | db["src"],
                       mod=da["mod"] or db["mod"])
            changed = True


def pts_digest(pts):
    """Stable fingerprint of a polyline, to break ties the same way in every run."""
    return hashlib.blake2b(np.ascontiguousarray(np.round(pts, 6), dtype=np.float64).tobytes(),
                           digest_size=8).digest()


def chain_key(u, v, pts):
    """Total order on chains that does not depend on insertion or iteration order.

    Two candidates that score exactly the same are ranked by their endpoints and then by
    their geometry, so "which tied candidate wins" is a property of the graph, not of the
    order the edges happen to sit in. Ties between identical chains are indistinguishable.
    """
    a, b = (u, v) if u <= v else (v, u)
    return (a, b, pts_digest(pts))


def _epsilon(a, b):
    """A deterministic per-edge offset in [0, 1e-6], so equal-weight paths have a unique order."""
    h = (int(round(a[0] * 1000)) * 73856093) ^ (int(round(a[1] * 1000)) * 19349663) \
        ^ (int(round(b[0] * 1000)) * 83492791) ^ (int(round(b[1] * 1000)) * 2971215073)
    return (h % 1000) * 1e-9


def small_loops(C, max_loop_len, max_loop_width=None):
    """-> (H, comps).

    H     simple graph of the chains, edge attributes k (multigraph key) and w (length).
    comps edge lists of every biconnected component with >= 3 chains and total length
          <= max_loop_len, i.e. every loop small enough to be an artefact, and (if
          max_loop_width is set) no wider than it. A wider loop is a real ring.
    """
    H = nx.Graph()
    # sorted, so that for parallel chains the same one (the largest key) stands for the pair
    # in every run, whatever order the edges were inserted in
    for u, v, k, d in sorted(C.edges(keys=True, data=True), key=lambda e: (e[0], e[1], e[2])):
        H.add_edge(u, v, k=k, w=geo.length(d["pts"]))
    comps = []
    for comp in nx.biconnected_component_edges(H):
        comp = list(comp)
        if len(comp) < 3 or sum(H[u][v]["w"] for u, v in comp) > max_loop_len:
            continue
        if geo.too_wide([C[u][v][H[u][v]["k"]]["pts"] for u, v in comp], max_loop_width):
            continue
        comps.append(comp)
    return H, comps


def hausdorff_to_alternative(C, H, u, v, k):
    """Hausdorff distance between chain (u,v,k) and the shortest path u -> v that avoids it.

    H is the simple graph from small_loops. None if there is no such path.
    Small means the rest of the loop already explains the chain.
    """
    H2 = H.copy()
    H2.remove_edge(u, v)
    try:
        path = nx.shortest_path(H2, u, v,
                                weight=lambda a, b, d: d["w"] + _epsilon(a, b))
    except nx.NetworkXNoPath:
        return None
    alt = np.vstack([geo.orient(C[a][b][H[a][b]["k"]]["pts"], a)
                     for a, b in zip(path[:-1], path[1:])])
    return LineString(C[u][v][k]["pts"]).hausdorff_distance(LineString(alt))


def pick_connector(C, cands, margin):
    """The most superfluous connector among `cands` [(u, v, k, su, sv)], or None.

    Canonical tie-break, so v0 and v2 choose the same chain from the same candidates.
    """
    conns = [c for c in cands if max(c[3], c[4]) < -margin]
    if not conns:
        return None
    return min(conns, key=lambda c: (min(c[3], c[4]),
                                     chain_key(c[0], c[1], C[c[0]][c[1]][c[2]]["pts"])))


def pick_redundant(C, H, cands):
    """(hausdorff, u, v, k) of the candidate its alternative path explains best, or None."""
    best = None
    for u, v, k, _, _ in cands:
        h = hausdorff_to_alternative(C, H, u, v, k)
        if h is None:
            continue
        cand = (h, chain_key(u, v, C[u][v][k]["pts"]), u, v, k)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    h, _, u, v, k = best
    return h, u, v, k


def n_loops(C):
    """Cyclomatic number of the chain multigraph (counts self-loops and parallel chains)."""
    return C.number_of_edges() - C.number_of_nodes() + nx.number_connected_components(C)


def snapshot(C):
    """id(pts) -> pts for every chain; lets a caller see which chains a step removed or added."""
    return {id(d["pts"]): d["pts"] for _, _, d in C.edges(data=True)}


def fmt(n, origin):
    return f"({n[0] - origin[0]:.1f},{n[1] - origin[1]:.1f})"


def chains_to_graph(C, isolated=()):
    """Chain multigraph -> vertex graph, as a reader of the written file sees it.

    Edges carry `mod` (True if the chain was rewritten). `isolated` nodes are kept.
    """
    G = nx.Graph()
    for _, _, d in C.edges(data=True):
        ks = [key(p) for p in d["pts"]]
        for a, b in zip(ks[:-1], ks[1:]):
            if a != b:
                prev = G.get_edge_data(a, b)
                G.add_edge(a, b, mod=bool(d["mod"] or (prev and prev["mod"])))
    G.add_nodes_from(isolated)
    return G


def chains_to_gdf(C, crs):
    """One LineString row per chain: Id (1..n), src_ids (comma-joined input Ids), modified."""
    rows = [{"Id": i + 1, "src_ids": ",".join(map(str, sorted(d["src"]))),
             "modified": bool(d["mod"]), "geometry": LineString(d["pts"])}
            for i, (_, _, d) in enumerate(C.edges(data=True))]
    if rows:
        return gpd.GeoDataFrame(rows, crs=crs)
    cols = pd.DataFrame({"Id": pd.Series(dtype="int64"), "src_ids": pd.Series(dtype="object"),
                         "modified": pd.Series(dtype="bool")})
    return gpd.GeoDataFrame(cols, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
