"""Vertex graph: an FGB layer as an undirected graph whose nodes are vertices.

Vertices closer than 1 mm are the same node, so lanes that share a vertex are connected.
Every edge remembers the input lane Ids it came from (attribute `src`).
"""
import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point

SNAP = 3             # vertex merge precision: 1 mm
SNAP_TOUCH = 0.05    # m; dangling end this close to another segment is treated as touching


def key(c):
    return (round(float(c[0]), SNAP), round(float(c[1]), SNAP))


def load_vertex_graph(path):
    """-> (nx.Graph, crs) for the LineString layer in `path`."""
    gdf = gpd.read_file(path)
    return vertex_graph(gdf), gdf.crs


def vertex_graph(gdf):
    """GeoDataFrame of LineStrings with an integer `Id` column -> nx.Graph.

    Dangling ends that touch another segment are snapped onto it.
    """
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    G = nx.Graph()
    for _, r in gdf.iterrows():
        ks = [key(c) for c in r.geometry.coords]
        for a, b in zip(ks[:-1], ks[1:]):
            if a == b:
                continue
            if G.has_edge(a, b):
                G[a][b]["src"].add(int(r.Id))
            else:
                G.add_edge(a, b, src={int(r.Id)})
    snap_dangles(G)
    return G


def snap_dangles(G, tol=SNAP_TOUCH):
    """Connect every dangling endpoint that touches another segment (distance <= tol).

    A lane clipped just short of the lane it runs into leaves the tip 0.x mm off
    the other polyline, with no vertex there. That hides a small loop as a
    hook. Project the tip onto the segment, split the segment at the projection
    and move the tip there, so the junction becomes a real vertex.

    Segments are looked up through an STRtree of the original edges, which is O(k log n)
    per tip instead of the O(n) scan that `_snap_dangles_brute` does. Edges created by
    earlier snaps in the same call are kept in `added` and checked too, so the result is
    identical to the brute-force scan. Projections use the segment in canonical vertex
    order, so the answer does not depend on which way the edge happens to be stored.
    """
    edges0 = list(G.edges())
    tips = [n for n in G if G.degree(n) == 1]
    if not tips or not edges0:
        G.graph["snapped"] = 0
        return 0

    ends = np.array(edges0, dtype=float)                     # (n, 2, 2)
    tree = STRtree(shapely.linestrings(ends))
    added = []                                               # segments split off during this call
    snaps = 0
    for n in tips:
        if n not in G or G.degree(n) != 1:
            continue          # an earlier snap already used this node; it is not a dangling tip now
        tip, best = Point(n), None
        x, y = n
        near = tree.query(shapely.box(x - tol, y - tol, x + tol, y + tol))
        for a, b in [edges0[i] for i in sorted(near)] + added:
            if n in (a, b) or not G.has_edge(a, b):
                continue      # the tip itself, or a segment an earlier snap split or removed
            seg = LineString([a, b] if a <= b else [b, a])
            p = seg.interpolate(seg.project(tip))
            d = tip.distance(p)
            if d <= tol and (best is None or d < best[0]):
                best = (d, a, b, p)
        if best is None:
            continue
        snaps += 1
        _, a, b, p = best
        k_new = key((p.x, p.y))
        (m,) = list(G[n])          # degree-1 tip; degree was re-checked above

        if k_new not in (a, b):
            src = set(G[a][b]["src"])
            G.remove_edge(a, b)
            G.add_edge(a, k_new, src=set(src))
            G.add_edge(k_new, b, src=set(src))
            added += [(a, k_new), (k_new, b)]

        if k_new != n:             # move the tip's edge onto the split point
            src = set(G[n][m]["src"])
            G.remove_edge(n, m)
            if G.has_edge(k_new, m):
                G[k_new][m]["src"] |= src
            else:
                G.add_edge(k_new, m, src=src)
                added.append((k_new, m))
            G.remove_node(n)
    G.graph["snapped"] = snaps
    return snaps


def _snap_dangles_brute(G, tol=SNAP_TOUCH):
    """The original O(tips x edges) scan, kept as the reference the STRtree version is tested against.

    Raises ValueError on graphs where an earlier snap turns a later tip into a degree-2 node;
    `snap_dangles` checks the degree instead.
    """
    snaps = 0
    for n in [x for x in G if G.degree(x) == 1]:
        if n not in G:
            continue
        tip, best = Point(n), None
        for a, b in G.edges():
            if n in (a, b):
                continue
            seg = LineString([a, b] if a <= b else [b, a])
            p = seg.interpolate(seg.project(tip))
            d = tip.distance(p)
            if d <= tol and (best is None or d < best[0]):
                best = (d, a, b, p)
        if best is None:
            continue
        snaps += 1
        _, a, b, p = best
        k_new = key((p.x, p.y))
        (m,) = list(G[n])

        if k_new not in (a, b):
            src = set(G[a][b]["src"])
            G.remove_edge(a, b)
            G.add_edge(a, k_new, src=set(src))
            G.add_edge(k_new, b, src=set(src))

        if k_new != n:
            src = set(G[n][m]["src"])
            G.remove_edge(n, m)
            if G.has_edge(k_new, m):
                G[k_new][m]["src"] |= src
            else:
                G.add_edge(k_new, m, src=src)
            G.remove_node(n)
    G.graph["snapped"] = snaps
    return snaps
