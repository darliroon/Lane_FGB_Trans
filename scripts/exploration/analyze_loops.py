"""Compare start vs target at lane level: find loops in start, show how target resolved them."""
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

D = Path(__file__).resolve().parent.parent.parent / "data"


def load(name):
    g = gpd.read_file(D / f"{name}.fgb")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    return g


def key(c):
    return (round(c[0], 3), round(c[1], 3))


def lane_graph(g):
    G = nx.MultiGraph()
    for i, r in g.iterrows():
        cs = list(r.geometry.coords)
        G.add_edge(key(cs[0]), key(cs[-1]), row=i, lid=int(r.Id), n=len(cs),
                   L=r.geometry.length, geom=r.geometry)
    return G


s, t = load("start"), load("target")
Gs, Gt = lane_graph(s), lane_graph(t)
ox, oy = s.total_bounds[:2]


def p(k):
    return f"({k[0]-ox:5.1f},{k[1]-oy:5.1f})"


# --- loops in start (cycle basis of undirected lane graph, incl. parallel edges)
print("=== loops in start (lane level) ===")
H = nx.Graph()
for u, v, d in Gs.edges(data=True):
    if H.has_edge(u, v):  # parallel lanes = 2-cycle
        print(f"bubble {p(u)}-{p(v)}: lanes {H[u][v]['lid']} & {d['lid']} "
              f"L={H[u][v]['L']:.1f}/{d['L']:.1f}")
    else:
        H.add_edge(u, v, **d)
for cyc in nx.cycle_basis(H):
    es = [H[a][b] for a, b in zip(cyc, cyc[1:] + cyc[:1])]
    print(f"cycle nodes={[p(x) for x in cyc]} lanes={[e['lid'] for e in es]} "
          f"L={[round(e['L'], 1) for e in es]}")

# --- lane matching: geometry identical / reversed / new
print("\n=== target lanes vs start ===")
sg = {i: r.geometry for i, r in s.iterrows()}
used = set()
for i, r in t.iterrows():
    tg = r.geometry
    match = None
    for j, g in sg.items():
        if g.equals_exact(tg, 1e-3):
            match = (j, "same")
        elif g.equals_exact(LineString(tg.coords[::-1]), 1e-3):
            match = (j, "reversed")
        if match:
            break
    if match:
        used.add(match[0])
        if match[1] != "same":
            print(f"t row{i} Id={r.Id}: {match[1]} of start row{match[0]} Id={s.Id[match[0]]}")
    else:
        cs = tg.coords
        print(f"t row{i} Id={r.Id}: NEW n={len(cs)} {p(key(cs[0]))}->{p(key(cs[-1]))} L={tg.length:.1f}")
for j in sorted(set(sg) - used):
    cs = sg[j].coords
    print(f"s row{j} Id={s.Id[j]}: REMOVED n={len(cs)} {p(key(cs[0]))}->{p(key(cs[-1]))} L={sg[j].length:.1f}")

# --- node degrees at junctions
print("\n=== junction nodes (degree>=3) ===")
for name, G in [("start", Gs), ("target", Gt)]:
    print(name, sorted((p(n), G.degree(n)) for n in G if G.degree(n) >= 3))
