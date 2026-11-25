"""Feasibility check: Steiner tree over terminals (degree-1 nodes) vs. hand-made target."""
import json
from pathlib import Path

import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from shapely.geometry import MultiLineString, LineString
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reads outputs/{start,target}_graph.json written by fgb_to_graph.py
OUT = Path(__file__).resolve().parent.parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


def load(name):
    d = json.load(open(OUT / f"{name}_graph.json"))
    pos = {n["id"]: (n["x"], n["y"]) for n in d["nodes"]}
    G = nx.Graph()
    for e in d["edges"]:
        u, v = e["source"], e["target"]
        G.add_edge(u, v, weight=LineString([pos[u], pos[v]]).length)
    return G, pos


def geom(G, pos):
    return unary_union(MultiLineString([[pos[u], pos[v]] for u, v in G.edges()]))


S, ps = load("start")
T, pt = load("target")
terms = [n for n in S if S.degree(n) == 1]
R = steiner_tree(S, terms, weight="weight", method="mehlhorn")

print("result: nodes", R.number_of_nodes(), "edges", R.number_of_edges(),
      "cycles", len(nx.cycle_basis(R)), "components", nx.number_connected_components(R))
gr, gt = geom(R, ps), geom(T, pt)
print(f"hausdorff(result, target) = {gr.hausdorff_distance(gt):.2f} m")
# how much of each lies within 0.5 m of the other
print(f"result length {gr.length:.0f} m, within 0.5m of target: {gr.intersection(gt.buffer(0.5)).length / gr.length:.0%}")
print(f"target length {gt.length:.0f} m, within 0.5m of result: {gt.intersection(gr.buffer(0.5)).length / gt.length:.0%}")

fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharex=True, sharey=True)
for ax, (G, pos, title) in zip(axes, [(S, ps, "start"), (R, ps, "steiner(start)"), (T, pt, "target")]):
    for u, v in G.edges():
        ax.plot(*zip(pos[u], pos[v]), color="#4C72B0", lw=1)
    ax.set_title(title)
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "steiner_compare.png", dpi=150)
