"""Convert lane-centerline FlatGeobuf files to node/edge graphs and plot them.

Nodes = every vertex of every LineString. Vertices with identical coordinates
(after rounding to 1 mm) are merged into one node, so lanes that share a
vertex are connected.
Edges = each pair of consecutive vertices in a LineString, directed in the
digitizing order. Each edge keeps the lane Id and row index it came from.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def fgb_to_graph(path, origin):
    gdf = gpd.read_file(path)
    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty.any():
        print(f"[{path.name}] skip {empty.sum()} rows with empty geometry, Id={gdf[empty].Id.tolist()}")
    gdf = gdf[~empty]

    ox, oy = origin
    G = nx.MultiDiGraph(name=path.stem)
    key2node = {}

    def node_of(xy):
        k = (round(xy[0], 3), round(xy[1], 3))
        if k not in key2node:
            nid = len(key2node)
            key2node[k] = nid
            G.add_node(nid, x=k[0] - ox, y=k[1] - oy)
        return key2node[k]

    for row_idx, row in gdf.iterrows():
        nodes = [node_of(c) for c in row.geometry.coords]
        for seg, (u, v) in enumerate(zip(nodes[:-1], nodes[1:])):
            if u == v:  # repeated consecutive vertex
                continue
            G.add_edge(u, v, lane_id=int(row.Id), row=int(row_idx), seg=seg)
    return G


def save(G):
    OUT.mkdir(exist_ok=True)
    data = {
        "nodes": [{"id": n, "x": d["x"], "y": d["y"]} for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
    }
    (OUT / f"{G.name}_graph.json").write_text(json.dumps(data, indent=1))


def draw(ax, G):
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    # nodes where lanes meet / end (degree != 2) are drawn larger
    key = [n for n in G if G.in_degree(n) + G.out_degree(n) != 2]
    mid = [n for n in G if n not in set(key)]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>",
                           arrowsize=6, width=0.8, edge_color="#4C72B0",
                           node_size=15)
    nx.draw_networkx_nodes(G, pos, nodelist=mid, ax=ax, node_size=8,
                           node_color="#F2C14E", edgecolors="black", linewidths=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=key, ax=ax, node_size=30,
                           node_color="#DD8452", edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=3)
    # one lane-Id label per lane, at its middle segment
    lanes = {}
    for u, v, d in G.edges(data=True):
        lanes.setdefault(d["row"], []).append((d["seg"], u, v, d["lane_id"]))
    for segs in lanes.values():
        segs.sort()
        _, u, v, lid = segs[len(segs) // 2]
        ax.text((pos[u][0] + pos[v][0]) / 2 + 0.8, (pos[u][1] + pos[v][1]) / 2,
                str(lid), fontsize=6, color="#C44E52", ha="left", va="center")
    ax.set_title(f"{G.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    ax.set_aspect("equal")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")


def main():
    names = sys.argv[1:] or ["start", "target"]
    paths = [DATA / f"{n}.fgb" for n in names]
    # shared local origin so all graphs use the same coordinate frame
    b = gpd.read_file(paths[0]).total_bounds
    origin = (b[0], b[1])
    graphs = [fgb_to_graph(p, origin) for p in paths]

    fig, axes = plt.subplots(1, len(graphs), figsize=(10 * len(graphs), 10))
    axes = [axes] if len(graphs) == 1 else axes
    for ax, G in zip(axes, graphs):
        save(G)
        draw(ax, G)
        print(f"[{G.name}] nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
              f"weakly_connected_components={nx.number_weakly_connected_components(G)}")
    fig.tight_layout()
    fig.savefig(OUT / "graphs.png", dpi=300)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
