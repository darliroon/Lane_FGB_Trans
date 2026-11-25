"""Graph statistics and similarity to a reference graph."""
import networkx as nx
from shapely.geometry import MultiLineString
from shapely.ops import unary_union


def stats(G):
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "loops": len(nx.cycle_basis(G)),
        "components": nx.number_connected_components(G),
        "dangling_ends": sum(1 for n in G if G.degree(n) == 1),
        "junction_degrees": sorted(G.degree(n) for n in G if G.degree(n) >= 3),
    }


def as_geom(G):
    return unary_union(MultiLineString([[a, b] for a, b in G.edges()]))


def compare(Gr, Gt, Gs, tol=0.5):
    """Result Gr against target Gt; Gs is the start graph (its dangling ends should survive)."""
    gr, gt = as_geom(Gr), as_geom(Gt)
    ends = lambda G: {n for n in G if G.degree(n) == 1}
    return {
        "same_dangling_ends_as_start": ends(Gr) == ends(Gs),
        f"result_within_{tol}m_of_target": round(gr.intersection(gt.buffer(tol)).length / gr.length, 3),
        f"target_within_{tol}m_of_result": round(gt.intersection(gr.buffer(tol)).length / gt.length, 3),
        "hausdorff_m": round(gr.hausdorff_distance(gt), 2),
    }
