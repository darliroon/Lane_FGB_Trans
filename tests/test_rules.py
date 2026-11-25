"""Loop-removal rules on small hand-made graphs and on data/start.fgb (kernel v0)."""
import networkx as nx

from lane_trans.chains import chains_to_graph, to_chains
from lane_trans.graph import load_vertex_graph, snap_dangles
from lane_trans.kernels import get_kernel
from lane_trans.metrics import compare, stats
from lane_trans.params import Params


def simplify(G, origin=(0.0, 0.0), kernel="v0", params=Params()):
    C = to_chains(G)
    log = get_kernel(kernel)(C, origin, params)
    return C, log, chains_to_graph(C)


def node_near(G, origin, x, y, tol=0.01):
    """Node whose local coordinates (x, y) match, avoiding rounding drift."""
    for n in G:
        if abs(n[0] - origin[0] - x) <= tol and abs(n[1] - origin[1] - y) <= tol:
            return n
    raise AssertionError(f"no node near ({x}, {y})")


# ------------------------------------------------------------------ snapping
def test_snap_dangles_joins_tip_touching_a_segment():
    """A tip 0.1 mm off a lane is snapped, turning the hidden hook into a real loop."""
    G = nx.Graph()
    G.add_edge((0.0, 0.0), (0.0, 10.0), src={1})
    G.add_edge((0.0, 5.0), (1.5, 5.5), src={2})
    G.add_edge((1.5, 5.5), (0.0, 6.0), src={2})
    assert snap_dangles(G) == 2   # both ends of the hook touch the lane
    assert G.degree((0.0, 5.0)) >= 1
    assert G.has_edge((0.0, 0.0), (0.0, 5.0))
    assert G.has_edge((0.0, 5.0), (0.0, 6.0))   # the hook now closes a real loop
    assert G.has_edge((0.0, 6.0), (0.0, 10.0))


def test_hook_layer_is_flattened_in_start(start, origin):
    """start.fgb: the lane-16 hook ending on lane 14 (0.0001 m away) must be merged."""
    G, _ = load_vertex_graph(start)
    _, log, Gr = simplify(G, origin)
    assert nx.cycle_basis(Gr) == []
    # the bulge of the hook at (24.8, 48.9) must be gone
    assert not any(abs(n[0] - origin[0] - 24.8) < 0.2 and abs(n[1] - origin[1] - 48.9) < 0.2
                   for n in Gr)
    assert any("(25.5,48.5)-(25.7,50.5)" in line for line in log)


# ------------------------------------------------------------------ v0 connector rule
def test_connector_not_through_lane_is_removed(start, origin):
    """start.fgb: the horizontal join at x~90 is a connector, not lane 36."""
    G, _ = load_vertex_graph(start)
    _, log, Gr = simplify(G, origin)
    assert any(line.startswith("connector") and "(94.6,52.8)-(88.9,53.6)" in line for line in log)
    left = node_near(Gr, origin, 86.34, 51.03, tol=0.1)
    right_top = node_near(Gr, origin, 93.9, 83.47, tol=0.1)
    assert nx.has_path(Gr, left, right_top)  # the two through-lanes survive


def test_v0_result_does_not_regress_against_target(start, target, origin):
    G, _ = load_vertex_graph(start)
    _, _, Gr = simplify(G, origin)
    Gs, _ = load_vertex_graph(start)
    Gt, _ = load_vertex_graph(target)
    c = compare(Gr, Gt, Gs)
    assert c["same_dangling_ends_as_start"] is True
    assert c["result_within_0.5m_of_target"] >= 0.90
    assert c["target_within_0.5m_of_result"] >= 0.90
    assert c["hausdorff_m"] <= 1.5
    assert stats(Gr)["loops"] == 0


# ------------------------------------------------------------------ safety net
def test_large_loop_is_kept():
    s = 300.0
    G = nx.Graph()
    ring = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        G.add_edge(a, b, src={1})
    _, _, Gr = simplify(G)
    assert len(nx.cycle_basis(Gr)) == 1


def test_large_loop_is_cut_when_max_loop_len_allows():
    s = 300.0
    G = nx.Graph()
    ring = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        G.add_edge(a, b, src={1})
    G.add_edge((0.0, 0.0), (-10.0, -10.0), src={2})   # give the ring a junction
    G.add_edge((s, s), (s + 10.0, s + 10.0), src={3})
    _, _, Gr = simplify(G, params=Params(max_loop_len=5000.0))
    assert len(nx.cycle_basis(Gr)) == 0


def test_parallel_bundle_is_averaged_not_deleted():
    G = nx.Graph()
    for i, off in enumerate((-0.5, 0.0, 0.5)):
        G.add_edge((1000.0, 0.0), (1000.0 + off, 10.0), src={i})
        G.add_edge((1000.0 + off, 10.0), (1000.0, 20.0), src={i})
    _, _, Gr = simplify(G)
    assert stats(Gr)["loops"] == 0
    assert Gr.number_of_edges() == 1  # three parallel chains become one average


def test_empty_graph_is_fine():
    _, log, Gr = simplify(nx.Graph())
    assert log == [] and Gr.number_of_nodes() == 0
