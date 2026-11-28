"""Loop-removal rules on small hand-made graphs and on data/start.fgb (kernel v0)."""
import networkx as nx
import numpy as np
import pytest

from lane_trans import geometry as geo
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
    assert any(line.startswith("connector") and "(88.9,53.6)-(94.6,52.8)" in line for line in log)
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


# ------------------------------------------------------------------ width limit
def _ring(n, w, d, center=(0.0, 0.0)):
    """n-node ring, w wide and d long, as a closed polygon cycle."""
    cx, cy = center
    pts = [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2),
           (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]
    if n == 1:   # a single self-loop chain: out along one side, back along the other
        return [(cx - w / 2, cy - d / 2), (cx - w / 2, cy + d / 2),
                (cx + w / 2, cy + d / 2), (cx + w / 2, cy - d / 2), (cx - w / 2, cy - d / 2)]
    return pts


def test_width_of_a_straight_line_is_zero():
    assert geo.width(np.array([(0.0, 0.0), (10.0, 0.0)])) < 1e-9


def test_width_is_the_short_side_of_the_ring():
    pts = np.array(_ring(4, 3.0, 20.0))
    assert abs(geo.width(pts) - 3.0) < 1e-6


def test_width_is_rotation_invariant():
    th = np.deg2rad(37.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = np.array(_ring(4, 3.0, 20.0)) @ R.T
    assert abs(geo.width(pts) - 3.0) < 1e-6


def test_wide_self_loop_is_kept_when_max_width_is_small():
    """A loop wider than max_loop_width is a real ring (roundabout), not an artefact."""
    G = nx.Graph()
    loop = _ring(1, 30.0, 30.0)
    for a, b in zip(loop, loop[1:]):
        G.add_edge(a, b, src={1})
    G.add_edge((0.0, -15.0), (0.0, -30.0), src={2})       # one exit: a 1-chain self-loop
    _, log, Gr = simplify(G, params=Params(max_loop_width=5.0))
    assert log == [] and len(nx.cycle_basis(Gr)) == 1


def test_narrow_self_loop_is_removed_when_max_width_is_small():
    G = nx.Graph()
    loop = _ring(1, 3.0, 30.0)
    for a, b in zip(loop, loop[1:]):
        G.add_edge(a, b, src={1})
    G.add_edge((-1.5, -15.0), (-1.5, -30.0), src={2})
    _, log, Gr = simplify(G, params=Params(max_loop_width=5.0))
    assert any(line.startswith("self-loop") for line in log)
    assert len(nx.cycle_basis(Gr)) == 0


def _parallel_pair(off):
    """Two chains between the same two junctions, `off` metres either side of the axis."""
    G = nx.Graph()
    for i, o in enumerate((-off, off)):
        G.add_edge((1000.0, 0.0), (1000.0 + o, 10.0), src={i})
        G.add_edge((1000.0 + o, 10.0), (1000.0, 20.0), src={i})
    G.add_edge((1000.0, 0.0), (1000.0, -10.0), src={9})    # junctions at both ends
    G.add_edge((1000.0, 20.0), (1000.0, 30.0), src={9})
    return G


def test_wide_parallel_bundle_is_kept_when_max_width_is_small():
    """Two lanes 6 m apart are a divided road, not duplicate geometry."""
    C, log, _ = simplify(_parallel_pair(3.0), params=Params(max_loop_width=5.0))
    assert log == []
    assert C.number_of_edges() == 4            # the two lanes both survive, plus the two exits


def test_narrow_parallel_bundle_is_merged_when_max_width_is_small():
    C, log, Gr = simplify(_parallel_pair(0.5), params=Params(max_loop_width=5.0))
    assert any(line.startswith("parallel") for line in log)
    assert stats(Gr)["loops"] == 0
    assert C.number_of_edges() == 1            # one lane; with no junction left it is all one chain


def test_wide_three_chain_loop_is_kept_when_max_width_is_small():
    G = nx.Graph()
    ring = _ring(3, 20.0, 40.0)
    for a, b in zip(ring, ring[1:] + ring[:1]):
        G.add_edge(a, b, src={1})
    G.add_edge((-10.0, 0.0), (-20.0, 0.0), src={2})
    G.add_edge((10.0, 0.0), (20.0, 0.0), src={3})
    G.add_edge((0.0, 20.0), (0.0, 30.0), src={4})
    _, log, Gr = simplify(G, params=Params(max_loop_width=5.0))
    assert log == [] and len(nx.cycle_basis(Gr)) == 1


def test_params_rejects_a_non_positive_max_loop_width():
    with pytest.raises(ValueError):
        Params(max_loop_width=0.0)
    with pytest.raises(ValueError):
        Params(max_loop_width=float("nan"))
    assert Params(max_loop_width=None).max_loop_width is None


def test_start_fgb_with_5m_width_limit_keeps_the_wide_loops(start, origin):
    """start.fgb: the 10.4 m loop and the 5.7 m bundle survive a 5 m width limit.

    Everything else is still handled, so the 3 remaining cycles are exactly:
    the wide 6-chain loop, the duplicate pair inside it, and lanes 32/33.
    """
    G, _ = load_vertex_graph(start)
    _, log, Gr = simplify(G, origin, params=Params(max_loop_width=5.0))
    assert not any("(88.9,53.6)-(94.6,52.8)" in line for line in log)   # the wide loop
    assert not any("(83.3,15.1)-(84.6,35.5)" in line for line in log)   # lanes 32/33
    assert any(line.startswith("parallel") and "(56.7,41.6)-(58.2,54.3)" in line for line in log)
    assert stats(Gr)["loops"] == 3


# ------------------------------------------------------------------ snapping: STRtree and the crash fix
def test_snap_matches_a_brute_force_scan_on_random_graphs():
    """The STRtree lookup must find exactly what scanning every edge found, in the same order."""
    import random
    import lane_trans.graph as gr

    random.seed(7)
    for _ in range(200):
        G = nx.Graph()
        pts = [(round(random.uniform(0, 10), 3), round(random.uniform(0, 10), 3)) for _ in range(10)]
        for i in range(0, 8, 2):
            G.add_edge(pts[i], pts[i + 1], src={i})
        segs = list(G.edges)
        for sid, _ in enumerate(range(random.randint(2, 8)), start=10):
            a, b = random.choice(segs)
            t = random.random()
            off = random.choice([0.0, 0.02, 0.049, 0.05, 0.051])
            base = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            tip = gr.key((base[0] + off, base[1]))
            far = gr.key((tip[0] + 3.0, tip[1] + 3.0))
            if tip != far and tip not in G:
                G.add_edge(tip, far, src={sid})
        A, B = G.copy(), G.copy()
        try:
            n_old = gr._snap_dangles_brute(A)
        except ValueError:
            continue   # the old implementation cannot handle this graph; covered below
        assert gr.snap_dangles(B) == n_old
        assert set(A.nodes) == set(B.nodes)
        assert {frozenset(e) for e in A.edges} == {frozenset(e) for e in B.edges}
        assert all(A[a][b]["src"] == B[a][b]["src"] for a, b in A.edges)


def test_snap_does_not_crash_when_a_tip_stops_being_dangling():
    """Three lanes: an earlier snap makes another tip degree 2. This used to raise ValueError."""
    G = nx.Graph()
    G.add_edge((6.704, 10.128), (6.976, 6.241), src={0})
    G.add_edge((6.752, 9.885), (5.868, 8.837), src={1})
    G.add_edge((6.753, 9.888), (9.319, 12.707), src={2})
    before = G.number_of_nodes()
    assert snap_dangles(G) == 1          # must not raise
    assert G.number_of_nodes() == before - 1   # that tip was moved onto the other lane
    assert all(G.degree(n) >= 1 for n in G)
