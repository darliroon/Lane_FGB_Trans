"""Contract every kernel must satisfy, and the registry that lists them."""
import geopandas as gpd
import numpy as np
import networkx as nx
import pytest

from lane_trans import KERNELS, convert
from lane_trans.api import OUTPUT_COLUMNS
from lane_trans.chains import chains_to_graph, to_chains
from lane_trans.graph import load_vertex_graph
from lane_trans.kernels import get_kernel, register
from lane_trans.metrics import compare
from lane_trans.params import Params

from conftest import V0_GOLDEN

ALL = sorted(KERNELS)


def run(tmp_path, kernel, src):
    out = tmp_path / f"out_{kernel.replace(':', '_')}.fgb"
    return convert(src, out, kernel), out


def test_expected_kernels_are_registered():
    assert {"v0", "v1", "v1:shortest", "v1:longest", "v1:hausdorff", "v1:weakest_tree"} <= set(KERNELS)


def test_v1_alias_is_the_weakest_tree_variant():
    assert KERNELS["v1"] is KERNELS["v1:weakest_tree"]


@pytest.mark.parametrize("kernel", ALL)
def test_removes_all_small_loops(tmp_path, kernel, start):
    rep, out = run(tmp_path, kernel, start)
    gdf = gpd.read_file(out)
    assert rep.loops_before == 11 and rep.loops_after == 0
    assert len(gdf) == rep.lanes_out > 0
    assert list(gdf.columns) == OUTPUT_COLUMNS + ["geometry"]
    assert set(gdf.geom_type) == {"LineString"}
    assert gdf.crs == gpd.read_file(start).crs
    # FlatGeobuf stores features in spatial-index order, so compare as a set
    assert sorted(gdf["Id"]) == list(range(1, len(gdf) + 1))
    Gr, _ = load_vertex_graph(out)
    assert nx.cycle_basis(Gr) == []


@pytest.mark.parametrize("kernel", ALL)
def test_keeps_the_dangling_ends(tmp_path, kernel, start, target):
    _, out = run(tmp_path, kernel, start)
    Gs, _ = load_vertex_graph(start)
    Gr, _ = load_vertex_graph(out)
    Gt, _ = load_vertex_graph(target)
    assert compare(Gr, Gt, Gs)["same_dangling_ends_as_start"] is True


@pytest.mark.parametrize("kernel", ALL)
def test_empty_input_gives_empty_output(tmp_path, kernel):
    p = tmp_path / "empty.fgb"
    gpd.GeoDataFrame({"Id": []}, geometry=gpd.GeoSeries([], crs=3857), crs=3857).to_file(p, driver="FlatGeobuf")
    rep = convert(p, tmp_path / "o.fgb", kernel)
    out = gpd.read_file(tmp_path / "o.fgb")
    assert len(out) == 0 and rep.lanes_out == 0
    assert list(out.columns) == OUTPUT_COLUMNS + ["geometry"] and out.crs == 3857


def test_v0_output_matches_the_frozen_v0_result(tmp_path, start):
    """v0 is a frozen baseline: its output on data/start.fgb must not change."""
    _, out = run(tmp_path, "v0", start)
    a, b = gpd.read_file(out), gpd.read_file(V0_GOLDEN)
    assert len(a) == len(b)
    assert a.union_all().symmetric_difference(b.union_all()).length < 1e-6


def test_v0_and_v1_differ_only_in_how_a_loop_is_cut(tmp_path, start):
    """Both start from the same snap/self-loop/parallel front end, so their parallel merges agree."""
    r0, _ = run(tmp_path, "v0", start)
    r1, _ = run(tmp_path, "v1", start)
    par = lambda log: [l for l in log if l.startswith("parallel")]
    assert par(r0.log)[:6] == par(r1.log)[:6]


# ------------------------------------------------------------------ registry
def test_unknown_kernel_lists_the_choices():
    with pytest.raises(ValueError, match="unknown kernel 'v9'.*v0"):
        get_kernel("v9")


def test_duplicate_registration_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register("v0")(lambda C, origin, params, on_step=None: [])


def test_registering_a_new_kernel_makes_it_usable(tmp_path, monkeypatch, start):
    calls = []

    def noop(C, origin, params, on_step=None):
        calls.append(C.number_of_edges())
        return ["nothing"]
    monkeypatch.setitem(KERNELS, "noop", noop)
    rep = convert(start, tmp_path / "o.fgb", "noop")
    assert calls and rep.log == ["nothing"] and rep.loops_after == rep.loops_before


def test_on_step_reports_every_change(start, origin):
    from lane_trans.chains import chains_to_graph, to_chains
    from lane_trans.params import Params

    G, _ = load_vertex_graph(start)
    C = to_chains(G)
    steps = []
    log = get_kernel("v0")(C, origin, Params(), on_step=lambda *a: steps.append(a))
    assert [s[0] for s in steps] == log
    msg, before, after, loops = steps[-1]
    assert loops == 0 and before != after


# ------------------------------------------------------------------ v2 is v0, driven faster
def _road(C):
    """The road as drawn: per chain, its two ends, the input lanes it carries, and its curve.

    Compared by geometry, not by vertex list: the two kernels may sample the same polyline
    at different vertices, which is not a difference in the road.
    """
    from shapely.geometry import LineString
    out = {}
    for _, _, d in C.edges(data=True):
        line = LineString(d["pts"])
        c = line.coords
        ends = tuple(sorted((tuple(np.round(c[0], 6)), tuple(np.round(c[-1], 6)))))
        out[ends] = (frozenset(d["src"]), line)
    return out


def _same_road(a, b, tol=1e-6):
    if set(a) != set(b):
        return False, f"different chains: {len(set(a) ^ set(b))}"
    for ends in a:
        if a[ends][0] != b[ends][0]:
            return False, f"{ends}: different src"
        if a[ends][1].hausdorff_distance(b[ends][1]) > tol:
            return False, f"{ends}: geometry differs"
    return True, ""


def _run_both(G, params=Params()):
    out = {}
    for name in ("v0", "v2"):
        C = to_chains(G)
        log = get_kernel(name)(C, (0.0, 0.0), params)
        out[name] = (_road(C), chains_to_graph(C), log)
    return out["v0"], out["v2"]


def test_v2_is_registered():
    assert "v2" in KERNELS


@pytest.mark.parametrize("params", [Params(), Params(max_loop_width=5.0), Params(max_loop_len=40.0),
                                    Params(connector_margin=0.0)],
                         ids=["default", "width5", "len40", "margin0"])
def test_v2_builds_the_same_road_as_v0_on_start(start, params):
    G, _ = load_vertex_graph(start)
    (seg0, Gr0, _), (seg2, Gr2, _) = _run_both(G, params)
    same, why = _same_road(seg0, seg2)
    assert same, why
    # same loops left (width5 leaves three on purpose), compared as sets: DFS order differs
    assert {frozenset(c) for c in nx.cycle_basis(Gr2)} == {frozenset(c) for c in nx.cycle_basis(Gr0)}
    assert sorted(n for n in Gr2 if Gr2.degree(n) == 1) == sorted(n for n in Gr0 if Gr0.degree(n) == 1)


def test_v2_builds_the_same_road_as_v0_on_random_graphs():
    """Grid coordinates make ties common, which is where the two kernels could drift apart."""
    import random
    random.seed(3)
    for _ in range(40):
        G = nx.Graph()
        grid = [(float(x), float(y)) for x in range(6) for y in range(6)]
        random.shuffle(grid)
        pts = grid[:random.randint(8, 20)]
        for i in range(len(pts) - 1):
            G.add_edge(pts[i], pts[i + 1], src={i})
        for j in range(random.randint(0, 6)):
            a, b = random.choice(pts), random.choice(pts)
            if a != b:
                G.add_edge(a, b, src={100 + j})
        (seg0, _, _), (seg2, _, _) = _run_both(G)
        same, why = _same_road(seg0, seg2)
        assert same, why


def test_v2_needs_no_more_steps_than_v0(start):
    """A change that removes one loop should not need more rounds than v0 takes."""
    G, _ = load_vertex_graph(start)
    (_, _, l0), (_, _, l2) = _run_both(G)
    assert len(l2) <= len(l0)
