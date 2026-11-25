"""Contract every kernel must satisfy, and the registry that lists them."""
import geopandas as gpd
import networkx as nx
import pytest

from lane_trans import KERNELS, convert
from lane_trans.api import OUTPUT_COLUMNS
from lane_trans.graph import load_vertex_graph
from lane_trans.kernels import get_kernel, register
from lane_trans.metrics import compare

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
    from lane_trans.chains import to_chains
    from lane_trans.params import Params

    G, _ = load_vertex_graph(start)
    C = to_chains(G)
    steps = []
    log = get_kernel("v0")(C, origin, Params(), on_step=lambda *a: steps.append(a))
    assert [s[0] for s in steps] == log
    msg, before, after, loops = steps[-1]
    assert loops == 0 and before != after
