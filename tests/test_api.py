"""convert(): file contract, input validation, thresholds. And the command line."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString

from lane_trans import KERNELS, Params, convert
from lane_trans.cli import main


def write(tmp_path, geoms, name="in.fgb", crs=3857):
    p = tmp_path / name
    gpd.GeoDataFrame({"Id": range(1, len(geoms) + 1)}, geometry=geoms, crs=crs).to_file(p, driver="FlatGeobuf")
    return p


# ------------------------------------------------------------------ convert()
def test_unknown_kernel(tmp_path, start):
    with pytest.raises(ValueError, match="unknown kernel 'v9'"):
        convert(start, tmp_path / "o.fgb", "v9")


def test_output_may_not_overwrite_input(tmp_path):
    p = write(tmp_path, [LineString([(0, 0), (1, 0)])])
    with pytest.raises(ValueError, match="differ"):
        convert(p, p)


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path / "nope.fgb", tmp_path / "o.fgb")


def test_multilinestring_is_rejected_with_a_clear_message(tmp_path):
    p = write(tmp_path, [MultiLineString([[(0, 0), (1, 1)], [(1, 1), (2, 0)]])])
    with pytest.raises(ValueError, match="only LineString.*MultiLineString"):
        convert(p, tmp_path / "o.fgb")


def test_geographic_crs_is_rejected(tmp_path):
    p = write(tmp_path, [LineString([(116.0, 39.0), (116.001, 39.0)])], crs=4326)
    with pytest.raises(ValueError, match="projected CRS"):
        convert(p, tmp_path / "o.fgb")


def test_missing_id_column_is_rejected(tmp_path):
    p = tmp_path / "noid.fgb"
    gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=3857).to_file(p, driver="FlatGeobuf")
    with pytest.raises(ValueError, match="'Id'"):
        convert(p, tmp_path / "o.fgb")


def test_loop_free_input_passes_through(tmp_path):
    p = write(tmp_path, [LineString([(0, 0), (10, 0), (20, 5)]), LineString([(20, 5), (30, 5)])])
    rep = convert(p, tmp_path / "o.fgb", "v0")
    out = gpd.read_file(tmp_path / "o.fgb")
    assert rep.loops_before == rep.loops_after == 0
    assert out.union_all().symmetric_difference(gpd.read_file(p).union_all()).length < 1e-6


def test_output_directory_is_created(tmp_path):
    p = write(tmp_path, [LineString([(0, 0), (1, 0)])])
    convert(p, tmp_path / "a/b/o.fgb")
    assert (tmp_path / "a/b/o.fgb").is_file()


def test_failed_kernel_leaves_no_output(tmp_path, monkeypatch, start):
    def boom(C, origin, params, on_step=None):
        raise RuntimeError("boom")
    monkeypatch.setitem(KERNELS, "boom", boom)
    with pytest.raises(RuntimeError):
        convert(start, tmp_path / "o.fgb", "boom")
    assert not (tmp_path / "o.fgb").exists()


# ------------------------------------------------------------------ thresholds
def test_max_loop_len_override_applies(tmp_path, start):
    # 1 m: no loop in start.fgb is that small, so nothing is cut
    rep = convert(start, tmp_path / "o.fgb", "v0", max_loop_len=1.0)
    assert rep.loops_after == rep.loops_before == 11


def test_overrides_do_not_leak_into_the_next_call(tmp_path, start):
    convert(start, tmp_path / "a.fgb", "v0", max_loop_len=1.0)
    rep = convert(start, tmp_path / "b.fgb", "v0")
    assert rep.loops_after == 0


@pytest.mark.parametrize("bad", [dict(max_loop_len=0), dict(max_loop_len=float("nan")),
                                 dict(connector_margin=2), dict(parallel_ratio=0.5)])
def test_invalid_thresholds_are_rejected(bad):
    with pytest.raises(ValueError):
        Params(**bad)


def test_invalid_threshold_through_convert(tmp_path, start):
    with pytest.raises(ValueError, match="parallel_ratio"):
        convert(start, tmp_path / "o.fgb", parallel_ratio=0.5)


# ------------------------------------------------------------------ CLI
def test_cli_runs_and_reports(tmp_path, capsys, start, target):
    out = tmp_path / "o.fgb"
    assert main([str(start), str(out), "--kernel", "v1", "--target", str(target), "-q"]) == 0
    text = capsys.readouterr().out
    assert "kernel=v1" in text and "loops 11 -> 0" in text and "vs target" in text
    assert out.is_file()


def test_cli_prints_the_step_log_unless_quiet(tmp_path, capsys, start):
    main([str(start), str(tmp_path / "o.fgb")])
    assert "connector" in capsys.readouterr().out


def test_cli_error_exits_2_without_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "missing.fgb"), str(tmp_path / "o.fgb")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_unknown_kernel(tmp_path, start):
    with pytest.raises(SystemExit) as e:
        main([str(start), str(tmp_path / "o.fgb"), "--kernel", "v9"])
    assert e.value.code == 2


def test_python_dash_m_entry_point(tmp_path, start):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "lane_trans", str(start), str(tmp_path / "o.fgb"), "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "loops 11 -> 0" in r.stdout
