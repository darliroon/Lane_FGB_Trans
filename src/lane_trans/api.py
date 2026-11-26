"""input.fgb -> output.fgb: remove small loops from a lane-centerline graph.

    from lane_trans import convert
    report = convert("start.fgb", "output.fgb", kernel="v0")

The file contract is the same whichever kernel runs; the kernel is the only thing that
changes (see lane_trans.kernels for the list and for how to add one).

Input   LineString layer in a projected CRS (units are metres), with an integer `Id` column.
Output  LineString layer, same CRS, columns Id (1..n), src_ids (comma-joined input Ids that
        the chain was built from), modified (True if its geometry was averaged). The input's
        other attribute columns are not carried over, because merged chains have no single
        source lane.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd

from .chains import chains_to_gdf, chains_to_graph, to_chains
from .graph import vertex_graph
from .kernels import get_kernel
from .metrics import stats
from .params import Params

DEFAULT_KERNEL = "v0"
OUTPUT_COLUMNS = ["Id", "src_ids", "modified"]


@dataclass
class Report:
    kernel: str
    lanes_in: int
    lanes_out: int
    loops_before: int
    loops_after: int      # loops longer than max_loop_len are kept on purpose
    seconds: float
    log: list = field(default_factory=list)


def read_input(path):
    """Read and validate the input layer -> (GeoDataFrame, origin)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    gdf = gpd.read_file(path)
    geoms = gdf.geometry[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if geoms.empty:
        return gdf, (0.0, 0.0)
    bad = sorted(set(geoms.geom_type) - {"LineString"})
    if bad:
        raise ValueError(f"{path.name}: only LineString is supported, found {', '.join(bad)}")
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError(f"{path.name}: needs a projected CRS in metres, got {gdf.crs}")
    if "Id" not in gdf.columns:
        raise ValueError(f"{path.name}: needs an integer 'Id' column")
    return gdf, tuple(float(v) for v in gdf.total_bounds[:2])


def convert(input_path, output_path, kernel=DEFAULT_KERNEL, *,
            max_loop_len=None, connector_margin=None, parallel_ratio=None,
            max_loop_width=None):
    """Read input_path (.fgb), remove its small loops with `kernel`, write output_path (.fgb).

    The keyword thresholds override the defaults in params.Params. Returns a Report.
    """
    run = get_kernel(kernel)
    given = dict(max_loop_len=max_loop_len, connector_margin=connector_margin,
                 parallel_ratio=parallel_ratio, max_loop_width=max_loop_width)
    params = Params(**{k: float(v) for k, v in given.items() if v is not None})
    src, dst = Path(input_path), Path(output_path)
    if dst.resolve() == src.resolve():
        raise ValueError("output_path must differ from input_path")
    gdf, origin = read_input(src)

    t0 = time.perf_counter()
    G = vertex_graph(gdf)
    C = to_chains(G)
    log = run(C, origin, params)
    out = chains_to_gdf(C, gdf.crs)
    seconds = time.perf_counter() - t0

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(dst, driver="FlatGeobuf")
    return Report(kernel=kernel, lanes_in=len(gdf), lanes_out=len(out),
                  loops_before=stats(G)["loops"], loops_after=stats(chains_to_graph(C))["loops"],
                  seconds=seconds, log=log)
