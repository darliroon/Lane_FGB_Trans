"""Web UI: draw any graph in the browser, press "convert", see the optimized graph.

    lane-trans-web [--host 127.0.0.1] [--port 8765]

The page sends the drawn graph (coordinates in metres) to /api/convert, which runs the same
pipeline as lane_trans.convert(): to_chains -> kernel. The optional request field "kernel"
selects the algorithm (default v0); the page itself does not expose it yet.
"""
import argparse
import json
import math
from pathlib import Path

import networkx as nx
from flask import Flask, jsonify, request

from ..api import DEFAULT_KERNEL
from ..chains import chains_to_graph, to_chains
from ..graph import key, snap_dangles
from ..kernels import KERNELS, get_kernel
from ..metrics import stats
from ..params import Params

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "example_graph.json"
MAX_NODES = 5000
ORIGIN = (0.0, 0.0)   # the page already works in local metres, so log coordinates need no offset
PARAM_NAMES = ("max_loop_len", "connector_margin", "parallel_ratio")


class InvalidGraph(ValueError):
    pass


def parse_graph(data):
    """{"nodes": [{id,x,y}], "edges": [{source,target}]} -> simple nx.Graph keyed by rounded (x, y)."""
    try:
        raw_nodes, raw_edges = data["nodes"], data["edges"]
        if len(raw_nodes) > MAX_NODES:
            raise InvalidGraph(f"too many nodes ({len(raw_nodes)} > {MAX_NODES})")
        pos = {}
        for n in raw_nodes:
            x, y = float(n["x"]), float(n["y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise InvalidGraph(f"node {n['id']} has a non-finite coordinate")
            pos[int(n["id"])] = key((x, y))
        edges = [(int(e["source"]), int(e["target"])) for e in raw_edges]
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, InvalidGraph):
            raise
        raise InvalidGraph(f"malformed graph: {e!r}")

    G = nx.Graph()
    G.add_nodes_from(pos.values())          # keep isolated nodes; they are not loops
    for i, (a, b) in enumerate(edges):
        if a not in pos or b not in pos:
            raise InvalidGraph(f"edge {i} refers to an unknown node")
        ka, kb = pos[a], pos[b]
        if ka == kb:                        # self edge or two nodes on the same spot
            continue
        if G.has_edge(ka, kb):
            G[ka][kb]["src"].add(i)
        else:
            G.add_edge(ka, kb, src={i})
    snap_dangles(G)   # tips that only touch a lane hide a loop as a hook
    return G


def parse_params(p):
    p = p or {}
    defaults = Params()
    try:
        vals = {k: float(p.get(k, getattr(defaults, k))) for k in PARAM_NAMES}
    except (TypeError, ValueError) as e:
        raise InvalidGraph(f"malformed params: {e!r}")
    try:
        return Params(**vals)
    except ValueError as e:
        raise InvalidGraph(str(e))


def graph_json(G):
    idx = {n: i for i, n in enumerate(G.nodes)}
    return {
        "nodes": [{"id": idx[n], "x": n[0], "y": n[1]} for n in G.nodes],
        "edges": [{"source": idx[a], "target": idx[b], "mod": bool(d.get("mod", False))}
                  for a, b, d in G.edges(data=True)],
    }


def create_app():
    app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    @app.errorhandler(InvalidGraph)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(Exception)
    def server_error(e):
        app.logger.exception("convert failed")
        code = getattr(e, "code", 500)
        return jsonify(error=f"{type(e).__name__}: {e}"), code if isinstance(code, int) else 500

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/config")
    def config():
        d = Params()
        return jsonify(defaults={k: getattr(d, k) for k in PARAM_NAMES},
                       kernels=sorted(KERNELS), kernel=DEFAULT_KERNEL,
                       example=EXAMPLE.exists())

    @app.get("/api/example")
    def example():
        if not EXAMPLE.exists():
            return jsonify(error="no example data"), 404
        return jsonify(json.loads(EXAMPLE.read_text()))

    @app.post("/api/convert")
    def convert():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise InvalidGraph("body must be a JSON object")
        try:
            run = get_kernel(data.get("kernel", DEFAULT_KERNEL))
        except (TypeError, ValueError) as e:
            raise InvalidGraph(str(e))
        G = parse_graph(data)
        params = parse_params(data.get("params"))

        isolated = [n for n in G if G.degree(n) == 0]
        C = to_chains(G)
        log = run(C, ORIGIN, params)
        Gr = chains_to_graph(C, isolated)
        return jsonify(graph=graph_json(Gr), log=log, stats={"before": stats(G), "after": stats(Gr)})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lane-trans-web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
