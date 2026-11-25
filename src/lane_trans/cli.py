"""Command line: lane-trans input.fgb output.fgb [--kernel NAME]."""
import argparse
import sys

from .api import DEFAULT_KERNEL, convert
from .graph import load_vertex_graph
from .kernels import KERNELS
from .metrics import compare
from .params import Params


def build_parser():
    d = Params()
    ap = argparse.ArgumentParser(prog="lane-trans",
                                 description="Remove small loops from lane centerlines: input.fgb -> output.fgb")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--kernel", default=DEFAULT_KERNEL, choices=sorted(KERNELS),
                    help="algorithm (default: %(default)s)")
    ap.add_argument("--max-loop-len", type=float, help=f"metres (default {d.max_loop_len:g})")
    ap.add_argument("--connector-margin", type=float, help=f"v0 only (default {d.connector_margin:g})")
    ap.add_argument("--parallel-ratio", type=float, help=f"default {d.parallel_ratio:g}")
    ap.add_argument("--target", help="reference .fgb; print how close the output is to it")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not print the per-step log")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        rep = convert(args.input, args.output, args.kernel, max_loop_len=args.max_loop_len,
                      connector_margin=args.connector_margin, parallel_ratio=args.parallel_ratio)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        print("\n".join(rep.log))
    print(f"kernel={rep.kernel}  lanes {rep.lanes_in} -> {rep.lanes_out}  "
          f"loops {rep.loops_before} -> {rep.loops_after}  {rep.seconds:.2f}s  -> {args.output}")
    if args.target:
        Gs, _ = load_vertex_graph(args.input)
        Gr, _ = load_vertex_graph(args.output)
        Gt, _ = load_vertex_graph(args.target)
        print("vs target:", compare(Gr, Gt, Gs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
