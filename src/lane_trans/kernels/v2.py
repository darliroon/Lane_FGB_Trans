"""v2: v0's rules, evaluated incrementally.

v0 changes one chain per round and then recomputes everything: every degree-2 node, every
parallel bundle, every biconnected component, every candidate's score. That is O(N) work per
round and O(N) rounds. v2 applies the same rules but only re-examines what a change touched:

  dissolve / self-loop / parallel  worklists of the nodes and node pairs a change touched
  loops                            biconnected components kept per block; after a change only
                                   the blocks that contain a touched node are recomputed
  scores                           connector scores are computed once per block, Hausdorff
                                   distances lazily and once per block

The decisions are v0's: the same `superfluous` score, the same connector-first-then-Hausdorff
choice over all small loops, the same thresholds, the same log lines. Every round still runs
dissolve -> self-loop -> parallel -> cut in that priority order. What can differ from v0:

  * the order of independent changes in the log (v0 takes the first one in networkx edge
    order, v2 the first one in its worklist), which does not change the result;
  * exact ties between two candidates are broken differently;
  * a self-loop chain that rule 1 left in place (too long, too wide) is ignored when the
    length of the loops around it is measured. v0 counts it into one neighbouring block,
    whichever its depth-first search happens to reach.

Registered as "v2".
"""
from collections import deque

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from .. import geometry as geo
from ..chains import fmt, n_loops, snapshot
from .registry import register
from .v0 import superfluous

_UNSET = object()


class _Block:
    """One biconnected component of the simple chain graph, and what v2 knows about it."""
    __slots__ = ("edges", "cands", "conn", "h")

    def __init__(self, edges):
        self.edges = edges    # [(u, v)]
        self.cands = ()       # [(u, v, k, superfluous_at_u, superfluous_at_v)] if it is a small loop
        self.conn = None      # the most superfluous connector among cands, or None
        self.h = _UNSET       # (hausdorff, u, v, k) of its best "redundant" candidate, filled lazily


def _hausdorff_to_alternative(C, Hb, u, v, k):
    """chains.hausdorff_to_alternative, on the block's own graph and without copying it.

    A simple path between two nodes of a biconnected block stays in the block, so searching
    the block alone finds the same shortest path as searching the whole graph.
    """
    view = nx.subgraph_view(Hb, filter_edge=lambda a, b: not ({a, b} == {u, v}))
    try:
        path = nx.shortest_path(view, u, v, weight="w")
    except nx.NetworkXNoPath:
        return None
    alt = np.vstack([geo.orient(C[a][b][Hb[a][b]["k"]]["pts"], a)
                     for a, b in zip(path[:-1], path[1:])])
    return LineString(C[u][v][k]["pts"]).hausdorff_distance(LineString(alt))


class _Net:
    """Chain graph C plus everything v2 keeps in step with it. All edits go through here."""

    def __init__(self, C, origin, params):
        self.C, self.origin, self.p = C, origin, params
        self.H = nx.Graph()          # simple graph of C without self-loops: k = chain key, w = length
        self.blk = {}                # frozenset({u, v}) -> id of the block that holds that edge
        self.blocks = {}             # id -> _Block, every block
        self.live = {}               # id -> _Block, the blocks that are small loops
        self.stale = set()           # ids of blocks that lost an edge
        self.dirty = set(C.nodes)    # nodes whose chains changed since the blocks were last built
        self.check = set(C.nodes)    # nodes that may now be degree 2
        self.loops = deque()         # nodes that may carry a removable self-loop
        self.pairs = deque()         # node pairs that may now be a parallel bundle
        self.queued = set()
        self._next = 0
        for u, v, k, d in C.edges(keys=True, data=True):
            if u == v:
                self.loops.append(u)
            else:
                self.H.add_edge(u, v, k=k, w=geo.length(d["pts"]))
        for u, v in self.H.edges:
            if C.number_of_edges(u, v) > 1:
                self._queue_pair(u, v)

    # ---------------------------------------------------------------- bookkeeping
    def _queue_pair(self, a, b):
        pair = frozenset((a, b))
        if pair not in self.queued:
            self.queued.add(pair)
            self.pairs.append(pair)

    def _touch(self, a, b):
        """Record that the chains between a and b changed; call after every edit of C."""
        self.dirty.update((a, b))
        self.check.update((a, b))
        if a == b:
            self.loops.append(a)
            return
        self._queue_pair(a, b)
        H, C = self.H, self.C
        if C.has_edge(a, b):
            keys = C[a][b]
            k = list(keys)[-1]       # v0 builds H last-one-wins over C.edges(), so does this
            H.add_edge(a, b, k=k, w=geo.length(keys[k]["pts"]))
        elif H.has_edge(a, b):
            H.remove_edge(a, b)
            bid = self.blk.pop(frozenset((a, b)), None)
            if bid is not None:
                self.stale.add(bid)
            for n in (a, b):
                if H.degree(n) == 0:
                    H.remove_node(n)

    # ---------------------------------------------------------------- rules 0-2
    def dissolve_all(self):
        """Concatenate the two chains at every degree-2 node (chains.dissolve, on touched nodes)."""
        C = self.C
        while self.check:
            n = self.check.pop()
            if n not in C or C.degree(n) != 2:
                continue
            es = list(C.edges(n, keys=True, data=True))
            if len(es) != 2 or any(u == v for u, v, _, _ in es):
                continue
            (_, a, ka, da), (_, b, kb, db) = es
            pa = geo.orient(da["pts"], a)        # a -> n
            pb = geo.orient(db["pts"], n)        # n -> b
            C.remove_edge(n, a, ka)
            self._touch(n, a)
            C.remove_edge(n, b, kb)
            self._touch(n, b)
            C.remove_node(n)
            C.add_edge(a, b, pts=np.vstack([pa, pb[1:]]), src=da["src"] | db["src"],
                       mod=da["mod"] or db["mod"])
            self._touch(a, b)

    def self_loop(self):
        """rules.remove_self_loop, on touched nodes."""
        C, p = self.C, self.p
        while self.loops:
            u = self.loops.popleft()
            if u not in C or not C.has_edge(u, u):
                continue
            for k, d in list(C[u][u].items()):
                L = geo.length(d["pts"])
                if C.degree(u) > 2 and L <= p.max_loop_len \
                        and not geo.too_wide([d["pts"]], p.max_loop_width):
                    C.remove_edge(u, u, k)
                    self._touch(u, u)
                    return f"self-loop   at {fmt(u, self.origin)} lanes={sorted(d['src'])} L={L:.1f}"
        return None

    def parallel(self):
        """rules.merge_parallel, on touched pairs."""
        C, p = self.C, self.p
        while self.pairs:
            pair = self.pairs.popleft()
            self.queued.discard(pair)
            a, b = tuple(pair)
            if not C.has_edge(a, b) or C.number_of_edges(a, b) < 2:
                continue
            es = list(C[a][b].items())
            lines = [geo.orient(d["pts"], a) for _, d in es]
            lens = [geo.length(l) for l in lines]
            s = sorted(lens)
            if s[0] + s[1] > p.max_loop_len or geo.too_wide(lines, p.max_loop_width):
                continue
            keep = [l for l, L in zip(lines, lens) if L <= p.parallel_ratio * min(lens)]
            src = set().union(*(d["src"] for _, d in es))
            for k, _ in es:
                C.remove_edge(a, b, k)
            C.add_edge(a, b, pts=geo.average(keep), src=src, mod=True)
            self._touch(a, b)
            return (f"parallel    {fmt(a, self.origin)}-{fmt(b, self.origin)} {len(es)} chains "
                    f"(averaged {len(keep)}) lanes={sorted(src)}")
        return None

    # ---------------------------------------------------------------- loops
    def _refresh(self):
        """Rebuild the blocks a change touched. Every other block is exactly as it was.

        A change can only alter the cycle structure inside the blocks that held the edges it
        removed, and the scores of the blocks that have a chain at a touched node.
        """
        if not (self.dirty or self.stale):
            return
        H = self.H
        ids, region = set(self.stale), set()
        for n in self.dirty:
            if n in H:
                for m in H[n]:
                    e = frozenset((n, m))
                    region.add(e)
                    if e in self.blk:
                        ids.add(self.blk[e])
        for bid in ids:
            b = self.blocks.pop(bid, None)
            self.live.pop(bid, None)
            if b is None:
                continue
            for u, v in b.edges:
                e = frozenset((u, v))
                if self.blk.get(e) == bid:
                    del self.blk[e]
                if H.has_edge(u, v):
                    region.add(e)
        self.dirty.clear()
        self.stale.clear()
        R = nx.Graph()
        R.add_edges_from(tuple(e) for e in region)
        for comp in nx.biconnected_component_edges(R):
            self._add_block(list(comp))

    def _add_block(self, edges):
        C, H, p = self.C, self.H, self.p
        bid = self._next
        self._next += 1
        b = self.blocks[bid] = _Block(edges)
        for u, v in edges:
            self.blk[frozenset((u, v))] = bid
        if len(edges) < 3 or sum(H[u][v]["w"] for u, v in edges) > p.max_loop_len:
            return
        if geo.too_wide([C[u][v][H[u][v]["k"]]["pts"] for u, v in edges], p.max_loop_width):
            return
        cands = []
        for u, v in edges:
            k = H[u][v]["k"]
            su, sv = superfluous(C, u, v, k, C[u][v][k])
            cands.append((u, v, k, su, sv))
        b.cands = cands
        conns = [c for c in cands if max(c[3], c[4]) < -p.connector_margin]
        b.conn = min(conns, key=lambda c: min(c[3], c[4])) if conns else None
        self.live[bid] = b

    def _best_redundant(self, b):
        if b.h is _UNSET:
            Hb = nx.Graph()
            for u, v in b.edges:
                Hb.add_edge(u, v, **self.H[u][v])
            best = None
            for u, v, k, _, _ in b.cands:
                h = _hausdorff_to_alternative(self.C, Hb, u, v, k)
                if h is not None and (best is None or h < best[0]):
                    best = (h, u, v, k)
            b.h = best
        return b.h

    def cut(self):
        """v0's cut: the best connector anywhere, else the best Hausdorff candidate anywhere."""
        C = self.C
        self._refresh()
        if not self.live:
            return None
        conns = [b.conn for b in self.live.values() if b.conn is not None]
        if conns:
            u, v, k, su, sv = min(conns, key=lambda c: min(c[3], c[4]))
            d = C[u][v][k]
            C.remove_edge(u, v, k)
            self._touch(u, v)
            return (f"connector   {fmt(u, self.origin)}-{fmt(v, self.origin)} "
                    f"lanes={sorted(d['src'])} superfluous={su:.2f},{sv:.2f}")
        best = None
        for b in self.live.values():
            h = self._best_redundant(b)
            if h is not None and (best is None or h[0] < best[0]):
                best = h
        if best is None:
            return None
        h, u, v, k = best
        d = C[u][v][k]
        C.remove_edge(u, v, k)
        self._touch(u, v)
        return (f"redundant   {fmt(u, self.origin)}-{fmt(v, self.origin)} "
                f"lanes={sorted(d['src'])} hausdorff_to_alt={h:.1f}")


@register("v2")
def run(C, origin, params, on_step=None):
    net = _Net(C, origin, params)
    log = []
    while True:
        net.dissolve_all()
        before = snapshot(C) if on_step else None
        msg = net.self_loop() or net.parallel() or net.cut()
        if msg is None:
            return log
        log.append(msg)
        if on_step:
            on_step(msg, before, snapshot(C), n_loops(C))
