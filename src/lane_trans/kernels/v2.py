"""v2: v0's rules, evaluated incrementally. Same decisions, same output, much less work.

v0 changes one chain per round and then recomputes the whole graph: every degree-2 node,
every parallel bundle, every biconnected component, every candidate's score. That is O(N)
work per round and O(N) rounds. v2 applies the same rules but only re-examines what a change
touched:

  dissolve / self-loop / parallel  only the nodes and node pairs the last change touched
  loops                            biconnected components kept per block; after a change only
                                   the blocks on a touched edge are rebuilt
  scores                           connector scores are computed once per block; a block's
                                   best Hausdorff candidate is computed lazily, once

Every round still runs dissolve -> self-loop -> parallel -> cut in that order, and every
choice is made by the same shared functions (`rules.merge_bundle`, `chains.pick_connector`,
`chains.pick_redundant`) over the same candidate set, in canonical order. Ties are broken by
`chains.chain_key`, which depends on the geometry only, never on edge insertion order, so the
two kernels pick the same chain in the same round and the outputs are identical.

The one construct v2 does not keep incrementally is an undeletable self-loop chain (too long
or too wide to remove by rule 1). v0 counts such a chain into the length of one neighbouring
block; v2 hands that round to v0's own `cut`, so the accounting stays identical.

Registered as "v2".
"""
from collections import deque

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from .. import geometry as geo
from ..chains import (chain_key, fmt, n_loops, pick_connector, pick_redundant,
                      snapshot, small_loops)
from ..pipeline import dissolve as _dissolve_unused  # noqa: F401  (keeps the import graph honest)
from ..rules import merge_bundle
from .registry import register
from .v0 import cut as _v0_cut
from .v0 import superfluous


class _Net:
    """The chain graph C and the bookkeeping v2 keeps in step with it.

    Every edit of C goes through this class, so the caches are updated by construction.
    """

    def __init__(self, C, origin, params):
        self.C, self.origin, self.p = C, origin, params
        self.H = nx.Graph()          # simple graph of C without self-loops: k = chain key, w = length
        self.blk = {}                # frozenset({u, v}) -> id of the block that holds that edge
        self.blocks = {}             # id -> (edges, cands, connector) for every block
        self.live = {}               # id -> (edges, cands, connector) for the blocks that are small loops
        self.stale = set()           # block ids that lost an edge
        self.dirty = set(C.nodes)    # nodes whose chains changed since the blocks were last built
        self.check = set(C.nodes)    # nodes that may now be degree 2
        self.pend = set()            # frozenset({u, v}): node pairs that may now be a bundle
        self.selfnodes = set()       # nodes with an undeleted self-loop chain
        self.hausdorff = {}          # block id -> (hausdorff, u, v, k) or None; dropped on touch
        self.full = False            # the block structure must be rebuilt from scratch
        self._next = 0
        for u, v, k, d in C.edges(keys=True, data=True):
            if u == v:
                self.selfnodes.add(u)
            else:
                self.H.add_edge(u, v, k=k, w=geo.length(d["pts"]))
        for u, v in self.H.edges:
            if C.number_of_edges(u, v) > 1:
                self.pend.add(frozenset((u, v)))

    # ---------------------------------------------------------------- bookkeeping
    def _touch(self, a, b):
        """Record that the chains between a and b changed; call after every edit of C."""
        self.dirty.update((a, b))
        self.check.update((a, b))
        self.hausdorff.clear()
        if a == b:
            (self.selfnodes.add if self.C.has_edge(a, a) else self.selfnodes.discard)(a)
            return
        self.pend.add(frozenset((a, b)))
        H, C = self.H, self.C
        if C.has_edge(a, b):
            k = max(C[a][b])         # v0's small_loops builds H in sorted key order; same winner
            H.add_edge(a, b, k=k, w=geo.length(C[a][b][k]["pts"]))
        elif H.has_edge(a, b):
            H.remove_edge(a, b)
            bid = self.blk.pop(frozenset((a, b)), None)
            if bid is not None:
                self.stale.add(bid)
            for n in (a, b):
                if H.degree(n) == 0:
                    H.remove_node(n)

    def _reset(self):
        """Rebuild H and drop all block state; the next _refresh starts from scratch."""
        C = self.C
        self.H = nx.Graph()
        for u, v, k, d in sorted(C.edges(keys=True, data=True), key=lambda e: (e[0], e[1], e[2])):
            if u != v:
                self.H.add_edge(u, v, k=k, w=geo.length(d["pts"]))
        self.blk.clear()
        self.blocks.clear()
        self.live.clear()
        self.stale.clear()
        self.hausdorff.clear()
        self.pend = {frozenset((u, v)) for u, v in self.H.edges if C.number_of_edges(u, v) > 1}
        self.dirty = set(C.nodes)
        self.check = set(C.nodes)
        self.full = False

    # ---------------------------------------------------------------- rules 0-2
    def dissolve_all(self):
        """Concatenate the two chains at every degree-2 node (chains.dissolve, incrementally)."""
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
        """rules.remove_self_loop: the first qualifying self-loop in canonical order."""
        C, p = self.C, self.p
        hits = []
        for u in sorted(self.selfnodes):
            if u not in C or C.degree(u) <= 2:
                continue
            for k, d in C[u][u].items():
                L = geo.length(d["pts"])
                if L <= p.max_loop_len and not geo.too_wide([d["pts"]], p.max_loop_width):
                    hits.append((chain_key(u, u, d["pts"]), u, k, d, L))
        if not hits:
            return None
        _, u, k, d, L = min(hits)
        C.remove_edge(u, u, k)
        self._touch(u, u)
        return f"self-loop   at {fmt(u, self.origin)} lanes={sorted(d['src'])} L={L:.1f}"

    def parallel(self):
        """rules.merge_parallel: the first mergeable bundle in canonical node-pair order.

        v0 walks the bundles in that same canonical order, so the same bundle is merged first
        and the two kernels stay in step even when a merge changes a neighbour's chains.
        """
        C = self.C
        while True:
            for pair in sorted(self.pend, key=lambda g: tuple(sorted(g))):
                a, b = tuple(sorted(pair))
                if not C.has_edge(a, b) or C.number_of_edges(a, b) < 2:
                    self.pend.discard(pair)
                    continue
                es = [(a, b, k, d) for k, d in C[a][b].items()]
                msg = merge_bundle(C, es, self.origin, self.p)
                if msg is None:
                    continue
                self.pend.discard(pair)
                self._touch(a, b)
                return msg
            return None

    # ---------------------------------------------------------------- loops
    def _refresh(self):
        """Rebuild the blocks a change touched; every other block is exactly as it was.

        A change can only alter the cycle structure inside the blocks that held the edges it
        removed, and the scores of the blocks that have a chain at a touched node.
        """
        if self.full:
            self._reset()            # rebuild everything, then continue as usual
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
            self.hausdorff.pop(bid, None)
            if b is None:
                continue
            for u, v in b[0]:
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
        for u, v in edges:
            self.blk[frozenset((u, v))] = bid
        if len(edges) < 3 or sum(H[u][v]["w"] for u, v in edges) > p.max_loop_len:
            self.blocks[bid] = (edges, (), None)
            return
        if geo.too_wide([C[u][v][H[u][v]["k"]]["pts"] for u, v in edges], p.max_loop_width):
            self.blocks[bid] = (edges, (), None)
            return
        cands = []
        for u, v in edges:
            k = H[u][v]["k"]
            su, sv = superfluous(C, u, v, k, C[u][v][k])
            cands.append((u, v, k, su, sv))
        entry = (edges, tuple(cands), pick_connector(C, cands, p.connector_margin))
        self.blocks[bid] = entry
        self.live[bid] = entry

    def _best_hausdorff(self, bid):
        if bid in self.hausdorff:
            return self.hausdorff[bid]
        entry = self.live[bid]
        Hb = nx.Graph()
        for u, v in entry[0]:
            Hb.add_edge(u, v, **self.H[u][v])
        best = pick_redundant(self.C, Hb, entry[1])
        # carry the canonical key so blocks are ranked exactly as v0 ranks candidates
        ranked = None if best is None else (best[0], chain_key(best[1], best[2],
                                                               self.C[best[1]][best[2]][best[3]]["pts"]),
                                            best[1], best[2], best[3])
        self.hausdorff[bid] = ranked
        return ranked

    def cut(self):
        """v0's cut: the best connector anywhere, else the best Hausdorff candidate anywhere."""
        C = self.C
        if self.selfnodes:
            # an undeletable self-loop chain counts into a neighbouring block's length in v0;
            # let v0 decide this round, then start the incremental state over
            msg = _v0_cut(C, self.origin, self.p)
            self.full = True
            return msg
        self._refresh()
        if not self.live:
            return None
        conns = [e[2] for e in self.live.values() if e[2] is not None]
        if conns:
            u, v, k, su, sv = pick_connector(C, conns, self.p.connector_margin)
            d = C[u][v][k]
            C.remove_edge(u, v, k)
            self._touch(u, v)
            if v < u:
                u, v, su, sv = v, u, sv, su
            return (f"connector   {fmt(u, self.origin)}-{fmt(v, self.origin)} "
                    f"lanes={sorted(d['src'])} superfluous={su:.2f},{sv:.2f}")
        best = None
        for bid in self.live:
            h = self._best_hausdorff(bid)
            if h is not None and (best is None or h < best):
                best = h
        if best is None:
            return None
        h, _, u, v, k = best
        d = C[u][v][k]
        C.remove_edge(u, v, k)
        self._touch(u, v)
        u, v = sorted((u, v))
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
