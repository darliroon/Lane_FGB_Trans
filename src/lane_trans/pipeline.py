"""The loop-removal loop every kernel runs; a kernel only supplies its `cut` rule."""
from .chains import dissolve, n_loops, snapshot
from .rules import merge_parallel, remove_self_loop


def simplify(C, origin, params, cut, on_step=None):
    """Edit chain graph C in place until no small loop is left; return the log lines.

    One change per round, cheapest rule first:
        self-loop chain -> parallel chains -> cut (kernel-specific)
    and degree-2 nodes are dissolved before every round.

    cut(C, origin, params) -> log line, or None when no small loop remains.
    on_step(msg, before, after, loops) is called after every change; before/after are
    snapshot() dicts of the chains and loops the cyclomatic number afterwards.

    Terminates: every change removes a chain or replaces a bundle of them by one.
    """
    log = []
    while True:
        dissolve(C)
        before = snapshot(C) if on_step else None
        msg = (remove_self_loop(C, origin, params) or merge_parallel(C, origin, params)
               or cut(C, origin, params))
        if msg is None:
            return log
        log.append(msg)
        if on_step:
            on_step(msg, before, snapshot(C), n_loops(C))
