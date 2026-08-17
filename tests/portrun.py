"""Run the port over a fixture's inputs, once per session.

Everything goes through :func:`wade.wade` — the production entry point —
with the fixture's jitter and permutation matrices passed on the ordinary
argument path. That is the whole point of accepting them there: a fixture
path that bypassed production code would validate a code path users never
run (``docs/porting-hazards.md`` hazard 2).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from conftest import load_fixture

import wade


def scalar(node) -> float:
    """A ``j_dbl`` emission parses to a length-1 array; take the value."""
    return float(np.asarray(node).reshape(-1)[0])


@lru_cache(maxsize=None)
def run_port(name: str):
    """Returns ``(fixture, WadeResult)``."""
    fx = load_fixture(name)
    p = fx["params"]
    nperms = int(p["nperms"])

    result = wade.wade(
        fx["counts"],
        fx["normalizer"]["data"],
        fx["cond"],
        nperms=nperms,
        tail_q=scalar(p["tail_q"]),
        noise=scalar(p["noise"]),
        norm_factor=scalar(p["norm_factor"]),
        jitter=fx["jitter"],
        perms=fx["perms"] if nperms > 0 else None,
        log2_scale=bool(p["log2_scale"]),
        weight=scalar(p["weight"]),
        gene_names=fx["gene_names"],
        n_exc_min=int(p["n_exc_min"]),
        n_tail=int(p["n_tail"]),
        keep_null=True,
        compute_scores=True,
        # Pinned to the NumPy path on purpose. Layers 0-8 exist to validate the
        # readable reference implementation against the R; if they silently ran
        # the compiled kernel once it was built, a disagreement would have two
        # candidate causes again — which is precisely the ordering constraint
        # ROADMAP.md is built around. The kernel is held to the same fixtures
        # separately, in test_kernel.py.
        backend="numpy",
    )
    return fx, result
