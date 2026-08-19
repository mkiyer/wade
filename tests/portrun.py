"""Run the port over a fixture's inputs, once per session.

Everything goes through :func:`wade.wade` — the production entry point — with
the fixture's jitter and permutation matrices passed on the ordinary argument
path.

**Parity runs use ``alternative="greater"``.** The R reference is one-sided
upward; WADE now defaults to two-sided. Comparing the port's default against a
one-sided oracle would be comparing two different tests, so the parity suite
pins the orientation R used and ``test_subset.py`` covers the two-sided
behaviour on its own terms.
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
        noise=scalar(p["noise"]),
        norm_factor=scalar(p["norm_factor"]),
        jitter=fx["jitter"],
        perms=fx["perms"] if nperms > 0 else None,
        gene_names=fx["gene_names"],
        n_exc_min=int(p["n_exc_min"]),
        n_tail=int(p["n_tail"]),
        alternative="greater",
        keep_null=True,
        subset=False,
        # Layers 0-7 validate the readable NumPy path against the R. If they
        # silently ran the compiled kernel once it was built, a disagreement
        # would have two candidate causes again. The kernel is held to the
        # same fixtures separately, in test_kernel.py.
        backend="numpy",
    )
    return fx, result
