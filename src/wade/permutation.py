"""The permutation null.

Inference is by exchangeability of the labels: draw a uniform permutation
of the condition vector, recompute the two axes for every gene, repeat.

Two properties of the construction are load-bearing
(``docs/algorithm.md`` section 4.1):

* **Both axes are tested on the same shuffles.** ``diff_mean`` and
  ``tail_mean`` each get their own null distribution and their own
  p-value. Neither gates the other and they are not combined.
* **One shuffle serves all genes.** Within permutation ``b`` the same
  label vector is applied to every gene, so the null preserves the
  gene-gene correlation structure of the data. An implementation drawing
  an independent shuffle per gene would compute a different — and, across
  correlated genes, anti-conservative — null.

This module is deliberately the slow, readable version. It is the
correctness baseline the Rust kernel is validated against, and the
ordering constraint in ``ROADMAP.md`` exists so that a disagreement with
the R has one candidate cause rather than two.
"""

from __future__ import annotations

import numpy as np

from .quantiles import probability_grid, tail_window_size, type7_quantiles
from .stats import split_groups

try:                                    # pragma: no cover - build-dependent
    from . import _kernel as _rust
except ImportError:                     # pragma: no cover
    _rust = None

__all__ = ["draw_perms", "validate_perms", "null_statistics", "available_backends",
           "HAVE_RUST_KERNEL"]

#: Whether the compiled kernel was built and imported. The package is fully
#: functional without it — the NumPy path is the correctness baseline and the
#: kernel is validated against it — just slower.
HAVE_RUST_KERNEL = _rust is not None


def available_backends() -> tuple[str, ...]:
    return ("numpy", "rust") if HAVE_RUST_KERNEL else ("numpy",)


def draw_perms(
    cond: np.ndarray,
    n_perms: int,
    seed: int | None = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``n_perms`` label permutations. Returns ``(n_perms, n_samples)``.

    Each row is a permutation of ``cond``'s entries, so the group sizes are
    preserved.

    Labels rather than index vectors, deliberately: an index permutation
    forces a 0-based/1-based convention into the fixture format, and a
    label matrix has no such ambiguity. It is also exactly what R's
    ``sample(cond)`` returns.
    """
    cond = np.asarray(cond)
    if rng is None:
        rng = np.random.default_rng(seed)
    out = np.empty((n_perms, cond.shape[0]), dtype=cond.dtype)
    for b in range(n_perms):
        out[b] = rng.permutation(cond)
    return out


def validate_perms(perms: np.ndarray, cond: np.ndarray, n_perms: int) -> np.ndarray:
    """Check a supplied permutation matrix is what it claims to be."""
    perms = np.asarray(perms)
    cond = np.asarray(cond)
    if perms.ndim != 2:
        raise ValueError(
            f"perms must be 2-D, shape (n_perms, n_samples); got {perms.shape}"
        )
    if perms.shape != (n_perms, cond.shape[0]):
        raise ValueError(
            f"perms must have shape (n_perms, n_samples) = "
            f"({n_perms}, {cond.shape[0]}); got {perms.shape}. Note the row axis is "
            f"the permutation axis."
        )
    want = np.sort(cond)
    got = np.sort(perms, axis=1)
    if not np.array_equal(got, np.broadcast_to(want, got.shape)):
        bad = int(np.flatnonzero(~np.all(got == want[None, :], axis=1))[0])
        raise ValueError(
            f"every row of perms must be a permutation of cond (group sizes are "
            f"preserved under label exchange); row {bad} is not."
        )
    return perms


def null_statistics(
    x: np.ndarray,
    perms: np.ndarray,
    tail_q: float,
    log2_scale: bool = False,
    weight: float = 1.0,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Null matrices for both axes. Returns ``(diff_mean, tail_mean)``, each ``(g, B)``.

    Only ``diff_mean`` and ``tail_mean`` are needed for the p-values, so
    this computes the two quantile grids, their difference, and those two
    reductions — not ``w1``, ``tail_conc``, ``fc`` or the auxiliary means.
    The grid, ``nprobs`` and ``k`` are fixed by the group sizes, which
    permutation preserves, so they are computed once and reused. That is a
    performance specialisation with no effect on the numbers: the
    expressions are the same expressions the observed path uses, computed
    in the same order, so observed and null statistics are commensurable
    in the strict sense.

    Parameters
    ----------
    backend
        ``"auto"`` uses the compiled kernel when it is available and the
        NumPy loop otherwise; ``"numpy"`` and ``"rust"`` force one. The
        two are held to elementwise agreement by the parity suite, so the
        choice is a performance decision only — and having both selectable
        is what makes that assertion possible.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    perms = np.asarray(perms)
    n_perms = perms.shape[0]
    g = x.shape[0]

    i1_0, i0_0 = split_groups(perms[0])
    nprobs = min(i1_0.size, i0_0.size)
    q = probability_grid(nprobs)
    k = tail_window_size(nprobs, tail_q)

    if backend not in ("auto", "numpy", "rust"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'rust'; got {backend!r}")
    if backend == "rust" and _rust is None:
        raise RuntimeError(
            "the compiled kernel is not available; build it with "
            "`pip install -e .` (needs cargo/rustc), or use backend='numpy'"
        )
    if backend != "numpy" and _rust is not None:
        d, t = _rust.null_statistics(
            x, np.ascontiguousarray(perms, dtype=np.int64), q, k, float(weight),
            bool(log2_scale),
        )
        # The kernel builds permutation-major so each permutation's writes stay
        # contiguous; transposing to the (genes, permutations) contract is a
        # view, so this costs nothing and — importantly at 3.2 GB — copies nothing.
        return d.T, t.T

    diff = np.empty((g, n_perms), dtype=np.float64)
    tail = np.empty((g, n_perms), dtype=np.float64)

    for b in range(n_perms):
        cb = perms[b]
        i1, i0 = split_groups(cb)
        xb = x
        if weight != 1.0:
            # Applied by condition, so the WEIGHTED group changes membership
            # every iteration. That is the reference's behaviour; whether it
            # is intended is open (docs/design-decisions.md O5).
            xb = x * np.where(cb == 0, weight, 1.0)[None, :]
        if log2_scale:
            xb = np.log2(xb + 1.0)
        d = type7_quantiles(xb[:, i1], q) - type7_quantiles(xb[:, i0], q)
        diff[:, b] = d.sum(axis=1) / nprobs
        tail[:, b] = d[:, :k].mean(axis=1)

    return diff, tail
