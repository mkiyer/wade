"""The permutation null.

Inference is by exchangeability of the labels: draw a uniform permutation
of the condition vector, recompute the two axes for every gene, repeat.

Two properties of the construction are load-bearing
(``docs/method.md`` section 6):

* **Both stages are tested on the same shuffles.** The mean-shift test and
  the subset test each get their own null and their own p-value. Neither
  gates the other and they are not combined.
* **One shuffle serves all genes.** Within permutation ``b`` the same
  label vector is applied to every gene, so the null preserves the
  gene-gene correlation structure of the data. An implementation drawing
  an independent shuffle per gene would compute a different — and, across
  correlated genes, anti-conservative — null.

The NumPy loops here are deliberately the slow, readable versions. They
are the correctness baselines the Rust kernel is validated against — both
the mean-shift null (:func:`null_statistics`) and the subset test's two
passes (:func:`_subset_null_numpy`, reached through
:func:`subset_null_backend`) have a compiled counterpart held to them
elementwise. The ordering constraint in ``ROADMAP.md`` exists so that a
disagreement with the R has one candidate cause rather than two.
"""

from __future__ import annotations

import numpy as np

from .quantiles import capped_nprobs, probability_grid, type7_quantiles
from .stats import split_groups

try:                                    # pragma: no cover - build-dependent
    from . import _kernel as _rust
except ImportError:                     # pragma: no cover
    _rust = None

__all__ = ["draw_perms", "validate_perms", "null_statistics", "available_backends",
           "subset_null_backend", "mean_diff_stat", "mean_diff_null",
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
    *,
    backend: str = "auto",
    max_probs: int | None = None,
) -> np.ndarray:
    """The mean-shift null. Returns a ``(genes, permutations)`` matrix.

    Only the signed grid area is needed, so this computes the two quantile
    grids, their difference and that one reduction. The grid and ``nprobs``
    are fixed by the group sizes, which permutation preserves, so they are
    computed once. The expressions are the ones the observed path uses,
    evaluated in the same order, so observed and null values are strictly
    commensurable — which matters because a gene whose observed value ties a
    null draw would otherwise fall on either side of the comparison.

    ``max_probs`` caps the grid exactly as :func:`wade.wade_stats` does, and
    must match what the observed statistic used — the p-value compares the
    two, so they have to share a quadrature.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    perms = np.asarray(perms)
    n_perms = perms.shape[0]
    g = x.shape[0]

    i1_0, i0_0 = split_groups(perms[0])
    nprobs = capped_nprobs(i1_0.size, i0_0.size, max_probs)
    q = probability_grid(nprobs)

    if backend not in ("auto", "numpy", "rust"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'rust'; got {backend!r}")
    if backend == "rust" and _rust is None:
        raise RuntimeError(
            "the compiled kernel is not available; build it with "
            "`pip install -e .` (needs cargo/rustc), or use backend='numpy'"
        )
    if backend != "numpy" and _rust is not None:
        # The kernel is parallel over genes — one sort per gene, then one
        # O(n) partition walk per permutation — and returns gene-major,
        # which is already the (genes, permutations) contract.
        return _rust.null_statistics(
            x, np.ascontiguousarray(perms, dtype=np.int64), q
        )

    out = np.empty((g, n_perms), dtype=np.float64)
    for b in range(n_perms):
        i1, i0 = split_groups(perms[b])
        d = type7_quantiles(x[:, i1], q) - type7_quantiles(x[:, i0], q)
        out[:, b] = d.sum(axis=1) / nprobs
    return out


def _mean_diff_weights(labels: np.ndarray) -> np.ndarray:
    """``+1/n1`` on cases, ``-1/n0`` on controls, so ``x @ w`` is exactly the
    difference of the two group means."""
    labels = np.asarray(labels)
    i1, i0 = split_groups(labels if labels.ndim == 1 else labels[0])
    return np.where(labels == 1, 1.0 / i1.size, -1.0 / i0.size)


def mean_diff_stat(x: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """The difference of group means, per gene, as one matrix-vector product.

    The ``stage1="gemm"`` observed statistic (``docs/scaling.md`` §3.1). On a
    **balanced** design this is exactly what ``mean_shift`` computes through
    the quantile grid (measured 2.1e-13 relative); it is evaluated as
    ``x @ w`` — the same arithmetic class as :func:`mean_diff_null` — so the
    two agree to BLAS rounding. Not bitwise: a matrix–vector product (GEMV)
    and a matrix–matrix product (GEMM) may round the same dot product
    differently, so a permutation that reproduces the observed labeling is
    not guaranteed to tie it exactly. On jittered continuous data exact ties
    have measure zero and the p-value is insensitive to 1e-15-level
    reroundings; the grid path remains the choice where last-ulp
    commensurability matters.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    return x @ _mean_diff_weights(np.asarray(cond))


def mean_diff_null(x: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """The mean-difference null as one GEMM. Returns ``(genes, permutations)``.

    ``docs/scaling.md`` §3.1: with ``W`` the ``(samples x B)`` signed
    indicator matrix, the entire stage-1 null is ``x @ W`` — measured
    139–185× faster than the permutation kernel at large ``n``. BLAS
    reassociates the sums, so this agrees with the grid statistic on a
    balanced design to ~1e-9 relative, **not bitwise** — which is why it is
    an opt-in beside the parity-pinned kernel, never the default, and why a
    chunked GEMM run agrees with an unchunked one to the same tolerance
    rather than exactly.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    perms = np.asarray(perms)
    if perms.ndim != 2 or perms.shape[1] != x.shape[1]:
        raise ValueError(
            f"perms must be (n_perms, n_samples) with {x.shape[1]} samples; got {perms.shape}"
        )
    split_groups(perms[0])
    return x @ np.ascontiguousarray(_mean_diff_weights(perms).T)


def _subset_null_numpy(xs, b_obs, perms, q, alternative):
    """Two passes over the permutations, on the shift-corrected matrix.

    Pass 1 estimates the null moments of the bridge at every width; pass 2
    needs them in order to standardize before maximizing, so it has to
    recompute. Storing every permutation's bridge instead would cost
    ``B * g * m`` doubles, which at realistic sizes is far larger than the
    two-pass recomputation is slow.
    """
    from .subset import _bridge_from

    g, m = b_obs.shape
    n_perms = perms.shape[0]

    s1 = np.zeros((g, m))
    s2 = np.zeros((g, m))
    for b in range(n_perms):
        bb = _bridge_from(xs, perms[b], q)
        s1 += bb
        s2 += bb * bb
    mu = s1 / n_perms
    sd = np.sqrt(np.maximum(s2 / n_perms - mu * mu, 0.0))

    # B_m is identically zero, so its width carries no information and its
    # null spread is zero; excluding it keeps the maximum honest.
    usable = sd > 0
    usable[:, -1] = False
    safe = np.where(usable, sd, np.inf)

    def reduce(z):
        if alternative == "greater":
            pass
        elif alternative == "less":
            z = -z
        else:
            z = np.abs(z)
        return z

    null = np.empty((g, n_perms), dtype=np.float64)
    for b in range(n_perms):
        z = reduce((_bridge_from(xs, perms[b], q) - mu) / safe)
        null[:, b] = z.max(axis=1)

    z_obs = reduce((b_obs - mu) / safe)
    return z_obs.max(axis=1), null, mu, sd, z_obs.argmax(axis=1) + 1


def subset_null_backend(xs, b_obs, perms, q, *, alternative="two-sided", backend="auto"):
    """Dispatch the subset test's permutation work. Returns
    ``(statistic, null, mu, sd, argmax_k)``.

    The kernel takes the same arguments and is held to the NumPy path
    elementwise (``tests/test_subset_kernel.py``); it parallelizes over
    genes, so each gene's two passes stay on one thread and accumulate over
    permutations in the same order the loop above does.
    """
    from .pvalues import ALTERNATIVES

    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    if backend not in ("auto", "numpy", "rust"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'rust'; got {backend!r}")
    if backend == "rust" and _rust is None:
        raise RuntimeError(
            "the compiled kernel is not available; build it with "
            "`pip install -e .` (needs cargo/rustc), or use backend='numpy'"
        )
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    if backend != "numpy" and _rust is not None:
        return _rust.subset_null(
            xs,
            np.ascontiguousarray(b_obs, dtype=np.float64),
            np.ascontiguousarray(perms, dtype=np.int64),
            np.ascontiguousarray(q, dtype=np.float64),
            alternative,
        )
    return _subset_null_numpy(xs, b_obs, perms, q, alternative)
