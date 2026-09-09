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

The NumPy loops here are deliberately the slow, readable versions. They are
the correctness baselines the Rust kernel is validated against: both the
mean-shift null (:func:`null_statistics`) and the subset test's two passes
(:func:`subset_null_backend`) have a compiled counterpart held to them
elementwise.
"""

from __future__ import annotations

from math import comb, log10

import numpy as np

from .quantiles import capped_nprobs, probability_grid, type7_quantiles
from .stats import split_groups
from .subset import bridge, log_ratio_curve

try:                                    # pragma: no cover - build-dependent
    from . import _kernel as _rust
except ImportError:                     # pragma: no cover
    _rust = None

__all__ = ["draw_perms", "validate_perms", "null_statistics",
           "subset_null_backend", "mean_diff_weights", "mean_diff_null",
           "strata_indices", "permutation_space", "detectability_floor",
           "HAVE_RUST_KERNEL"]

#: Whether the compiled kernel was built and imported. The package is fully
#: functional without it — the NumPy path is the correctness baseline and the
#: kernel is validated against it — just slower.
HAVE_RUST_KERNEL = _rust is not None


def _resolve_backend(backend: str) -> str:
    """``"auto"`` becomes whichever of ``"rust"`` / ``"numpy"`` is available."""
    if backend not in ("auto", "numpy", "rust"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'rust'; got {backend!r}")
    if backend == "rust" and _rust is None:
        raise RuntimeError(
            "the compiled kernel is not available; install a published wheel or "
            "build from source (needs cargo/rustc), or use backend='numpy'"
        )
    return "numpy" if backend == "numpy" or _rust is None else "rust"


def orient(z: np.ndarray, alternative: str) -> np.ndarray:
    """A statistic oriented so that larger is more extreme under
    ``alternative``: unchanged for ``"greater"``, negated for ``"less"``,
    absolute for ``"two-sided"``."""
    if alternative == "greater":
        return z
    if alternative == "less":
        return -z
    if alternative == "two-sided":
        return np.abs(z)
    raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less'; got {alternative!r}")


def strata_indices(strata, n_samples: int) -> list[np.ndarray]:
    """Sample indices grouped by stratum label, in order of first appearance.

    Any hashable labels are accepted (study names, batch ids, integers), and
    the grouping is by equality. The order is deterministic — first
    appearance, not sorted — so a run is reproducible whatever the label type.
    """
    strata = np.asarray(strata)
    if strata.ndim != 1:
        raise ValueError(f"strata must be a 1-D per-sample vector, got shape {strata.shape}")
    if strata.shape[0] != n_samples:
        raise ValueError(
            f"strata must have one entry per sample: expected {n_samples}, "
            f"got {strata.shape[0]}"
        )
    seen: dict = {}
    for j, lab in enumerate(strata.tolist()):
        seen.setdefault(lab, []).append(j)
    return [np.asarray(v, dtype=np.intp) for v in seen.values()]


def permutation_space(cond: np.ndarray, strata=None) -> dict:
    """How many distinct label assignments the design admits, and the p-value
    floor that implies.

    Unrestricted, the count is ``C(n, n1)``. Restricted to strata it is
    ``prod_s C(n_s, k_s)``, and any stratum containing only one class
    contributes a factor of 1 — no freedom at all — so a design stratified
    into many small studies can have a permutation space too small to
    support the p-values it is asked for.

    Returns ``n_strata``, ``log10_space``, ``p_floor`` (the smallest
    attainable p-value, ``1 / space``), and ``uninformative_strata`` (those
    with only one class present).
    """
    cond = np.asarray(cond)
    if strata is None:
        groups = [np.arange(cond.shape[0], dtype=np.intp)]
    else:
        groups = strata_indices(strata, cond.shape[0])
    log10_space = 0.0
    uninformative = 0
    for idx in groups:
        c = cond[idx]
        n_s = int(c.size)
        k_s = int(np.sum(c == 1))
        if k_s == 0 or k_s == n_s:
            uninformative += 1
            continue
        log10_space += log10(comb(n_s, k_s))
    return {
        "n_strata": len(groups),
        "log10_space": log10_space,
        "p_floor": 10.0 ** (-log10_space),
        "uninformative_strata": uninformative,
    }


def detectability_floor(n_case: int, n_ctrl: int, k: int) -> float:
    """The scale of the smallest p-value a label-permutation test can resolve
    for a signal carried by ``k`` samples (``docs/method.md`` §6).

    Shuffling labels puts all ``k`` affected samples in one group with
    probability ``C(n_case, k) / C(n_case + n_ctrl, k)``, whatever the effect
    size, the statistic or the number of permutations. It is a **scale, not a
    bound**: measured on count data the median gene lands within a factor of
    two of it and a substantial minority below it, so use it to judge a
    design and never clamp a gene's p-value to it. Where it exceeds your alpha
    the signal is undetectable by this family of methods.

    >>> round(detectability_floor(77, 18, 15), 6)
    0.031963

    It is driven by *imbalance* rather than size, so balancing the groups buys
    far more than adding cases — the same 95 samples, split evenly:

    >>> round(detectability_floor(48, 47, 15), 8)
    9.9e-06
    """
    n_case, n_ctrl, k = int(n_case), int(n_ctrl), int(k)
    if min(n_case, n_ctrl) < 1:
        raise ValueError("both groups must be non-empty")
    if not 1 <= k <= n_case + n_ctrl:
        raise ValueError(f"k must be between 1 and n_case + n_ctrl; got {k}")
    if k > n_case:
        # The affected samples cannot all land in the case group at all, so
        # that configuration has probability zero and imposes no floor.
        return 0.0
    return comb(n_case, k) / comb(n_case + n_ctrl, k)


def draw_perms(
    cond: np.ndarray,
    n_perms: int,
    seed: int | None = 1,
    rng: np.random.Generator | None = None,
    strata=None,
) -> np.ndarray:
    """Draw ``n_perms`` label permutations. Returns ``(n_perms, n_samples)``.

    Each row is a permutation of ``cond``'s entries (labels, not index
    vectors), so the group sizes are preserved.

    ``strata`` restricts the shuffle: labels are permuted **within** each
    stratum, so every stratum keeps its own case/control counts. This is the
    restricted permutation a batch-structured cohort needs; a stratum with
    only one class present contributes no freedom, and
    :func:`permutation_space` counts what is left.
    """
    cond = np.asarray(cond)
    if rng is None:
        rng = np.random.default_rng(seed)
    out = np.empty((n_perms, cond.shape[0]), dtype=cond.dtype)
    if strata is None:
        for b in range(n_perms):
            out[b] = rng.permutation(cond)
        return out
    groups = strata_indices(strata, cond.shape[0])
    for b in range(n_perms):
        row = out[b]
        for idx in groups:
            row[idx] = rng.permutation(cond[idx])
    return out


def validate_perms(perms: np.ndarray, cond: np.ndarray, n_perms: int,
                   strata=None) -> np.ndarray:
    """Check a supplied permutation matrix is what it claims to be.

    With ``strata``, the check tightens: permuting within strata preserves the
    global label multiset *and* every stratum's own counts, so a row that
    moved a label across strata passes the global test and is still wrong.
    Both are checked.
    """
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
    ok = np.all(np.sort(perms, axis=1) == np.sort(cond)[None, :], axis=1)
    if not ok.all():
        bad = int(np.argmin(ok))
        raise ValueError(
            f"every row of perms must be a permutation of cond (group sizes are "
            f"preserved under label exchange); row {bad} is not."
        )
    if strata is not None:
        for idx in strata_indices(strata, cond.shape[0]):
            want_s = int(np.sum(cond[idx] == 1))
            got_s = np.sum(perms[:, idx] == 1, axis=1)
            if not np.all(got_s == want_s):
                bad = int(np.flatnonzero(got_s != want_s)[0])
                raise ValueError(
                    f"restricted permutation must preserve each stratum's case count: "
                    f"row {bad} has {int(got_s[bad])} cases in a stratum that has "
                    f"{want_s}. Labels were moved across strata."
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
    if perms.ndim != 2 or perms.shape[0] == 0:
        raise ValueError(f"perms must be a non-empty (n_perms, n_samples) matrix; got {perms.shape}")
    n_perms = perms.shape[0]
    g = x.shape[0]

    i1_0, i0_0 = split_groups(perms[0])
    nprobs = capped_nprobs(i1_0.size, i0_0.size, max_probs)
    q = probability_grid(nprobs)

    if _resolve_backend(backend) == "rust":
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


def mean_diff_weights(labels: np.ndarray) -> np.ndarray:
    """``+1/n1`` on cases and ``-1/n0`` on controls, so ``x @ w`` is exactly
    the difference of the two group means. ``labels`` may be one condition
    vector or a ``(n_perms, n_samples)`` permutation matrix."""
    labels = np.asarray(labels)
    i1, i0 = split_groups(labels if labels.ndim == 1 else labels[0])
    return np.where(labels == 1, 1.0 / i1.size, -1.0 / i0.size)


def mean_diff_null(x: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """The mean-difference null as one matrix product, ``(genes, permutations)``.

    The stage-1 null of ``stage1="gemm"`` and ``"saddlepoint"``: with ``W``
    the ``(samples x B)`` signed indicator matrix, the entire null is
    ``x @ W``. BLAS reassociates the sums, so this agrees with the grid
    statistic on a balanced design to ~1e-9 relative, not bitwise, which is
    why it is opt-in beside the kernel and never the default.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    perms = np.asarray(perms)
    if perms.ndim != 2 or perms.shape[0] == 0 or perms.shape[1] != x.shape[1]:
        raise ValueError(
            f"perms must be a non-empty (n_perms, n_samples) matrix with "
            f"{x.shape[1]} samples; got {perms.shape}"
        )
    return x @ np.ascontiguousarray(mean_diff_weights(perms).T)


def _bridge_from(x: np.ndarray, labels: np.ndarray, q: np.ndarray) -> np.ndarray:
    i1, i0 = split_groups(labels)
    return bridge(log_ratio_curve(type7_quantiles(x[:, i1], q),
                                  type7_quantiles(x[:, i0], q)))


def _subset_null_numpy(xs, b_obs, perms, q, alternative):
    """Two passes over the permutations, on the shift-corrected matrix.

    Pass 1 estimates the null moments of the bridge at every width; pass 2
    needs them in order to standardize before maximizing, so it has to
    recompute. Storing every permutation's bridge instead would cost
    ``B * g * m`` doubles, which at realistic sizes is far larger than the
    two-pass recomputation is slow.
    """
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

    null = np.empty((g, n_perms), dtype=np.float64)
    for b in range(n_perms):
        z = orient((_bridge_from(xs, perms[b], q) - mu) / safe, alternative)
        null[:, b] = z.max(axis=1)

    z_obs = orient((b_obs - mu) / safe, alternative)
    return z_obs.max(axis=1), null, mu, sd, z_obs.argmax(axis=1) + 1


def subset_null_backend(xs, b_obs, perms, q, *, alternative="two-sided", backend="auto"):
    """Dispatch the subset test's permutation work. Returns
    ``(statistic, null, mu, sd, argmax_k)``.

    The kernel takes the same arguments and is held to the NumPy path
    elementwise (``tests/test_subset_kernel.py``); it parallelizes over
    genes, so each gene's two passes stay on one thread and accumulate over
    permutations in the same order the loop above does.
    """
    orient(np.zeros(1), alternative)          # validates `alternative`
    perms = np.asarray(perms)
    if perms.ndim != 2 or perms.shape[0] == 0:
        raise ValueError(f"perms must be a non-empty (n_perms, n_samples) matrix; got {perms.shape}")
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    if _resolve_backend(backend) == "rust":

        return _rust.subset_null(
            xs,
            np.ascontiguousarray(b_obs, dtype=np.float64),
            np.ascontiguousarray(perms, dtype=np.int64),
            np.ascontiguousarray(q, dtype=np.float64),
            alternative,
        )
    return _subset_null_numpy(xs, b_obs, perms, q, alternative)
