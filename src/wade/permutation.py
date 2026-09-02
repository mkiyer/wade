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

__all__ = ["draw_perms", "validate_perms", "null_statistics",
           "subset_null_backend", "mean_diff_stat", "mean_diff_null",
           "strata_indices", "permutation_space", "detectability_floor",
           "HAVE_RUST_KERNEL"]

#: Whether the compiled kernel was built and imported. The package is fully
#: functional without it — the NumPy path is the correctness baseline and the
#: kernel is validated against it — just slower.
HAVE_RUST_KERNEL = _rust is not None


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
    floor that implies — ``docs/limits.md`` §1, restricted to strata.

    Unrestricted, the count is ``C(n, n1)``. **Restricted permutation
    multiplies a product of much smaller numbers instead**: within-stratum
    exchangeability gives ``prod_s C(n_s, k_s)``, and any stratum containing
    only one class contributes a factor of 1 — no freedom at all. A design
    stratified into many small studies can therefore have a permutation space
    too small to support the p-values it is asked for, which is exactly the
    kind of arithmetic this package surfaces rather than hides.

    Returns ``n_strata``, ``log10_space``, ``p_floor`` (the smallest
    attainable p-value, ``1 / space``), and ``uninformative_strata`` (those
    with only one class present).
    """
    from math import comb, log10

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
    for a signal carried by ``k`` samples — ``docs/limits.md`` §1 and §4.1.

    Shuffling labels puts all ``k`` affected samples in one group with
    probability ``C(n_case, k) / C(n_case + n_ctrl, k)``, whatever the effect
    size, the statistic or the number of permutations.

    **It is a scale, not a bound**, and this docstring claimed otherwise until
    it was measured on 2026-09-02: with noise the relabellings that place all
    ``k`` in cases scatter around the observed rather than tying it, and ``k``
    itself is a random variable in count data. 21% of planted 3-sample genes
    and 42% of planted 6-sample genes came in *under* this number at
    ``B = 40,000`` (``limits.md`` §4.1 has the table). Use it to judge a
    design; never clamp a gene's p-value to it. Where this exceeds your alpha the
    signal is undetectable by this family of methods, and that is a property of
    the *design*: it is why a study with 18 controls cannot find a 5% subtype
    however dramatic the subtype is.

    ``limits.md`` tells the reader to check this before running anything, so
    here it is as one call instead of a formula to transcribe:

    >>> round(detectability_floor(77, 18, 15), 6)
    0.031963

    It is driven by *imbalance* rather than size, so balancing the groups buys
    far more than adding cases — the same 95 samples, split evenly:

    >>> round(detectability_floor(48, 47, 15), 8)
    9.9e-06
    """
    from math import comb

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

    Each row is a permutation of ``cond``'s entries, so the group sizes are
    preserved.

    Labels rather than index vectors, deliberately: an index permutation
    forces a 0-based/1-based convention into the fixture format, and a
    label matrix has no such ambiguity. It is also exactly what R's
    ``sample(cond)`` returns.

    ``strata`` restricts the shuffle: labels are permuted **within** each
    stratum, so every stratum keeps its own case/control counts. This is the
    restricted permutation that a batch-structured cohort needs — permuting
    study labels freely across studies tests exchangeability the design does
    not have (``docs/limits.md`` §2.2), and the same machinery is what
    donor-level permutation needs for single cell (``docs/scaling.md`` §5.3).
    A stratum with only one class present contributes no freedom;
    :func:`permutation_space` counts what is left.

    ``strata=None`` is bitwise what it always was — the unrestricted branch
    consumes the generator exactly as before, so existing seeds reproduce.
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
    want = np.sort(cond)
    got = np.sort(perms, axis=1)
    if not np.array_equal(got, np.broadcast_to(want, got.shape)):
        bad = int(np.flatnonzero(~np.all(got == want[None, :], axis=1))[0])
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
