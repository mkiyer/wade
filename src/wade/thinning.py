"""The count-native shift correction for the subset stage — ``docs/method.md`` §10.3–10.4.

Stage 2 tests "is a global fold change an adequate explanation?" by first
making the two groups look identical *under that hypothesis*, then shuffling
labels. For continuous data that was a division: divide the case values by the
fitted fold change and the groups are exchangeable if the shift is the whole
story. For counts at low expression a division does not do that — a zero does
not divide, and a fold change in a negative-binomial mean is not a
multiplicative shift of the negative-binomial distribution — and the result,
measured, was a test that called genuine 2× shifts "not a global shift" most of
the time below a few counts and most of the time at 20 counts with large
cohorts.

The count-native operation is **binomial thinning**: keep each read of the
higher group with probability ``1/f``. That is what a sample would have looked
like at ``1/f`` of its sequencing depth, it maps NB(``f·μ``, ``φ``) to
NB(``μ``, ``φ``) exactly (Poisson–Gamma is closed under thinning), zeros stay
zeros and integers stay integers. This module owns three pieces:

* :func:`fit_fold_change` — ``f`` as *the thinning factor that makes the two
  groups' interquartile means agree*. It is unbiased under an NB fold change
  by construction (it asks "after thinning, do the middles agree?", assuming
  nothing about how the middle scales) and out of reach of any subset below
  25%. Two simpler estimators were measured and rejected: the interquartile
  mean ratio is biased for skewed counts (2.08 for a true 2), and the mean
  ratio is moved to 1.35 by a 5% subset at 8×. Accuracy matters because an
  inaccurate ``f`` on a true global shift is *anti*-conservative (30% off →
  0.26–0.28 false subsets at 200 v 200).
* :func:`thin_counts` — the thinning itself, drawn **once**, before any
  permutation, like the jitter; inference is conditional on the draw.
* :func:`one_count` — the pseudocount for the log-ratio curve: one count,
  expressed in each sample's normalized units. A count of zero means "less
  than one"; without this the tie-breaking jitter turns it into a log-ratio
  of 7–10 against any nonzero count, and one such node can halve
  ``affected_fraction`` at any sample size.

Everything here works on the **raw count matrix**, which is why WADE takes
one. ``thin=False`` falls back to the division, which is kept only so
``tests/test_scale.py`` can demonstrate why thinning replaced it.
"""

from __future__ import annotations

import numpy as np

from .stats import split_groups

__all__ = ["midmean", "fit_fold_change", "thin_counts", "one_count"]


def gene_chunks(g: int, gene_chunk: int | None) -> list[slice]:
    """Row slices of at most ``gene_chunk + 1`` genes; one slice when ``None``.

    The chunked paths are **bit-identical** to the unchunked ones — chunking
    is a memory layout, never a numerical choice (``docs/scaling.md`` §2.2) —
    so this helper is shared to keep every consumer slicing the same way.

    **No chunk is ever a single row.** NumPy's row reductions take a
    different code path for a ``(1, m)`` array than for the same row inside a
    larger matrix, and the two can differ in the last ulp — measured on
    ``D.sum(axis=1)`` at m = 40. Reductions over blocks of two or more rows
    are layout-independent (verified 2026-08-20, ``tests/test_chunking.py``),
    so ``gene_chunk`` must be at least 2 and a trailing one-row remainder is
    folded into the last chunk.
    """
    if gene_chunk is None or g <= 1:
        return [slice(0, g)]
    gene_chunk = int(gene_chunk)
    if gene_chunk < 2:
        raise ValueError(
            f"gene_chunk must be at least 2, got {gene_chunk}: a single-row chunk "
            f"changes NumPy's reduction path and the last bits of every sum"
        )
    chunks = [slice(s, min(s + gene_chunk, g)) for s in range(0, g, gene_chunk)]
    if len(chunks) > 1 and chunks[-1].stop - chunks[-1].start == 1:
        chunks[-2:] = [slice(chunks[-2].start, g)]
    return chunks

#: Bisection range for the fitted fold change, in natural log. 1024× covers
#: anything a global shift plausibly is; a gene absent from one group pins at
#: the bound and is a stage-1 finding, not a stage-2 one.
MAX_LOG_FOLD = float(np.log(1024.0))


def midmean(x: np.ndarray) -> np.ndarray:
    """Row-wise mean of the middle half of the values — the interquartile mean.

    Robust to anything living in the top or bottom quarter, which is where a
    subset of fewer than 25% of the samples lives, and defined where a median
    is not: a gene that is zero in most samples still has a middle half with a
    mean.

    Computed by **selection**, not a full sort — ``np.partition`` at the two
    quartile bounds picks out exactly the middle set — and the block is copied
    contiguous before reducing. That last part is load-bearing rather than
    tidy: the fold-change fit reduces per-side row *subsets* whose sizes depend
    on the chunk boundaries, and reductions over fresh C-contiguous rows are
    independent of how many rows share the array where F-ordered ones are not
    (``docs/scaling.md`` §2.2).
    """
    x = np.ascontiguousarray(x)
    n = x.shape[1]
    lo, hi = n // 4, n - n // 4
    part = np.partition(x, (lo, max(hi - 1, lo)), axis=1)
    return np.ascontiguousarray(part[:, lo:hi]).mean(axis=1)


def thin_counts(
    counts: np.ndarray,
    cond: np.ndarray,
    fold: np.ndarray,
    rng: np.random.Generator,
    *,
    gene_chunk: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Binomial-thin the higher group of every gene by its fitted fold change.

    ``fold[g] >= 1`` thins the **case** columns of gene ``g`` with keep
    probability ``1/fold[g]``; ``fold[g] < 1`` thins the **control** columns
    with keep probability ``fold[g]``. Nothing is ever scaled up, and a
    ``fold`` of exactly 1 leaves the gene untouched. Returns a new
    integer-valued float matrix of the same shape.

    ``gene_chunk`` bounds the transient working set (``docs/scaling.md``
    §2.2) and is **bit-identical** to the unchunked call: the case blocks of
    every chunk are drawn first and the control blocks after, so the one
    generator's stream is consumed in exactly the order the single
    full-matrix call consumes it. ``out`` is an optional preallocated
    ``(genes, samples)`` output — any dtype that holds the counts exactly
    (the chunked driver passes int32 to halve the resident matrix); the
    values are identical whatever the dtype.
    """
    counts = np.asarray(counts, dtype=np.float64)
    cond = np.asarray(cond)
    fold = np.asarray(fold, dtype=np.float64)
    if fold.shape != (counts.shape[0],):
        raise ValueError(f"fold must have one value per gene; got {fold.shape} for {counts.shape[0]} genes")
    if not np.all(np.isfinite(fold)) or np.any(fold <= 0):
        raise ValueError("fold must be finite and positive")
    i1, i0 = split_groups(cond)
    keep_case = np.where(fold >= 1.0, 1.0 / fold, 1.0)
    keep_ctrl = np.where(fold < 1.0, fold, 1.0)
    chunks = gene_chunks(counts.shape[0], gene_chunk)
    if out is None:
        out = np.empty(counts.shape, dtype=np.float64)
    elif out.shape != counts.shape:
        raise ValueError(f"out must have the counts' shape {counts.shape}, got {out.shape}")
    # Two phases, not one loop: the unchunked call draws every gene's case
    # block and then every gene's control block from one stream, and the
    # chunked call must consume that stream in the same order to stay
    # bit-identical.
    for ch in chunks:
        c1 = counts[ch][:, i1]
        if np.any(c1 != np.round(c1)):
            raise ValueError("thinning needs integer counts; pass raw counts, not a normalized matrix")
        out[ch][:, i1] = rng.binomial(c1.astype(np.int64), keep_case[ch][:, None])
    for ch in chunks:
        c0 = counts[ch][:, i0]
        if np.any(c0 != np.round(c0)):
            raise ValueError("thinning needs integer counts; pass raw counts, not a normalized matrix")
        out[ch][:, i0] = rng.binomial(c0.astype(np.int64), keep_ctrl[ch][:, None])
    return out


def _fit_fold_change_alpha(counts, cond, alpha, *, seed, iters, max_log_fold,
                           gene_chunk, backend="numpy"):
    """The fit on a per-cell affine scale — ``docs/scaling.md`` §3.3, fix 2.

    ``wade()``'s fold-change fit normalizes **without** the jitter, and
    jitter-free ``tpm_like`` is exactly ``x[g, j] = counts[g, j] * alpha[g, j]``
    with ``alpha = norm_factor / (normalizer * lib)`` fixed. So nothing needs
    re-normalizing inside the bisection: the untouched group's interquartile
    mean is computed once, and each step only redraws the thinned group and
    multiplies by its alpha — half the binomial draws and no full-matrix
    normalization, with the interquartile means read by selection rather than
    a full sort.

    Chunk-invariant like the general path: the two per-step streams are
    consumed in gene order across chunks, each restricted to the genes whose
    side it thins (a deterministic set, fixed before the loop).

    ``backend="rust"`` runs the bisection in the compiled kernel — parallel
    over genes, each gene's stream a function of ``(seed, global gene
    index)`` alone, so equally chunk-invariant. **Not bitwise against the
    NumPy path** (no two binomial samplers consume randomness alike), which
    is why it is opt-in and never dispatched automatically: the realized
    ``f`` for a given seed depends on the backend, and a backend must never
    change an answer silently.
    """
    counts = np.asarray(counts, dtype=np.float64)
    i1, i0 = split_groups(np.asarray(cond))
    g, n = counts.shape
    chunks = gene_chunks(g, gene_chunk)
    if backend not in ("numpy", "rust"):
        raise ValueError(f"fit backend must be 'numpy' or 'rust'; got {backend!r}")

    up = np.empty(g, dtype=bool)
    nothing = np.empty(g, dtype=bool)
    m_static = np.empty(g)                         # the untouched group's midmean
    for ch in chunks:
        cc = counts[ch]
        if np.any(cc != np.round(cc)):
            raise ValueError("fit_fold_change needs integer counts; pass raw counts, not a normalized matrix")
        a = alpha(ch)
        # A zero library size makes its column's alpha infinite, so a zero
        # count there scales to NaN — anticipated, and discarded with the top
        # quarter by the selection (np.partition orders NaN last).
        with np.errstate(invalid="ignore"):
            m1 = midmean(cc[:, i1] * a[:, i1])
            m0 = midmean(cc[:, i0] * a[:, i0])
        u = m1 >= m0
        up[ch] = u
        nothing[ch] = (m1 == 0) & (m0 == 0)
        m_static[ch] = np.where(u, m0, m1)

    if backend == "rust":
        from .permutation import _rust

        if _rust is None:
            raise RuntimeError(
                "the compiled kernel is not available; build it with "
                "`pip install -e .` (needs cargo/rustc), or use fit_backend='numpy'"
            )
        case_mask = np.zeros(n, dtype=bool)
        case_mask[i1] = True
        ctrl_mask = np.zeros(n, dtype=bool)
        ctrl_mask[i0] = True
        half = np.empty(g)
        for ch in chunks:
            active = np.where(up[ch][:, None], case_mask[None, :], ctrl_mask[None, :])
            half[ch] = _rust.fit_bisect(
                np.ascontiguousarray(counts[ch]), np.ascontiguousarray(alpha(ch)),
                active, m_static[ch], int(seed), int(ch.start), int(iters),
                float(max_log_fold))
        f = np.exp(half)
        f = np.where(nothing, 1.0, f)
        return np.where(up, f, 1.0 / f)

    lo = np.zeros(g)
    hi = np.full(g, float(max_log_fold))
    d = np.empty(g)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep = np.exp(-mid)
        r_case = np.random.default_rng(seed)
        r_ctrl = np.random.default_rng(seed)
        for ch in chunks:
            cc = counts[ch]
            a = alpha(ch)
            u = up[ch]
            rows_up = np.flatnonzero(u)
            rows_dn = np.flatnonzero(~u)
            m_thin = np.empty(cc.shape[0])
            with np.errstate(invalid="ignore"):        # 0 * inf on a dead sample
                if rows_up.size:
                    sub = np.ix_(rows_up, i1)
                    t = r_case.binomial(cc[sub].astype(np.int64),
                                        keep[ch][rows_up][:, None])
                    m_thin[rows_up] = midmean(t * a[sub])
                if rows_dn.size:
                    sub = np.ix_(rows_dn, i0)
                    t = r_ctrl.binomial(cc[sub].astype(np.int64),
                                        keep[ch][rows_dn][:, None])
                    m_thin[rows_dn] = midmean(t * a[sub])
            # d is m1 - m0 after thinning, exactly as the general path forms it.
            d[ch] = np.where(u, m_thin - m_static[ch], m_static[ch] - m_thin)
        too_little = np.where(up, d > 0, d < 0)
        lo = np.where(too_little, mid, lo)
        hi = np.where(too_little, hi, mid)
    f = np.exp(0.5 * (lo + hi))
    f = np.where(nothing, 1.0, f)
    return np.where(up, f, 1.0 / f)


def fit_fold_change(
    counts: np.ndarray,
    cond: np.ndarray,
    *,
    alpha,
    seed: int = 0,
    iters: int = 16,
    max_log_fold: float = MAX_LOG_FOLD,
    gene_chunk: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Per gene, the thinning factor at which the two groups' interquartile
    means agree — ``docs/method.md`` §10.3.

    Bisection on ``log f`` over ``[0, max_log_fold]``, thinning the higher
    group inside the loop. The same ``seed`` is used at every evaluation —
    common random numbers — so the objective is monotone in ``f`` up to the
    thinning algorithm's own discreteness, which sixteen halvings of a 10-bit
    range absorb.

    ``alpha`` is a callable from a row slice to that block's
    ``(rows, samples)`` **per-cell scale**, so the normalized matrix is
    ``counts * alpha``. Jitter-free ``tpm_like`` is exactly that —
    ``alpha = norm_factor / (normalizer * lib)`` — which is what lets the fit
    re-normalize nothing: the untouched group's interquartile mean is computed
    once and each step redraws only the group being thinned. The fit works on
    the **jitter-free** scale deliberately: a hundredth of a count has no
    business in a fold-change estimate, and on an all-zero gene it would decide
    which group is "higher" and keep deciding it at every step.

    ``gene_chunk`` bounds the working set and is **bit-identical** to the
    unchunked fit: the two per-step streams are consumed in gene order across
    chunks, each restricted to the genes whose side it thins. ``backend =
    "rust"`` runs the bisection in the kernel — deterministic and equally
    chunk-invariant, but **not bitwise against NumPy** (no two binomial
    samplers consume randomness alike), which is why it is opt-in and never
    auto-dispatched.

    Returns ``f`` with the convention of :func:`thin_counts`: ``f >= 1`` means
    cases are higher and get thinned by ``1/f``; ``f < 1`` means controls are
    higher and get thinned by ``f``. A gene whose two middles are both zero
    returns ``1``: nothing to correct, and nothing to say.
    """
    return _fit_fold_change_alpha(counts, cond, alpha, seed=seed, iters=iters,
                                  max_log_fold=max_log_fold,
                                  gene_chunk=gene_chunk, backend=backend)


def one_count(
    normalizer,
    lib_sizes: np.ndarray,
    shape: tuple[int, int],
    *,
    norm_factor: float,
) -> np.ndarray:
    """The normalized value of one count, per cell: ``norm_factor / (L_gj * l_j)``.

    This is the pseudocount added before the log-ratio curve is taken, so
    that "+1" means one count in *that sample's* units rather than 1 TPM. A
    sample with a zero library size has no meaningful count scale (every
    value in it is the ``norm_factor`` artefact of ``tpm_like``) and gets 0.
    """
    from .normalize import _broadcast_normalizer

    norm = _broadcast_normalizer(normalizer, shape)
    lib = np.asarray(lib_sizes, dtype=np.float64)
    if lib.shape != (shape[1],):
        raise ValueError(f"lib_sizes must have one value per sample; got {lib.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = norm_factor / (norm * lib[None, :])
    pc = np.broadcast_to(pc, shape).copy()
    pc[:, lib == 0] = 0.0
    return pc
