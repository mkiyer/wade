"""Normalizers: raw counts to a comparable scale, with the continuity jitter.

These are separate, individually callable functions rather than part of
the statistic (``docs/design-decisions.md`` S3): a normalizer produces a
matrix, the statistic consumes one, and :func:`wade.wade` composes them.
A new normalizer can be added without touching the test.

Why the entry point takes raw counts (``docs/design-decisions.md`` S2)
-----------------------------------------------------------------------
The continuity jitter is applied **at count precision, before division**.
In :func:`tpm_like` the jitter is added to the counts and a matching
per-cell correction appears in the denominator, so a pre-normalized
matrix cannot reproduce it: once counts have been divided by a normalizer
and a library size, you no longer know what one count was worth in that
cell. A caller who supplies an already-normalized matrix is running a
variant of the test whose ties were never broken and whose zeros are
still exactly equal — see :func:`wade.wade_from_matrix`, which says so.

The jitter is drawn **once**, before any permutation. Inference is then
conditional on that one realised draw, which is what makes a given seed
reproduce a given result exactly (``docs/algorithm.md`` section 4.4). It
must never be re-drawn inside the permutation loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["draw_jitter", "library_sizes", "tpm_like", "cpm", "rle"]

DEFAULT_NOISE = 0.01
DEFAULT_NORM_FACTOR = 1.0e6


def _as_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(
            f"counts must be a 2-D genes x samples array, got shape {counts.shape}"
        )
    if counts.size and np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite")
    return counts


def _broadcast_normalizer(normalizer, shape: tuple[int, int]) -> np.ndarray:
    """Accept the two forms R accepts, but check the shape rather than recycle.

    R divides ``counts / normalizer`` and relies on recycling, which has no
    length check: passing a per-*sample* vector where a per-*gene* vector
    belongs recycles cleanly whenever ``g * n`` is divisible by ``n`` and
    returns wrong numbers with no diagnostic (``docs/r-implementation.md``
    section 1). That is the one input error in the R file that produces
    plausible output, so this raises instead.
    """
    g, n = shape
    normalizer = np.asarray(normalizer, dtype=np.float64)
    if normalizer.ndim == 1:
        if normalizer.shape[0] != g:
            raise ValueError(
                f"a vector normalizer must have one value per gene: expected "
                f"length {g}, got {normalizer.shape[0]}. (A length-{n} vector "
                f"would be per-sample, which is the wrong axis; R recycles it "
                f"silently, this does not.)"
            )
        normalizer = normalizer[:, None]
    elif normalizer.ndim == 2:
        if normalizer.shape != shape:
            raise ValueError(
                f"a matrix normalizer must be genes x samples {shape}, "
                f"got {normalizer.shape}"
            )
    else:
        raise ValueError("normalizer must be a per-gene vector or a genes x samples matrix")
    if np.any(normalizer == 0):
        raise ValueError("normalizer contains zeros; division would be undefined")
    if not np.all(np.isfinite(normalizer)):
        raise ValueError("normalizer must be finite")
    return normalizer


def draw_jitter(
    shape: tuple[int, int],
    noise: float = DEFAULT_NOISE,
    seed: int | None = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw the continuity jitter: ``Uniform(0, noise)``, genes x samples.

    The jitter serves three purposes. It breaks ties in sparse, zero-heavy
    data, where many samples share a count of zero and the quantile grid
    would otherwise degenerate into flat runs. It makes every entry
    strictly positive, so ``fc`` and ``log2(fc)`` stay finite even for a
    gene with all-zero counts in one group. And it gives a defined value
    where the library size is zero.

    ``seed=None`` draws from ambient state without touching any global RNG,
    mirroring R's ``seed = NULL`` mode.

    Note this cannot reproduce R's draw for the same integer seed, and is
    not meant to: R's Mersenne-Twister and NumPy's PCG64 are different
    algorithms (``docs/porting-hazards.md`` hazard 2). Exact cross-language
    agreement requires passing the realised matrix in as ``jitter=``.
    """
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.uniform(0.0, noise, size=shape)


def _resolve_jitter(jitter, shape, noise, seed, rng) -> np.ndarray:
    if jitter is None:
        return draw_jitter(shape, noise=noise, seed=seed, rng=rng)
    jitter = np.asarray(jitter, dtype=np.float64)
    if jitter.shape != shape:
        raise ValueError(
            f"supplied jitter must be genes x samples {shape}, got {jitter.shape}. "
            f"Pass a 2-D array, never a flat vector plus dimensions — the two "
            f"languages fill a flat vector in opposite orders "
            f"(docs/porting-hazards.md hazard 8)."
        )
    if not np.all(np.isfinite(jitter)):
        raise ValueError("supplied jitter must be finite")
    return jitter


def library_sizes(counts: np.ndarray, normalizer) -> np.ndarray:
    """Per-sample size factors: ``sum_g counts[g, j] / normalizer[g, j]``.

    R's ``wade_lib_size()``. This is *not* a read count and is not on any
    interpretable scale — its units depend entirely on what the normalizer
    is.

    Computed from the **unjittered** counts, which is what makes
    :func:`tpm_like`'s per-cell denominator correction necessary.

    Note this depends on which genes are in the matrix: change the gene set
    and every normalized value changes. That is a real reproducibility
    surface, matching R's ``wade_run()``, which sizes libraries on the
    subset matrix (``docs/design-decisions.md`` S2).
    """
    counts = _as_counts(counts)
    norm = _broadcast_normalizer(normalizer, counts.shape)
    return (counts / norm).sum(axis=0)


def tpm_like(
    counts: np.ndarray,
    normalizer,
    lib_sizes: np.ndarray | None = None,
    *,
    noise: float = DEFAULT_NOISE,
    norm_factor: float = DEFAULT_NORM_FACTOR,
    seed: int | None = 1,
    jitter: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Normalizer- and library-scaled TPM-like units. Port of ``wade_normalize()``.

    Takes a **count** measure and a **normalizer** measure per gene; the
    arithmetic is identical and only the pair changes. Splice-junction
    counts over intron count gives "sjTPM"; total counts over effective
    length gives standard TPM.

    .. math::
        X_{gj} = \\kappa \\,
          \\frac{(C_{gj} + \\eta_{gj}) / L_{gj}}
                {\\ell_j + \\eta_{gj} / L_{gj}}

    **The denominator is per-cell, and this is the subtlest arithmetic in
    the method.** The naive reading is that it should be ``lib_sizes[j]``,
    giving ordinary TPM. It is not. ``lib_sizes`` was computed from the
    *unjittered* counts while the numerator uses jittered ones, so the two
    are inconsistent; the consistent library size would add the *whole
    column's* jitter contribution, but what is added is only **this gene's
    own**. The denominator is therefore the library size in the
    counterfactual where gene ``g`` alone received jitter.

    Three consequences, all real and all documented rather than fixed:

    * **Column sums are not exactly** ``norm_factor``. The output is
      TPM-*like*, not TPM. A port that "fixes" this to restore the constant
      has changed every number.
    * Substituting ``lib_sizes[j]`` for the per-cell denominator changes
      values by around one part in 10^6 and leaves within-row rank ordering
      untouched — the dangerous shape of error, since the quantile grid
      reads within-row order. It passes every plausibility check and fails
      an exact parity test.
    * **A gene that is the entire library normalizes to exactly**
      ``norm_factor``, and so does every gene in an all-zero sample, where
      numerator and denominator are equal. That is an artefact of the
      algebra, not a sensible value for an empty library; see
      ``docs/r-implementation.md`` section 2.

    Parameters
    ----------
    jitter
        A genes x samples array used **in place of** the internal draw.
        This is the production argument path, not a test-only branch: the
        parity fixtures and real calls go through the same code, or the
        parity suite would be validating code nobody runs.
    """
    counts = _as_counts(counts)
    norm = _broadcast_normalizer(normalizer, counts.shape)
    if lib_sizes is None:
        lib = (counts / norm).sum(axis=0)
    else:
        lib = np.asarray(lib_sizes, dtype=np.float64)
        if lib.shape != (counts.shape[1],):
            raise ValueError(
                f"lib_sizes must have one value per sample: expected shape "
                f"({counts.shape[1]},), got {lib.shape}"
            )

    nz = _resolve_jitter(jitter, counts.shape, noise, seed, rng)
    y = (counts + nz) / norm
    denom = nz / norm + lib[None, :]
    return norm_factor * y / denom


def cpm(
    counts: np.ndarray,
    *,
    noise: float = DEFAULT_NOISE,
    norm_factor: float = DEFAULT_NORM_FACTOR,
    seed: int | None = 1,
    jitter: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Counts per million.

    New work: there is no reference implementation of this in the R, so it
    has nothing in ``reference/`` to be tested against and is validated on
    its own terms.

    The jitter is applied at count precision, as in :func:`tpm_like` —
    that part is the load-bearing one, since it is what breaks ties before
    any division. But the library size here is the **consistent** one,
    computed from the jittered counts, so columns sum to exactly
    ``norm_factor``. That deliberately differs from :func:`tpm_like`, whose
    per-cell denominator is a quirk of the original reproduced for parity
    rather than a design to propagate.
    """
    counts = _as_counts(counts)
    nz = _resolve_jitter(jitter, counts.shape, noise, seed, rng)
    jittered = counts + nz
    lib = jittered.sum(axis=0)
    if np.any(lib == 0):
        raise ValueError("a sample has zero total count; CPM is undefined for it")
    return norm_factor * jittered / lib[None, :]


def rle(
    counts: np.ndarray,
    *,
    noise: float = DEFAULT_NOISE,
    seed: int | None = 1,
    jitter: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Relative log expression (median-of-ratios) size-factor normalization.

    New work, as :func:`cpm` is.

    The usual formulation restricts the geometric-mean reference to genes
    detected in every sample, because the reference is undefined (zero or
    ``-inf`` in logs) for a gene with any zero. **The continuity jitter
    interacts with that restriction and the interaction is not inherited
    from anywhere**, so it is stated rather than defaulted
    (``docs/design-decisions.md`` S3):

    Jitter is added at count precision first, which makes every entry
    strictly positive, so the reference would technically be defined for
    every gene. That is not used. A gene whose "detection" is a jitter
    draw of order 0.01 carries no information, and admitting it would let
    the noise term set the size factors. The reference is therefore taken
    over genes whose **raw** counts are strictly positive in every sample,
    exactly as it would be without jitter; the jitter then only perturbs
    the ratios of those genes.

    On sparse data that set can be small or empty, which is a real
    limitation of RLE on this kind of input rather than a bug here, and it
    raises rather than silently falling back.
    """
    counts = _as_counts(counts)
    nz = _resolve_jitter(jitter, counts.shape, noise, seed, rng)
    jittered = counts + nz

    if reference is None:
        detected = np.all(counts > 0, axis=1)
        n_detected = int(detected.sum())
        if n_detected == 0:
            raise ValueError(
                "RLE needs at least one gene with a strictly positive raw count in "
                "every sample to build the geometric-mean reference, and this matrix "
                "has none. On zero-heavy data use tpm_like() or cpm(), or supply an "
                "explicit `reference`."
            )
        log_ref = np.log(jittered[detected]).mean(axis=1)
        ratios = np.log(jittered[detected]) - log_ref[:, None]
    else:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (counts.shape[0],):
            raise ValueError(
                f"reference must have one value per gene: expected shape "
                f"({counts.shape[0]},), got {reference.shape}"
            )
        usable = reference > 0
        if not np.any(usable):
            raise ValueError("reference has no strictly positive entries")
        ratios = np.log(jittered[usable]) - np.log(reference[usable])[:, None]

    size_factors = np.exp(np.median(ratios, axis=0))
    if np.any(size_factors <= 0) or not np.all(np.isfinite(size_factors)):
        raise ValueError("RLE produced a non-positive or non-finite size factor")
    return jittered / size_factors[None, :]
