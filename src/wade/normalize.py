"""Normalization: raw counts to a comparable scale, with the continuity jitter
(``docs/method.md`` §7).

The jitter is applied **at count precision, before division**: it is added
to the counts and a matching per-cell correction appears in the denominator,
so a pre-normalized matrix cannot reproduce it. It is drawn **once**, before
any permutation, so inference is conditional on that one draw and a seed
reproduces a result exactly. It must never be re-drawn inside the loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["draw_jitter", "library_sizes", "tpm_like"]

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
    """A per-gene vector or a genes x samples matrix, as ``(g, 1)`` or
    ``(g, n)``. The shape is checked rather than recycled: a per-*sample*
    vector where a per-*gene* one belongs is the one input error that would
    otherwise produce plausible wrong numbers."""
    g, n = shape
    normalizer = np.asarray(normalizer, dtype=np.float64)
    if normalizer.ndim == 1:
        if normalizer.shape[0] != g:
            raise ValueError(
                f"a vector normalizer must have one value per gene: expected "
                f"length {g}, got {normalizer.shape[0]}. (A length-{n} vector "
                f"would be per-sample, which is the wrong axis.)"
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

    It breaks ties in sparse, zero-heavy data, where many samples share a
    count of zero and the quantile grid would otherwise degenerate into flat
    runs, and it makes every entry strictly positive, so fold changes stay
    finite. ``rng`` takes precedence over ``seed``; ``seed=None`` draws from
    ambient state.
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
            f"supplied jitter must be genes x samples {shape}, got {jitter.shape}"
        )
    if not np.all(np.isfinite(jitter)):
        raise ValueError("supplied jitter must be finite")
    return jitter


def library_sizes(counts: np.ndarray, normalizer) -> np.ndarray:
    """Per-sample size factors: ``sum_g counts[g, j] / normalizer[g, j]``.

    Computed from the **unjittered** counts. Not on any interpretable scale:
    the units depend on the normalizer, and the value depends on which genes
    are in the matrix.
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
    """Normalizer- and library-scaled TPM-like units.

    Takes a count measure and a normalizer measure per gene: total counts
    over effective length gives TPM, a constant normalizer gives CPM.

    .. math::
        X_{gj} = \\kappa \\,
          \\frac{(C_{gj} + \\eta_{gj}) / L_{gj}}
                {\\ell_j + \\eta_{gj} / L_{gj}}

    **The denominator is per-cell.** ``lib_sizes`` is computed from the
    unjittered counts while the numerator uses jittered ones, and what is
    added to the library size is only *this gene's* jitter contribution, so
    the denominator is the library size in a counterfactual where gene ``g``
    alone received jitter. Consequences: column sums are not exactly
    ``norm_factor`` (the output is TPM-*like*), and a gene that is an entire
    library — every gene of an all-zero sample — normalizes to exactly
    ``norm_factor``, which is why :func:`wade.wade` refuses empty libraries.

    ``jitter`` supplies the draw in place of ``seed`` / ``rng``, on the same
    argument path :func:`wade.wade` uses.
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
