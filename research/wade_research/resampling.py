"""Numerical label-bank helpers, conditional on a justified permutation design.

These routines do not establish exchangeability or certify FDR.
"""
import numpy as np


def label_coefficients(labels):
    """Control minus case, with one unit of mass in each group."""
    labels = np.atleast_2d(np.asarray(labels))
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    ncase = labels.sum(axis=1, keepdims=True)
    ncontrol = labels.shape[1] - ncase
    if np.any(ncase == 0) or np.any(ncontrol == 0):
        raise ValueError("Both patient groups are required")
    return (1 - labels) / ncontrol - labels / ncase


def permutation_bank(labels, count, seed):
    """Fixed-budget engineering bank; observed labels occupy row zero."""
    labels = np.asarray(labels)
    label_coefficients(labels)
    rng = np.random.default_rng(seed)
    return np.array([labels, *[rng.permutation(labels) for _ in range(count)]])


def score_bank(engine, weights, *, bank_size=128, refinements=3):
    """Same bounded-bank and outward-roundoff rule as the frozen scientific pilot."""
    weights = np.asarray(weights)
    if (weights.ndim != 2 or len(weights) < 2
            or not isinstance(bank_size, (int, np.integer)) or bank_size < 2
            or not isinstance(refinements, (int, np.integer)) or refinements < 0):
        raise ValueError("Require observed and shuffled rows, bank_size >= 2, and refinements >= 0")
    records, observed = [], []
    splits = capped = 0

    def evaluate_bank(shuffles):
        nonlocal splits, capped
        bank = np.r_[weights[:1], shuffles]
        compressed = engine.compress(bank)
        result = None
        for level in range(refinements+1):
            try:
                refined = engine.scores(bank, atol=2e-5*engine.unit/10**level, rtol=2e-4/10**level)
            except MemoryError:
                if result is not None:
                    capped += 1
                    break
                if len(shuffles) == 1:
                    raise
                splits += 1
                middle = len(shuffles)//2
                evaluate_bank(shuffles[:middle]); evaluate_bank(shuffles[middle:])
                return
            result = refined
            lower, upper = result.permutation_p_bounds(compressed, 2e-8*engine.unit)
            if lower == upper:
                break
        records.append(result)
        observed.append((result.lower[0], result.upper[0]))

    for start in range(1, len(weights), bank_size-1):
        evaluate_bank(weights[start:start+bank_size-1])
    tol = 2e-8*engine.unit
    gap = max(0., max(a for a, b in observed)-min(b for a, b in observed))
    if gap > 2*tol:
        raise ArithmeticError('Observed brackets disagree beyond the roundoff allowance')
    lo = np.maximum(0, np.r_[max(a for a, b in observed), *[r.lower[1:] for r in records]]-tol)
    hi = np.r_[min(b for a, b in observed), *[r.upper[1:] for r in records]]+tol
    assert np.all(lo <= hi)
    return lo, hi, dict(bank_splits=splits, precision_capped_banks=capped,
                       roundoff_padding=tol, observed_roundoff_gap=gap)
