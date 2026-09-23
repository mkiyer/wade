"""Descriptive effects on the same log-expression scale as the test.

These are not separate tests, fitted biological shifts, or outlier probabilities.
"""
import numpy as np


def quantile_mean(sorted_values, lower, upper):
    """Exact integral of an empirical quantile function over a probability band."""
    n = sorted_values.shape[-1]
    edges = np.arange(n + 1, dtype=float) / n
    weights = np.maximum(0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
    return sorted_values @ (weights / (upper - lower))


def describe(expression, cond):
    case = np.sort(expression[:, cond == 1], axis=1)
    control = np.sort(expression[:, cond == 0], axis=1)
    bulk = np.median(case, axis=1) - np.median(control, axis=1)
    upper = quantile_mean(case, .9, 1) - quantile_mean(control, .9, 1) - bulk
    lower = quantile_mean(case, 0, .1) - quantile_mean(control, 0, .1) - bulk
    return bulk, upper, lower
