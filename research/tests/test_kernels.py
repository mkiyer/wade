"""Refactor replay and independent checks of the research calculation.

These are numerical tests, not biological calibration or model-selection gates.
The archived implementation generated the replay fixture before extraction.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, wasserstein_distance

from wade_research import (
    BetaKernelArea, label_coefficients, permutation_bank, point_scores, score_bank,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/kernel-baseline.json").read_text())


@pytest.mark.parametrize("row", FIXTURE["cases"], ids=lambda r: f"case{r['example']}-h{r['prior_shape']}")
def test_extraction_replays_frozen_area_bounds_and_points(row):
    engine = BetaKernelArea(row["counts"], row["totals"],
                            size_factors=row["size_factors"], prior_shape=row["prior_shape"],
                            node_cache_mb=1)
    lo, hi, diagnostics = score_bank(engine, label_coefficients(row["labels"]),
                                     bank_size=8, refinements=3)
    np.testing.assert_array_equal(lo, row["lower"])
    np.testing.assert_array_equal(hi, row["upper"])
    assert diagnostics == row["diagnostics"]
    points = np.asarray(row["counts"]) / row["size_factors"]
    np.testing.assert_array_equal(point_scores(points, row["labels"]), row["point_scores"])


@pytest.mark.parametrize("h", [1., .5])
def test_area_brackets_independent_cdf_quadrature(h):
    c = np.array([0, 2, 8, 30, 0, 1, 4])
    n = np.array([100, 200, 300, 100, 100, 200, 300])
    sf = np.array([.3, 1., 3., .3, 3., 1., .3])
    labels = np.array([1, 1, 1, 1, 0, 0, 0])
    engine = BetaKernelArea(c, n, size_factors=sf, prior_shape=h)
    weights = label_coefficients(labels)
    bounds = engine.scores(weights, rtol=1e-5)
    # Independently integrate complete group CDFs, without the adapter's
    # primitives, mesh, cache or patient-state compression.
    scale = n / sf
    a, b = c + h, n - c + h
    knots = np.unique((scale * beta.ppf(np.array([.01, .5, .99])[:, None], a, b)).ravel())
    def integrand(x):
        f = beta.cdf(x / scale, a, b)
        return max(0., f[labels == 0].mean() - f[labels == 1].mean())
    value, error = quad(integrand, 0., scale.max(), points=knots,
                        epsabs=1e-8, epsrel=1e-9, limit=2000)
    assert error < 1e-6
    assert bounds.lower[0] - error <= value <= bounds.upper[0] + error


@pytest.mark.parametrize("h", [1., .5])
def test_normalization_scales_entire_distribution_and_area(h):
    c = np.array([0, 1, 10, 0, 1, 30])
    n = np.array([100, 100, 1000, 100, 100, 1000])
    sf = np.array([.1, 1., 10., .1, 1., 10.])
    engine = BetaKernelArea(c, n, size_factors=sf, prior_shape=h)
    scaled = BetaKernelArea(c, n, size_factors=sf * 7, prior_shape=h)
    x = np.linspace(0, 50, 101)
    np.testing.assert_allclose(engine.evaluate(x)[0], scaled.evaluate(x / 7)[0], atol=1e-14)
    np.testing.assert_allclose(engine.mean, scaled.mean * 7, rtol=2e-15)
    weights = label_coefficients([1, 1, 1, 0, 0, 0])
    a, b = engine.scores(weights), scaled.scores(weights)
    assert a.lower[0] <= b.upper[0] * 7 + 1e-10
    assert b.lower[0] * 7 <= a.upper[0] + 1e-10
    np.testing.assert_allclose((n / sf) * (c / n), c / sf)


@pytest.mark.parametrize("h", [1., .5])
def test_zero_kernels_have_positive_width_and_identical_groups_cancel(h):
    engine = BetaKernelArea([0] * 6, [1000] * 6, size_factors=np.ones(6), prior_shape=h)
    assert np.all(engine.mean > 0)
    width = engine.scale * (beta.ppf(.975, engine.a, engine.b) - beta.ppf(.025, engine.a, engine.b))
    assert np.all(width > 0)
    np.testing.assert_array_equal(engine.evaluate([0, 1000])[0], [[0, 1]])
    result = engine.scores(label_coefficients([1, 1, 1, 0, 0, 0]))
    np.testing.assert_array_equal(result.lower, 0)
    np.testing.assert_array_equal(result.upper, 0)


def test_cache_is_exact_and_keeps_different_scales_distinct():
    c, n, sf = [0, 0, 2, 2, 10, 10], [100] * 6, [1, 2, 1, 2, 1, 2]
    plain = BetaKernelArea(c, n, size_factors=sf, prior_shape=.5)
    cached = BetaKernelArea(c, n, size_factors=sf, prior_shape=.5, node_cache_mb=1)
    assert len(cached.a) == 6
    labels = permutation_bank([1, 1, 1, 0, 0, 0], 15, 51)
    weights = label_coefficients(labels)
    expected = plain.scores(weights)
    for _ in range(2):
        actual = cached.scores(weights)
        np.testing.assert_array_equal(actual.lower, expected.lower)
        np.testing.assert_array_equal(actual.upper, expected.upper)
    assert cached.node_cache.stats()["hits"] > 0


def test_point_area_uses_exact_unequal_group_masses_and_ties():
    x = np.array([0., 0., 1., 9., 1., 3., 3.])
    labels = permutation_bank([1, 1, 1, 0, 0, 0, 0], 19, 51)
    up, total = point_scores(x, labels), point_scores(x, labels, upward=False)
    for i, selected in enumerate(labels.astype(bool)):
        distance = wasserstein_distance(x[selected], x[~selected])
        signed = x[selected].mean() - x[~selected].mean()
        assert total[i] == pytest.approx(distance)
        assert up[i] == pytest.approx((distance + signed) / 2, abs=1e-14)


@pytest.mark.parametrize("change", [
    {"counts": [-1, 0]}, {"counts": [.5, 0]}, {"totals": [0, 100]},
    {"counts": [101, 0]}, {"size_factors": [0, 1]},
    {"size_factors": [1]}, {"prior_shape": 0}, {"prior_shape": True},
])
def test_ambiguous_or_invalid_measurement_inputs_rejected(change):
    args = dict(counts=[0, 1], totals=[100, 100], size_factors=[1, 1], prior_shape=1.)
    args.update(change)
    with pytest.raises(ValueError):
        BetaKernelArea(**args)
