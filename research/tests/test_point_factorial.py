"""Six-lane arithmetic against both independent point implementations."""
import shutil

import numpy as np
import pytest

from wade._wasserstein import W1State, infer
from wade.permutation import draw_perms
from wade.pvalues import bh_adjust
from wade_research.point_factorial import FactorialState, factorial_infer, build_backend, load_backend, LANES
from wade_research.point_workflow import DirectionalState


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not shutil.which("rustc"):
        pytest.skip("benchmark-only acceleration requires rustc")
    return load_backend(build_backend(tmp_path_factory.mktemp("point-native")))


@pytest.mark.parametrize("nc,n0", [(3, 4), (50, 500)])
def test_six_lanes_match_oracles_including_sequential_stops(native, nc, n0):
    rng = np.random.default_rng(908)
    x = rng.integers(0, 20, (9, nc+n0))/rng.lognormal(0, .5, nc+n0)
    x[0] = 0
    x[1, :nc] *= 10
    cond = np.r_[np.ones(nc, int), np.zeros(n0, int)]
    bank = draw_perms(cond, 143, seed=902)
    reference = FactorialState(x, cond, stop=11)
    reference.update(bank)
    for batch in (1, 29, 143):
        actual = FactorialState(x, cond, stop=11)
        for start in range(0, len(bank), batch):
            actual.update(bank[start:start+batch], native)
        np.testing.assert_array_equal(actual.exceed, reference.exceed)
        np.testing.assert_array_equal(actual.tried, reference.tried)
    for scale, expression in enumerate((x, np.log2(x+1))):
        directional = DirectionalState(expression, cond, stop=11)
        unsigned = W1State(expression, cond, stop=11)
        directional.update(bank)
        unsigned.update(bank, backend="numpy")
        for key in ("observed", "exceed", "tried"):
            np.testing.assert_array_equal(getattr(reference, key)[:, scale*3:scale*3+2], getattr(directional, key))
            np.testing.assert_array_equal(getattr(reference, key)[:, scale*3+2], getattr(unsigned, key))


def test_complete_inference_matches_public_baseline_and_keeps_four_families(native):
    x = np.random.default_rng(159).poisson(2, (60, 20)).astype(float)
    x[0] = 0
    x[1, :10] *= 8
    cond = np.r_[np.ones(10, int), np.zeros(10, int)]
    stats, p, q, k, t = factorial_infer(x, cond, maximum=501, seed=88, native=native, batch=19)
    for scale, expression in enumerate((x, np.log2(x+1))):
        old = infer(expression, cond, maximum=501, seed=88, strata=None, backend="numpy", batch=71)
        for got, want in zip((stats[:, scale*3+2], p[:, scale*3+2], k[:, scale*3+2], t[:, scale*3+2]), old):
            np.testing.assert_array_equal(got, want)
    for lanes in LANES.values():
        np.testing.assert_array_equal(q[:, lanes], bh_adjust(p[:, lanes].ravel()).reshape(-1, len(lanes)))


def test_native_wrapper_rejects_invalid_state_and_labels(native):
    state = FactorialState(np.ones((3, 6)), [1, 1, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="permutations"):
        state.update(np.ones((2, 6)), native)
    state.order[0, 0] = 99
    with pytest.raises(ValueError, match="native state"):
        state.update(np.array([[1, 1, 1, 0, 0, 0]]), native)
