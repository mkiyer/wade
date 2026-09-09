"""Restricted (within-stratum) permutation — ``docs/method.md`` §10.2.

A cohort assembled from several studies is not exchangeable across them:
shuffling labels freely tests a hypothesis nobody holds, and reports batch
structure as biology. ``strata=`` shuffles labels **within** each stratum
instead, holding the batch axis fixed.

What is pinned here: the unrestricted path is untouched (bitwise, so every
existing seed reproduces), restricted draws preserve every stratum's case
count, supplied permutations are validated against the strata, the
permutation space and its p-value floor are computed and recorded, and a
planted batch effect that fools the unrestricted null is controlled by the
restricted one.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.permutation import draw_perms, permutation_space, strata_indices, validate_perms

N1 = N0 = 40
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]


def test_strata_indices_groups_by_first_appearance():
    st = np.array(["b", "a", "b", "a", "c"])
    idx = strata_indices(st, 5)
    assert [i.tolist() for i in idx] == [[0, 2], [1, 3], [4]]
    with pytest.raises(ValueError, match="one entry per sample"):
        strata_indices(st, 4)
    with pytest.raises(ValueError, match="1-D"):
        strata_indices(np.zeros((2, 2)), 4)


def test_unrestricted_draw_is_bitwise_unchanged():
    """strata=None must consume the generator exactly as before."""
    a = draw_perms(COND, 25, seed=7)
    b = draw_perms(COND, 25, seed=7, strata=None)
    np.testing.assert_array_equal(a, b)


def test_restricted_draw_preserves_every_stratum():
    st = np.array([f"s{i % 5}" for i in range(N1 + N0)])
    perms = draw_perms(COND, 50, seed=3, strata=st)
    assert perms.shape == (50, N1 + N0)
    for idx in strata_indices(st, N1 + N0):
        want = int((COND[idx] == 1).sum())
        assert np.all((perms[:, idx] == 1).sum(axis=1) == want)
    # globally it is still a permutation of cond, so the plain check passes too
    validate_perms(perms, COND, 50)
    validate_perms(perms, COND, 50, strata=st)
    # and it does shuffle *something*
    assert not np.array_equal(perms[0], COND)


def test_validate_rejects_labels_moved_across_strata():
    st = np.r_[np.zeros(N1 + N0 - 20, int), np.ones(20, int)]
    perms = draw_perms(COND, 10, seed=1)          # unrestricted: crosses strata
    with pytest.raises(ValueError, match="preserve each stratum"):
        validate_perms(perms, COND, 10, strata=st)


def test_permutation_space_counts_the_freedom_that_is_left():
    # unrestricted: C(80, 40)
    unrestricted = permutation_space(COND)
    assert unrestricted["n_strata"] == 1
    assert unrestricted["log10_space"] == pytest.approx(23.03, abs=0.05)  # C(80,40)

    # four balanced strata of 20: C(20,10)^4, far smaller
    st = np.array([f"s{i % 4}" for i in range(N1 + N0)])
    restricted = permutation_space(COND, st)
    assert restricted["n_strata"] == 4
    assert restricted["log10_space"] < unrestricted["log10_space"]
    assert restricted["p_floor"] > unrestricted["p_floor"]

    # a stratum holding only one class gives no freedom at all, and says so
    one_class = np.where(COND == 1, "cases", "ctrls")
    collapsed = permutation_space(COND, one_class)
    assert collapsed["uninformative_strata"] == 2
    assert collapsed["log10_space"] == 0.0 and collapsed["p_floor"] == 1.0


def test_wade_records_the_restricted_space_in_params_and_manifest():
    rng = np.random.default_rng(0)
    counts = rng.poisson(50, (25, N1 + N0)).astype(float)
    st = np.array([f"s{i % 4}" for i in range(N1 + N0)])
    res = wade.wade(counts, np.ones(25), COND, nperms=100, seed=1, strata=st)
    np.testing.assert_array_equal(res.params["strata"], st)
    man = wade.manifest(res)["design"]["permutation_space"]
    assert man["restricted"] is True and man["n_strata"] == 4
    assert man["uninformative_strata"] == 0
    # unrestricted runs record the unrestricted space
    plain = wade.wade(counts, np.ones(25), COND, nperms=100, seed=1)
    assert plain.params["strata"] is None
    assert wade.manifest(plain)["design"]["permutation_space"]["restricted"] is False


def test_restricted_permutation_controls_a_planted_batch_effect():
    """The point of the feature. Two studies, the case/control split
    confounded with study (study A is 80% cases, B is 20%), and study B
    shifted 4x on every gene. **Nothing is differential within a study.**
    The unrestricted null must fire; the restricted null must not.

    Unit library sizes on purpose: with TPM-like normalization a uniform
    shift of every gene is divided straight back out, so the planted batch
    effect would vanish before any test saw it (``CONTRIBUTING.md`` on
    composition). Unit libraries make a 4x shift stay a 4x shift.
    """
    rng = np.random.default_rng(11)
    g, n = 40, 150
    study = np.array(["A"] * n + ["B"] * n)
    cond = np.r_[np.ones(int(0.8 * n), int), np.zeros(n - int(0.8 * n), int),
                 np.ones(int(0.2 * n), int), np.zeros(n - int(0.2 * n), int)]
    assert cond.sum() == n                        # balanced overall, confounded with study
    base = rng.gamma(10.0, 5.0, (g, 1))
    counts = np.c_[rng.poisson(np.broadcast_to(base, (g, n))),
                   rng.poisson(np.broadcast_to(4.0 * base, (g, n)))].astype(float)

    kw = dict(nperms=300, seed=2, subset=False, lib_sizes=np.ones(2 * n))
    free = wade.wade(counts, np.ones(g), cond, **kw)
    held = wade.wade(counts, np.ones(g), cond, strata=study, **kw)
    rate_free = float((free.padj_mean_shift <= 0.05).mean())
    rate_held = float((held.padj_mean_shift <= 0.05).mean())
    assert rate_free > 0.5, f"the confound should fool the unrestricted null; got {rate_free:.2f}"
    assert rate_held <= 0.15, f"restricted permutation should control it; got {rate_held:.2f}"


def test_wade_contrast_carries_per_sample_arguments_onto_its_reordering():
    """wade_contrast reorders the matrix into [cases..., controls...]. Every
    per-sample argument has to be reordered with it — applying one positionally
    to the reordered columns lands it on the wrong samples, and when the
    lengths happen to match it does so *silently* (found by the package audit,
    2026-08-20).

    A paired design is the case that matters: with subject strata and
    alternating case/control columns, the un-reindexed version destroyed every
    pair and made each stratum condition-pure, so within-stratum shuffling
    became a no-op and the null collapsed to nothing — with no error.
    """
    rng = np.random.default_rng(3)
    n = 20
    counts = rng.poisson(60, (25, n)).astype(float)
    names = np.array([f"s{j}" for j in range(n)], dtype=object)
    subject = np.array([f"subj{j // 2}" for j in range(n)])
    case = [names[j] for j in range(1, n, 2)]
    ctrl = [names[j] for j in range(0, n, 2)]

    res = wade.wade_contrast(counts, 2.0, case, ctrl, sample_names=names,
                             nperms=30, subset=False, strata=subject)
    lookup = dict(zip(names.tolist(), subject.tolist()))
    got = res.params["strata"]
    assert [lookup[s] for s in res.sample_names.tolist()] == list(got)
    # every stratum keeps one case and one control, so the shuffle is real
    for idx in strata_indices(got, n):
        assert int((np.asarray(res.cond)[idx] == 1).sum()) == 1

    # lib_sizes and jitter travel the same way
    lib = np.arange(1.0, n + 1.0)
    jit = np.tile(np.arange(n, dtype=float), (25, 1))
    res2 = wade.wade_contrast(counts, 2.0, case, ctrl, sample_names=names,
                              nperms=0, subset=False, lib_sizes=lib, jitter=jit)
    order = [int(s[1:]) for s in res2.sample_names.tolist()]
    np.testing.assert_array_equal(res2.jitter[0], np.asarray(order, dtype=float))

    # an already-subset argument cannot be reindexed, so it is refused
    with pytest.raises(ValueError, match="must cover every sample"):
        wade.wade_contrast(counts, 2.0, case, ctrl, sample_names=names,
                           nperms=0, subset=False, strata=subject[:10])
