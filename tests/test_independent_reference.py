"""Where the fixtures' authority comes from, now that R is gone.

``tests/fixtures/*.json`` were generated from the original R implementation.
That implementation has been **retired and deleted** (2026-09-04): keeping a
frozen copy of a dead program in the repository so that a test could point at
it was not provenance, it was an heirloom. But deleting it without doing
anything else would leave twenty files of recorded numbers whose only
justification was "a program we no longer have said so", which is worse.

**This file is what replaces it.** Every quantity the layer suites compare
against is re-derived here from an authority that is neither R nor WADE:

===========================  ====================================================
layer                        independent authority
===========================  ====================================================
1  probability grid          the closed form, ``linspace(1, 0, m)``
2  normalization             the closed form in ``method.md`` §7
3  quantile grids            :func:`numpy.quantile` -- a separately written,
                             separately maintained type-7 implementation
4  reductions                definitions: ``D = Q1 - Q0``, and means of it
5  permutation null          layer 3 and 4 re-applied under permuted labels
6  empirical p-values        the add-one counting definition
7  BH                        :func:`scipy.stats.false_discovery_control`
===========================  ====================================================

Two of these (3 and 7) are genuinely independent *implementations* written by
other people; the rest are the definitions themselves, which is the strongest
authority available. What none of them is, is WADE. A bug in WADE cannot make
this file pass.

That is the whole point, and it is why this file must never import anything
from ``wade`` -- doing so would quietly turn it into a regression test that
proves the code agrees with itself. The layer suites (``test_layer1..7``) do
the WADE-vs-fixture comparison; this file establishes that the fixture is
right in the first place. Neither is sufficient alone.

The remaining provenance is behavioural rather than numerical and lives
elsewhere: brute-force enumeration in ``test_layer9_validation.py`` and
``test_saddlepoint.py``, and the head-to-head against COPA, OS, ORT, MOST,
LSOSS, the t-test, Wilcoxon and waddR in ``notebooks/benchmark.qmd``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from conftest import PARITY_SCENARIOS, assert_close, load_fixture

LAYER = "independent"

pytestmark = pytest.mark.parity


def _groups(fx):
    cond = np.asarray(fx["cond"])
    return np.flatnonzero(cond == 1), np.flatnonzero(cond == 0)


def _type7_via_numpy(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """The quantile grid, from NumPy rather than from WADE.

    ``numpy.quantile(method="linear")`` *is* Hyndman-Fan type 7 and is an
    independently written implementation of it. The only adaptation is order:
    WADE's grid runs from probability 1 down to 0, because position 0 must be
    the group maximum (``quantiles.py``), while NumPy wants ascending probs.
    Reversing in and back out changes no value.
    """
    return np.quantile(x, probs[::-1], axis=1, method="linear").T[:, ::-1]


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer1_grid_is_the_closed_form(name):
    """``linspace(1, 0, m)``, descending, with ``m = 1`` degenerating to 1.0."""
    fx = load_fixture(name)
    m = fx["q"].size
    want = np.array([1.0]) if m == 1 else np.linspace(1.0, 0.0, m)
    assert_close(want, fx["q"], 0.0, f"{name}: probability grid", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer2_normalization_is_the_closed_form(name):
    """The formula in ``method.md`` §7, written out here rather than called.

    The denominator is *not* ``lib_sizes[j]``, and that is the whole subtlety:
    the library sizes came from the unjittered counts while the numerator is
    jittered, so the consistent denominator adds back this gene's own jitter
    contribution and nothing else. Re-deriving it independently is the only
    way that stays checked once the reference implementation is gone.
    """
    fx = load_fixture(name)
    counts, jitter, lib = fx["counts"], fx["jitter"], fx["lib_sizes"]
    norm = fx["normalizer"]
    norm = np.asarray(norm["data"] if isinstance(norm, dict) else norm, dtype=float)
    if norm.ndim == 1:
        norm = norm[:, None]
    nf = float(np.ravel(fx["params"]["norm_factor"])[0])

    with np.errstate(divide="ignore", invalid="ignore"):
        share = (counts + jitter) / norm
        want = nf * share / (lib[None, :] + jitter / norm)
    assert_close(want, fx["tpm"], 1e-12, f"{name}: tpm_like closed form", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer3_quantile_grids_agree_with_numpy(name):
    """The one layer with a genuine second implementation available."""
    fx = load_fixture(name)
    i1, i0 = _groups(fx)
    for idx, key in ((i1, "Q1"), (i0, "Q0")):
        if len(idx) < 1 or not fx[key].size:
            continue
        got = _type7_via_numpy(fx["tpm"][:, idx], fx["q"])
        assert_close(got, fx[key], 1e-13, f"{name}: {key} vs numpy.quantile", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer4_reductions_are_their_definitions(name):
    """`D`, `mean_shift`, `w1` and the group means, straight from `Q1` and `Q0`.

    Note `cond1.mean` is the mean of the **quantile grid**, not of the raw
    columns. The two coincide only when that group is the smaller one and the
    grid therefore reads every sample; when it is the larger group the grid
    interpolates it, and the difference reaches 20% on these fixtures. That is
    the quadrature-versus-mean distinction of ``docs/pvalue-review.md``, and
    asserting the raw column mean here would silently redefine the statistic.
    """
    fx = load_fixture(name)
    st = fx["stats"]
    q1, q0 = fx["Q1"], fx["Q0"]
    assert_close(q1 - q0, fx["D"], 0.0, f"{name}: D = Q1 - Q0", LAYER)
    assert_close(fx["D"].mean(axis=1), st["diff.mean"], 1e-13,
                 f"{name}: mean_shift = mean(D)", LAYER)
    assert_close(np.abs(fx["D"]).mean(axis=1), st["w1"], 1e-13,
                 f"{name}: w1 = mean(|D|)", LAYER)
    assert_close(q1.mean(axis=1), st["cond1.mean"], 1e-13,
                 f"{name}: case mean = mean(Q1)", LAYER)
    assert_close(q0.mean(axis=1), st["cond0.mean"], 1e-13,
                 f"{name}: control mean = mean(Q0)", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer5_null_is_the_same_reduction_under_permuted_labels(name):
    """The permutation null re-derived through NumPy, one permutation at a time.

    This is the layer where a port is most likely to go wrong in a way no
    endpoint reveals -- an off-by-one in the label indexing, or a grid built
    from the wrong group -- so it is worth re-deriving rather than trusting
    that layers 3 and 4 passing implies it.
    """
    fx = load_fixture(name)
    perms = fx["perms"]
    null = fx.get("null", {})
    if not np.size(perms) or "perm_dm" not in null or not np.size(null["perm_dm"]):
        pytest.skip("this scenario has no permutations")

    tpm, q = fx["tpm"], fx["q"]
    want = np.empty((tpm.shape[0], perms.shape[0]))
    for b, labels in enumerate(perms):
        labels = np.asarray(labels)
        j1, j0 = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        want[:, b] = (_type7_via_numpy(tpm[:, j1], q)
                      - _type7_via_numpy(tpm[:, j0], q)).mean(axis=1)
    assert_close(want, null["perm_dm"], 1e-12,
                 f"{name}: mean-shift null under permuted labels", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer6_exceedances_and_the_empirical_p_are_counting(name):
    """`nexc` is a count and the empirical p-value is `(1 + nexc)/(B + 1)`.

    Only the unrefined genes are checked against the counting form: a refined
    p-value is by definition not this number, and the GPD branch is checked
    where it is defined rather than approximated here.
    """
    fx = load_fixture(name)
    pv, null = fx.get("pvalues", {}), fx.get("null", {})
    if "nexc_diff" not in pv or not np.size(fx["perms"]):
        pytest.skip("this scenario has no permutations")

    stat, draws = fx["stats"]["diff.mean"], null["perm_dm"]
    want = (draws >= stat[:, None]).sum(axis=1)
    np.testing.assert_array_equal(want, np.asarray(pv["nexc_diff"]),
                                  err_msg=f"{name}: exceedance counts")

    b = draws.shape[1]
    refined = np.zeros(stat.size, dtype=bool)
    idx = np.asarray(pv.get("refined_diff_i0", np.array([], dtype=int)), dtype=int)
    if idx.size:
        refined[idx] = True
    plain = ~refined
    assert_close((1 + want[plain]) / (b + 1), np.asarray(pv["p_diff"])[plain],
                 1e-13, f"{name}: empirical p = (1 + nexc)/(B + 1)", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_layer7_bh_agrees_with_scipy(name):
    """The second genuine outside implementation in the suite."""
    fx = load_fixture(name)
    pv, frame = fx.get("pvalues", {}), fx.get("frame", {})
    checked = 0
    for p_key, adj_key in (("p_diff", "padj.diff"), ("p_tail", "padj.tail")):
        p, adj = pv.get(p_key), frame.get(adj_key)
        if p is None or adj is None or not np.size(p) or not np.size(adj):
            continue
        p = np.asarray(p, dtype=float)
        if not np.all((p >= 0.0) & (p <= 1.0)):
            continue
        assert_close(false_discovery_control(p, method="bh"), adj, 1e-13,
                     f"{name}: {adj_key} vs scipy.false_discovery_control", LAYER)
        checked += 1
    if not checked:
        pytest.skip("this scenario has no adjusted p-values")


def test_this_file_does_not_import_wade():
    """The guard that keeps this file honest.

    If this module ever imports WADE, it stops being an independent check and
    becomes WADE agreeing with itself -- which would look identical in the
    test report and prove nothing. Asserted rather than left to review.
    """
    import pathlib

    # Built at run time: spelling the needle literally would make this file
    # contain the very string it forbids, and the guard would fail on itself.
    needle = "wade"
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    code = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ")):
            code.append(stripped)
    offenders = [ln for ln in code if ln.split()[1].split(".")[0] == needle]
    assert not offenders, f"this file must not import the package: {offenders}"
