"""Fixture loading and the deviation ledger for the parity suite.

The suite is built **inside-out** on the design in
``docs/implementation-notes.md``.

**What it is for, now that the statistic it originally validated has been
replaced.** The tail-window machinery is gone, so parity with R on
``tail.mean`` and ``tail.conc`` no longer tests anything that runs. What the
fixtures still pin is the *shared machinery underneath* every statistic —
normalization, the type-7 quantile grids, the permutation null, the GPD
refinement and BH — which the redesign uses unchanged and which is exactly
where a silent cross-language disagreement would do the most damage. The organizing principle:
intermediate quantities localize a disagreement, endpoint quantities only
detect one. If ``padj_tail`` differs, the cause could be the quantile
type, the tail window, the grid orientation, the exceedance broadcast,
the GPD branch, the floor, or BH — seven candidates and no information.
If ``Q1``, ``Q0`` and ``D`` have already been asserted, six of those are
eliminated before the endpoint is compared. So **stop at the first
failing layer**: a suite reporting six failures from one root cause has
told you less than one reporting the innermost.

Two things this suite refuses to do:

* **No bitwise assertions** except on integers, ranks and branch flags.
  Floating-point addition is not associative, so ``rowSums``,
  ``numpy.sum`` (pairwise) and a hand-written loop legitimately differ in
  the last bits. Merely reversing a summation order changes about two
  thirds of genes at a relative magnitude of 1e-14
  (``docs/implementation-notes.md`` hazard 10).
* **The R is not the oracle for ``tail_conc`` or for a one-sample
  group.** In both cases a correct port disagrees with ``wade.R``, and a
  suite that enforces agreement enforces the bug. Those live in
  ``test_divergences.py``.

Every comparison is recorded in a ledger and the worst-case relative
deviation is printed at the end of the run. That number is what
distinguishes an exact port from a close one; "the assertions passed" is
not the same claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

#: Scenarios the port is expected to reproduce.
#: The `weighted` and `log2scaled` fixtures exercised wade.R's `weight` and
#: `log2_scale` pre-transforms, which no reference call site ever used and
#: which the redesign does not carry. The fixtures remain on disk as a record.
PARITY_SCENARIOS = [
    "tiny", "main", "even_larger", "vecnorm", "m2", "nperms0",
    "gate_closed", "refine", "zerolib", "nonames", "tailconc", "tiesheavy",
]

#: Scenarios where a correct port MUST disagree with the reference.
DIVERGENT_SCENARIOS = ["onesample"]

#: Tolerances. Tight by intent: at 1e-12 relative a disagreement is a real
#: bug and one at 1e-14 is summation order. The normalization layer gets
#: the tightest of all, because the naive lib_sizes-only denominator fails
#: it by about 1.7e-6 relative — generous tolerance there would let the
#: single subtlest arithmetic error in the method through.
TOL_NORMALIZATION = 1e-12
TOL_GRID = 1e-12
TOL_REDUCTION = 1e-12
TOL_NULL = 1e-12
TOL_PVALUE = 1e-12
TOL_BH = 1e-10        # hazard 3: one-ULP library differences are expected here

_MISSING = {"NA", "NaN", "Inf", "-Inf"}


def _parse_scalar(dec: str, hexs: str) -> float:
    """Parse one value, using hex as authoritative and dec as a cross-check.

    The R writes both channels: ``%.17g`` for humans and ``%a`` for
    exactness. R cannot verify its own decimal output — measured, R's
    ``as.numeric`` is not a correctly-rounded parser and reads back one
    ULP off even at ``%.20g`` — so the check belongs here, in the
    consumer, where the parser *is* correctly rounded.
    """
    if hexs in _MISSING:
        if dec != hexs:
            raise AssertionError(f"fixture channel mismatch: dec={dec!r} hex={hexs!r}")
        return {"NA": math.nan, "NaN": math.nan,
                "Inf": math.inf, "-Inf": -math.inf}[hexs]
    exact = float.fromhex(hexs)
    approx = float(dec)
    if approx != exact:
        raise AssertionError(
            f"fixture 17-digit decimal does not round-trip to its exact value: "
            f"dec={dec!r} -> {approx!r}, hex={hexs!r} -> {exact!r}"
        )
    return exact


def _parse_dbl_vec(node) -> np.ndarray:
    dec, hexs = node["dec"], node["hex"]
    assert len(dec) == len(hexs), "dec/hex channels differ in length"
    return np.array([_parse_scalar(d, h) for d, h in zip(dec, hexs)], dtype=np.float64)


def _parse_dbl_mat(node) -> np.ndarray:
    rows, cols = node["shape"]
    dec, hexs = node["dec"], node["hex"]
    assert len(dec) == rows and len(hexs) == rows, "matrix row count disagrees with shape"
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        assert len(dec[i]) == cols, f"row {i} has {len(dec[i])} entries, shape says {cols}"
        for j in range(cols):
            out[i, j] = _parse_scalar(dec[i][j], hexs[i][j])
    return out


def _parse_int_mat(node) -> np.ndarray:
    rows, cols = node["shape"]
    arr = np.array(node["data"], dtype=np.int64)
    if arr.size == 0:
        # An empty permutation matrix (nperms = 0) parses to shape (0,);
        # the declared shape is what carries the sample count.
        return arr.reshape(rows, cols)
    assert arr.shape == (rows, cols), f"int matrix shape {arr.shape} != declared {(rows, cols)}"
    return arr


def _walk(node):
    """Recursively convert the emitted JSON into numpy arrays."""
    if isinstance(node, dict):
        keys = set(node)
        if keys == {"dec", "hex"}:
            return _parse_dbl_vec(node)
        if keys == {"shape", "dec", "hex"}:
            return _parse_dbl_mat(node)
        if keys == {"shape", "data"}:
            return _parse_int_mat(node)
        return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, list):
        # Empty arrays in these fixtures are always index or count vectors
        # (no refined genes, no permutations), so give them an array type
        # rather than leaving a bare list with no .size / .shape.
        if not node:
            return np.array([], dtype=np.int64)
        if all(isinstance(v, int) and not isinstance(v, bool) for v in node):
            return np.array(node, dtype=np.int64)
        return [_walk(v) for v in node]
    return node


_CACHE: dict[str, dict] = {}


def load_fixture(name: str) -> dict:
    if name not in _CACHE:
        path = FIXTURE_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing fixture {path}. Regenerate with:\n"
                f'  export PATH="/usr/local/bin:$PATH"\n'
                f"  cd reference/R && RENV_CONFIG_SANDBOX_ENABLED=FALSE "
                f'RENV_PATHS_CACHE="$HOME/Library/Caches/org.R-project.R/R/renv/cache" '
                f"Rscript ../../tools/r/generate_fixtures.R"
            )
        with path.open() as fh:
            _CACHE[name] = _walk(json.load(fh))
    return _CACHE[name]


# ---------------------------------------------------------------------
# The deviation ledger
# ---------------------------------------------------------------------

_LEDGER: list[tuple[str, str, float, float, int]] = []


def max_rel_dev(got, want) -> tuple[float, int]:
    """Worst relative deviation and the index where it occurs.

    Relative to the reference value, falling back to absolute where the
    reference is exactly zero (a relative deviation from zero is not
    defined and clamping the denominator would flatter the port).
    Non-finite entries are compared for exact agreement instead and
    excluded from the magnitude.
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: got {got.shape}, want {want.shape}")

    g_nan, w_nan = np.isnan(got), np.isnan(want)
    if not np.array_equal(g_nan, w_nan):
        bad = int(np.flatnonzero(g_nan != w_nan)[0])
        raise AssertionError(
            f"missing-value positions differ: index {bad} is "
            f"{'nan' if g_nan[bad] else got[bad]} in the port and "
            f"{'nan' if w_nan[bad] else want[bad]} in R"
        )
    g_inf, w_inf = np.isinf(got), np.isinf(want)
    if not np.array_equal(g_inf, w_inf) or not np.array_equal(got[g_inf], want[w_inf]):
        raise AssertionError("infinite values differ in position or sign")

    ok = ~(g_nan | g_inf)
    if not np.any(ok):
        return 0.0, -1
    g, w = got[ok], want[ok]
    denom = np.abs(w)
    rel = np.where(denom > 0, np.abs(g - w) / np.where(denom > 0, denom, 1.0), np.abs(g - w))
    i = int(np.argmax(rel))
    return float(rel[i]), int(np.flatnonzero(ok)[i])


def assert_close(got, want, tol: float, label: str, layer: str = ""):
    """Compare, record the deviation, and assert it is within tolerance."""
    dev, idx = max_rel_dev(got, want)
    _LEDGER.append((layer, label, dev, tol, idx))
    if dev > tol:
        g = np.asarray(got, dtype=np.float64).ravel()
        w = np.asarray(want, dtype=np.float64).ravel()
        raise AssertionError(
            f"{label}: worst relative deviation {dev:.3e} exceeds tolerance {tol:.1e} "
            f"at flat index {idx} (port {g[idx]!r} vs R {w[idx]!r})"
        )
    return dev


def record(label: str, dev: float, tol: float, layer: str = ""):
    _LEDGER.append((layer, label, dev, tol, -1))


@pytest.fixture(scope="session")
def ledger():
    return _LEDGER


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report the worst-case relative deviation per layer, and overall.

    ``ROADMAP.md``: report the worst-case relative deviation observed, not
    merely that assertions passed.
    """
    if not _LEDGER:
        return
    tr = terminalreporter
    tr.write_sep("=", "WADE parity: worst-case relative deviation vs the R reference")

    by_layer: dict[str, list[tuple[str, float, float]]] = {}
    for layer, label, dev, tol, _ in _LEDGER:
        by_layer.setdefault(layer or "(unlabelled)", []).append((label, dev, tol))

    def _key(name: str):
        return (0, int(name.split(" ")[1])) if name.startswith("layer ") else (1, 0)

    overall = 0.0
    total = 0
    for layer in sorted(by_layer, key=_key):
        entries = by_layer[layer]
        worst_label, worst_dev, worst_tol = max(entries, key=lambda e: e[1])
        # Layers in parentheses are diagnostics — the size of an error the
        # suite is built to catch — and are not deviations the port exhibits.
        if not layer.startswith("("):
            overall = max(overall, worst_dev)
            total += len(entries)
        tr.write_line(
            f"  {layer:<34s} n={len(entries):<4d} worst={worst_dev:9.3e}  "
            f"(tol {worst_tol:.0e})  <- {worst_label}"
        )
    tr.write_line("")
    tr.write_line(
        f"  {'WORST OVER ALL PARITY COMPARISONS':<34s} {overall:9.3e}   "
        f"({total} comparisons)"
    )
    tr.write_line(
        "  Reference: 1e-12 relative is a real bug; 1e-14 is summation order "
        "(implementation-notes.md hazard 10)."
    )
