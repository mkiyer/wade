"""How accurate is the computed area-test p-value, and does its own error estimate know?

For each design, simulates one transcriptome with the confirmation's generator (Gamma- or
lognormal-Poisson biology), multiplies every case by 2, 3, 4 or 8 in a quarter of the genes and
one or two cases by 20 in a tenth (lumpy totals, where a saddlepoint strains), computes every gene's
two-sided area-test p-value by the double saddlepoint with its error estimate, on the values the
test shuffles (counts thinned to the shallowest depth, jittered), and picks up to 10 genes per
decade of p from 0.1 to 1e-6. Their exact permutation p-value comes from every
labelling when the design has at most 3 million, and otherwise from 20 million random ones.
Reports how the error estimate tracks the actual error, and, per design, the share of genes
whose estimate exceeds the default relative_error (0.2) and so are shuffled instead.

    PYTHONPATH=src:research python research/pvalue_accuracy.py
"""
from itertools import combinations
from math import comb
from pathlib import Path
import json

import numpy as np

import wade
from wade._means import area_values, mean_saddlepoint
from wade_research.point_simulation import generate

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = [(3, 52), (5, 52), (8, 52), (5, 5), (8, 8), (12, 12), (445, 5), (445, 10), (50, 500)]
ENUMERATE, SHUFFLES, BLOCK = 3_000_000, 20_000_000, 20_000


def exact(x, n1, observed):
    """Two-sided exact permutation p-values: the share of labellings whose case total lies at
    least as far from its average as the observed one, over every labelling or a large sample."""
    n = x.shape[1]
    small = min(n1, n - n1)
    centre = n1 * x.mean(axis=1)
    reach = np.abs(observed - centre) - 1e-9 * np.abs(x).sum(axis=1)
    rng = np.random.default_rng(7)
    hits, drawn = np.zeros(len(x)), 0
    everything = comb(n, small) <= ENUMERATE
    groups = combinations(range(n), small) if everything else None
    while True:
        if everything:
            chunk = np.array([g for _, g in zip(range(BLOCK), groups)])
            if not len(chunk):
                break
        else:
            if drawn >= SHUFFLES:
                break
            if small * small < n:                               # draw indices, drop the rare repeats
                chunk = rng.integers(0, n, (2 * BLOCK, small))
                ordered = np.sort(chunk, axis=1)
                chunk = chunk[~(ordered[:, 1:] == ordered[:, :-1]).any(axis=1)][:BLOCK]
            else:
                chunk = np.argpartition(rng.random((BLOCK, n)), small, axis=1)[:, :small]
        smaller = x[:, chunk].sum(axis=2)                       # genes x labellings: the smaller group's total
        total = smaller if small == n1 else x.sum(axis=1)[:, None] - smaller
        hits += (np.abs(total - centre[:, None]) >= reach[:, None]).sum(axis=1)
        drawn += len(chunk)
    return hits / drawn, hits, everything


def main():
    cfg = json.loads((ROOT / "research/protocols/confirmation.json").read_text())
    rows = []
    for d, (n1, n0) in enumerate(DESIGNS):
        setting = dict(n_case=n1, n_control=n0, panel="complete_null", biology=("gamma", "correlated_lognormal")[d % 2],
                       seed=[2026092900, d, 0])
        data = generate(cfg, setting)
        counts, cond = data["counts"].copy(), data["cond"]
        plant = np.random.default_rng([2026092900, d, 0, 1])
        shifted = plant.random(len(counts)) < 0.25
        counts[np.ix_(shifted, cond == 1)] *= plant.choice([2, 3, 4, 8], shifted.sum())[:, None]
        for g in np.flatnonzero(~shifted & (plant.random(len(counts)) < 0.1)):
            counts[g, plant.choice(n1, plant.integers(1, 3), replace=False)] *= 20
        keep = np.flatnonzero(wade.filter_counts(counts, min_count=3, min_samples=2))
        factors = wade.fit_normalization(counts).size_factors
        x = area_values(counts[keep], keep, factors, 1)
        p, error = mean_saddlepoint(x, cond, direction="both")
        rng = np.random.default_rng([d, 0])
        pick = np.concatenate([rng.choice(np.flatnonzero((p >= 10.0 ** -(k + 1)) & (p < 10.0 ** -k)),
                                          min(10, np.sum((p >= 10.0 ** -(k + 1)) & (p < 10.0 ** -k))), replace=False)
                               for k in range(1, 6)])
        truth, hits, everything = exact(x[pick], n1, x[pick][:, :n1].sum(axis=1))
        resolved = everything | (hits >= 20)
        for g, t in zip(pick[resolved], truth[resolved]):
            rows.append((n1, n0, p[g], error[g], t))
        print(f"{n1:3d} v {n0:3d}: {len(keep)} tested genes, estimate > 0.2 for "
              f"{np.mean(error > 0.2):.1%}; checked {resolved.sum()} against "
              f"{'every labelling' if everything else f'{SHUFFLES:.0e} shuffles'}", flush=True)
    report(np.array(rows, dtype=object))


def report(rows):
    ratio = np.array([r[2] / r[4] for r in rows])
    estimate = np.array([r[3] for r in rows])
    off = (ratio < 1 / 1.5) | (ratio > 1.5)
    print("\nerror estimate    genes  median |actual error|  outside 0.67-1.5x")
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, np.inf)):
        k = (estimate >= lo) & (estimate < hi)
        if k.any():
            print(f"  {lo:4.2f}-{hi:<5.2g}  {k.sum():6d}  {np.median(np.abs(ratio[k] - 1)):20.0%}  {np.mean(off[k]):16.0%}")
    kept = estimate <= 0.2
    print(f"\nkept by the default rule (estimate <= 0.2): {kept.sum()} of {len(rows)} genes, "
          f"{np.mean(off[kept]):.1%} outside 0.67-1.5x; shuffled instead: {np.mean(off[~kept]):.1%} of theirs were")
    for n1, n0 in dict.fromkeys((r[0], r[1]) for r in rows):
        k = np.array([(r[0], r[1]) == (n1, n0) for r in rows])
        print(f"  {n1:3d} v {n0:3d}: kept {np.mean(kept[k]):4.0%} of checked genes; of those kept, "
              f"{np.mean(off[k & kept]) if (k & kept).any() else 0:.0%} outside 0.67-1.5x")


if __name__ == "__main__":
    main()
