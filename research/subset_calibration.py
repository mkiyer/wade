"""Is the subset test calibrated under the null, gene abundance by abundance, and what does jitter change?

Complete-null transcriptomes from the confirmation's generator (5 per design, alternating
biology) at 500:50, 50:500 and 250:250. The subset test (two-sided, 99,999 shuffles) runs on
count / size_factor as it stands and with each count's seeded Uniform(0, 1) jitter, as wade()
applies it. Reports observed over expected p-values at 0.01 and 0.001 by baseline abundance:
zero counts cannot move with the mean shift, and without jitter the null moves them anyway.

    PYTHONPATH=src:research python research/subset_calibration.py
"""
from pathlib import Path
import json

import numpy as np

import wade
from wade._means import expression, jittered
from wade._subsets import subset_test
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
DESIGNS, REPLICATES = ("500v50", "50v500", "250v250"), 5


def main():
    cfg = json.loads((ROOT / "research/protocols/confirmation.json").read_text())
    print("observed / expected share of p <= 0.01 and p <= 0.001 on complete nulls, by mean count")
    for design in DESIGNS:
        settings = [s for s in datasets(cfg) if s["panel"] == "complete_null" and s["id"].startswith(
            f"complete_null-{design}-")][:REPLICATES]
        results = {False: ([], []), True: ([], [])}
        for setting in settings:
            data = generate(cfg, setting)
            counts, cond = data["counts"], data["cond"]
            keep = np.flatnonzero(wade.filter_counts(counts, min_count=3, min_samples=2))
            factors = wade.fit_normalization(counts).size_factors
            for jitter in (False, True):
                c = jittered(counts[keep], keep, setting["permutation_seed"]) if jitter else counts[keep].astype(float)
                x = expression(c, factors, "linear")
                p, _, _ = subset_test(x, cond, "linear", direction="both", stop=25, maximum=99_999,
                                      seed=setting["permutation_seed"])
                results[jitter][0].append(p)
                results[jitter][1].append(counts[keep].mean(axis=1))
        for jitter, (ps, means) in results.items():
            p, m = np.concatenate(ps), np.concatenate(means)
            cells = []
            for lo, hi in ((0, 5), (5, 50), (50, np.inf)):
                k = (m >= lo) & (m < hi)
                cells.append(f"mean {lo}-{hi:g} (n={k.sum()}): "
                             + " ".join(f"{np.sum(p[k] <= a) / (a * k.sum()):.2f}x" for a in (1e-2, 1e-3)))
            print(f"  {design:7s} {'jittered' if jitter else 'counts  '}  " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
