"""Is the subset test calibrated, and does it lose significance as a subset grows into the mean shift?

Simulates Gamma-Poisson genes (mean 50 counts; unit size factors) in which a share f
of the cases is raised by a fold and the rest match the controls, with tight
(CV2 0.05) or realistic (CV2 0.5, as in the confirmation) biological spread, plus
nulls in each direction: no change, a fold shared by every case, and a subset in
the opposite direction only. f = 100% is itself a null: every case carries the same
fold. Reports the side the split chose, the share of cases it selects and the share of
genes whose subset has p <= 0.05.

    PYTHONPATH=src:research python research/subset_significance.py
"""
import numpy as np

import wade

GENES, SHUFFLES = 200, 999


def design(rng, n1, n0, multiplier, cv2, mean=50):
    latent = rng.gamma(1 / cv2, mean * cv2, (GENES, n1 + n0))
    latent[:, :n1] *= multiplier
    return rng.poisson(latent)


def run(label, n1, n0, multiplier, cv2, seed):
    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    names = [f"s{i}" for i in range(n1 + n0)]
    counts = wade.as_counts(design(rng, n1, n0, multiplier, cv2), sample_names=names)
    unit = wade.Normalization.from_size_factors(counts, dict.fromkeys(names, 1.0), source="unit")
    r = wade.wade(counts, cond, normalization=unit, max_permutations=SHUFFLES, seed=seed)
    up, down = r.subset_log2fc > 0, r.subset_log2fc < 0
    print(f"  {label:34s} subset up {np.mean(up):4.0%} / down {np.mean(down):4.0%} of genes, "
          f"{np.mean(r.subset_fraction):4.0%} of cases; significant {np.mean(r.subset_pvalue <= .05):4.0%}")


def main():
    n1 = n0 = 250
    case = np.arange(n1)
    print(f"{n1}:{n0}, {GENES} genes per row, {SHUFFLES} shuffles; nominal 5% under the null")
    run("unchanged", n1, n0, np.ones(n1), .5, 1)
    run("every case x2", n1, n0, np.full(n1, 2.), .5, 2)
    run("every case x0.5", n1, n0, np.full(n1, .5), .5, 3)
    run("10% x16 up, rest unchanged", n1, n0, np.where(case < 25, 16, 1.), .5, 4)
    run("10% x0.25 down, rest unchanged", n1, n0, np.where(case < 25, .25, 1.), .5, 5)
    run("20% x16 up, rest x0.5", n1, n0, np.where(case < 50, 16, .5), .5, 6)
    for cv2, spread in ((.05, "tight"), (.5, "realistic")):
        for fold in (4, 100):
            for seed, f in enumerate((.05, .2, .5, .8, .95, 1.)):
                run(f"{spread}, {f:.0%} of cases x{fold}", n1, n0, np.where(case < round(f * n1), fold, 1.), cv2, 10 + seed)


if __name__ == "__main__":
    main()
