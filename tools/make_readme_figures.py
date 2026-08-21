"""Regenerate the figures the README embeds. Deterministic; no download.

    conda activate wade
    python tools/make_readme_figures.py

Writes docs/figures/{gene,volcano,stages}.png with the matplotlib backend so
the output is stable across machines up to font rendering.

**Negative-binomial counts**, which is what WADE is for (docs/method.md §10):
dispersion 0.1 (biological CV ~0.32), per-gene mean drawn log-uniformly from
1 to 1000 counts so the panel spans the range where the log scale is
noise-dominated as well as the range where it is not. 300 v 300, B = 2000,
3,045 genes:

* 3,000 null genes;
* 15 genes with a **global 2x** shift;
* 15 genes with **15% of cases at 8x** — chosen because their log2 fold change
  is the same as the global genes', which is the situation the method exists
  for: the same fold change, a different shape;
* 15 genes with **5% of cases at 8x**, which sit near the combinatorial floor
  at this design and show what "found at raw p, not at FDR" looks like.

**The matrix has to be big enough for composition not to dominate.** A first
draft used 600 null genes, so the 45 signal genes were 7% of the library mass:
the case libraries inflated by 6.6%, every null gene acquired a -0.09 log2 fold
change, and 186 of them cleared BH on the mean-shift stage. That is the effect
docs/limits.md 2.4 describes, and at 300 v 300 a systematic 6% shift is easily
significant. At 3,000 null genes the signal is 1.5% of the mass, the null
genes' median log2 fold change is -0.02, and the figure shows the method
rather than the artifact. Real matrices have 20,000 genes; toy ones need to be
told this.

Numbers printed by this script are what it measured.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import wade

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
PHI = 0.1


def _nb(rng, mu, size):
    """Negative binomial with dispersion PHI, as a Poisson-Gamma mixture."""
    return rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)


def figure_example(seed: int = 0, n: int = 300, nperms: int = 2000):
    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n, int), np.zeros(n, int)]
    genes, names = [], []
    # Per-gene expression level, log-uniform over 1..1000 counts.
    mu_null = 10 ** rng.uniform(0, 3, 3000)
    for i, mu in enumerate(mu_null):
        genes.append(np.r_[_nb(rng, mu, n), _nb(rng, mu, n)])
        names.append(f"null_{i}")
    mu_signal = 10 ** rng.uniform(0.5, 3, 45)
    for i, mu in enumerate(mu_signal[:15]):
        genes.append(np.r_[_nb(rng, 2 * mu, n), _nb(rng, mu, n)])
        names.append(f"global_{i}")
    k = 15
    for frac, tag in ((0.15, "subset15"), (0.05, "subset5")):
        for i, mu in enumerate(mu_signal[k:k + 15]):
            case = _nb(rng, mu, n)
            idx = rng.choice(n, round(frac * n), replace=False)
            case[idx] = _nb(rng, 8 * mu, idx.size)
            genes.append(np.r_[case, _nb(rng, mu, n)])
            names.append(f"{tag}_{i}")
        k += 15
    counts = np.array(genes)
    return wade.wade(counts, np.full(counts.shape[0], 1.0), cond, nperms=nperms,
                     gene_names=names, seed=1, n_boot=200)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = figure_example()

    # The panel that explains the method: the same fold change, a different shape.
    fig = wade.plot_gene(res, gene=["global_0", "subset15_0", "subset5_0"],
                         backend="matplotlib", width=10.5, height=6.0)
    fig.savefig(OUT / "gene.png", dpi=120)

    fig = wade.plot_volcano(res, stage="both", label=["global_0", "subset15_0", "subset5_0"],
                            backend="matplotlib", width=11, height=4.8)
    fig.savefig(OUT / "volcano.png", dpi=120)

    fig = wade.plot_stages(res, label=["global_0", "subset15_0", "subset5_0"],
                           backend="matplotlib", width=6.4, height=5.6)
    fig.savefig(OUT / "stages.png", dpi=120)

    from wade.plotting import stages_data
    print("quadrant counts:", stages_data(res).counts())
    for tag, sl in (("null", slice(0, 3000)), ("global x2", slice(3000, 3015)),
                    ("subset 15% x8", slice(3015, 3030)), ("subset 5% x8", slice(3030, 3045))):
        print(f"  {tag:>14}: log2fc {np.median(res.log2_fc[sl]):+.2f}  "
              f"p_mean {np.median(res.p_mean_shift[sl]):.4f}  p_subset {np.median(res.p_subset[sl]):.4f}  "
              f"affected {np.median(res.affected_fraction[sl]):.2f} "
              f"[{np.median(res.ci_affected_fraction[0, sl]):.2f}, {np.median(res.ci_affected_fraction[1, sl]):.2f}]  "
              f"f-hat {np.median(res.fitted_fold_change[sl]):.2f}")
    # The three genes the figures actually draw, because those are the numbers
    # the README quotes beside them. Printing the class medians alone is how
    # that prose went stale once already.
    for name in ("global_0", "subset15_0", "subset5_0"):
        i = res.gene_index(name)
        print(f"  {name:>14}: log2fc {res.log2_fc[i]:+.2f}  "
              f"p_mean {res.p_mean_shift[i]:.2g}  p_subset {res.p_subset[i]:.2g}  "
              f"affected {res.affected_fraction[i]:.2f}")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
