# WADE — Wasserstein Area Differential Expression

A two-group differential-expression test for RNA-seq counts that answers
**two** questions instead of one:

1. **Is there a difference?** — the ordinary question, which any DE method
   answers.
2. **What kind of difference is it?** — a shift affecting every sample, or a
   pronounced change confined to a small subset of one group.

The second question is the point. A gene altered in 5% of cases and a gene
shifted 2× in all of them can produce the same mean difference, and a
first-moment test cannot tell them apart. WADE separates them, and never asks
you to declare in advance which you are looking for: there is no percentile
cutoff, no window width, no tuning parameter describing the shape of the
effect.

For each gene it compares the two groups' whole quantile functions. Stage 1
tests the mean shift; stage 2 tests whether a global fold change is an
*adequate* explanation, against a null built by binomial thinning of the raw
counts; three threshold-free descriptors then say how much of the group
differs, which way, and by how many folds. Inference is by label permutation
with a tail refinement and BH-FDR. [`docs/method.md`](docs/method.md) is the
full account.

**Scope: raw counts.** WADE takes raw counts, not TPM or CPM, and is for
discrete count data: the tie-breaking jitter and the thinning null both need
counts.

## Install

```bash
pip install wade-rnaseq              # NumPy only; compiled kernel included
pip install 'wade-rnaseq[all]'       # + polars (results out), plotly and matplotlib (figures), SciPy (saddlepoint)
```

The distribution is `wade-rnaseq` (the name `wade` on PyPI belongs to an
unrelated project); the import name is `wade`. Wheels cover Linux (x86-64,
ARM64), macOS (Apple silicon, Intel) and Windows (x64) for Python 3.10+. On
any other platform pip builds from source, which needs a Rust toolchain
([rustup](https://rustup.rs)); WADE still runs without the compiled kernel,
about 80× slower on the subset test. To build from a clone see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

> Not on PyPI yet. Until the first upload:
> `pip install git+https://github.com/mkiyer/wade` (needs Rust).

## Use it

```python
import polars as pl, wade

counts  = pl.read_csv("counts.tsv", separator="\t")   # gene id + one column per sample, RAW counts
samples = pl.read_csv("samples.tsv", separator="\t")  # sample_id, condition, ...

cond = wade.condition(samples, key="sample_id", column="condition",
                      case="tumor", control="normal")

res = wade.wade(counts, normalizer=1.0, cond=cond, nperms=2000, seed=1)
wade.write_results(res, "results.tsv")               # + results.manifest.json
```

`counts` may be a NumPy array, a polars or pandas DataFrame, a sparse matrix,
or a `wade.Counts`; WADE reads no files. `normalizer` is a per-gene vector
(gene length gives TPM-like values), a matrix, a scalar (`1.0` gives CPM), or
the name of a column carried alongside the counts. The alignment of samples
to the condition is strict and names the offenders.

| column | question it answers |
|---|---|
| `p_mean_shift`, `padj_mean_shift` | stage 1: is average expression different? |
| `p_subset`, `padj_subset` | stage 2: is a global shift an **inadequate** explanation? |
| `affected_fraction` | what fraction of samples differ? `1.0` = all of them |
| `direction` | `+1` all up, `-1` all down, `0` two-sided |
| `subset_log2_fc` | by how many folds does that fraction differ? |
| `log2_fc`, `mean_shift` | the global fold change and the signed quantile area |
| `z_mean_shift`, `z_subset` | permutation z-scores, for ranking once p-values hit the floor |

Four patterns, read off the two p-values:

| `p_mean_shift` | `p_subset` | what you are looking at |
|---|---|---|
| significant | — | a **global shift**; an ordinary DE method finds this too |
| significant | significant | a **subset**, strong enough to move the mean |
| — | significant | a distributional change with **no net mean shift**: a balanced subset, or a variance change |
| — | — | not differential |

There is deliberately no categorical label: turning continuous statistics
into classes needs thresholds, which is what this design exists to avoid.

```python
from wade import plot_gene, plot_volcano, plot_stages, plot_drivers   # needs wade-rnaseq[plot]

plot_gene(res, gene="MYC")                     # what does this gene's difference look like?
plot_volcano(res, stage="both", label=8)       # which genes?
plot_stages(res)                               # what kind of difference?
plot_drivers(res, "MYC", counts)               # should I believe this one?
```

![plot_gene: a global 2× gene, a 15% subset at 8×, and a 5% subset at 8×](https://raw.githubusercontent.com/mkiyer/wade/main/docs/figures/gene.png)

The first two genes have the **same log₂ fold change**. The mean-shift test
cannot tell them apart; the curve, `p_subset` and `affected_fraction` (0.98
against 0.15) can. A flat curve is a global fold change; a curve that sits at
zero and then climbs is a subset.

## Before you run it

Two things are arithmetic on your **design**, and no amount of data or
permutations moves them ([`docs/method.md`](docs/method.md) §10).

- **Resolution.** The grid has `min(n_case, n_ctrl)` points. `affected_fraction`
  is quantitative above roughly 100 per group and only qualitative below 50.
- **Detectability.** If `k` samples carry a signal, no permutation test can
  resolve much below `C(n1, k) / C(n1 + n0, k)`. It is driven by *imbalance*:
  balancing the groups helps far more than adding cases.
  `wade.detectability_floor(n_case, n_ctrl, k)` computes it.

And one honest caveat: stage 2's p-values are conservative by a measured 4–80×
where they are safe, and can be anti-conservative in another regime. Ranking
survives; calling with an FDR loses real subset genes. Stage 1 has an exact
alternative (`stage1="saddlepoint"`); stage 2 is the open problem
([`docs/pvalue-review.md`](docs/pvalue-review.md), [`ISSUES.md`](ISSUES.md)).

## Documentation

- [`docs/manual.md`](docs/manual.md) — **start here**: install, run, read the
  output, the options that matter, what to check before trusting a result.
- [`docs/method.md`](docs/method.md) — what WADE computes and why, with
  every measurement, and what it cannot do.
- [`docs/pvalue-review.md`](docs/pvalue-review.md) — the p-value estimators
  and the open problem, written for a statistician.
- [`notebooks/demo.qmd`](notebooks/demo.qmd) — the demo on synthetic data
  with planted ground truth, the source of the figures above;
  [`notebooks/benchmark.qmd`](notebooks/benchmark.qmd) — head to head against
  COPA, OS, ORT, MOST, LSOSS, the *t*-test, Wilcoxon and waddR.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`ISSUES.md`](ISSUES.md) — for
  developers.

## Status

The statistic, both stages, the descriptors with bootstrap intervals,
permutation inference with tail refinement and BH, Rust kernels for both
permutation loops (bitwise against the NumPy path), the figures, and the
data-in / results-out boundary are implemented and tested: 854 tests in about
20 s. At 20,000 genes, 100 v 100 and 2,000 permutations a full run takes
about 17 s on a 16-core laptop; a 30,000-gene × 80,000-sample cohort
completes in 30 minutes at a 56 GB peak. Every fixture value is re-derived
from NumPy, SciPy and the closed forms by a test that is forbidden from
importing WADE.

## Provenance and license

Extracted from the MCTP cfRNA analysis codebase, where it began as a set of
functions called HITLIB. No patient-derived data is included. GPL-3; see
[`LICENSE`](LICENSE).
