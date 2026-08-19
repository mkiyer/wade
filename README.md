# WADE — Wasserstein Area Differential Expression

A two-group differential-expression test that answers **two** questions instead
of one:

1. **Is there a difference?** — the ordinary question, which any DE method
   answers.
2. **What kind of difference is it?** — a shift affecting every sample, or a
   pronounced change confined to a small subset of one group.

The second question is the point. A gene altered in 5% of cases and a gene
shifted 2× in all of them can produce the same mean difference, and a
first-moment test cannot tell them apart. WADE separates them, and **never asks
you to declare in advance which you are looking for** — there is no percentile
cutoff, no window width, no tuning parameter describing the shape of the effect.

For each gene it compares the two groups' whole quantile functions on a shared
grid of `min(n_case, n_ctrl)` probabilities. Inference is by label permutation,
with a Generalized Pareto fit refining p-values whose empirical resolution has
run out, then BH-FDR.

## Install

```bash
mamba env create -f mamba_env.yaml
conda activate wade
pip install -e . --no-build-isolation
pytest                       # ~25 s
```

The Rust kernel is optional. Without a toolchain the package installs and runs
on the NumPy path, which is the correctness baseline the kernel is validated
against — just slower.

## Use it

```python
import numpy as np
from wade import wade

# counts: genes x samples, RAW counts (see "Why raw counts" below)
# normalizer: a per-gene vector (e.g. gene length) or a genes x samples matrix
# cond: 1 = case, 0 = control, one entry per column
res = wade(counts, normalizer, cond, nperms=2000)
```

### Reading the result

| column | question it answers |
|---|---|
| `mean_shift` | how much did average expression move? (signed) |
| `p_mean_shift`, `padj_mean_shift` | is that shift significant? |
| `p_subset`, `padj_subset` | is a global shift an **inadequate** explanation? |
| `affected_fraction` | what fraction of samples differ? `1.0` = all of them |
| `direction` | `+1` all up, `-1` all down, `0` two-sided |
| `log2_fc`, `w1` | fold change; 1-Wasserstein distance |

Four patterns, read off the two p-values and the two descriptors:

| `p_mean_shift` | `p_subset` | what you are looking at |
|---|---|---|
| significant | — | a **global shift**; an ordinary DE method finds this too |
| significant | significant | a **subset**, strong enough to move the mean. `affected_fraction` says how much of the group, `direction` says which way |
| — | significant | a distributional change with **no net mean shift**: a balanced subset, or a variance change |
| — | — | not differential |

There is deliberately no categorical label and no `interpret()` function.
Turning continuous statistics into classes needs thresholds, which is what this
design exists to avoid.

```python
res.columns()          # dict of equal-length arrays
res.to_polars()        # if polars is installed
```

### Worked example

```python
import numpy as np
from wade import wade

rng = np.random.default_rng(0)
n1 = n0 = 200
cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]

genes = []
genes += [np.r_[rng.lognormal(3, .6, n1), rng.lognormal(3, .6, n0)]        # null
          for _ in range(300)]
genes += [np.r_[rng.lognormal(3, .6, n1) * 2, rng.lognormal(3, .6, n0)]    # global 2x
          for _ in range(20)]
for _ in range(20):                                                         # 5% subset
    case = rng.lognormal(3, .6, n1)
    case[rng.choice(n1, 10, replace=False)] *= 8
    genes.append(np.r_[case, rng.lognormal(3, .6, n0)])

counts = np.array(genes)
res = wade(counts, np.full(counts.shape[0], 4.0), cond, nperms=500)

for name, sl in (("null", slice(0, 300)), ("global", slice(300, 320)),
                 ("subset", slice(320, 340))):
    print(f"{name:<8s} p_mean={np.median(res.p_mean_shift[sl]):.3f}  "
          f"p_subset={np.median(res.p_subset[sl]):.3f}  "
          f"affected={np.median(res.affected_fraction[sl]):.2f}")
```

The planted 5% subset shows the point: the mean-shift test largely misses it
while the subset test finds it, and `affected_fraction` recovers roughly 0.05.
A genuine global shift is the mirror image — found by the mean test, and
correctly **not** flagged as a subset.

### Choosing direction

```python
wade(..., alternative="two-sided")   # default: differences either way
wade(..., alternative="greater")     # only elevation in cases
wade(..., alternative="less")        # only reduction in cases
```

### Why raw counts

The entry point takes **raw counts**, not a normalized matrix. WADE adds a tiny
continuity jitter at *count precision* before dividing, which breaks ties in
zero-heavy data where many samples share a count of zero and the quantile grid
would otherwise degenerate into flat runs. Once counts have been divided by a
normalizer and a library size that perturbation cannot be reconstructed.

`wade_from_matrix(x, cond)` accepts an already-normalized matrix and documents
what it costs: no tie-breaking, and no guaranteed positivity.

Normalizers ship as separate functions — `tpm_like`, `cpm`, `rle` — so the
choice is explicit and swappable.

## Before you run it: two things to check

Both are arithmetic on your **design**, not properties of the software, and
neither is fixable with more data or more permutations.

**Resolution.** The grid has `min(n_case, n_ctrl)` points, so nothing finer than
`1/m` is estimable. `affected_fraction` is quantitative above roughly 100 per
group and only qualitative below about 50.

**The combinatorial floor.** If `k` samples carry a signal, shuffling puts all
of them in one group with probability `C(n1,k)/C(n1+n0,k)` — and **no
permutation test can return a p-value below that.** At 77 cases vs 18 controls
the floor is still 0.032 with 15 affected samples. It is driven by group
*imbalance*: balancing the groups helps far more than adding cases.

[`docs/limits.md`](docs/limits.md) has both in full, plus the checklist of when
to reach for something else.

## Documentation

- [`docs/method.md`](docs/method.md) — what WADE computes and why. The contract.
- [`docs/limits.md`](docs/limits.md) — what it cannot do; read before running.
- [`docs/implementation-notes.md`](docs/implementation-notes.md) — parity with
  the R original, the cross-language traps, the Rust kernel's boundary.
- [`ROADMAP.md`](ROADMAP.md) — what is planned.

## Status

Implemented and tested: the statistic, both stages, the characterization, the
normalizers, permutation inference with GPD refinement and BH, and a Rust kernel
for the permutation loop. **499 tests.**

Ported from an R implementation that remains in `reference/` as the oracle the
golden fixtures were generated from. Worst-case relative deviation across 325
parity comparisons: **9.2e-15**, with the normalized matrix, the quantile grids
and the permutation null bit-for-bit identical.

Not yet built: plotting, file I/O, and a demo notebook. See the roadmap.

## Provenance

Extracted from the MCTP cfRNA analysis repository at commit `828f2f1c`, where it
began as a small set of functions called HITLIB. No patient-derived data is
included; everything here is synthetic. Details in
[`reference/PROVENANCE.md`](reference/PROVENANCE.md).

## License

GPL-3. See [`LICENSE`](LICENSE).
