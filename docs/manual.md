# WADE user manual

How to install WADE, run it, and read what comes back. If you want to know
*why* it works the way it does, read [`method.md`](method.md); this file is
about using it.

---

## Contents

1. [Install](#1-install)
2. [The sixty-second version](#2-the-sixty-second-version)
3. [Your inputs](#3-your-inputs)
4. [Reading the output](#4-reading-the-output)
5. [The options that matter](#5-the-options-that-matter)
6. [Large cohorts](#6-large-cohorts)
7. [Before you trust a result](#7-before-you-trust-a-result)
8. [Diagnostics and figures](#8-diagnostics-and-figures)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Install

WADE is a Python package with a compiled Rust core. Its only hard dependency
is NumPy. The distribution is named `wade-rnaseq` (the name `wade` on PyPI
belongs to an unrelated project); the import name is `wade`.

```bash
pip install wade-rnaseq
```

That is all most people need. The wheels carry the compiled kernel for Linux
(x86-64 and ARM64), macOS (Apple silicon and Intel) and Windows (x64), and one
wheel covers every Python from 3.10 up (GIL-enabled builds; a free-threaded
build compiles from source).

> **Not on PyPI yet.** Until the first upload, install from a wheel built by
> CI (the `wheel-*` artifacts on any green run) or from source, which needs
> a Rust toolchain: `pip install git+https://github.com/mkiyer/wade`.

**Optional extras**, each additive:

| extra | what it adds |
|---|---|
| `wade-rnaseq[io]` | `to_frame()` and `write_results()` (polars) |
| `wade-rnaseq[plot]` | the five figures (plotly and matplotlib; either alone works) |
| `wade-rnaseq[saddlepoint]` | `stage1="saddlepoint"`, the exact stage-1 p-value (SciPy) |
| `wade-rnaseq[all]` | all of the above |

Nothing in the statistic imports any of them. `import wade` on a bare NumPy
install works and stays fast.

On a platform with no wheel, pip builds from source, which needs a Rust
toolchain ([rustup](https://rustup.rs), or `conda install rust`; on Windows
also the MSVC Build Tools). Without the compiled kernel WADE still runs, on
the NumPy path the kernels are checked against, about 80× slower on the
subset test. To confirm you got the fast build:

```python
import wade
assert wade.permutation.HAVE_RUST_KERNEL
```

To develop WADE, see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

---

## 2. The sixty-second version

```python
import numpy as np
from wade import wade

# counts:     genes x samples, RAW integer counts
# normalizer: per-gene vector (e.g. gene length), or 1.0 for CPM
# cond:       1 = case, 0 = control, one entry per column
res = wade(counts, normalizer=1.0, cond=cond, nperms=2000, seed=1)

df = res.to_frame()                    # needs wade-rnaseq[io]
hits = df.filter(df["padj_subset"] < 0.05)
```

Two questions are answered separately and reported separately.

**Is average expression different?** That is `p_mean_shift`, the ordinary
test, the thing any conventional differential-expression method computes.

**Is the difference confined to a subset of samples?** That is `p_subset`, and
it is why WADE exists. A gene elevated in 15% of tumours produces a weak
average signal that ordinary methods miss, and a strong subset signal.

There is no combined p-value and no categorical label, by design. You read
both columns.

---

## 3. Your inputs

### Counts

**Raw counts, genes by samples.** Not TPM, not CPM, not log anything. WADE
normalizes internally because two of its steps need count precision: the
tie-breaking jitter, and a subset stage whose null is built by binomial
thinning of reads. Handing it a pre-normalized matrix silently breaks both.

Accepted: a NumPy array, a polars or pandas DataFrame, a SciPy sparse matrix,
or a `wade.Counts`. **WADE reads no files.** Your reader reads them, which
keeps file formats out of the statistic. From a frame, the first column is
the gene id when it is non-numeric (or the pandas index, when it is
labelled), and every other numeric column is a sample; a non-numeric
leftover raises. For a transposed matrix, an integer id column, or a
featureCounts layout with annotation columns in the middle, call
`wade.as_counts(obj, genes="columns")`, `as_counts(obj, id_column="Geneid")`
or `as_counts(obj, sample_columns=[...])` yourself first.

WADE is for **discrete count data**. Continuous input is not a target and is
neither tested nor tuned for.

### The normalizer

A per-gene vector, a full genes-by-samples matrix, or a scalar.

* Gene length gives TPM-like values.
* `normalizer=1.0` gives CPM.
* Other size factors go in through `lib_sizes=`, not here.

### The condition vector

`1` for case, `0` for control, one entry per column of the counts. If your
assignment lives in a sample metadata table:

```python
cond = wade.condition(sample_meta, key="sample_id", column="group",
                      case="tumour", control="normal")
```

Alignment is strict and names the offenders rather than silently dropping
them. `wade.wade_contrast(counts, normalizer, case_samples, ctrl_samples)`
takes two groups by name or index instead.

---

## 4. Reading the output

`res.to_frame()` gives one row per gene. The columns that matter:

| column | meaning |
|---|---|
| `mean_shift` | the signed area between the quantile functions, which is the difference of group means |
| `p_mean_shift`, `padj_mean_shift` | stage 1: is average expression different |
| `subset_stat` | the scan statistic over the standardized bridge |
| `p_subset`, `padj_subset` | stage 2: is the difference confined to a subset |
| `affected_fraction` | the effective fraction of the case group that differs. `1.0` is a global change, `0.05` a 5% subset |
| `direction` | `+1` all up, `-1` all down, `0` two-sided |
| `subset_log2_fc` | how large the change is *within* the affected subset |
| `log2_fc` | the global fold change |
| `z_mean_shift`, `z_subset` | ranking scores, see below |
| `nexc_*`, `refined_*` | how many null draws reached the observed statistic, and whether the p-value came from the tail model rather than that count |

Benjamini-Hochberg is applied **within each stage separately**. There is no
multiplicity correction across the two stages, because they answer different
questions and you should decide which one you are asking.

### The z columns are for ranking, not for calling

`z_mean_shift` and `z_subset` are the observed statistic standardized against
that gene's own null, the analogue of a normalized enrichment score. On large
cohorts thousands of genes tie at the p-value floor while their separations
from the null differ by orders of magnitude, and the z keeps ordering them.

**They are not calibrated tail probabilities.** Do not push them through a
normal cumulative distribution function. The null of a maximum statistic is
not normal.

### Writing results out

```python
wade.write_results(res, "results.tsv")     # plus a JSON manifest beside it
```

The manifest records every parameter, the realized grid size, the permutation
count and the version, so a result can be traced back to the run that made it.

---

## 5. The options that matter

Most defaults are right. These five are worth a decision.

### `nperms` (default 2000)

Sets the resolution of every p-value. The empirical p-value cannot go below
`1/(nperms+1)`, and the tail refinement extends that to `1/(nperms*n_tail)`,
which is 2e-6 at the defaults. On a 20,000-gene run Benjamini-Hochberg decides
near 2.5e-6, so 2,000 permutations is the sensible floor rather than a
generous choice. Raising it costs linearly.

### `stage1="saddlepoint"` (opt-in, needs SciPy)

Replaces stage 1's permutation p-value with an exact one computed in closed
form. No permutations, no tail extrapolation, and the floor drops from 2e-6 to
`1/C(n, n1)`, which is around 1e-23 for 40 against 40. Validated against 1e8
brute-force permutations at median accuracy 0.91 to 1.15 where the default
path is 3 to 4,906 times conservative.

Costs about 60 times the stage-1 permutation loop it replaces, which is 0.7
minutes at 40 against 40 and 50 minutes at 3,000 against 3,000, for 20,000
genes. Works at **any** group balance. It changes reported numbers, so it is
opt-in and named. Use it when stage 1 is what you care about and your cohort
is under about a thousand per group.

### `alternative` (default `"two-sided"`)

`"greater"` or `"less"` for a one-sided test. Applies to stage 1. Stage 2 is
always upper-tailed, because its statistic is a maximum.

### `strata`

Restricts permutation to within strata. **This is the right null when your
cohort is assembled from several studies, batches or protocols**, because
shuffling labels across them tests an exchangeability the design does not
have. It costs resolution, and the cost is arithmetic rather than a matter of
opinion:

```python
wade.permutation_space(cond, strata=batch)   # the realized space and its floor
```

Any stratum holding a single class contributes a factor of one, so a badly
stratified design can collapse the space entirely. Check before running.

### `seed`

Set it. The jitter, the permutations, the thinning and the bootstrap all
derive from it, so a seed reproduces a result exactly. `seed=None` draws from
ambient state and is not reproducible.

---

## 6. Large cohorts

WADE is built for cohorts far larger than a typical differential-expression
run. Measured on a 16-core laptop, 2,000 permutations throughout:

| workload | time |
|---|---|
| 2,000 genes, 120 samples | 0.5 s |
| 20,000 genes, 200 samples | 8 s |
| 20,000 genes, 2,000 samples | 88 s |
| 30,000 genes, 80,000 samples | 30 min |

**Memory is the binding constraint, not time.** Measured peaks, from the
installed wheel:

| genes | samples | count matrix | peak |
|---|---|---|---|
| 5,000 | 2,000 | 0.08 GB | 1.8 GB |
| 20,000 | 500 | 0.08 GB | 2.6 GB |
| 20,000 | 2,000 | 0.32 GB | 6.0 GB |
| 30,000 | 80,000 | 19.2 GB | 56 GB |

The ratio is dominated by fixed working space on small inputs and settles near
**three times the count matrix** on large ones, which is the number to budget
with. The result you get back is about 1.1 times the input.

Two things that are *not* true and are worth knowing:

* **The permutation count is nearly free in memory.** Going from 500 to 2,000
  permutations moved the peak by 2%. Raise it if you want resolution.
* **`gene_chunk` is not the lever it looks like.** At 20,000 by 2,000 it buys
  about 8%, because the peak lives in stage 2's working space rather than in
  the matrices it chunks. It is still exactly bit-identical and still worth
  using, just not the answer to an out-of-memory error.

If you are tight on memory, `subset=False` is the real lever: it drops the
peak by a third, at the cost of the stage that makes WADE worth running.

Three knobs:

* **`gene_chunk`** processes genes in blocks. Results are **bit-identical** to
  the one-pass run, asserted rather than assumed, so this is purely a memory
  layout choice and never a numerical one.
* **`max_probs`** (default 2,000) caps the quantile grid. This is the one
  default that changes an answer, and it is recorded in the result and the
  manifest. Above the cap, `affected_fraction` resolves to `1/max_probs`; the
  rule is `m >= 2.5/pi_min` for the smallest subset fraction you care about
  ([`method.md`](method.md) §1).
* **`fit_backend="rust"`** moves the fold-change fit into the kernel, several
  times faster. Opt-in because the two backends draw their thinning
  differently and fit slightly different fold changes for the same seed.

On a balanced design `stage1="gemm"` computes stage 1 as one matrix product,
which is much faster at large `n`; it changes `mean_shift` from the grid
quadrature to the exact mean difference, so it is opt-in too. Both permutation
loops are already compiled and threaded. You do not need to opt into that.

---

## 7. Before you trust a result

Read [`method.md`](method.md) §10. Two things decide whether WADE can answer
your question at all, and neither is about the software.

### The detectability floor

If `k` samples carry a signal, label shuffling puts all of them in the case
group with probability `C(n1,k)/C(n1+n0,k)`. **That is the scale of the
smallest p-value your design can resolve**, whatever the effect size and
whatever the method. Check it first:

```python
wade.detectability_floor(n_case=100, n_ctrl=10, k=10)     # 0.37
```

At 100 cases against 10 controls, a 10% subset cannot reach significance no
matter how strong the effect. Stage 1 is fine at that geometry; stage 2 needs
roughly half the cases to share the change. This is the single most useful
thing to compute before a run.

It is a **scale, not a hard bound**. Roughly 21 to 42 percent of genes fall
below it in real data, so never use it to clamp a p-value.

**Worked example.** 3,000 genes, 50 cases against 50 controls, 60 genes given
a 6× change in `k` of the cases, 2,000 permutations, Benjamini-Hochberg at
0.05. Only `k` changes between rows:

| affected cases | floor | subset genes found | false positives | `affected_fraction` |
|---|---|---|---|---|
| 8 of 50 | 2.9e-3 | **0** of 60 | 0 | 0.18 (true 0.16) |
| 15 of 50 | 8.9e-6 | 50 of 60 | 0 | 0.32 (true 0.30) |
| 25 of 50 | 5.2e-10 | 59 of 60 | 0 | 0.53 (true 0.50) |

The top row finds nothing, and the software is not at fault: at 8 affected
cases no relabelling test can produce a p-value small enough to survive
correction across 3,000 genes. The genes are still *ranked* correctly, they
just cannot be *called*. Notice also that `affected_fraction` recovers the
planted fraction in every row, including the one where nothing is
significant, and that there are no false positives anywhere.

### Stage 2's p-values are conservative

This is the honest caveat and it is worth stating plainly. Stage 1 has an
exact answer available. Stage 2's tail model is 4 to 80 times conservative
where it is safe, and can be anti-conservative in the other regime, with no
way to tell at 2,000 permutations which regime a given gene is in.

**For ranking genes this does not matter**, because the ordering survives. For
calling with a false discovery rate it costs you real subset genes, which is
the direction that loses discoveries rather than manufacturing them. Treat a
`p_subset` near your threshold as a lower bound on the evidence. The full
account is in [`pvalue-review.md`](pvalue-review.md).

### Checklist: when not to use WADE

Work down this list before running. Any "yes" is a reason to stop.

1. **Is the smaller group under 10?** The grid has fewer than 10 points. Stop.
2. **Is the detectability floor above your threshold** for the smallest
   subset you care about? Nothing in this family can find it. Stop, or
   rebalance.
3. **Do you have a covariate or batch effect** that tracks the contrast?
   Detect and refuse, or stratify with `strata=`.
4. **Do you have repeated measures?** Aggregate to one value per subject
   first; `strata=` does not solve this.
5. **Do you need a fraction estimate** and have fewer than ~50 per group?
   `affected_fraction` will separate global from concentrated but will not
   tell you 2% from 5%.
6. **Is a large share of your genes strongly differential?** The
   characterization will be distorted by composition. The subset test's
   significance is not.
7. **Do you actually have a location shift?** If the question is "is the
   mean different", an ordinary DE method answers it and is better understood.
   WADE's value is the second question, not the first.

---

## 8. Diagnostics and figures

```python
res.gene_detail("MYC")            # the quantile curves the statistics came from
wade.subset_drivers(res, "MYC")   # which samples drive the subset call
wade.library_qc(counts)           # per-library depth, complexity, concentration
```

`subset_drivers` earns its place: on real data a handful of anomalous
libraries can top the subset ranking across thousands of genes. **Check the
drivers before believing a hit.** If one gene's subset is the same three
libraries as everyone else's, you have found a technical artefact.

### Figures

Figures need `wade-rnaseq[plot]`. Four of them, each answering one question, and
a fifth that wires two of the views together in a live notebook:

| function | the question it answers |
|---|---|
| `plot_gene(res, gene="MYC")` | *What does this gene's difference look like?* The log-ratio curve, with the two quantile functions beneath it. A flat curve is a global fold change; a curve at zero that climbs is a subset. The dashed line is the fitted global shift the subset test argues against. |
| `plot_volcano(res, stage="subset")` | *Which genes?* Effect size against significance, coloured by `affected_fraction`. `stage="both"` puts the two stages side by side. |
| `plot_stages(res)` | *What kind of difference?* Stage 1 against stage 2: the reading table of §2, drawn, with a count per quadrant. |
| `plot_drivers(res, "MYC", counts)` | *Should I believe this one?* The driving samples against the cohort's median and interquartile range: the gene's share of each library, complexity, depth. Needs the raw counts passed back, because the result does not carry them. |
| `plot_linked(res, "subset", label=8)` | The volcano and the gene panel in one widget: click a point. Needs a live Jupyter kernel and does not survive export to HTML. |

Every figure takes `backend="plotly"` (interactive, the default when
installed) or `"matplotlib"` (static, for a manuscript), `theme="light"` /
`"dark"` / `"high-contrast"`, `title=`, `width=` and `height=`. Either axis of
a cloud takes any result column, so on a large cohort where p-values saturate:

```python
wade.plot_volcano(res, stage="subset", x="subset_log2_fc", y="z_subset", label=8)
```

`label=n` names the top *n* genes (placed so they do not overlap);
`label=["MYC", ...]` names specific ones. `meta="gene_name"` names points by a
column carried from the input frame instead of the gene id. `n_boot > 0`
draws the bootstrap intervals as error bars on the labelled points.

The figures are a convenience layer: every number a figure draws is in
`res.columns()` or in `wade.subset_drivers` / `wade.library_qc`, and every
figure's arrays are available as a table (`wade.plotting.volcano_data(res,
"subset").table()`), so exporting and plotting in another tool loses you
nothing.

---

## 9. Troubleshooting

**`ImportError: stage1='saddlepoint' needs SciPy`** — install SciPy, or drop
the option. It is the only path that needs it.

**A vector normalizer is rejected.** WADE checks the length rather than
recycling. A per-sample vector where a per-gene vector belongs is the one
input error that would otherwise produce plausible wrong numbers.

**`wade.permutation.HAVE_RUST_KERNEL` is `False`.** You are on the NumPy
fallback, correct but about 80 times slower on the subset test. Install a
published wheel, or build from source with a Rust toolchain on the path.

**Everything is significant.** On large cohorts significance saturates, and it
is not a bug: with tens of thousands of samples a tiny real difference is
detectable. Rank by `z_subset` and `subset_log2_fc` rather than thresholding
on the p-value, and check the drivers.

**Nothing is significant.** Compute `detectability_floor` for your geometry
before concluding anything about the biology. The design may not admit the
answer.

**Results changed between runs.** Set `seed`. Without it the jitter and
permutations come from ambient state.

---

## Where to read more

| document | what it covers | for whom |
|---|---|---|
| [`method.md`](method.md) | the statistic, both stages, normalization, why each choice, and what it cannot do (§10, **read before your first run**) | anyone evaluating WADE |
| [`pvalue-review.md`](pvalue-review.md) | the p-value estimators and the open problem | statisticians |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | building, testing, layout, and the traps | developers |
| [`../ISSUES.md`](../ISSUES.md) | the work queue | everyone |
