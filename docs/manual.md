# WADE 0.2.0 manual

For every gene WADE reports two things and tests each:

- the **area** between cases and controls: the difference in their means, in
  reads per sample, with its fraction and its fold;
- the **subset** of cases that departs from the one fold every case would
  share.

This manual is the user reference. The [method](method.md) gives exact
definitions; [validation](validation.md) gives the measured performance.

## Running WADE on a dataset

```python
import wade

data = wade.as_counts(frame, id_column="gene_id", sample_columns=sample_ids)     # raw counts
cond = wade.condition(sample_sheet, key="sample_id", column="group", case="tumour", control="normal")

qc = wade.library_qc(data)                                        # 1. check the libraries
norm = wade.fit_normalization(data)                               # 2. one size factor per sample (RLE)
# norm = wade.fit_normalization(data, reference_genes=controls)   #    or RLE on internal control genes
keep = wade.filter_counts(data, min_count=3, min_samples=2)       # 3. the tested genes
result = wade.wade(data, cond, normalization=norm, test_genes=keep, direction="up")   # 4. both tests
table = result.to_frame().sort("area_padj")
```

1. **Check the libraries.** Decide technical exclusions from depth and
   detection before testing ([before running](#before-running)).
2. **Fit one size factor per sample** with the normalizer that suits the data
   ([size factors](#size-factors)). WADE uses the size factors twice:
   - it divides counts by them for the effects;
   - before both tests, it thins every sample until all size factors equal
     the smallest ([depth equalization](#depth-equalization)).
3. **Choose the tested genes** with a filter that ignores the labels. The
   filter never refits the size factors.
4. **Run both tests,** with the settings chosen in advance
   ([settings](#settings)).
   - Rank markers among the genes with a small `area_padj` by `area`,
     `log2fc` or `area_fraction`.
   - Read subsets where their test is validated
     ([validated range](#the-subset-tests-validated-range)), and rank them
     by `subset_rarity` ([rarity](#rarity-ranking-the-subsets)).

The [demo notebook](../notebooks/demo.qmd) runs these steps on simulated
counts, with RLE on all genes and with RLE on internal control genes.

## Settings

`wade.wade(counts, cond, *, ...)` takes these settings. Choose them before
looking at any result.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `counts` | | Raw counts, genes × samples ([input](#input)) |
| `cond` | | One label per sample, 1 for a case and 0 for a control, or a `Condition` from `wade.condition()` |
| `direction` | `"both"` | The alternative of both tests: `"both"`, `"up"` or `"down"` ([direction](#direction)) |
| `normalization` | `"rle"` | `"rle"`, `"library"`, or a fitted `Normalization` ([size factors](#size-factors)) |
| `test_genes` | `None` | Gene IDs or a Boolean mask of the tested genes. `None` tests every gene with reads |
| `pvalues` | `"auto"` | `"auto"` computes the area test by a saddlepoint and shuffles where its error estimate exceeds `relative_error`; `"permutation"` shuffles every p-value ([p-values](#p-values)) |
| `fdr` | `0.05` | The false discovery rate at which BH decides. Shuffles stop once each gene's decision is settled. `None` finishes every run |
| `relative_error` | `0.2` | The precision of every p-value: a shuffled gene stops at its ⌈1/relative_error²⌉-th exceedance |
| `max_permutations` | `999_999` | The shuffle budget. The smallest shuffled p-value is 1/(`max_permutations` + 1) |
| `seed` | `1` | Seeds the jitter, the thinning and the shuffles: an integer, a sequence of integers, or `None` for a fresh run. `result.params["seed"]` records it |
| `strata` | `None` | One label per sample (study, site, batch), or a list of such factors ([strata](#strata-study-site-or-batch)) |
| `gene_names`, `sample_names` | `None` | Labels for an array input |
| `gene_chunk` | automatic | How many genes are held at once: memory, never the answer |

`wade.wade_contrast(counts, case_ids, control_ids, **settings)` selects the
two groups by sample ID and runs `wade()` on them.

- A fitted `Normalization` keeps its factors; a method name fits on the
  selected samples.
- `strata=` gives one label per sample of `counts`, in its order.

## Before running

Fix the contrast, the biological population and the independent unit before
looking at results.

- Keep one observation per unit: one library per patient and sample class.
  Cells or spots from one donor are not independent patients.
- Decide technical sample exclusions in advance, and never remove cases
  because they weaken a result.
- Expression-derived labels (PAM50, RNA-called fusions) can select on the
  outcome, and unknown annotations are not negatives.

**Check the libraries first.** Every sample is thinned until its size factor
equals the smallest, so one failed or very shallow library costs every gene
power.

- `wade.library_qc(counts, top_n=10)` returns each library's raw depth
  (`depth`), detected genes (`n_detected`, `detected_fraction`) and the
  share of reads held by its top genes (`top_share`).
- Remove failed libraries by a rule set before testing, one that does not
  look at the results.
- After a run, `result.params["depth_equalized"]` is the share of its reads
  the median sample kept, and `result.params["size_factor_median"]` is the
  median size factor in each group.

## Input

```python
import wade

# NumPy arrays have genes in rows and samples in columns.
result = wade.wade(counts, [1, 1, 1, 0, 0, 0], gene_names=gene_ids, sample_names=sample_ids)

# Frames and sample sheets are aligned by ID, strictly.
data = wade.as_counts(frame, id_column="gene_id", sample_columns=sample_ids)
cond = wade.condition(sample_sheet, key="sample_id", column="group", case="tumour", control="normal")
result = wade.wade(data, cond)
```

- Raw integer gene counts are the validated input. Negative or missing
  values are errors that name the cell.
- Non-integer counts (Salmon or RSEM estimates) are accepted. Both tests
  round them to whole reads; the effects use them as given.
- At least two samples per group are required. This is an input guard, not a
  claim of adequate power.
- Samples must be independent, or exchangeable within strata
  ([strata](#strata-study-site-or-batch)). Pairing and repeated measures are
  not validated yet.
- **Arrays** have genes in rows; `as_counts(..., genes="columns")`
  transposes. Sparse matrices are accepted through `toarray()` and
  materialized.
- **Frames.** `id_column` names the gene ID column, and `sample_columns` the
  columns that hold counts. An integer ID column (Entrez IDs) is numeric and
  must be named, or it is read as a sample.
- **Sample sheets.** `wade.condition()` reads the labels by sample name. A
  third level in `column` raises unless `drop_other=True`; a sample with no
  label, or a label with no sample, raises.

## Strata: study, site or batch

When samples come from several studies, sites or batches, pass them as
strata. Labels are then exchanged only within each stratum.

- The area test is exact under the null that labels are exchangeable within
  strata, whatever differs between them: depth, a gene-specific batch scale,
  composition.
- The subset test is as exact as without strata: approximate, through its
  estimated fold.

```python
result = wade.wade(counts, cond, strata=meta["study"])                 # one factor
result = wade.wade(counts, cond, strata=[meta["study"], meta["sex"]])  # combinations of factors
cond = wade.condition(meta, key="library", column="group", case="tumor",
                      control="normal", strata="study")                # read by sample name
result = wade.wade(counts, cond)
```

- **Why.** When a study effect lines up with the condition (one study holds
  most of the cases), unstratified tests call the study effect. TCGA and GTEx
  prostate normals, relabelled at random with case shares that differ by
  study, were called at p ≤ 0.01 in 42.5% of genes by the unstratified area
  test; stratified, it held its level
  ([validation](validation.md#strata)).
- **What changes.**
  - Each stratum is thinned to its own smallest size factor.
  - The area test sums within-stratum differences with Cochran–Mantel–Haenszel
    weights.
  - The subset test takes one fold out of every case and moves each stratum
    to the cohort's level.
  - `case_mean` and `control_mean` become Mantel–Haenszel standardized means,
    so the area is still their difference.
- **Strata with one group.** A stratum holding only cases or only controls
  cannot move a label. Its samples are left out of the tests and the
  effects, with a warning. A gene with reads only in those samples is not
  tested (`status == "all_zero"`). If no stratum holds both groups, `wade()`
  raises an error.
- **Missing labels.** A missing stratum label raises, however the strata are
  given.
- **The cost.** The effective size falls from n₁n₀/n to Σ n₁ₛn₀ₛ/nₛ.
  `result.params["strata"]` records each stratum's cases and controls, the
  samples left out and both effective sizes.
- **What strata cannot do.**
  - They cannot separate a study effect from a condition that coincides with
    it.
  - They cannot bring in external controls: GTEx normals, stratified by
    study, form a stratum with no cases and drop out. Pooling them as
    controls assumes the studies' normals are exchangeable. TCGA
    normal-adjacent prostate and GTEx prostate differ in 78% of genes by the
    area test ([validation](validation.md#strata)).

## Size factors

Every normalization ends as one size factor per sample: the sample's depth as
the normalizer measures it. WADE uses the size factors twice.

- **For the effects** it divides each count by its sample's size factor, so
  `case_mean` and `control_mean` are reads per sample at the cohort's typical
  depth.
- **Before both tests** it thins every sample until all size factors equal
  the smallest ([depth equalization](#depth-equalization)).

Raw library totals enter only if you choose them (`normalization="library"`).
Fit once, then choose the tested genes without refitting:

```python
norm = wade.fit_normalization(data)                              # RLE on all genes
norm = wade.fit_normalization(data, reference_genes=controls)    # RLE on internal control genes
norm = wade.Normalization.from_size_factors(data, factors, source="spike-ins")   # fitted elsewhere
keep = wade.filter_counts(data, min_count=3, min_samples=2)
result = wade.wade(data, cond, normalization=norm, test_genes=keep)
```

| Situation | Use |
| --- | --- |
| Typical bulk gene counts | The default RLE: DESeq2's median-of-ratios on the genes with reads in every sample |
| Internal control genes (housekeeping genes, spike-ins) known not to change | `fit_normalization(..., reference_genes=ids)` |
| Size factors fitted elsewhere | `Normalization.from_size_factors(data, factors, source=...)`: complete positive exposures, depth included |
| Total counts are the intended reference | `normalization="library"` |
| Per-sample abundance for reporting | `wade.cpm(counts)`, `wade.tpm(counts, lengths)`; not inputs to the test |

**Internal control genes.** `reference_genes=` fits RLE on the listed genes
alone.

- Each sample's size factor is the median, over the controls, of its count's
  ratio to that gene's geometric mean across the samples.
- Only controls with reads in every sample enter; `norm.used_reference_genes`
  lists them.
- The controls must not change between the groups. A low-variance panel is
  not automatically a valid one.
- A handful of controls gives noisier size factors than hundreds, so use as
  many validated controls as the assay has.
- The thinning then equalizes the depth the controls measure.

**When many genes change.** RLE on all genes assumes most genes do not change.
When many genes shift one way, the size factors absorb part of that shift, and
unchanged genes look changed.

- With 10% of genes doubled in one group, the area test called 19.6% of
  unchanged genes at p ≤ 0.05
  ([validation](validation.md#normalization-when-many-genes-change)).
- This holds for any DE tool that uses size factors.
- Balanced changes, as many genes up as down, are harmless.
- When a global shift is expected, normalize by internal control genes or
  external factors.

**Filtering.** `wade.filter_counts(counts, min_count=1, min_samples=1,
min_total=0)` keeps the genes with at least `min_count` reads in at least
`min_samples` samples and at least `min_total` reads in all. A filter must be
label-independent: never choose thresholds by group differences or q-values.
Genes outside the filter keep their descriptions but have no test
(`status == "filtered"`).

## What WADE reports

**The area** is the signed area between the case and control quantile
functions of normalized counts. It is exactly the difference in group means:
`area = case_mean − control_mean`, in reads per sample at the cohort's
typical depth. It scales with expression, so it ranks genes by how many reads
separate the groups.

Two more views describe the same difference:

- **The area fraction** divides the area by the total area under both
  curves: (case_mean − control_mean)/(case_mean + control_mean). It lies
  between −1 and 1 and reads the same at every expression level: 1/3 is a
  doubling, −1/3 a halving, and 1 a gene with no reads in the controls.
- **The fold**, `log2fc`, is log2(case_mean / control_mean), the ratio of the
  same two means. It needs no pseudocount: +1 is a doubling, and a gene with
  no reads in the controls has `log2fc = +inf`, one with none in the cases
  `-inf`.

The three views are one difference, and `area_pvalue` and `area_padj` test all
three.

**The subset** asks whether the cases are more unequal than the controls.

- If every case carried one common fold of the controls' distribution, the
  two groups' Lorenz curves of count + 1 would coincide.
- WADE finds where the cases' curve falls furthest below the controls'
  (`lorenz_gap`). That point splits the cases into those that carry less than
  their share and those that carry more.
- The smaller side, which departs further per case, is the subset: above the
  rest of the cases (`subset_side == "up"`) or below them (`"down"`).
  `subset_fraction` is its share of the cases.
- `alignment_log2fc` is the common fold: the Wasserstein-1 fold that aligns
  the cases with the controls. `subset_log2fc` is the subset's fold over what
  the common fold predicts for it.
- `subset_pvalue` and `subset_padj` test the null that every case carries
  one common fold.
- A gene whose cases are no more unequal than its controls has no subset:
  `subset_side` is `"none"`, `subset_fraction` 0, `subset_log2fc` NaN and
  `subset_pvalue` 1. A gene with no reads has no subset either.
- The subset is read on the jittered normalized counts ([jitter](#jitter));
  only its test reads the thinned counts.

Any combination can occur:

- a raised area with a subset further up: an amplified subgroup among raised
  tumours;
- a lowered area with a subset raised above the rest: a fusion-driven
  subgroup in tumours that otherwise lose the gene;
- no area with a strong subset: a gene switched on in a tenth of tumours.

No subset size is chosen in advance, and noise also has a largest gap, so read
the subset with its q-value. One gene reports one subset; a raised and a
lowered subset in the same gene share one split.

## Rarity: ranking the subsets

In a large, heterogeneous cohort nearly every gene's subset is significant:
tumour purity and subtype alone make tumours more unequal than normals. Nor
do the q-values rank the strongest subsets:

- With `fdr` set, discoveries report a bound.
- A labelling that keeps all of a subset's members among the cases
  reproduces its split, so its p-value cannot fall far below the share of
  such labellings: 26.1% for 12 tumours with 52 controls. A subset that
  small can be among the rarest and still not be called
  ([validation](validation.md#rarity)).

`subset_rarity` ranks the subsets instead.

- **What it is.** The share of genes whose subset is at least as rare.
  - Among the genes of its group, take the share whose Lorenz gap at this
    gene's cut is at least this gene's gap. At that cut, every gene's subset
    has the same size and side.
  - Rank that share across the group.
  - In TCGA prostate, ETV4's subset ranks first and ETV1's third in the
    validated range ([validation](validation.md#rarity)).
- **Why size-matched.** A gap can be no larger than the share of the cases
  below its cut: a lowered subset's own share, or a raised subset's
  remainder. So gaps of subsets of different sizes and sides do not compare.
- **Groups.** Genes in the subset test's validated range (the lower group
  averages at least 32 reads per sample) are ranked against each other, and
  genes below it against each other. Low-count genes' gaps run larger and
  would otherwise crowd the top. `result.params["rarity_groups"]` counts
  each group.
- **Special values.** A gene without a subset has rarity 1; an untested gene
  has NaN.
- **What it is not.** Rarity describes and ranks, relative to this cohort's
  genes; it is not a test. Never select genes by it before BH. Ordering BH's
  discoveries by it is fine.

```python
rare = result.rare_subsets(n=25)                    # the 25 rarest subsets in the validated range
low = result.rare_subsets(n=25, validated=False)    # the 25 rarest below it
load = result.sample_load(drivers=5)                # each case's load and its five heaviest genes
```

**Rare subsets.** `rare_subsets(n=25, validated=True)` lists the rarest
subsets, each with its members and with the other listed genes that share
members with it. `validated=False` lists the genes below the validated
range. It returns a dict of parallel lists, rarest first:

| Key | Value |
| --- | --- |
| `gene` | The gene |
| `rarity` | Its `subset_rarity` |
| `side` | `"up"` or `"down"` |
| `size` | The number of cases in the subset |
| `members` | The subset's cases, furthest out first |
| `shared` | The other listed genes that share members with it, as (gene, cases in common), most first |

- Genes whose members coincide are one programme: another tissue, a subtype
  or an artefact. In TCGA prostate, the seminal-vesicle genes SEMG1 and
  SEMG2 share 6 tumours.
- A driver can share members too, so read how many are shared: ETV1's subset
  holds 8 of ISG15's tumours.

**Sample load.** `sample_load(drivers=5)` asks which cases carry more than
their share of the rare subsets. It returns a dict over the cases:

| Key | Value |
| --- | --- |
| `sample` | The case |
| `load` | Its load; the loads average exactly 1 |
| `subsets` | The number of subsets it belongs to |
| `drivers` | Its `drivers` heaviest genes, most weight first |

- **The weights.** Every tested gene with a subset, in either rarity group,
  spreads a weight, −log(rarity), over its members. A singleton gives all of
  it to one case; a subset of 40% of the cases gives a little to each.
- **The scale.** A case's load is the weight it receives over what it would
  receive if subset members were drawn at random, so the loads average
  exactly 1.
- **Reading it.**
  - In TCGA prostate the heaviest libraries carry another tissue's or cell
    type's programme: skeletal muscle, gut defensins, immunoglobulins.
  - A light contamination may not stand out. A breast library mixed into a
    prostate tumour as 5% of its reads raised that tumour's load, but not to
    the top ([validation](validation.md#rarity)).
  - A subtype carries many co-regulated genes too, so read the drivers
    before acting.
  - The load is a diagnostic, not a test: it flags nothing and removes
    nothing.

With strata, both read the cases in the strata that hold both groups.

## Direction

```python
wade.wade(data, cond)                    # both tests two-sided
wade.wade(data, cond, direction="up")    # is the gene higher in cases, as a whole or in a subset?
```

`direction` sets the alternative of both tests.

- `"both"` (default) tests the area two-sided and counts subsets on either
  side.
- `"up"` asks whether cases are higher: a larger area, and a subset above
  the rest of the cases. For markers that should rise in cases, this is the
  declared question.
- `"down"` asks the reverse.

A one-sided subset test compares the observed subset only with chance subsets
on its own side. Declare the direction before looking; do not run `"both"` and
report one side as if it had been declared.

## Depth equalization

Read counts carry Poisson noise that shrinks, relative to the mean, as depth
grows. Samples at different depths are therefore not exchangeable at low
counts, even after normalization: a group sequenced half as deep looks
different even where nothing differs.

Before both tests, WADE thins every sample binomially until all size factors
equal the smallest. Each read of sample j is kept with chance min(s)/s_j,
where s are the size factors of the normalization you fitted. Counts are
rounded to whole reads first.

- The thinned samples share one size factor. When the size factors are
  right, the area test is exact at every expression level.
- Genes with too few reads to trust thin to mostly zeros and are washed out
  rather than called. Well-expressed genes lose little.
- The thinning follows the normalizer: with internal control genes, it
  equalizes the depth the controls measure.
- The subset test then takes the common fold out of the higher group by
  thinning it further. It draws the thinning four times and scores each
  shuffle by the mean of the four walks (a walk scans the sorted cases for
  the largest gap between the two groups' Lorenz curves), so that which
  reads one draw keeps does not decide a call. This costs four walks per shuffle.
- With strata, each stratum is thinned to its own smallest size factor.
- The reported means, areas and folds use every read, and the subset's
  description reads the jittered normalized counts. Only the tests use the
  thinned counts.

**The seed.** The thinning is drawn from streams seeded like the jitter and
the shuffles, so a run is reproducible from its seed.

- Calls near a threshold move between seeds. The subset test's move most,
  because which reads its thinning keeps can decide a call near the
  threshold. On TCGA prostate, among genes at 128 reads and above, two seeds
  agreed on 90% of the subset calls either made
  ([validation](validation.md#the-subset-test-under-the-count-model)).
- A subset's side, share, fold and rarity do not depend on the thinning.
  They move with the seed only through the jitter, and slightly: rarity's
  Spearman correlation between two seeds on TCGA prostate was 1.000
  ([validation](validation.md#rarity)).
- Rank subsets by rarity, and read a call near the threshold as one draw.

## P-values

- **The subset test is shuffled.** Label shuffles preserve group sizes, and
  every gene, in both tests, sees one seeded stream of them. Each gene follows
  Besag and Clifford's rule: it stops at its ⌈1/relative_error²⌉-th exceedance
  (25 at the default `relative_error=0.2`), whose count estimates p to that
  relative standard error, or at `max_permutations` shuffles.
- **The area test is computed** with `pvalues="auto"` (default).
  - The exact permutation distribution of the cases' total has a
    double-saddlepoint tail, with no shuffles and no floor above the chance
    of the single most extreme labelling.
  - Its error is estimated from the saddlepoint's own next term and from how
    coarse the exact distribution is there.
  - A gene whose estimate exceeds `relative_error` is shuffled instead. In a
    small design this shuffles the genes far enough in the tail that only a
    handful of labellings reach them.
- **`pvalues="permutation"`** shuffles every p-value, including the area
  test's.

**Shuffles stop at the decision.** `fdr` (default 0.05) is the false
discovery rate at which BH decides; choose it before looking.

- **How it works.** Each gene's run has a bound on the p-value it will finish
  with, one that holds whatever its remaining shuffles show. Once that bound
  falls below BH's threshold, the gene stops.
- **The guarantee.** BH then rejects exactly the genes it would reject had
  every run finished, on every run, so the false discovery rate is the same
  (Fischer, Barry and Ramdas, *Electronic Journal of Statistics*, 2026).
- **The saving.** A gene BH rejects needs only about
  25 / threshold shuffles, instead of running to `max_permutations`
  ([validation](validation.md#decision-based-stopping)).
- **What is reported.**
  - The genes BH rejects report their bound as the p-value, at most the
    threshold, and a q-value of at most `fdr`. `result.bounded` marks them.
  - Every other gene's p-value and q-value is exactly the finished one.
  - Deciding at `fdr` or any higher level gives the same decisions. A
    stricter level needs a new run.
  - Rank discoveries by their effects, not by their p-values.
- **`fdr=None`** finishes every run, for when exact small p-values matter.

**The budget.** `max_permutations=999_999` lets a shuffled p-value reach
10⁻⁶, so an isolated BH discovery at q ≤ 0.05 stays possible among up to
50,000 genes. The manifest's `minimum_discovery_rank` says how many genes
must sit at the shuffled floor before any can pass at `write_results`'
`alpha`. Only genes near BH's
threshold, or a lone discovery, need the budget's depth. A smaller
`relative_error` buys precision with shuffles: a gene that stops at its
exceedance needs about 1/(relative_error² · p) of them.

Fix the seed, budget, precision and `fdr` before looking, and never rerun
borderline genes until they pass.

## The subset test's validated range

Under the count model the subset test is calibrated where the lower group
averages at least 32 reads per sample
([validation](validation.md#the-subset-test-under-the-count-model)).

- **What fails below 32.** A fold that every case carries, at low counts. The
  fold is estimated from the data and held fixed, and its error reads as a
  subset. Zero-rich genes are not the problem: a gene that reads zero in half
  the tumours stays near its level (at most 1.25% at p ≤ 0.01).
- **What the test does about depth.** It thins every sample to equal depth,
  and takes the common fold out of the higher group by thinning. A difference
  in depth between the groups therefore does not read as a subset.
- **The warning.** `wade()` warns how many tested genes fall below 32, and
  `result.params["subset_validated_genes"]` counts those at or above. Genes
  below are still tested and reported.

To keep subset calls only where the test is validated, select those genes and
recompute BH within them:

```python
import numpy as np

lower = np.minimum(result.case_mean, result.control_mean)
ok = result.tested & (lower >= result.params["subset_validated_reads"])
subset_q = np.full(len(ok), np.nan)
subset_q[ok] = wade.bh_adjust(result.subset_pvalue[ok])
```

With `fdr` set, the called genes' p-values are bounds, so this BH is
conservative; `fdr=None` gives finished p-values.

The area test needs no such limit.

## Jitter

Each count receives independent Uniform(0, 1) jitter, drawn from a stream
seeded by the run's seed and the gene, so no two values tie. Counts are
discrete, and many samples share one, above all zero; the walk places its
boundaries only between distinct values. The jitter moves a count by less than
one, so it barely changes well-expressed genes; for a gene full of zeros,
which zeros fall inside a subset depends on the seed.

## Read, plot and export

| Column | Meaning |
| --- | --- |
| `gene` | The gene's name |
| `status` | `tested`; `filtered` (outside `test_genes`); `all_zero` (no reads in the samples the tests use) |
| `case_mean`, `control_mean` | Mean normalized count per group: reads per sample at the cohort's typical depth. With strata, Mantel–Haenszel standardized means over the strata holding both groups, so the area is still their difference |
| `area` | `case_mean − control_mean` |
| `area_fraction` | `area / (case_mean + control_mean)`, between −1 and 1 |
| `log2fc` | `log2(case_mean / control_mean)`: the fold of the group means, ±inf where a group has no reads |
| `area_pvalue`, `area_padj` | The area test's p-value, on depth-equalized counts, and its BH q-value over the tested genes |
| `alignment_log2fc` | The common fold: the Wasserstein-1 fold of log(count + 1) that aligns the cases with the controls (with strata, the weighted median of the strata's own) |
| `lorenz_gap` | The subset statistic: how far the cases' Lorenz curve of count + 1 falls below the controls' (0 without a subset) |
| `subset_side` | `"up"` above the rest of the cases, `"down"` below them, `"none"` |
| `subset_fraction` | The subset's share of the cases (0 without a subset) |
| `subset_log2fc` | The subset's fold over what the common fold predicts for it, on count + 1 (NaN without a subset) |
| `subset_pvalue`, `subset_padj` | The test that every case carries one common fold, and its BH q-value over the tested genes (p = 1 without a subset) |
| `subset_rarity` | The share of genes in its group whose subset is at least as rare (1 without a subset, NaN untested) |

The subset columns read the jittered normalized counts; only `area_pvalue` and
`subset_pvalue` come from the thinned counts. Each test has its own BH family
over the tested genes.

**A gene's subset.** `result.subset(gene)` returns a `Subset`:

| Field | Value |
| --- | --- |
| `side` | `"up"`, `"down"` or `"none"` |
| `samples` | The subset's cases, furthest out first |
| `fraction` | Its share of the cases |
| `log2fc` | Its fold over what the common fold predicts for it (NaN without a subset) |
| `pvalue`, `padj` | The gene's `subset_pvalue` and `subset_padj` |

**More of the result.**

- `result.tested` marks the tested genes; `result.bounded` marks, per test,
  the p-values that are bounds.
- `result.area_permutations` and `result.subset_permutations` record the
  shuffles each test used (0 for a computed area test); the exceedances sit
  beside them.
- `result.params` records the run: the settings, `depth_equalized`,
  `size_factor_median`, `strata`, `subset_validated_reads`,
  `subset_validated_genes`, `rarity_groups` and `permutation_space`.
- `wade.permutation_space(cond, strata=None)` returns how many labellings a
  design allows (`log10_space`) and the smallest p-value they allow
  (`p_floor`); `result.params["permutation_space"]` holds it for the run.
- Methods that take a gene accept a name or a position. A name matches
  first, so with integer gene IDs, `result.subset(7)` is the gene named 7.

```python
result.subset("ETV4")             # the subset's cases, its side, share, fold, p-value and q-value
result.gene_detail("ETV4")        # exact quantile intervals of the jittered normalized counts
result.plot_gene("ETV4")          # ECDFs and quantile differences on log2, with the fold and q-values
result.plot_volcano(label=["ETV1", "ETV4"])   # area fraction against the area q-value
result.plot_patterns()            # each gene's subset share against its fold
wade.write_results(result, "results.tsv")     # also writes results.manifest.json
```

- **`gene_detail`** reads every sample, without strata. Its `.area` is the
  signed area of the jittered normalized counts, within the jitter of `area`
  when there are no strata.
- **Plots** use matplotlib by default or `backend="plotly"`, with themes
  `light`, `dark` and `high-contrast`, and take a `title`. They plot log2 of
  the jittered normalized count + 1. `plot_volcano` and `plot_patterns` take
  `alpha`, the q-value that marks a call, and `label`: the genes to name, or
  how many of the smallest q-values.
  `import wade` loads no plotting library.
- **Tables.** `result.to_frame()` returns a Polars table (the `io` extra).
  `write_results(result, path, alpha=0.05)` writes TSV/CSV, Parquet or Arrow
  by the extension, and a manifest beside it.
- **The manifest** records the normalization, references, test families,
  sample order, strata, seed, the depth kept, group size-factor medians and every
  stopping record. `alpha` sets only its counts of significant genes and its
  `minimum_discovery_rank`; it is independent of `fdr`.

## Limits

- **The subset test is validated where the lower group averages at least 32
  reads per sample.** Below that it can be anti-conservative when every case
  carries a fold, because it plugs in an estimated fold. Recompute BH within
  the validated genes to keep only those calls
  ([validated range](#the-subset-tests-validated-range)). The area test holds
  at every expression level.
- **Every sample is thinned until its size factor equals the smallest.** One
  failed or very shallow library costs every gene power. Remove failed
  libraries first, by a rule set before testing
  ([before running](#before-running)).
- **Calls near a threshold depend on the seed,** the subset test's most. Fix
  the seed before looking, rank subsets by rarity, and read a call near the
  threshold as one draw ([depth equalization](#depth-equalization)).
- **RLE absorbs part of a change shared by many genes,** so unchanged genes
  look changed, in WADE as in any DE tool that uses size factors. When a
  global shift is expected, normalize by internal control genes or external
  factors ([size factors](#size-factors)).
- **The area test's null is one distribution for both groups.** It is built
  to detect a difference in means, but it can also reject when the groups
  differ only in spread. Such genes' area stays near zero, so rank calls by
  their area.
- **Unbalanced designs cost power.** With ten cases per control, the area
  test called 9% of genes raised 1.5× in every case
  ([validation](validation.md#confirmation-protocol-version-7)). Few controls
  also limit the subset test: every shuffle that keeps a small subset's cases
  out of the controls repeats its split, so its p-value cannot fall far below
  the share of such shuffles. Rank such subsets by rarity.
- **In large, heterogeneous cohorts the subset test rejects almost
  everywhere:** tumour purity alone makes the cases more unequal than the
  controls. There `subset_rarity`, not the q-value, ranks the subsets.
- **A subset is a description.** A small `subset_padj` means the cases are
  more unequal than relabelling explains. It does not say which patients are
  individually outliers, that a subset is clinically separable, or how
  prevalent it is.
- **Samples must be exchangeable under the null once depth is equalized,**
  across the cohort or within each stratum. Batch confounded with the groups
  breaks this, and normalization does not repair it; what strata repair is
  set out under [strata](#strata-study-site-or-batch). Pairing and repeated
  measures are not validated yet.
