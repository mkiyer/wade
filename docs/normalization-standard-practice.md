# Normalization practice for WADE 0.2.0

This is the migrated operating interface. The earlier proposed menus and
experiments remain in the [source archive](../archive/README.md) and linked
research reports; they are not additional production modes.

```python
import wade
result = wade.wade(counts, condition)  # bulk default: RLE
```

| Requirement | Production path | Interpretation |
| --- | --- | --- |
| Typical bulk gene counts with a justified stable reference | Default RLE | Cohort-fitted relative exposure |
| A protocol supplies validated internal controls | `fit_normalization(counts, reference_genes=ids)` | The specified panel defines the reference |
| A separately justified normalization or exposure | `Normalization.from_size_factors(...)` | Import complete positive relative exposure factors |
| Total-count exposure is the intended reference | `normalization="library"` | Cohort-centered library totals; inspect composition assumptions |
| Independent relative abundance per library | `wade.cpm(counts)` | Each column sums to one million |
| Independent length-adjusted abundance | `wade.tpm(counts, lengths)` | Requires positive assay-appropriate lengths |
| Sparse single-cell, spatial or cfRNA inference | No automatic sparse mode | Use a separately validated design and external factors; broad inference certification is deferred |

RLE uses information across samples. CPM and TPM use one library plus fixed
feature/length definitions; zeros do not require a new per-sample formula.
Their outputs are abundance summaries, not alternate raw-count inputs to the
WADE test. The `library` inference mode uses relative exposure with geometric
mean one, not a million-count transform, before the log offset. External
factors should likewise describe the **complete exposure**, rather than only
a composition multiplier that still needs library depth.

## Fit once, then select the test family

First perform sample QC and define the measured/reference feature universe.
This may include justified assay-level gene QC. Fit normalization on that
universe. Then define a label-independent test mask and reuse the fitted
object; do not silently change its reference when changing test genes.

```python
norm = wade.fit_normalization(counts, reference_genes=validated_controls)
keep = wade.filter_counts(counts, min_count=3, min_samples=5)
result = wade.wade(counts, condition, normalization=norm, test_genes=keep)
print(norm.used_reference_genes)
```

The thresholds above are an example, not universal defaults. Counts stay in
raw gene-by-sample form. Use named matrices, or explicit gene/sample names, to
make alignment and filtering reviewable. The fitted factors have geometric
mean one on their fit cohort; selecting a sample subset preserves that scale.
The [manual](manual.md) documents external factors and alignment in full.

## What the evidence supports

The frozen whole-transcriptome study supports the single-test workflow in its
specified bulk scenarios, including reference/test overlap and pooled filtering.
It does not establish a universal normalization guarantee. In the difficult
500:500 development screen, RLE gives WADE about 20.0% mean FDP; true known
exposures lower it to 3.46% on the same matrices. A majority of unchanged genes
and a large reference universe do not ensure negligible factor bias at large n.
See the [measured production assessment](../archive/development-20260923/docs/production-migration-0.2.0.md).

Validated controls can identify a useful reference. Low variance alone cannot
certify invariance, and no automatic control nomination is included. The
[mean–variance study](../archive/development-20260923/docs/mean-variance-normalization-results.md) evaluates richer
nomination strategies without treating a stable-looking panel as independent
biological truth.

The [E2 study](../archive/development-20260923/docs/operating-procedure-e2.md) investigated TMMwsp, positive-count
ratios and clustered pooling. Clustered pooling can supply external factors,
but successful factor estimation is not by itself evidence that raw individual
cells are independent replicates or that depth-confounded inference is valid.
No scran dependency or implicit clustering is introduced. Sparse and depth
robustness remain explicit future validation work.

No noise is added in the migrated procedure. Earlier jitter/thinning experiments
remain research evidence; they do not justify selectable public noise knobs or
a claim that noise repairs reference bias. The method tests distributional
exchangeability and reports its fixed transformations and assumptions.
