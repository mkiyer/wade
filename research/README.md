# Research

Validation code for the release; excluded from distributions and never
imported by the package.

| File | Purpose |
| --- | --- |
| `count_model.py` | The count model's exact reference: real tumours split at random (a split per gene) and thinned binomially by design, with known folds and depths. Measures the area test against the same test on unthinned counts, the normalization under composition change, and the area test on TCGA prostate with its own labels |
| `rarity.py` | Rarity on the TCGA examples, against the rank by the gap alone: the markers' ranks, subsets planted by thinning into real tumours (folds 1.5–8, and 16–32 in `strong`), each gene's own DNA alterations among its subset's members (GISTIC, cBioPortal), a contamination mixed into one library, and stability across seeds |
| `strata.py` | A prototype of both tests with labels exchanged only within strata (the stratified stream, thinning and saddlepoint; five subset constructions), and its vetting: exact enumeration, realistic designs against 20 million labellings, the count model with studies and batch effects, TCGA sites and TCGA with GTEx as real nulls, the real contrast by site, and cost ([plan](../docs/strata-plan.md)) |
| `subset_range.py` | Whether the subset test can reach below 128 reads per sample: depth equalization and thinning registrations against the rescaled test (shipped until 25 October 2026) and an exact oracle, on the count model's arms (folds, depth gaps, the TCGA design's imbalance), real zero-inflated genes and planted subsets; and how the shipped test's calls move with the seed on TCGA prostate |
| `subset_pooled.py` | Whether combining K thinning draws (their p-values averaged, or the gap averaged) keeps the subset test calibrated and steadies its calls: the count model's grid, zero-inflated genes, planted subsets, and seeds 1 and 2 on TCGA prostate |
| `stopping.py` | Decision-based stopping on the TCGA examples: whether `fdr` rejects exactly what finished runs reject (on 3,000 genes, or the whole prostate transcriptome), and its shuffles and wall time |
| `pvalue_accuracy.py` | The computed area-test p-value against every labelling or 20 million shuffles, across small, unbalanced and balanced designs, and how well its own error estimate tracks its error |
| `subset_recovery.py` | Whether the subsets recover the planted cases, on the side they were planted, and the subset test, on the confirmation's matrices |
| `subset_significance.py` | Calibration of the subset test, and its significance as a subset grows toward the whole cohort |
| `confirm.py`, `protocols/confirmation.json` | The frozen fresh-seed confirmation: simulated 16,384-gene transcriptomes through the public API, with a two-sided Wilcoxon comparator. The protocol file records version 6; version 7 is pending |
| `wade_research/point_simulation.py` | The count generator shared by the confirmation and the studies |

```sh
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/count_model.py [shipped composition real]
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/stopping.py [prostate breast] [exact full whole]
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/rarity.py [biology planted strong cnv contamination stability]
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/strata.py [exact accuracy model sites gtex real layout fold common heterogeneous registration thinned speed]
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/subset_range.py [grid bimodal power seeds]
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/subset_pooled.py [grid bimodal power seeds]
PYTHONPATH=src:research python research/pvalue_accuracy.py
PYTHONPATH=src:research python research/subset_recovery.py
PYTHONPATH=src:research python research/subset_significance.py
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
```

Results and their interpretation are in [validation](../docs/validation.md).

The research retired on 30 September 2026 is in git at commit `e4fd67f`.
That covers the subset null study, the location tests,
the Welch saddlepoint, the subset calibration, and the count model's other
sections. Its ideas are recorded in
[research ideas](../docs/research-ideas.md). Earlier research is at commit
`f07d5f9`.
