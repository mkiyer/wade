# Research

Validation code for the release; excluded from distributions and never
imported by the package.

| File | Purpose |
| --- | --- |
| `count_model.py` | The count model's exact reference: real tumours split at random (a split per gene) and thinned binomially by design, with known folds and depths. Measures the area test against the same test on unthinned counts, the normalization under composition change, and the area test on TCGA prostate with its own labels |
| `pvalue_accuracy.py` | The computed area-test p-value against every labelling or 20 million shuffles, across small, unbalanced and balanced designs, and how well its own error estimate tracks its error |
| `subset_recovery.py` | Whether the subsets recover the planted cases, on the side they were planted, and the subset test, on the confirmation's matrices |
| `subset_significance.py` | Calibration of the subset test, and its significance as a subset grows toward the whole cohort |
| `confirm.py`, `protocols/confirmation.json` | The frozen fresh-seed confirmation: simulated 16,384-gene transcriptomes through the public API, with a two-sided Wilcoxon comparator. The protocol file records version 6; version 7 is pending |
| `wade_research/point_simulation.py` | The count generator shared by the confirmation and the studies |

```sh
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/count_model.py [shipped composition real]
PYTHONPATH=src:research python research/pvalue_accuracy.py
PYTHONPATH=src:research python research/subset_recovery.py
PYTHONPATH=src:research python research/subset_significance.py
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
```

Results and their interpretation are in [validation](../docs/validation.md).

The research retired on 30 September 2026 is in git at tag
`analysis-2026-09-30`. That covers the subset null study, the location tests,
the Welch saddlepoint, the subset calibration, and the count model's other
sections. Its ideas are recorded in
[research ideas](../docs/research-ideas.md). Earlier research is at commit
`f07d5f9`.
