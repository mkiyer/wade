# Research

Validation code for the release; excluded from distributions and never
imported by the package.

| File | Purpose |
| --- | --- |
| `confirm.py`, `protocols/confirmation.json` | The frozen fresh-seed confirmation: 120 simulated 16,384-gene transcriptomes through the public API, with a two-sided Wilcoxon comparator |
| `pvalue_accuracy.py` | The computed mean-test p-value against every labelling or 20 million shuffles, across small, unbalanced and balanced designs, and how well its own error estimate tracks its error |
| `subset_calibration.py` | The subset test's null calibration by gene abundance, with and without count jitter |
| `subset_recovery.py` | Whether the subsets recover the planted cases, on the side they were planted, and the subset test, on the confirmation's matrices |
| `subset_significance.py` | Calibration of the subset test, and its significance as a subset grows into the mean shift |
| `subset_null_study.py` | The subset test's null: the released construction, the derived one and ablations, on real TCGA genes under random labels, simulations, planted patterns and TCGA with its labels ([derivation](../docs/subset-null-derivation.md)). Extra sections answer the two reviews: other normalizers, thinning, bimodal nulls and their mechanism, the Lorenz statistic, genotype-bimodal genes, registrations, Berger–Boos, the centered bootstrap, power at matched size and at the full budget, and the mean test ([plan](../docs/subset-null-plan.md)) |
| `welch_saddlepoint.py` | A saddlepoint for the Welch-studentized permutation test of equal means (prototype): against exact enumeration, against shuffles, and its solver on heavy-tailed genes ([plan](../docs/subset-null-plan.md), section 4.8) |
| `location_tests.py` | Which location test the DE test can calibrate: the mean, Welch, Wilcoxon and Brunner–Munzel permutation tests, each under its own null, on subsets and on shifts ([plan](../docs/subset-null-plan.md), section 4.8) |
| `count_model.py` | The count model's exact reference: real tumours split at random (a split per gene) and thinned binomially by design, with known folds and exposures. Measures the package's tests, a prototype rank test, thinning constructions (the Laplace fold; equalized, optimized-depth and higher-group thinning), count-exact DE scores, power on subsets planted by thinning, the thinning draw's effect, the TCGA designs' own exposure imbalance, small designs, bimodal and zero-inflated genes, and the normalization under composition change ([plan](../docs/subset-null-plan.md), section 3.5 and step 1) |
| `wade_research/point_simulation.py` | The count generator shared by the confirmation and the studies |

```sh
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
PYTHONPATH=src:research python research/pvalue_accuracy.py
PYTHONPATH=src:research python research/subset_calibration.py
PYTHONPATH=src:research python research/subset_recovery.py
PYTHONPATH=src:research python research/subset_significance.py
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json PYTHONPATH=src:research python research/subset_null_study.py
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json PYTHONPATH=src:research python research/subset_null_study.py \
    review2 mechanism2 lorenz bimodalgenes bergerboos registrations matched budget
PYTHONPATH=src:research python research/welch_saddlepoint.py
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json PYTHONPATH=src:research python research/subset_null_study.py large250
PYTHONPATH=src:research python research/location_tests.py
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/count_model.py [splits reference laplace power scores realistic depthchoice seeds pairwise
                                    small bimodal realbimodal robust composition summary]
```

Results and their interpretation are in [validation](../docs/validation.md).
Earlier research is in git history at commit `f07d5f9`.
