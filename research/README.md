# Research

Validation code for the release; excluded from distributions and never
imported by the package.

| File | Purpose |
| --- | --- |
| `confirm.py`, `protocols/confirmation.json` | The frozen fresh-seed confirmation: 120 simulated 16,384-gene transcriptomes through the public API, with a one-sided Wilcoxon comparator |
| `subset_limits.py` | Single-gene study of where upward detection is possible, including a hindsight-optimal cutoff as an information bound |
| `subset_recovery.py` | Precision and recall of the learned subsets against the planted cases, on the confirmation's matrices |
| `wade_research/point_simulation.py` | The count generator shared by all three |

```sh
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
PYTHONPATH=src:research python research/subset_limits.py gamma 8
PYTHONPATH=src:research python research/subset_recovery.py --confirmation .release/0.2.0/confirmation-new
```

Results and their interpretation are in [validation](../docs/validation.md).
Earlier research is in git history at commit `f07d5f9`.
