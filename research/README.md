# Kernel research workspace

This is the single workspace for the next kernel development steps. It is
separate from `src/wade`, excluded from release distributions, and never imported
by the public package. Follow the [canonical scientific plan](../docs/kernel-validation-roadmap.md).
No new experiment or statistical candidate was introduced by the cleanup.

## What is implemented

| Module | Responsibility |
| --- | --- |
| `wade_research/kernels.py` | Binomial–Beta patient kernels on the normalized-count axis; equal-patient group areas with numerical bounds. |
| `wade_research/points.py` | Matched empirical directional/unsigned area controls. |
| `wade_research/resampling.py` | Group weights, label banks and bounded numerical scoring of those banks. |
| `wade_research/_cache.py` | Exact CDF reuse within one immutable kernel instance. |

The earlier adapter and prior-contrast subclass are consolidated into one
`BetaKernelArea`. Size factors and prior shape are required inputs: normalization
is explicit and no prior is silently selected. Shapes 1 and 0.5 reproduce the
two previously studied candidates; neither is a production recommendation.
The numerical routines do not fit normalization, priors or a biological null.
Label permutation remains conditional on a justified exchangeable design.

```python
import wade
from wade_research import BetaKernelArea, label_coefficients

# counts: complete genes × patients raw count matrix; condition: 1/0 labels.
norm = wade.fit_normalization(counts)
kernel = BetaKernelArea(
    counts[gene_index], counts.sum(axis=0),
    size_factors=norm.size_factors,
    prior_shape=1,  # Explicit replay of one historical candidate, not a default.
)
bounds = kernel.scores(label_coefficients(condition))
```

`bounds` measures a numerical integration interval, not biological uncertainty
or a p-value. The full kernel uses A=(N/s)P; supplying the same RLE factors as
the point control puts both on the same abundance axis. The current evaluator
supports linear area. Log kernels, fold-change/subset estimators and complete
kernel-workflow inference remain scientific work in the plan.

## Verification and provenance

The historical `archive/` and execution `.release/` directories are local evidence,
not versioned dependencies. Tests and the numerical fixture run without them.

From the repository root, using an environment with the test dependencies:

```sh
PYTHONPATH=src:research python -m pytest -q research/tests
PYTHONPATH=src:research python -m pytest -q
```

The tests replay six pre-cleanup kernel/point cases exactly, compare areas with
independent CDF quadrature, check normalization of the whole kernel, nonzero
zero-count uncertainty, equal mass, cache isolation and unequal-group point
areas. Fixture provenance is in `tests/fixtures/kernel-baseline.json`.
These are correctness checks, not FDR calibration or model selection.

The extracted code originates in the [archived adapter](../archive/development-20260923/tools/kernel_benchmark_adapter.py),
[prior contrast](../archive/development-20260923/tools/reference_kernel.py),
[cache](../archive/development-20260923/tools/kernel_cache.py), and
[point control](../archive/development-20260923/tools/kernel_scientific_pilot.py).
Other kernel families, fitted-reference prototypes, simulators and completed
study runners remain in that archive. Reuse their ideas deliberately; do not
import archived modules into this workspace.

New studies should have one small runner and a frozen configuration, with
execution artifacts under `.release/`. Shared calculations belong here, not
inside successive experiment scripts. Add evidence to the existing plan/archive
index instead of creating another parallel roadmap. No further caching or Rust
work is needed before the scientific benefit is established.
