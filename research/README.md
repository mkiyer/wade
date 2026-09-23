# Kernel research workspace

This is the single workspace for the next kernel development steps. It is
separate from `src/wade`, excluded from release distributions, and never imported
by the public package. Follow the [canonical scientific plan](../docs/kernel-validation-roadmap.md).
No statistical candidate was selected by the cleanup or the gate-B instruments.

## What is implemented

| Module | Responsibility |
| --- | --- |
| `wade_research/kernels.py` | Binomial–Beta patient kernels on the normalized-count axis; equal-patient group areas with numerical bounds. |
| `wade_research/points.py` | Matched empirical directional/unsigned area controls. |
| `wade_research/resampling.py` | Group weights, label banks and bounded numerical scoring of those banks. |
| `wade_research/_cache.py` | Exact CDF reuse within one immutable kernel instance. |
| `wade_research/assessment.py` | Checked group curves, effect contrasts, descriptive walk, membership/ranking scores and kernel interval diagnostics. |
| `assess_kernels.py` | One frozen truth-recovery and historical-null replay runner; no imports from archived code. |

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
supports linear area. Research descriptions measure mean ratios and selected-tail
contrasts; they do not identify latent mixture components. Log kernels, validated
subset estimators and complete kernel-workflow inference remain scientific work.

## Gate-B assessment

Run the analytic instrument tests without external data:

```sh
PYTHONPATH=src:research python -m pytest -q research/tests/test_assessment.py
```

With the original local study artifacts present, run the bounded replay:

```sh
PYTHONPATH=src:research python research/assess_kernels.py \
  --output .release/0.2.0/truth-recovery
```

The runner refuses to overwrite results. It freezes the
[protocol](protocols/truth-recovery.json), executable sources and input hashes
before evaluation. It writes per-profile effects, per-cohort scorecards, interval
coverage/width, normalization errors, and the null replay with denominators.
The source artifacts under `.release/` are required for this replay; a clean
checkout can still run every analytic and numerical test.

Points use exact empirical quantile curves. Kernel descriptions use equal-mass
quantile nodes for each patient, with 64/128-node convergence checks and an
analytic transport-error bound plus an explicit floating-point allowance.
Inverse Beta CDFs are checked against the CDF; failing requests use bisection.
Detection retains the saved bounded CDF integrator's conservative p-values,
and does not use the description approximation. Area-error scores also use that
CDF integral; curve/localization scores retain their explicit quadrature limits.

To render the exact latent counterexample for review:

```sh
PYTHONPATH=research python research/plot_assessment.py \
  --output .release/0.2.0/truth-recovery/review
```

The PNG/PDF and their hash manifest mark user approval as pending. The completed
[gate-B results](../docs/kernel-validation-roadmap.md#b-build-the-small-truth-recovery-and-failure-replay-benchmark)
record the selected run and its limits. Existing data are diagnostic evidence,
not new independent calibration or a selected production method.

The scorecard keeps three contrasts separate: the whole-group mean ratio,
the selected-tail versus control mean ratio, and the planted affected-group
versus control mean ratio. Missing zero ratios remain missing, without a hidden
pseudocount. Membership precision/recall use fractional selected patient mass;
average precision assesses patient-mean ranking separately. A null profile's
descriptive tail is not an individual significance call. Shared biological tails
are null for differential membership, and broad upshifts affect all cases.

An analytic regression exposes a crucial limit: adding a 10% spike to an
otherwise unchanged, heterogeneous case distribution can make the upper walk
select almost every case. Its cutoff is an observable curve description,
**not generally an estimator of the planted subset fraction**. This limitation
exists before any counts, normalization or kernels are introduced.

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
