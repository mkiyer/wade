# WADE documentation

The checkout contains a working **point-based package baseline** and a separate
**kernel research workspace**. Kernels remain intended for 0.2.0; they have not
been selected or validated for production.

## Current documentation

| Page | Purpose |
| --- | --- |
| [Design and validation plan](kernel-validation-roadmap.md) | The one scientific roadmap: agreed sequence, unresolved model choices and acceptance gates. |
| [Manual](manual.md) | Using the current package, inputs, filtering, design and interpretation. |
| [Method](method.md) | Exact definitions of the current point-based test and inference. |
| [Normalization](normalization-standard-practice.md) | RLE, designated controls, supplied factors and per-sample conversions. |
| [TCGA examples](tcga-examples.md) | Gene-count inputs, cohort definitions and annotation provenance. |
| [Notebook builds](notebook-builds.md) | Reproduce and review the six notebooks against the current API. |
| [Change notes](changes-0.2.0.md) | The migration already implemented; kernel changes remain pending. |

[PLAN.md](../PLAN.md) is the release queue; [ISSUES.md](../ISSUES.md) lists open
issues without the experiment chronology. The [contributor guide](../CONTRIBUTING.md)
explains the code layout and verification commands.

## Research and evidence

`archive/` and `.release/` are optional local evidence, excluded from Git and
release distributions. Links into them are available only in the original
research workspace. The current specification, code and numerical fixtures
are versioned and work without either directory.

New kernel work belongs in [research/](../research/README.md). Completed studies
and their competing prototypes are in the [development archive](../archive/development-20260923/README.md).
Their reports remain evidence, including negative results; they are not active
implementation instructions.

The latest comparison reports are [the scientific pilot](../archive/development-20260923/docs/kernel-scientific-pilot.md),
[paired downsampling](../archive/development-20260923/docs/kernel-downsampling.md), and
[minority-depth stress](../archive/development-20260923/docs/kernel-depth-stress.md).
All existing result/figure approvals retain their original pending status.
