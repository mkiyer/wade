# Contributing to WADE

Read [the release plan](PLAN.md) before changing inference. The current package
is the unsigned log-W1 baseline; the release requires directional biomarker
discovery. Research must evaluate that biological target, not promote unsigned
testing because it recovers more arbitrary distribution changes.
Kernel research is deferred. There is no two-stage compatibility path.

## Repository layout

| Location | What belongs here |
| --- | --- |
| `src/wade/`, `rust/src/` | The current public package and its native implementation. |
| `tests/` | Independent tests of that package, inputs, normalization and exports. |
| `research/wade_research/`, `research/tests/` | Candidate point workflow, deferred kernels and independent research checks. |
| `tools/` | Notebook, data-input, comparison and release helpers using the current package. |
| `docs/` | Current usage, measured performance and the deferred kernel derivation record. |
| `notebooks/` | Six current notebook sources; scientific claims must match their actual method. |
| `archive/` | Optional local studies and snapshots; excluded from Git and distributions. |
| `.release/` | Local data, frozen executions, source/result hashes and review artifacts. |

Do not add imports from `archive/` to active code. Keep shared research
calculations out of study runners. Do not introduce public kernel/prior options
until the method has passed its scientific gates. Preserve failed experiments
with their scope; a passing numerical test is not evidence of FDR control.

## Development and verification

```sh
python -m pip install -e ".[test,plot]"
cargo clippy --release -- -D warnings
python -m pytest -q
python tools/smoke_install.py
python tools/check_workspace.py
```

Pytest includes package tests and research correctness tests, with the research
import path configured in `pyproject.toml`. It excludes the archive. Production
checks compare distances and stopping times with independent oracles, verify the
sequential p-value distribution, and exercise ties, strata, filtering, BH,
alignment, chunk/batch invariance and exports. Kernel checks add archived replay,
independent quadrature, full-distribution normalization and cache isolation.
The production import still requires only NumPy; research uses SciPy.

Historical fixture audits and the superseded sequential runner's tests are
preserved with their source in [the archive](archive/README.md). They are no
longer counted as tests of the current package. The current implementation has
its own independent distance, stopping-rule and BH tests.

`tools/review_production.py` replays the sealed point-method transcriptome study
through the public API. Use the exact source and input fingerprints for historical
claims. A changed method needs fresh calibration and its own scientific record.

## Release

Exclude `research/`, `archive/`, `.release/` and the private workbench from the
release distribution. Pass source, wheel, fallback and cross-platform CI checks.
Reexecute all six notebooks after final scientific selection, including the
TCGA prostate/breast gene-count analyses. Native comparator reuse requires
identical, checksummed inputs and settings; old results cannot be renamed into
new method results.

Inspect all final figures and numerical exports and obtain user approval of
results, illustrations and claims. Passing tests or renders is not that approval.
Publish/tag only with final authorization. The release queue is [PLAN.md](PLAN.md);
open questions are in [ISSUES.md](ISSUES.md).
