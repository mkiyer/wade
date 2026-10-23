# Contributing to WADE

## Layout

| Location | Contents |
| --- | --- |
| `src/wade/` | The package: `api.py` (`wade`, `wade_contrast`, `WadeResult`), `_means.py` (the area, the depth equalization and the area test), `_subsets.py` (the subset and its test), normalization, filtering, permutations, p-values, input/output, diagnostics and plotting |
| `rust/src/lib.rs` | The native subset walk and registration: each gene's split, and the shuffles of the subset test |
| `tests/` | Package tests against independent oracles |
| `research/` | The count-model reference, the fresh-seed confirmation runner, and the p-value and subset studies; excluded from distributions |
| `tools/` | TCGA inputs and notebook support, the install smoke test and the workspace check |
| `notebooks/` | Demo, prostate and breast examples (Quarto) |
| `docs/` | Manual, method, validation, changes and the TCGA data contract |

The research behind this version, including the ideas not adopted, is
recorded in [research ideas](docs/research-ideas.md); its code is at commit
`e4fd67f`, the analysis freeze of 30 September 2026. Earlier research (the two-stage method, measurement
kernels, the direction and scale comparison) is at commit `f07d5f9`.

## Build and test

Use the `wade` conda environment, which provides Python, Rust and maturin.

```sh
python -m pip install -e ".[test,plot]" --no-build-isolation
cargo clippy --release -- -D warnings
python -m pytest -q
python tools/smoke_install.py
python tools/check_workspace.py
```

## Rules

- The native walk is the only implementation of the subset statistic. Tests hold
  the split and the registration to an exact rational-arithmetic oracle, the
  shuffled area test to exact mean differences, the computed area test to exact
  enumeration of every labelling, and the thinning to its binomial law.
- A setting a user must choose needs a meaning a user can state. Prefer a
  principled derivation to a tuned constant: the area test's switch between
  computing and shuffling follows from `relative_error`, not from a group size.
- `import wade` needs only NumPy. Import optional libraries inside the
  functions that use them.
- A change to the statistic, the null, the hypothesis families or a default
  needs a new confirmation: bump the protocol version and seed root in
  `research/protocols/confirmation.json` and rerun `research/confirm.py`.
- Every number in the documentation comes from a run of a script in
  `research/` (at the tag named, for research since retired) or a rendered
  notebook, never from reasoning alone.
- Prefer deleting to adding. A parameter or code path needs a use case.

## Confirmation

```sh
WADE_TCGA_CONFIG=/path/to/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/count_model.py
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
PYTHONPATH=src:research python research/pvalue_accuracy.py
PYTHONPATH=src:research python research/subset_recovery.py
PYTHONPATH=src:research python research/subset_significance.py
```

The confirmation (protocol version 6) took 26 minutes of `wade()` time on a
quiet 16-core laptop and 72 on a busy one, for the procedure before depth
equalization. It refuses to overwrite an existing output directory
(`--resume` continues one).

## Notebooks

```sh
quarto render notebooks/demo.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render notebooks/examples.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render notebooks/examples-breast.qmd
```

The TCGA notebooks need local data described in the
[data contract](docs/tcga-examples.md). Rendered HTML is not tracked.
`notebooks/rna100k.qmd` is a private workbench, excluded from distributions.

## Release

1. Set the version in `pyproject.toml` and `CITATION.cff`; update `docs/changes-*.md`.
2. Render the notebooks and review every figure and table.
3. Tag `vX.Y.Z` on `main`. CI checks that the tag matches `pyproject.toml`,
   builds and tests wheels for every platform, and publishes to PyPI through
   the trusted publisher. A PyPI version can never be reused.
