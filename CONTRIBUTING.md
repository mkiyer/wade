# Contributing to WADE

## Layout

| Location | Contents |
| --- | --- |
| `src/wade/` | The package: `api.py` (`wade`, `wade_contrast`, `WadeResult`), `_wasserstein.py` (areas and the permutation engine), normalization, filtering, permutations, p-values, input/output, diagnostics and plotting |
| `rust/src/lib.rs` | The native permutation kernel; the same ordered sums as the NumPy path |
| `tests/` | Package tests against independent oracles |
| `research/` | The fresh-seed confirmation runner and the single-gene limits study; excluded from distributions |
| `tools/` | TCGA inputs and notebook support, the install smoke test and the workspace check |
| `notebooks/` | Demo, prostate and breast examples (Quarto) |
| `docs/` | Manual, method, validation, changes and the TCGA data contract |

Earlier research (the two-stage method, measurement kernels, the direction
and scale comparison) is preserved in git history at commit `f07d5f9`.

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

- The NumPy path is the reference implementation. The native kernel must give
  identical areas and stopping records; the tests check both backends against
  an exact rational-arithmetic oracle.
- `import wade` needs only NumPy. Import optional libraries inside the
  functions that use them.
- A change to the statistic, the null, the hypothesis families or a default
  needs a new confirmation: bump the protocol version and seed root in
  `research/protocols/confirmation.json` and rerun `research/confirm.py`.
- Every number in the documentation comes from a run of a script in
  `research/` or a rendered notebook, never from reasoning alone.
- Prefer deleting to adding. A parameter or code path needs a use case.

## Confirmation

```sh
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-new
PYTHONPATH=src:research python research/subset_limits.py gamma 8
```

The confirmation takes about 30 minutes on a 16-core laptop and refuses to
overwrite an existing output directory (`--resume` continues one).

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
