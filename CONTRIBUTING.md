# Contributing to WADE

## Layout

| Location | Contents |
| --- | --- |
| `src/wade/` | The package: `api.py` (`wade`, `wade_contrast`, `WadeResult`, strata), `_means.py` (the area, the depth equalization, the area test and its saddlepoints), `_subsets.py` (the subset and its test), `_rarity.py` (rarity and the sample load), normalization, filtering, the shuffle stream, p-values, input/output, diagnostics and plotting |
| `rust/src/lib.rs` | The native subset walk and registration: each gene's split, and the shuffles of the subset test |
| `tests/` | Package tests against independent oracles |
| `research/` | The count-model reference, the confirmation, and the studies of p-values, subsets, decision-based stopping, rarity and strata; excluded from distributions |
| `tools/` | TCGA inputs and notebook support, the install smoke test and the workspace check |
| `notebooks/` | Demo, prostate and breast examples (Quarto) |
| `docs/` | Manual, method, validation, changes, research ideas and the TCGA data contract |

The research behind this release is in `research/`. Its
[README](research/README.md) lists each script, its sections and its command.
Outputs go to `.release/0.2.0/`, which is local and untracked. The ideas,
including those not adopted, are in [research ideas](docs/research-ideas.md).
Research retired on 30 September 2026 is at commit `e4fd67f`; earlier
research (the two-stage method, measurement kernels, the direction and scale
comparison) is at commit `f07d5f9`.

## Build and test

Use the `wade` conda environment (`conda activate wade`), never `base`. It
provides Python, Rust and maturin.

```sh
python -m pip install -e ".[test,io,plot]" --no-build-isolation
cargo clippy --release -- -D warnings
python -m pytest -q
python tools/smoke_install.py
python tools/check_workspace.py
```

The install builds the native kernel. After a change under `rust/`, rebuild
it with `maturin develop --release`.

**macOS 27 (Xcode 27.1) linker defect.** The new linker can leave the native
module's string table off the 8-byte alignment that macOS 27 requires, and
`import wade` then fails with "mis-aligned LINKEDIT string pool". Linking
with chained fixups avoids it:

```sh
CONDA_PREFIX=~/sw/miniforge3/envs/wade maturin develop --release -- -C link-arg=-Wl,-fixup_chains
```

Pass the flag through maturin as above, so that it reaches only the final
link. Setting it in `RUSTFLAGS` breaks the build: the flag also reaches the
proc-macro crates that rustc loads. This is a local workaround: CI's build
configuration is unchanged, so revisit it if CI moves to Xcode 27.

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
  needs a new [confirmation](#confirmation).
- Every number in the documentation comes from a run of a script in
  `research/` (for retired research, at the commit named: `e4fd67f` or
  `f07d5f9`) or a rendered notebook, never from reasoning alone.
- Claim only what is proven, and label the rest as measured.
- Prefer deleting to adding. A parameter or code path needs a use case.

## Research runs

- Run from the repository root, in the `wade` environment, with
  `PYTHONPATH=src:research`. The TCGA sections need
  `WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json`.
- The machine is shared. Cap threads on long runs (`RAYON_NUM_THREADS=6`).
- Check `uptime` before timing anything, and record the load beside a timing
  taken while other jobs run.
- NumPy's Poisson sampler consumes a variable number of draws. Changing one
  pattern's rates shifts every later draw in a script, so later rows change
  within Monte Carlo noise.
- Name each output with its date, e.g. `stopping-4draws-20261025.txt`.

## Confirmation

A confirmation is predeclared and frozen before any of its data exist.

1. Add `research/protocols/confirmation-vN.json`: each check's quantity, its
   acceptable and unacceptable values, the size that tells them apart with
   power 0.9, and a new seed root.
2. Point `PROTOCOL`, and its entry in `FROZEN`, in `research/confirm.py` at
   the new file. Commit the protocol and the script together.
3. Render the TCGA notebooks with the frozen package. The biology arm copies
   their tables from `notebooks/results/`, and `arm_biology` names the commit
   that rendered them.
4. Run, then score:

```sh
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \
    python research/confirm.py --output .release/0.2.0/confirmation-vN
PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation-vN --evaluate
```

- The count-model, real-label and strata arms read the TCGA bundles through
  `WADE_TCGA_CONFIG`.
- A run refuses to start with uncommitted changes to the frozen files: the
  package, the kernel, the script, its protocol and the research code it
  imports.
- It refuses to overwrite an existing output directory. `--resume` continues
  one, only with the package and script it began with.
- A run scores the checks when it ends; `--evaluate` scores them again from
  the saved outputs.
- Version 7 took about 4.2 hours with six threads on a shared 16-core laptop.

## Notebooks

```sh
quarto render notebooks/demo.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render notebooks/examples.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render notebooks/examples-breast.qmd
```

The TCGA notebooks need local data described in the
[data contract](docs/tcga-examples.md). Rendered HTML is not tracked.
`notebooks/rna100k.qmd` is a private workbench on the RNA100K compendium,
excluded from distributions. Its config, also named by `WADE_TCGA_CONFIG`,
gives the `samples`, `counts` and `genes` paths.

## Release

1. Set the version in `pyproject.toml` and `CITATION.cff`; update `docs/changes-*.md`.
2. Render the notebooks and review every figure and table.
3. The maintainer tags `vX.Y.Z` on `main`. CI checks that the tag matches
   `pyproject.toml`, builds and tests wheels for every platform, and
   publishes to PyPI through the trusted publisher. A PyPI version can never
   be reused.
