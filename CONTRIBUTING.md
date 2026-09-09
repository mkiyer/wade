# Contributing

How to build WADE from source, run its tests, find your way around, and the
things that have cost time before. The method is in
[`docs/method.md`](docs/method.md); the work queue is
[`ISSUES.md`](ISSUES.md).

## Build and test

```bash
git clone https://github.com/mkiyer/wade && cd wade
mamba env create -f mamba_env.yaml && conda activate wade   # Rust toolchain included
pip install -e . --no-build-isolation                        # builds the kernel
pytest -q                                                    # ~20 s
```

Without conda: a Python 3.10+ virtual environment plus a Rust toolchain
([rustup](https://rustup.rs); on Windows also the MSVC Build Tools), then
`pip install -e ".[all,test]"`. The `--no-build-isolation` flag only saves
pip fetching maturin into a temporary environment; drop it if maturin is not
installed. `maturin develop --release` is the fastest rebuild loop once
maturin is on your path.

```bash
pytest -q -m "not kernel"        # if you built without the Rust toolchain
pytest -q -m "not validation"    # skip the simulations (~8 s)
cargo clippy --release -- -D warnings
python -c "import wade; assert wade.permutation.HAVE_RUST_KERNEL"
```

CI runs the suite on three OSes × two Pythons, at the declared NumPy floor,
with the kernel deleted, and builds every wheel on a native runner. The
development environment has every optional dependency installed and
therefore cannot see the class of bug where a test assumes one; reproduce a
bare install with a venv and `pip install -e ".[test,io]"`.

## Layout

```
src/wade/
  api.py          wade(), wade_contrast(), WadeResult — the one driver, over gene chunks
  io.py           as_counts(), condition(), write_results(), manifest — the data boundary
  normalize.py    tpm_like() and the continuity jitter
  quantiles.py    the descending probability grid and type-7 quantiles
  stats.py        wade_stats(): the grids, mean_shift, w1, fc
  subset.py       the bridge, the subset test, affected_fraction, direction, subset_log2_fc, the bootstrap
  thinning.py     fit_fold_change(), thin_counts(), one_count() — the count-native shift correction
  permutation.py  draw_perms(), the two nulls, kernel dispatch, permutation_space(), detectability_floor()
  pvalues.py      empirical p, GPD refinement, BH
  saddlepoint.py  stage1="saddlepoint": the exact permutation tail of a subset sum
  diagnostics.py  wade_gene(), library_qc(), subset_drivers()
  plotting/       theme.py (every colour and label) → data.py (pure NumPy dataclasses)
                  → _plotly.py / _matplotlib.py (thin renderers) → __init__.py (the plot_* functions)
rust/src/lib.rs   three kernels: null_statistics, subset_null, fit_bisect
tests/            layered suite; tests/fixtures/*.json are permanent committed data
tools/            reproduction scripts: bench_scaling, pvalue_study, competitors
notebooks/        demo.qmd (the README's figures), benchmark.qmd, rna100k.qmd
```

**Plotting is an extension.** `theme.py` imports nothing, `data.py` imports
no backend, the renderers import their library inside their functions, and
`import wade` stays NumPy-only; tests pin all of that, and a test greps the
package for colour literals outside `theme.py`. Every figure's arrays are a
dataclass with a `table()`, which is the seam a third renderer would be
written against (about 100 lines).

## Rules

**Nothing changes a reported number.** `pytest -q` is the check, and the
parity ledger it prints at the end (worst relative deviation over every
fixture comparison) is the sharper one. Paths that do change numbers are
opt-in and named: `stage1="gemm"`, `stage1="saddlepoint"`,
`fit_backend="rust"`. `backend="auto"` must never let the machine pick the
answer.

**Measure before writing.** Every number in a docstring or document was
measured, not reasoned. Three claims written from theory turned out wrong
when run (the scan's argmax "obeys the arcsine law"; "the participation ratio
works on the log scale" without the fourth moment; "permutation gives the
right null for the subset test"). Build a planted-ground-truth simulation and
check.

**Prefer deleting to adding.** Before adding a parameter or a code path, ask
whether an existing one covers the case: CPM is `normalizer=1.0`; the chunked
driver with one chunk *is* the one-pass driver; a demonstration of why a
design was rejected belongs in a test, not on the entry point.

**A numerical routine checks its own answer.** Four separate bugs in the
saddlepoint solver each returned a plausible wrong p-value rather than an
error, and each was caught only because the final answer is verified against
both saddlepoint equations before it is returned. Anything approximating a
tail gets the same treatment.

## Things that will bite you

**Tolerances go on the quantity's scale, never per element on something
that crosses zero.** `mean_shift` is a difference of two large nearly equal
means, and the bridge passes through zero; a per-element relative tolerance
on either measures cancellation, not arithmetic. Which BLAS and which libm
NumPy links decides summation order and the last bit of `log2`, and one
machine has one of each: a `1e-12` per-element tolerance passed for a
fortnight on the development laptop and failed on the first CI run under a
different BLAS, and a `1e-15` one passed on arm64 and failed on Linux. Use
`conftest.assert_close_scaled`.

**Chunking is bit-identical only because every coupling is structural.**
The jitter is drawn once and indexed per chunk; library sizes are always one
full-matrix pass (a column sum's association depends on blocking); the
fold-change fit and the thinning consume their random streams in gene order
across chunks (iteration-outer, chunk-inner; case blocks then control blocks).
And no chunk is ever a single row: NumPy reduces a `(1, m)` array by a
different code path than the same row inside a larger matrix, at last-ulp
cost, so `gene_chunks` refuses `gene_chunk=1` and folds a one-row remainder
into the previous chunk.

**The fit kernel is deliberately not bitwise.** No two binomial samplers
consume randomness alike, so `fit_backend="rust"` fits a slightly different
fold change for the same seed (measured max |log ratio| 0.035). Its
guarantees are different ones: deterministic given the seed, chunk- and
thread-invariant (each gene's stream is a function of `(seed, global gene
index)`), and statistically held to the NumPy path.

**Composition couples every gene.** Library-size normalization divides each
gene by a column total every gene contributes to, so a simulation with a
large fraction of strongly differential genes produces nonsense null genes
(five strongly-up genes among 205 moved null genes' `direction` from ~0 to
near −1). Keep the signal fraction of the *library mass* under ~10%, or pass
`lib_sizes=np.ones(n)` to bypass normalization.

**The subset stage needs `B ≥ 500` to refine at all** and the empirical floor
is `1/(B+1)`, so a figure of the subset stage on a few hundred genes needs
`nperms=2000` or no subset gene clears BH. A planted 5% subset in 200 cases
sits near the combinatorial floor and is found at raw `p` but not at FDR;
that is the floor doing its job.

**The fixtures cannot be regenerated, and do not need to be.** The R
implementation they were generated from was deleted. `tests/fixtures/*.json`
are permanent committed data whose correctness is established by
`tests/test_independent_reference.py`, which re-derives every value from
`numpy.quantile`, SciPy's BH and the closed forms and is forbidden from
importing WADE. Do that work first if you ever add a fixture; restore one
from git, never rebuild. Each double is stored as a decimal *and* a hex
float and the loader asserts they agree.

**plotly has quiet failure modes.** `add_hline` / `add_vline` / `add_vrect`
with `row=`/`col=` skip subplots that do not yet contain data, so add shapes
after the traces. An explicit range on one axis breaks autorange on axes
`matches`-ed to it. `shared_yaxes="rows"` shares every row; for one row use
`update_yaxes(matches=...)`. `make_subplots` silently drops an empty
`subplot_titles` entry. In matplotlib a colour bar is an `Axes`, so
`len(fig.axes)` is one more than the number of panels.

**The kernel's boundary.** The kernels return full `genes × permutations`
null matrices, not p-values: that is what the GPD needs, and it is the only
place a kernel bug is cleanly separable from a p-value bug. Type 7 is
reimplemented in Rust term for term with the no-interpolation guard on ties;
accumulation is sequential; parallelism is over genes so no accumulation
crosses a thread. Permutations are an input, a `(n_perms, samples)` label
matrix, on the same argument path production uses.

## Test-suite conventions

- Layers 0–7 (`test_layer*.py`) validate the NumPy path against the fixtures
  inside-out: intermediate quantities localize a disagreement, endpoints only
  detect one. Stop at the first failing layer. `test_kernel.py` holds the
  kernel to the same fixtures separately.
- Parity runs pin `alternative="greater"`; the package defaults to
  two-sided.
- Every fixture comparison goes through `assert_close`, which records into a
  ledger printed at the end of the run. Report the worst-case deviation, not
  "the assertions passed". 1e-12 relative is a real bug; 1e-14 is summation
  order.
- Every plotting test parametrizes over `available_backends()`, which yields
  zero cases when neither backend is installed. Never index into it.

## Notebooks

`quarto render notebooks/demo.qmd` regenerates `docs/figures/*.png` and the
numbers the README quotes; it is the master source for both.
`benchmark.qmd` takes about 20 minutes, almost all of it one permutation
sweep shared by nine methods. Quarto is installed separately; the conda
environment supplies the Jupyter kernel.
