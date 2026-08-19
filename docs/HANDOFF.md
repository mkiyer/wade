# Session handoff

Written to let a new session resume without re-deriving anything. Read
[`../README.md`](../README.md) and [`method.md`](method.md) first; this file
covers only what is *not* obvious from them.

---

## 1. Where things stand

The statistic is finished and tested. **499 tests pass in about 25 seconds.**

```bash
export PATH="/usr/local/bin:$PATH"          # only if you need R
conda activate wade
pytest -q                                   # everything
pytest -q -m "not slow"                     # skip the validation simulations
```

The package is `src/wade/`. Nine modules, all small:

| module | what it owns |
|---|---|
| `quantiles.py` | the probability grid and type-7 quantiles |
| `stats.py` | `wade_stats()` — the grids, `mean_shift`, `w1`, `fc` |
| `subset.py` | the bridge, the subset test, `affected_fraction`, `direction` |
| `permutation.py` | the nulls; dispatches to the Rust kernel |
| `pvalues.py` | empirical p, GPD refinement, BH, `alternative` |
| `normalize.py` | `tpm_like` (ported), `cpm`, `rle` (new) |
| `diagnostics.py` | `wade_gene()` — the per-gene curves for plotting |
| `api.py` | `wade()`, `wade_from_matrix()`, `wade_contrast()` |
| `rust/src/lib.rs` | the mean-shift permutation kernel |

## 2. What is NOT built, in priority order

See [`../ROADMAP.md`](../ROADMAP.md) for the detail. Short version:

1. **Plotting** — nothing exists. Blocks the notebook.
2. **I/O** — nothing exists. No readers, no writers; `wade_contrast()` takes
   integer column indices rather than sample names.
3. **Demo notebook** — nothing exists.
4. **Rust kernel for the subset test** — 98% of runtime at scale.

## 3. Things that will bite you, learned the hard way

Each of these cost real time to discover. They are not in the code comments
because they are about *reasoning*, not implementation.

### Composition couples every gene

Library-size normalization divides each gene by a column total every gene
contributes to. **Any simulation with a large fraction of strongly differential
genes will produce nonsense null genes.** Measured: five strongly-up genes among
205 moved null genes' `direction` from ~0 to near −1, because the case libraries
inflated 8.4% and every other gene's log-ratio shifted by log2(1.084).

I wasted a cycle on this twice — once diagnosing a "bug" that was the effect,
once writing a test whose premise it violated. **When simulating, keep the
signal fraction realistic (under ~10%), or use `wade_from_matrix` to bypass
normalization entirely.** The subset *test* is immune (its statistic is
invariant to a global offset); the characterization is not.

### Claims about the statistic must be measured, not reasoned

Three assertions I wrote from theory turned out to be wrong when run:

- The scan's argmax "obeys the arcsine law and is unstable" — false for the
  *bridge*, which is pinned at both ends. It is stable but not shape-robust.
- "π̂ works on the log scale" — true for global changes, false for small subsets
  until the fourth moment replaced the second.
- "Permutation gives the right null for the subset test" — false, and it cost
  14–19% false positives on genuine fold changes.

The pattern: build a planted-ground-truth simulation and check, before writing
it down.

### The R sandbox is fiddly

Only needed to regenerate `tests/fixtures/`. Three load-bearing details in
[`../reference/R/README.md`](../reference/R/README.md): R 4.6.1 is not on the
default `PATH`, you must run from `reference/R/`, and `renv::restore()` succeeds
and *then* errors on a socket — run it twice. Fixtures are committed, so the
suite runs with no R present.

### Test-suite conventions

- Parity runs pin `alternative="greater"`, because the R oracle is one-sided and
  the port defaults to two-sided. Comparing the default against R would be
  comparing two different tests.
- `portrun.py` pins `backend="numpy"` so layers 0–7 validate the readable path;
  `test_kernel.py` holds the kernel to the same fixtures separately.
- Every comparison goes through `assert_close`, which records into a ledger
  printed at the end of the run. **Report the worst-case deviation, not "the
  assertions passed"** — that number is what distinguishes an exact port from a
  close one.

## 4. Decisions already made — do not relitigate

Reasoning is in `method.md`; this is the index.

| decision | where |
|---|---|
| Raw counts in, not a normalized matrix (the jitter needs count precision) | `method.md` §8 |
| Two stages, reported separately; no combined p-value and no categorical label | `method.md` §5 |
| The subset test's null is the **fitted global shift**, not no-difference | `method.md` §3 |
| `affected_fraction` uses the **fourth** moment on the **log** curve | `method.md` §4 |
| Two-sided by default, `alternative` for one-sided | `method.md` §9 |
| The tail window, `tail_conc`, `F` and the rank scores are retired | `method.md` §7 |
| Higher Criticism and max-Z lost the detector comparison | commit `dcba920` |
| Names: `mean_shift`, `p_subset`, `affected_fraction`, `direction`; no aliases | commit `ff22cbb` |

The detector and shape-test prototypes were deleted once their findings were
documented; they are recoverable at commits `dcba920` and `664cfd5` if a design
choice ever needs re-litigating with the original evidence.

## 5. Suggested first move

Start with `plot_gene()`. It is small, it unblocks the notebook, and it forces
you to look at real curves — which is the fastest way to build intuition for
what `affected_fraction` and `direction` are actually measuring.

```python
import numpy as np, wade
rng = np.random.default_rng(0)
n = 200
cond = np.r_[np.ones(n, int), np.zeros(n, int)]
case = rng.lognormal(3, .6, n); case[rng.choice(n, 10, replace=False)] *= 8
x = np.r_[case, rng.lognormal(3, .6, n)]
d = wade.wade_gene(x, cond)          # d.p, d.y1, d.y0, d.cum, d.r
```

`d.r` against `d.p` is the figure. Flat means a global fold change; zero-then-
climbing means a subset. Plot one of each side by side and the method explains
itself.
