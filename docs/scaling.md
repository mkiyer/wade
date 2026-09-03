# Scaling WADE

Measurements, hypotheses and directions for running WADE on cohorts far larger
than it was built for. Every number was measured — the 2026-08-19 baselines by
scratchpad scripts whose recipes are `tools/bench_scaling.py` now, and each
section's **"Landed 2026-08-20"** block records what shipped from it, how it
was validated, and the before/after. §§2.1–2.2 and 3.1–3.4 are implemented;
§2.3 (residents), §4 (p-value resolution) and §5 (single cell) are still
agenda. **§7 is what the real cohort measured**, and it is what reordered the
agenda.

The method is in [`method.md`](method.md); what WADE cannot do is in
[`limits.md`](limits.md); the work queue is [`../ROADMAP.md`](../ROADMAP.md).
**This file is the research agenda behind ROADMAP §2** — it exists so the
reasoning and the failed ideas survive, not just the task list.

---

## 0. The target, and the one after it

**Now:** ~30,000 genes x ~80,000 samples, a real dataset the user has.

**Next:** single-cell and spatial transcriptomics — thousands to millions of
cells, sparse and zero-inflated. Not to be implemented yet, but §5 records what
that regime demands, because two of the choices below (grid capping, gene
chunking) are exactly the ones that make it reachable and one of them (§5.3) is
a statistical trap that no amount of engineering fixes.

---

## 1. The wall, measured

Reproduce with `tools/bench_scaling.py` (per-stage: default; whole pipeline:
`--full`, plus `--max-probs / --gene-chunk / --stage1 / --fit-backend /
--n-boot` for each landed change) and `tools/bootstrap_scheme_study.py` for
§6's table.

Per-stage scaling at G = 1,000 genes, B = 200 permutations, NB counts
(dispersion 0.1), seconds:

| n/group | m | normalize | observed | mean null | f fit | thin | subset | total | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 0.00 | 0.00 | 0.03 | 0.25 | 0.01 | 0.02 | 0.32 | 0.02 GB |
| 500 | 500 | 0.00 | 0.02 | 0.16 | 1.23 | 0.03 | 0.13 | 1.58 | 0.09 GB |
| 2,000 | 2,000 | 0.01 | 0.09 | 0.72 | 4.85 | 0.11 | 0.56 | 6.35 | 0.34 GB |
| 6,000 | 6,000 | 0.05 | 0.34 | 2.26 | 15.57 | 0.32 | 1.75 | 20.28 | 1.03 GB |

Extrapolated to 30,000 x 80,000 at B = 2,000: **about 3.7 hours, needing about
163 GB** against the 128 GB on the development machine. It does not run.

**After the queue landed (2026-08-20), measured, not extrapolated**: a full
`wade()` at exactly that shape — NB counts, 30,000 × (40,000 v 40,000),
B = 2,000, `gene_chunk=2000`, `stage1="gemm"`, `fit_backend="rust"`,
`max_probs` at its default 2,000 —

```
python tools/bench_scaling.py --genes 30000 --n 40000 --nperms 2000 \
    --gene-chunk 2000 --stage1 gemm --fit-backend rust
# wade() total   1780.53 s   m = 2000   peak RSS 55.59 GB
```

**29.7 minutes and 55.6 GB.** The wall-clock is dominated by the subset
kernel's B × O(n) per gene, as §3.3's closing note predicts; the memory
peak is the full-matrix residents of §2.3. The dataset runs.

**The wall is not speed, it is `m`.** The quantile grid is
`m = min(n_case, n_ctrl)`, so at 40,000 per group **every `(genes x m)` array is
the size of the data matrix**:

| array | size at G=30,000, m=40,000 |
|---|---|
| counts, normalized matrix | 19.2 GB each |
| `WadeStats.Q1`, `.Q0`, `.D` | 9.6 GB each |
| subset `r`, `b`, `r_test`, `b_test`, `mu`, `sd` | 9.6 GB each |
| thinned counts, corrected matrix | 19.2 GB each |
| the two nulls (`genes x B`) | 0.5 GB each — not the problem |

`method.md` §1 calls `m` "a property of the design, not a parameter". At 40,000
that property is a liability, and §2.1 is the consequence.

---

## 2. Memory

### 2.1 Cap the quantile grid — the change that unlocks everything else

**Hypothesis.** Nothing needs an affected fraction resolved to 1/40,000.
Capping `m` divides every `(genes x m)` array *and* every per-permutation grid
readout by the same factor — 40x at m = 1,000.

**Evidence.** n = 4,000 per group (so the full grid is m = 4,000), NB counts at
mean 50, 40 genes per cell, median over genes:

| planted | statistic | m=4000 | m=2000 | m=1000 | m=500 | m=200 | m=100 |
|---|---|---|---|---|---|---|---|
| global 2x | affected_fraction | 0.9960 | 0.9958 | 0.9956 | 0.9951 | 0.9937 | 0.9919 |
| 5% at 8x | affected_fraction | 0.0484 | 0.0485 | 0.0487 | 0.0491 | 0.0497 | 0.0506 |
| 1% at 8x | affected_fraction | 0.0093 | 0.0095 | 0.0096 | 0.0098 | 0.0104 | 0.0111 |
| 0.1% at 8x | affected_fraction | 0.0014 | 0.0014 | 0.0014 | **0.0024** | 0.0055 | 0.0106 |
| 0.025% at 8x | affected_fraction | 0.0010 | 0.0012 | 0.0017 | 0.0028 | 0.0059 | 0.0112 |
| global 2x | mean_shift | 49.94 | 49.95 | 49.96 | 49.97 | 50.06 | 50.19 |
| 5% at 8x | mean_shift | 17.48 | 17.58 | 17.77 | 18.17 | 19.39 | 21.21 |
| 1% at 8x | mean_shift | 3.34 | 3.41 | 3.57 | 3.86 | 4.98 | 6.72 |
| 0.1% at 8x | mean_shift | 0.42 | 0.50 | 0.56 | 0.92 | 1.96 | 3.69 |

**The rule: `m` must be roughly `2.5 / smallest fraction of interest`.**
m = 1,000 is faithful down to a 0.1% subset — 80 samples in 80,000. A global
shift survives m = 100. Note the last row of the 0.025% block: at 1 affected
sample in 4,000 even the *full* grid reads 0.0010 against a true 0.00025, which
is the `1/m` resolution limit of `limits.md` §1 doing exactly what it says.

**The cost falls on `mean_shift` for concentrated signals**, and it is
one-sided: a coarse uniform-in-`p` grid gives the extreme node weight `1/m`
while its value is huge, so the quadrature inflates. 21.21 against 17.48 at
m = 100 for a 5% subset. **Inference is unaffected** — the permutation null
inherits whatever quadrature the observed value used — but the number stops
being an estimate of the mean difference, which is what §3.1 is really about.

**Risk.** `affected_fraction`'s resolution becomes a parameter rather than a
property of the design. That is a real change to the contract in `method.md`
§1 and must be stated there, not buried. Mitigation: record the realized `m` in
the result and the manifest, refuse silently degrading it, and default
`max_probs` high enough (2,000) that the common case is unaffected.

**Validate by:** rerunning `tests/test_subset.py` and `tests/test_scale.py`
with the cap active at a size where the uncapped grid is also computable, and
asserting the table above rather than remembering it.

**Landed 2026-08-20** (`max_probs`, default 2,000, `None` uncapped). Threaded
through `wade_stats`, both nulls, `subset_test`, `characterization_ci` and
`wade_gene`; both kernels now accept a grid capped below the design (and
still refuse one above it); the realized `m` is `result.nprobs` and is
recorded with `max_probs` in `params` and the manifest; the contract change
is stated in `method.md` §1. `tests/test_grid_cap.py` pins: bitwise identity
when the cap does not bind, backend agreement on a capped grid, faithfulness
of `affected_fraction`, the one-sided `mean_shift` inflation, and stage-2
level on genuine global shifts under the cap (the §6 thinning-interaction
question — level held at 0.03–0.08 in the suite-sized check).

The fidelity table above was **re-measured through the shipped path**
(scratchpad `cap_fidelity.py`, same design) and reproduces to the third
decimal — worst affected_fraction drift at m = 1,000 for fractions ≥ 0.1% is
0.0003; mean_shift 49.94 → 50.19 (global), 17.51 → 21.29 (5% at 8×) across
4,000 → 100, matching the prototype's 17.48 → 21.21.

Runtime effect at G = 1,000, n = 6,000 v 6,000, B = 200 (same recipe as §1,
now `tools/bench_scaling.py`): total 19.8 s uncapped → 18.7 s at m = 1,000
(subset stage 1.73 → 0.98 s, peak RSS 1.85 → 1.47 GB). As predicted, the cap
is not a speed change at these sizes — the fit and the per-permutation sorts
scale with `n`, not `m` — it is the memory change that makes the target's
`(genes x m)` arrays 40× smaller and gene chunking (§2.2) worth doing.

### 2.2 Chunk over genes

**Hypothesis.** Peak memory should not depend on the number of genes at all.
Library sizes are column sums over the whole matrix, so one pass computes them
and every subsequent pass — normalization, observed statistics, both nulls, the
fold-change fit, the bootstrap — can run in blocks of a few thousand genes.

**Why it is safe.** The permutation matrix is drawn once and shared across
chunks, so "one shuffle serves all genes" (`method.md` §6) survives untouched.
Nothing about a gene's statistic depends on another gene except through the
library sizes, which are fixed before chunking begins.

**Risk.** Low; this is an implementation detail with no numerical consequence.
The one trap is the jitter: it is drawn once for the whole matrix and must be
*indexed* per chunk, never redrawn.

**Payoff.** Removes the 19.2 GB matrix copies from the peak entirely — at
2,000 genes per chunk each working matrix is 1.28 GB.

**Landed 2026-08-20** (`wade(..., gene_chunk=N)`), and as of 2026-08-21 it
is the *only* counts driver — `gene_chunk=None` is one chunk over the whole
matrix, so there is no second pipeline to keep in step. Bit-identical by
construction — `assert_array_equal`, not tolerance, in
`tests/test_chunking.py` — because every coupling was made structural:

* the jitter is drawn once and indexed per chunk, as planned;
* **two couplings the plan missed were in the thinning's random streams.**
  The fold-change fit draws fresh case/control generators inside each
  bisection step, and `thin_counts` draws case blocks then control blocks
  from one stream. Both were restructured to consume their streams in gene
  order across chunks (iteration-outer/chunk-inner in the fit; two phases in
  the thinning), which is bit-identical because sequential chunked
  `Generator` calls reproduce a full-matrix call's stream exactly (verified
  for `uniform` and `binomial`, scratchpad `verify_fit_refactor.py`, which
  also holds the refactors bitwise against the pre-refactor code);
* library sizes stay a full-matrix pass — a column sum's association depends
  on blocking, so chunking it would move every normalized value by an ulp;
* **no chunk is ever a single row.** NumPy's row reductions take a different
  code path for a `(1, m)` array than for the same row inside a larger
  matrix (measured: last-ulp drift in `D.sum(axis=1)` at m = 40); blocks of
  ≥ 2 rows are layout-independent (verified k = 2..24 over two shapes), so
  `gene_chunk >= 2` and a trailing one-row remainder folds into the last
  chunk;
* the thinned matrix is held int32 (counts are integers, values exact) at
  half the float64 footprint; the residents that remain full-size are
  counts, jitter, `tpm` and the pseudocount — all carried by `WadeResult` —
  plus the two `(genes × B)` nulls.

Measured at G = 5,000, n = 6,000 v 6,000, B = 200 (`tools/bench_scaling.py
--full` / `--gene-chunk 1000`): **peak RSS 7.18 → 3.82 GB, 104 → 98 s** —
chunking is slightly faster, not slower (chunk-sized transients stay in
cache). At the 30,000 × 80,000 target the projected residents are counts +
jitter + tpm + pseudocount (19.2 GB each) + thinned int32 (4.8 GB) ≈ 82 GB
against 128 GB — the dataset fits; the end-to-end run should be measured
once §3.3 lands, because the fit still dominates wall-clock. The obvious
next memory cut, if one is needed: the jitter and the pseudocount are both
derivable (a seeded stream; a rank-1 outer product) and could stop being
materialized residents — that is §2.3's question in a sharper form.

### 2.3 float32 storage

**Hypothesis.** The stored count and normalized matrices can be float32 (9.6 GB
instead of 19.2 GB) while every statistic is still accumulated in float64.

**Risk.** Real and needs measuring. Counts above 2^24 lose integer exactness in
float32; the thinning needs integers; the parity fixtures are float64
end-to-end. Likely a storage-only option with the kernels widening on read.

**Open question.** Whether it earns its complexity once §2.1 and §2.2 land, or
whether the answer is simply "the matrix is 19.2 GB, own it".

**Sharpened 2026-08-20, not implemented.** After §2.1 and §2.2, the peak is
no longer transients but the four full `genes × samples` float64 residents
`WadeResult` carries — counts, jitter, `tpm`, pseudocount — plus the int32
thinned counts during the run. The jitter is a seeded stream and the
pseudocount a rank-1 outer product (`norm_factor / (normalizer · lib)`), so
neither has to be materialized; whether to make them lazy is a question about
the result object's contract (`gene_detail`, the parity fixtures' explicit
`jitter=` path), not about the kernels. That, rather than float32, is the
next memory cut if one is ever needed — the target now fits without either.

---

## 3. Time

### 3.1 Stage 1 as one matrix multiply

**The identity.** On a **balanced** design `mean_shift` is *exactly* the
difference of group means. Measured, max relative deviation:

| design | deviation | |
|---|---|---|
| 200 v 200 | 2.1e-13 | identical to float noise |
| 300 v 100 | 17x | differs |
| 500 v 50 | 36x | differs |

So on a balanced design the statistic needs **no sort and no grid**, and the
entire permutation null is `X @ W` with `W` the signed indicator matrix,
`(samples x B)`:

| | current kernel | GEMM | speedup | max rel dev |
|---|---|---|---|---|
| G=2,000, n=4,000, B=500 | 3.35 s | 0.02 s | **139x** | 1.9e-9 |
| G=5,000, n=8,000, B=500 | 17.83 s | 0.10 s | **185x** | 3.9e-9 |

At the target, with a measured 241 GFLOP/s: **~40 s** for the whole stage-1
null, against ~1.5 hours today.

**Risk.** The 1e-9 is not bitwise: BLAS reassociates where the kernel sums
sequentially left-to-right for R parity (`implementation-notes.md` hazard 10).
This must therefore be an **opt-in fast path**, with the parity-pinned kernel
remaining the default and the reference.

**The deeper question this raises.** `method.md` §2 already apologizes for the
quadrature on unbalanced designs — "a grid quadrature that over-weights the
extremes of the larger group ... it should be read as signed quantile area
rather than a drop-in mean estimate". Combined with §2.1, where a capped grid
makes the quadrature drift further from the mean difference, there is a case
for **defining `mean_shift` as the difference of group means outright**. That
would be faster, simpler, exactly interpretable, and identical on balanced
designs — but it breaks parity with the R reference on unbalanced fixtures,
which is a decision about what the fixtures are for, not a performance call.
**Do not make it silently.**

**Landed 2026-08-20** as `wade(..., stage1="gemm")`, opt-in, balanced designs
only (refused otherwise), grid path unchanged and still the default. The
observed statistic and the null are both evaluated as products with the
signed indicator (`mean_diff_stat` / `mean_diff_null`), so they stay
commensurable; `result.mean_shift` then reports the statistic the p-value
actually tested (the exact mean difference) and the grid quadrature stays
available as `result.stats.mean_shift`. Re-measured through the shipped
path: **134× / 225×** at (G=2,000, n=4,000, B=500) / (G=5,000, n=8,000,
B=500), max relative deviation 2.5e-9; exceedance counts identical to the
grid path, so p-values move only through the GPD's smooth tail fit
(`tests/test_stage1_gemm.py`). Two prices, both stated in the docstring: not
bitwise (BLAS reassociates), and in gemm mode a chunked run agrees with an
unchunked one only approximately, because dgemm's blocking depends on the
matrix shape.

**How approximately, and in what units** — re-measured 2026-09-02 across two
BLASes, because the answer is not the machine's to keep private. numpy from
PyPI links Apple Accelerate; numpy from conda-forge links OpenBLAS, and the
summation order is theirs, not ours. Against the literal mean difference, and
chunked against unchunked, at all five sites in `tests/test_stage1_gemm.py`:

| quantity | Accelerate | OpenBLAS |
|---|---|---|
| disagreement / statistic's scale | 1.2e-14 – 1.8e-14 | 1.2e-14 – 1.6e-14 |
| worst **per-element** relative | 2.7e-12 | 0 – 5.7e-13 |

The two rows are the same arithmetic read two ways. `mean_shift` is a
difference of two large nearly equal group means, so an element where the
groups almost cancel — one gene reads −1.13 among values spanning ±957 —
carries no relative precision of its own: one ulp of the sums behind it is
2.7e-12 of *that element* and 1.8e-14 of the statistic. **The scale row is the
contract; the element row is the cancellation.** The tests assert the first,
and had asserted the second, which is why they passed for a fortnight on
OpenBLAS — where chunked gemm happens to come out exactly bitwise — and would
have failed on the first CI run under Accelerate. Under a §2.1 cap the GEMM statistic is arguably the
better number — it stays the exact mean difference where the capped
quadrature drifts.

### 3.2 One sort per gene in the mean-shift kernel

**Hypothesis.** The mean-shift kernel re-sorts both groups for every
permutation. The subset kernel already proves the alternative: sort the gene's
row once, then obtain each permutation's two sorted groups by a single O(n)
partition walk, because a permutation only re-partitions the same values.

**Payoff.** ~10x, and it is the path that applies where §3.1's GEMM does not
(unbalanced designs, and any future non-linear stage-1 statistic).

**Evidence it works:** the subset kernel is bitwise identical to its NumPy path
over 111 comparisons and 80x faster.

**Landed 2026-08-20.** The mean-shift kernel now shares the subset kernel's
design: parallel over genes, one sort per gene, one O(n) two-ended partition
walk per permutation, type-7 read through the same precomputed plans, and it
returns gene-major so the Python-side transpose is gone. The quantile
arithmetic and the sequential grid summation are unchanged term for term, so
outputs are bitwise what the permutation-major kernel produced — every
existing kernel and parity test passes untouched. Measured (same recipe as
§3.1's table): 2.15 → 0.36 s at (G=1,000, n=6,000, B=200), 7.16 → 1.11 s at
(G=2,000, n=4,000, B=500), 41.3 → 5.5 s at (G=5,000, n=8,000, B=500) —
**6–7.5×**, and it applies on unbalanced designs and capped grids where
§3.1's GEMM does not.

### 3.3 The fold-change fit is the largest single term

At n = 6,000 it is 15.57 s of a 20.28 s total — **more than everything else
combined.** Sixteen bisection steps, each thinning the whole matrix and
re-normalizing it.

**Three candidate fixes, in increasing order of work:**

1. **Start from a bracket instead of `[0, log 1024]`.** The interquartile-mean
   ratio is one pass and lands within ~5% (measured: 2.08 for a true 2.00), so
   a bracket of `ratio / 1.3` to `ratio * 1.3` needs ~6 steps instead of 16.
2. **Fit in count space.** The objective compares interquartile means on the
   normalized scale, but normalization is a per-column affine map — the
   comparison could be done on counts with the column factors folded into the
   comparison, removing the re-normalization from the loop entirely.
3. **Move it into the kernel**, where the per-gene sort is already paid for by
   the subset pass that follows.

**Constraint on all three:** `f̂` must stay unbiased under an NB fold change.
An inaccurate `f̂` is *anti-conservative* (30% off gives 0.26–0.28 false
subsets), and `tests/test_thinning.py` pins it.

**Landed 2026-08-20**, as fix 2 plus fix 3; fix 1 (the bracket) was
deliberately **not** taken — a bracket that misses the root converges to a
wrong `f̂`, and a wrong `f̂` is anti-conservative, so a ~2× saving was not
worth a correctness cliff. Measured at (G = 1,000, n = 6,000 v 6,000): the
fit went **15.2 s → 6.3 s → 0.44 s**.

* **Fix 2, the affine path** (`fit_fold_change(..., alpha=...)`, what
  `wade()` now uses): jitter-free `tpm_like` is exactly
  `x = counts * alpha` with `alpha = norm_factor / (normalizer · lib)`
  fixed, so nothing is re-normalized inside the bisection — the untouched
  group's interquartile mean is computed once, each step redraws only the
  thinned group (half the binomial draws), and the interquartile means are
  read by `np.partition` selection instead of a full sort. 15.2 → 6.3 s;
  after it, the **binomial draw is ~85% of what remains** (measured 0.64 s
  of a 0.75 s iteration), and NumPy's sampler is single-threaded — which is
  what makes fix 3 the real fix. The `normalize=` path survives unchanged
  for objectives that are not affine (the tests' unit-normalizer closures).
  One trap worth recording: the affine path reduces per-side row *subsets*,
  so its interquartile mean had to be made independent of row grouping —
  fresh C-contiguous reductions are (pinned by a test); F-ordered ones are
  not (see §2.2's single-row finding).
* **Fix 3, the kernel** (`wade(..., fit_backend="rust")`): the bisection
  loop parallel over genes, each gene's stream a function of
  `(seed, global gene index)` alone — deterministic, chunk- and
  thread-schedule-invariant. 6.3 → **0.44 s (14×; 34× against the start)**.
  **Opt-in, unlike the other two kernels**, because it cannot be bitwise
  against NumPy (no two binomial samplers consume randomness alike): the
  realized `f̂` for a given seed differs between backends by up to the
  bisection cell (measured max |log ratio| 0.035), so auto-dispatch would
  let the machine choose the answer. Held to the NumPy path statistically
  (`tests/test_fit_kernel.py`), and stage-2 level on genuine global shifts
  holds under it.

With every fast path on (`max_probs` default, `stage1="gemm"`,
`fit_backend="rust"`), a full `wade()` at (G = 1,000, 6,000 v 6,000,
B = 200) is **2.27 s against 19.8 s** at the start of the day, and
(G = 5,000, `gene_chunk=1000`) is 11.7 s against 104 s, at 3.7 GB peak
against 7.2. The remaining wall-clock at the 30,000 × 80,000 target is the
**subset kernel's** B × O(n) per gene — intrinsic for dense data (§5.1's
zero-run walk is the answer for sparse) — projected ~40 min of a ~50-min
total at B = 2,000.

### 3.4 `n_boot` is serial and embarrassingly parallel

200 bootstrap replicates cost about as much as the rest of a 20,000-gene run.
Each replicate is independent; the loop is NumPy in `subset.characterization_ci`.
A thread pool or a kernel would make it nearly free.

**Related open question (see §6):** whether the bootstrap's resampling with
replacement interacts badly with the quantile grid, since duplicated samples
create flat runs in the quantile function — the same degeneracy the continuity
jitter exists to break.

**Landed 2026-08-20** (`characterization_ci(..., threads=)`, default all
cores): the replicate index sets are predrawn (which is also what makes the
call chunk-invariant, §2.2), each replicate's arithmetic is self-contained
and writes its own rows, so threading is **bit-identical** at any thread
count. Measured at the HANDOFF's size (20,000 genes, 100 v 100,
`n_boot=200`): 19.7 → 4.0 s (**4.9×** — fancy-indexing holds the GIL; the
sorts release it). The §6 open question is answered below: no measurable
tie degradation, and resampling with replacement stays the default.

---

## 4. P-value resolution

### 4.0 What landed 2026-08-20: ranking without resolution

The rna100k plasma contrast made the pile-up real (18.8% of genes at the
mean-shift floor; nearly every gene subset-significant), and the *ordering*
half of the problem is now addressed without new inference machinery:
**permutation z-scores** (`z_mean_shift`, `z_subset` — the observed
statistic against its own gene's null mean and sd, the GSEA-NES analogue,
free because the null matrices are in hand) and **`subset_log2_fc`** (the
subset's magnitude: mean of the log-ratio curve over the affected fraction,
quantile-matched). The recipe on saturated cohorts: filter by `padj`, rank
by magnitude or z. What these deliberately do **not** provide is a
calibrated tail probability — that remains this section's agenda, below.

### 4.1 Why this becomes urgent at scale, and not before

At 30,000 genes, BH at 0.05 needs resolution to roughly `1e-6` — which is
almost exactly today's floor, `1/(B · n_tail) = 2e-6` at the defaults. At
80,000 samples effects will be enormous and **p-values will pile at the floor
and stop ranking anything.** The argument for better resolution is *ordering
genes*, not believing the number.

The combinatorial floor is no longer the binding constraint: at 40,000 per
group, `C(n1,k)/C(n,k)` is astronomically small for any interesting `k`, so
tiny permutation p-values are genuinely attainable rather than an artefact.

### 4.2 What exists today

**GPD tail refinement** (`method.md` §6): fit a Generalized Pareto to the top
`n_tail = 250` null draws and read the p-value off it, floored at
`1/(B · n_tail)`. Cheap, already implemented, parametric extrapolation. The
floor is a deliberate honesty constraint, not a numerical guard.

### 4.3 fgsea's adaptive multilevel splitting

Read from the source of fgsea 1.38.0 (`src/ScoreRuler.cpp`,
`src/fgseaMultilevelSupplement.cpp`) on 2026-08-19. It is **not** MCMC in the
posterior-sampling sense; it is multilevel splitting for rare-event estimation,
with an MCMC move as its inner step.

**The algorithm.** Maintain a population of `sampleSize` (~1,000) random gene
sets. Then repeat:

1. score every element; **discard the lower half and duplicate the upper half**
   (`duplicateSampleElements`), recording the discarded scores as the level;
2. move each survivor by an MCMC step — swap one gene out of the set and one
   in, **accept only if the score stays above the current threshold**
   (`updateElement`);
3. the threshold ratchets up to the population median; stop when it passes the
   observed score, or when the move-acceptance rate falls below 1%.

After `k` rounds the survivors represent the distribution conditioned on
`score >= threshold_k`, and `P ~ (1/2)^k`. A p-value of 1e-100 is just
k ≈ 332 rounds.

**The estimator is not naively `2^-k`.** Because each level is the *sample*
median, its exceedance probability is a Beta order statistic, so they accumulate
`E[log U]` rather than `log E[U]`: `adjLogPval = k * betaMeanLog(halfSize,
sampleSize) + betaMeanLog(remainder + 1, sampleSize)`. This is what makes
`log p` approximately unbiased instead of systematically optimistic.

**The error bar is explicit and good:**

```
log2err = sqrt( floor(-log2(p) + 1) * (trigamma((N+1)/2) - trigamma(N+1)) ) / log(2)
```

At `p = 1e-100` with `N = 1,000` this is ≈ 0.83 — the p-value is known to
within a factor of about 1.8. The error grows only as `sqrt(k)` and shrinks as
`1/sqrt(N)`.

**How it would map onto WADE.** Cleanly. fgsea's "swap one gene in the set"
becomes "**swap one case label with one control label**", which is a uniform
random walk on label assignments with the permutation distribution as its
stationary distribution. Accept the swap iff the statistic stays above the
current threshold. Everything else — the halving, the Beta correction, the
error formula — carries over unchanged.

**What it costs us, and the design tension.** It is **per gene**: each gene has
its own threshold and therefore its own chain. WADE's whole inference design is
"one shuffle serves all genes", which preserves the gene–gene correlation
structure of the joint null. Multilevel splitting abandons that for the refined
genes. Marginal p-values stay valid; the joint null does not. Since refinement
would only fire where the empirical p-value has run out — exactly where GPD
fires now — the practical exposure is limited, but it must be stated.

**Should we believe 1e-100?** The permutation p-value at 1e-100 is a
well-defined combinatorial quantity and the estimator above genuinely estimates
it. But it is a statement about an exchangeability model that is certainly
wrong at that precision — batch, ancestry, RIN, capture efficiency. **Model
error dominates long before 1e-20.** Use these for ordering; never quote one as
evidence strength. If they are reported at all, report the `log2err` beside
them, as fgsea does.

### 4.4 Saddlepoint approximation for stage 1 — possibly better than either

**Hypothesis.** If `mean_shift` is (or becomes, §3.1) a **linear** statistic —
the sum of the case subset — then its permutation distribution is the
distribution of a sum over a random subset without replacement, for which a
classical **saddlepoint approximation** (Robinson 1982) gives *relative-error*
accuracy far into the tail at `O(n)` per gene **with no permutations at all.**

**Measured 2026-09-02 (§4.5): it works, and it is the most accurate of the
four** — within 1% of brute force down to 1e-6 and within 30% at 1e-8, with no
permutations at all. And the constraint is **linearity, not balance**: for any
group sizes `T = A(1/n1 + 1/n0) - V/n0` with `A` the case-subset sum, so the
tail probability is a subset-sum problem whatever the design. What rules it
out on an unbalanced design is that `mean_shift` is then the grid quadrature —
an L-statistic — rather than the mean difference (§3.1, `limits.md` §2.5).
Taking §3.1's open decision to define `mean_shift` as the mean difference
outright is therefore also the decision that makes the saddlepoint universal.

**Why this is attractive.** It would give stage 1 essentially unlimited
p-value resolution for the cost of a root-find per gene — cheaper than 2,000
permutations, let alone 332 rounds of multilevel splitting. It also degrades
gracefully: the saddlepoint is exact in the limit and its error is bounded
relative, not absolute, so the far tail is where it is *best*.

**What it does not cover.** Stage 2's statistic is a maximum over widths of a
standardized bridge — emphatically not linear, no saddlepoint. Stage 2 would
still need GPD or multilevel.

**Validate by:** comparing saddlepoint p-values against brute-force permutation
at a scale where 1e7 permutations are affordable (small `n`, few genes), across
skewed and zero-heavy count distributions where the normal approximation fails.
The claim to test is *relative* accuracy at 1e-6 and below, not agreement in
the bulk.

### 4.5 The comparison, run

**Answered 2026-09-02**, `tools/pvalue_study.py`. 40 v 40, log-normal
expression, stage 1's `mean_shift`, upper tail. Ground truth is **1e9
streamed permutations** — 46 of 64 genes resolve with at least ten
exceedances, and the ladder of planted shifts spans true `p` from 2.3e-3 to
1.3e-7. Each candidate is then given 2,000 permutations, or none.

Ratio of the estimate to the truth, and the worst `|log10|` error in the bin:

| true `p` | n | empirical B=2k | GPD B=2k | saddlepoint | multilevel |
|---|---|---|---|---|---|
| 1e-3 – 1e-2 | 4 | 1.32x  (0.17) | 3.20x  (0.70) | **0.99x  (0.00)** | 0.97x  (0.08) |
| 1e-4 – 1e-3 | 9 | 3.28x  (0.93) | 12.4x  (1.65) | **1.00x  (0.00)** | 0.93x  (0.09) |
| 1e-5 – 1e-4 | 10 | 21.8x  (1.66) | 75.5x  (2.16) | **1.00x  (0.01)** | 0.95x  (0.10) |
| 1e-6 – 1e-5 | 7 | 115x  (2.61) | 270x  (2.72) | **1.00x  (0.02)** | 0.94x  (0.12) |
| 1e-7 – 1e-6 | 8 | 2502x  (3.61) | 2118x  (3.85) | **1.01x  (0.04)** | 1.10x  (0.12) |
| 1e-8 – 1e-7 | 2 | 25113x  (4.62) | 9899x  (4.22) | **1.11x  (0.26)** | 0.93x  (0.29) |

| method | cost, 64 genes | resolution reached |
|---|---|---|
| empirical, B = 2,000 | ~0 s | `1/(B+1)` = 5e-4, as advertised |
| GPD, B = 2,000 | ~0 s | **~1e-4 in practice**, not the nominal 2e-6 |
| saddlepoint | 17 s (unoptimized) | none — no sampling at all |
| multilevel | 0.5 s | none — error grows as `sqrt(k)` |

**The GPD refinement does not reach its own floor, and is conservative by
orders of magnitude below about 1e-4.** That was the surprise. It is not
merely censored at `1/(B·n_tail)`; on this statistic it never gets there,
bottoming out around 5e-5 to 1e-3 while the truth is 1e-7 or smaller.

**The cause is the `xi <= 0` branch, and it is doing exactly what it was
written to do.** For all 17 genes with true `p` below 1e-5 the fitted shape is
negative (median `xi` = −0.21), so all 17 take the exponential limit. A GPD
with negative shape has a hard upper bound at `-sigma/xi`; `gpd_tail_p`
substitutes the exponential rather than let an observation past that bound
collapse to a machine-epsilon p-value (its docstring says so). The permutation
null of a mean difference on 80 samples *is* bounded — the statistic is
maximal when the largest `n1` values are all cases — so a negative shape is
the correct fit and the exponential is a far heavier tail than the truth. The
guard against being wildly anti-conservative is what makes it wildly
conservative instead.

**Both alternatives work, and both are accurate where the GPD is not.** The
saddlepoint is essentially exact — within 1% to 1e-6, within 30% at 1e-8 —
for `O(n)` per gene and no permutations. Multilevel splitting holds 0.93x to
1.10x throughout at 8 ms per gene, which is what fgsea's own `log2err` would
predict.

**What this costs today.** The refinement fires when a gene has fewer than
`n_exc_min` exceedances, which at B = 2,000 is `p` below roughly 5e-3 — and BH
across 20,000 genes decides at around 2.5e-6. Genes whose true `p` is 1e-5 to
1e-7 are therefore both *the ones BH is ruling on* and the ones being reported
75x to 2000x too large. The ordering survives, because the distortion is
monotone; the calling does not. §7.1's 18.8% of genes at the floor on the real
contrast is the same phenomenon seen from the other side.

**Scope.** One design (40 v 40, balanced), continuous data, upper tail. An
unbalanced design is not covered and cannot be, for stage 1: it is an
L-statistic there rather than a subset sum (§4.4), so neither the saddlepoint
nor this reduction applies. Stage 2 is below.

### 4.6 Stage 2, which is a different problem

Same design, NB counts, `k` affected cases of 40 at 8x, ground truth 2e8
streamed permutations. **The ladder here is `k`, not the effect size** — at a
fixed `k = 8`, subsets of 2x through 25x all returned `p` between 2.5e-4 and
5e-4, because the gene had reached the combinatorial floor for `k = 8` and no
effect size buys depth a design has not got. 29 of 64 genes resolve.

| true `p` | n | empirical B=2k | GPD B=2k | multilevel |
|---|---|---|---|---|
| 1e-3 – 1e-2 | 10 | 1.47x  (0.24) | **1.15x  (0.38)** | |
| 1e-4 – 1e-3 | 7 | 2.86x  (0.68) | **0.80x  (1.71)** | |
| 1e-5 – 1e-4 | 5 | 28.4x  (1.63) | 4.09x  (0.98) | **0.94x median,** |
| 1e-6 – 1e-5 | 3 | 171x  (2.33) | **1.43x  (0.43)** | **worst 2.5x,** |
| 1e-7 – 1e-6 | 3 | 3447x  (3.58) | 15.5x  (2.04) | **over the 12** |
| 1e-8 – 1e-7 | 1 | 7139x  (3.85) | 80.6x  (1.91) | **genes < 1e-4** |

**The GPD is far better on stage 2 than on stage 1, and the mechanism says
why.** §4.5 blamed the `xi <= 0` exponential substitution. Stage 1's fitted
shape is negative for **100%** of deep-tail genes (median −0.21) and 100% take
that branch; stage 2's is negative for **71%** (median −0.07) and 57% take it.
A maximum over widths has a heavier null tail than a mean difference, so the
GPD form fits and the substitution bites less often. The prediction and the
measurement agree: the damage tracks how often the branch fires.

**So the p-value problem is worse for stage 1 than for stage 2**, which is the
opposite of what the stages' relative importance would suggest. Stage 1 is
75-2000x conservative where BH decides; stage 2 is 4-80x, on few genes.

**Multilevel splitting works on stage 2, and needs one thing first.** Stage 2's
statistic is `max_k (B_k - mu_k)/sigma_k` with the moments estimated from *the
same permutations the tail is read from*, so it is **not a fixed function of a
label assignment** — the observed value moves 3% between permutation samples
at B = 2,000 (0.09% at 500,000, which is why brute force streams blocks that
large and lets each carry its own observed value). Multilevel samples the
distribution *conditioned* on exceeding a level, whose moments are not the
null's, so pointed at stage 2 as it stands it would standardize by the wrong
numbers and estimate the tail of a different statistic, silently.

Freezing `mu` and `sigma` from one uniform sample fixes it, and is better than
what is done now: they are bulk quantities converging as `1/sqrt(B)` while the
tail is a rare-event quantity, and estimating them separately untangles two
error sources that are currently conflated. With that done, multilevel holds
**0.94x median and 2.5x worst** over the 12 genes below 1e-4, at 1.1 s/gene —
a thousand times slower per gene than on stage 1, because a label swap changes
the whole quantile function and there is no O(1) update.

`tools/pvalue_study.py` captures stage 2's real inputs by wrapping
`subset_null_backend` for one call rather than rebuilding them from the six
pieces `wade()` assembles, which would be six chances to diverge silently. The
frozen evaluator reproduces the kernel to **0.000e+00** on both the observed
labels and a random permutation.

**What this leaves.** Stage 2's deep bins hold 1 to 3 genes: enough to show the
GPD degrading, not enough to price it. An unbalanced design is untested for
both stages. And shipping either alternative means `subset_null_backend`
taking frozen moments, which changes what stage 2 reports.

---

---

## 5. Single cell and spatial — the regime after this one

Thousands to millions of cells, sparse and zero-inflated. Not to be implemented
yet. Three things to think about now, because two of them are free if the work
in §2 is done right and the third cannot be engineered away.

### 5.1 Sparsity is an opportunity, not just a storage format

At >90% zeros, WADE's core operation — a per-gene quantile over all cells —
becomes much cheaper, not just smaller. **A gene's sorted values are a run of
zeros followed by the sorted non-zeros.** So:

- the sort is `O(nnz log nnz)`, not `O(n log n)`;
- a quantile at `p < n_zero/n` is exactly zero (or the pseudocount) — no lookup
  needed;
- the per-permutation partition of §3.2 only has to walk the **non-zeros**,
  provided the count of zeros landing in each group is tracked, which is a
  single dot product of the label vector with the gene's zero mask.

This makes the natural single-cell implementation *sub-linear in cells* per
permutation. It also makes the grid cap of §2.1 essential rather than merely
helpful: `m = min(n1, n0)` would be 500,000.

The count-native machinery already fits: zeros stay zeros under binomial
thinning, and the one-count pseudocount is exactly the right treatment of a
zero on the log scale (`method.md` §10.4).

### 5.2 A sparse `Counts` is already half-designed

`wade.as_counts` accepts anything with `.toarray()` today and densifies it. The
change would be to keep the sparse representation and teach the kernels the
zero-run structure above — the accepting boundary does not need to move.

### 5.3 The trap: cells are not exchangeable, donors are

**This is the important one.** A million cells from twenty donors is not a
million independent observations. WADE's null is built by permuting labels, and
labels attach to *donors*, not cells; permuting cell labels would test a
hypothesis nobody holds and would return p-values driven entirely by
pseudo-replication. `limits.md` §2.2 already says this about repeated measures,
and single cell is that problem at extreme multiplicity.

Two honest options, both worth writing down before any code:

1. **Pseudobulk.** Aggregate to one profile per donor per cell type, then run
   WADE with `n` = number of donors. The cell count buys a *better estimate per
   donor*, not more samples. This is what most current best practice does, and
   it makes the design small again — which puts the combinatorial floor
   (`limits.md` §1) right back in the foreground.
2. **Restricted permutation.** Permute donor labels, keeping each donor's cells
   together. This preserves exchangeability and *does* use the within-donor
   distribution — which is exactly what WADE's subset stage is about, and is
   arguably the most interesting thing WADE could offer single cell: **"is this
   gene altered in a subset of *cells* within the affected donors?"** That is a
   genuinely different question from pseudobulk DE, and the machinery for it —
   a distributional test with a shape stage — already exists here.

Option 2 is a research direction with a real claim behind it. It should not be
started until §2 and §3 land, but it is the reason to do them well.

---

## 6. Standing questions to answer along the way

- **Does the bootstrap's resampling with replacement degrade the quantile
  grid?** Duplicated samples create flat runs — the same degeneracy the
  continuity jitter exists to break (`method.md` §8). `affected_fraction` is a
  functional of the curve's shape, so ties could bias it. Measure the bootstrap
  distribution against a subsampling (`m`-out-of-`n`, without replacement)
  alternative before trusting narrow intervals.

  **Answered 2026-08-20** (`tools/bootstrap_scheme_study.py`; 200 v 200, NB
  dispersion 0.1, 200 datasets × 400 replicates per cell, truth = the
  estimator's sampling distribution over 2,000 independent datasets). Four
  schemes on raw counts with wade's jitter and pseudocount: **resample**
  (with replacement, what ships), **poisson** (keep every sample, redraw
  each count as Poisson(count) — proposed by the user as a fair,
  tie-free resample), **hybrid** (resample columns, then Poisson-redraw the
  drawn counts), and **m-of-n** (n/2 without replacement, √(m/n)-rescaled).
  95% CI coverage of the sampling-distribution median / (bootstrap SD ÷
  true SD):

  | scenario, estimand | resample | poisson | hybrid | m-of-n |
  |---|---|---|---|---|
  | null nb(50), aff | 0.96 / 0.86 | 0.98 / 0.74 | 1.00 / 1.03 | 0.81 / 0.93 |
  | null nb(50), lfc | 0.94 / 1.00 | **0.56 / 0.41** | 0.95 / 1.07 | 0.84 / 1.01 |
  | global 2× nb(50), aff | 0.91 / 0.76 | 0.83 / 0.79 | 0.98 / 1.23 | 0.77 / 0.75 |
  | global 2× nb(50), lfc | 0.98 / 1.01 | **0.54 / 0.37** | 0.98 / 1.07 | 0.85 / 1.01 |
  | subset 5% 8× nb(50), dir | 0.94 / 1.16 | **0.66 / 0.45** | 0.99 / 1.28 | 0.77 / 1.21 |
  | global 2× nb(5), aff | 0.94 / 0.93 | 0.61 / 1.00 | 0.82 / 1.18 | 0.81 / 0.94 |
  | global 2× pois(50), aff | 0.94 / 0.92 | 0.64 / 1.82 | 0.79 / 2.03 | 0.81 / 0.91 |

  **The ties question is answered: no.** With-replacement coverage is
  0.91–1.00 everywhere; the flat runs the duplicates create do not bias the
  intervals measurably (the jitter travels with the duplicated columns, and
  the quartic form of `affected_fraction` suppresses the floor).

  **Poisson-only resampling is anti-conservative on overdispersed counts**
  — half-nominal coverage and bootstrap SDs at 0.3–0.5× the truth on NB
  data — for a structural reason: redrawing counts as Poisson(count)
  reproduces the *measurement* noise but holds the sample panel fixed, so
  the between-sample (biological, NB overdispersion) component of the
  estimator's sampling variance is simply absent. It is calibrated exactly
  where its assumption is true (pure Poisson data, lfc: 0.93 / 0.99). Real
  cohorts are the overdispersed case, so it cannot replace resampling.

  **The hybrid is the interesting one**: at moderate/high counts on NB data
  it never under-covers and fixes resample's mild SD under-dispersion on
  `affected_fraction` (0.86 → 1.03, 0.76 → 1.23) at ~5–10% extra width —
  the Poisson redraw de-duplicates the tied columns, which was the sound
  half of the proposal. But it double-counts the Poisson component, and at
  low counts (nb(5): 0.82) or pure Poisson (0.79) that extra noise floor
  biases `affected_fraction`'s replicates and costs real coverage.
  **Verdict: resampling with replacement stays the default; not wired as an
  option** — a scheme that under-covers in plausible regimes invites
  misuse, and the shipped scheme was never the miscalibrated one. m-of-n
  under-covers broadly (0.77–0.85) for these non-linear functionals.
- **Does the grid cap interact with the thinning?** The fold-change fit uses
  interquartile means, which are grid-free, but the bridge is not. Check the
  false-subset table of `method.md` §10.3 at a capped grid.
- **What `B` is right at 80,000 samples?** The empirical floor `1/(B+1)` and
  the GPD floor both scale with `B`, but so does runtime. If §4.4 lands, stage
  1 may need no permutations at all and `B` becomes a stage-2-only parameter.
- **Is `w1` worth carrying at this scale?** It is reported, tested, and
  consumed by nothing (`ROADMAP.md` open questions). It costs a full `(g x m)`
  reduction.

---

## 7. What the real cohort measured

`notebooks/rna100k.qmd`, first run 2026-08-20: 30,976 genes × 83,047 libraries
of splice-junction counts, `gene_num_introns` as the normalizer, harmonized
metadata joined by `library` (228 CPTAC_MEL libraries not yet harmonized). The
full-cohort run is gated in the notebook; the smoke contrast is plasma
malignant v non-malignant, 1,343 v 1,662, **126 s** end to end. Five findings,
each of which changed a priority.

### 7.1 Significance saturates, and it is not a bug

The contrast called **29,206 of 30,976 genes** subset-significant at BH 0.05.
Two controls settle what that is:

* **Permuted labels** — the same matrix, same depth spread, same study
  mixture, labels shuffled once: **0 mean-shift and 53 subset genes** at BH
  0.05, raw `p_subset < 0.05` for **3.9%** of genes against a nominal 5%. The
  machinery is calibrated on this data.
* **One study only** (`pw_pdac_bloodev`, 283 v 117): still **~70%** of genes
  called, with its own depth imbalance *reversed*. Holding study fixed does
  not deflate it.

So at *n* in the hundreds-to-thousands, stage 2's null — *one* global fold
change explains the whole distribution — is a **point null**, and every real
deviation from it, however small and whether biological or technical, rejects.
**Significance has saturated as an instrument.** The operative outputs are the
rankings and the descriptors, which is what `subset_log2_fc` and the
permutation z-scores were added for (§4.0).

### 7.2 A few anomalous libraries drive many genes' subsets

Chasing the magnitude ranking's top found FLI1, EWSR1, TRGC2 and LYVE1 sharing
**23–24 of their top 25 driver libraries** — and FLI1 with the *unrelated*
TRGC2 sharing 24, *more* than the FLI1/EWSR1 "fusion pair" shared. That is what
falsified a tempting Ewing-sarcoma reading.

The mechanism is library composition, not normalization: those libraries detect
**5,563 genes against 11,114** elsewhere and hold **36% of their mass in ten
genes** (cohort median 15%), with raw FLI1 junction counts of ~297,000 against
~1,600. Blood-cell transcripts dominating a plasma prep. At cohort scale the
skew is severe: **some libraries sit in the top 25 of a third of all genes**
(10,229 of 30,976; 258 expected if uniform).

**`ETV4` is the counter-example** and survives every control: zero overlap with
those libraries, ordinary complexity (10,483 genes detected, 13.9% top-10
share), *below*-median depth (2.1M), median **0 TPM** across all 3,005 plasma
libraries, and its top 25 are **25 distinct patients, all PDAC, replicated
across two independent studies**.

**The obvious diagnostic was measured and refuted.** "How often do this gene's
drivers drive everything else?" does *not* flag the artefact: FLI1's and
TRGC2's drivers appear in 375 of 6,769 up-gene subsets against a cohort median
of 1,769 — far *below* average — because a low-complexity library is zero in
~25,000 genes and therefore ranks at the **bottom** of most genes' orderings.
What discriminates is the drivers' **own library complexity**, which is why
`library_qc()` and `subset_drivers()` exist and why a recurrence score does not.

### 7.3 Group-associated depth

Median depth 8.1M (malignant) against 4.8M (non-malignant) in the plasma
contrast — 1.7× — and *reversed* (6.8M v 8.9M) in the single-study one. Cohort
depth spans 0.8M–89M, 17× between the 5th and 95th percentiles.

Library-size normalization removes depth's first moment. What nothing removes
is the **second**: after normalization a shallow library's values are noisier
than a deep one's, so group-associated depth is a genuine distributional
difference between the groups — technical, but real, and a distributional test
will see it. Binomial thinning does run here, but it is stage 2's per-gene
*fold-change* correction and does not equalize depth. The candidate guard is
**depth-equalizing thinning as preprocessing** — thin every library to a common
depth, exact for counts, at the price of discarded reads — and the experiment
that would justify it is a planted depth imbalance with false-positive rates
per stage. ETV4's drivers sitting *below* median depth says depth is not a
simple monotone confound, so this needs measuring rather than assuming.

### 7.4 The p-value floor, observed

**18.8% of genes** sat at the mean-shift resolution floor. That is §4.1's
predicted pile-up, on real data. It is also why §4 is *demoted* rather than
urgent: the ordering it would buy is already supplied by magnitude and the
z-scores.

### 7.5 47% zeros

Real sparsity, but not single cell's >90%. §5.1's zero-run walk would roughly
**halve** the subset stage here, not decimate it — worth having, not urgent.

### 7.6 What this demands, in order

1. **A rank-recovery benchmark on real background** — plant known subset and
   global signals into real non-malignant plasma libraries (so the background
   keeps every technical wart) and measure whether the ranking recovers them
   and where the technical tail begins. The direct test of "are the top-ranked
   genes the true ones".
2. **The depth experiment** of §7.3.
3. **Restricted permutation** — landed 2026-08-21 as `strata=`; the defensible
   contrasts on this cohort hold study fixed.
4. **P-value resolution** (§4), demoted as above.
