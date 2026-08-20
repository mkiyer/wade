# Scaling WADE

Measurements, hypotheses and directions for running WADE on cohorts far larger
than it was built for. Nothing here is implemented; everything here that claims
a number was measured on 2026-08-19 by the scripts named beside it.

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

`prototypes/` has none of this; the scripts lived in the session scratchpad.
Reproduce with the recipe at the end of each section.

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

### 2.3 float32 storage

**Hypothesis.** The stored count and normalized matrices can be float32 (9.6 GB
instead of 19.2 GB) while every statistic is still accumulated in float64.

**Risk.** Real and needs measuring. Counts above 2^24 lose integer exactness in
float32; the thinning needs integers; the parity fixtures are float64
end-to-end. Likely a storage-only option with the kernels widening on read.

**Open question.** Whether it earns its complexity once §2.1 and §2.2 land, or
whether the answer is simply "the matrix is 19.2 GB, own it".

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

### 3.2 One sort per gene in the mean-shift kernel

**Hypothesis.** The mean-shift kernel re-sorts both groups for every
permutation. The subset kernel already proves the alternative: sort the gene's
row once, then obtain each permutation's two sorted groups by a single O(n)
partition walk, because a permutation only re-partitions the same values.

**Payoff.** ~10x, and it is the path that applies where §3.1's GEMM does not
(unbalanced designs, and any future non-linear stage-1 statistic).

**Evidence it works:** the subset kernel is bitwise identical to its NumPy path
over 111 comparisons and 80x faster.

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

### 3.4 `n_boot` is serial and embarrassingly parallel

200 bootstrap replicates cost about as much as the rest of a 20,000-gene run.
Each replicate is independent; the loop is NumPy in `subset.characterization_ci`.
A thread pool or a kernel would make it nearly free.

**Related open question (see §6):** whether the bootstrap's resampling with
replacement interacts badly with the quantile grid, since duplicated samples
create flat runs in the quantile function — the same degeneracy the continuity
jitter exists to break.

---

## 4. P-value resolution

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

### 4.5 The comparison to run

One experiment settles the p-value question: pick a design small enough that
**brute force is the ground truth** (say 40 v 40, where 1e8 permutations is
hours but feasible for a handful of genes), then compare, at true p-values
spanning 1e-3 to 1e-9:

| method | cost | resolution | assumption |
|---|---|---|---|
| empirical | `O(B)` | `1/(B+1)` | none |
| GPD refinement | `O(B)` | `1/(B·n_tail)` | tail is Generalized Pareto |
| multilevel splitting | `O(k·N·iters)` per gene | unbounded, error `~sqrt(k/N)` | MCMC mixes |
| saddlepoint (stage 1) | `O(n)` per gene | unbounded | statistic is linear |

Report each method's error against the brute-force truth. That table, filled
in, is the deliverable.

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
- **Does the grid cap interact with the thinning?** The fold-change fit uses
  interquartile means, which are grid-free, but the bridge is not. Check the
  false-subset table of `method.md` §10.3 at a capped grid.
- **What `B` is right at 80,000 samples?** The empirical floor `1/(B+1)` and
  the GPD floor both scale with `B`, but so does runtime. If §4.4 lands, stage
  1 may need no permutations at all and `B` becomes a stage-2-only parameter.
- **Is `w1` worth carrying at this scale?** It is reported, tested, and
  consumed by nothing (`ROADMAP.md` open questions). It costs a full `(g x m)`
  reduction.
