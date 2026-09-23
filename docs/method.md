# WADE 0.2.0 method

One question: does a gene's expression distribution differ between two groups?
The test is unsigned empirical Wasserstein distance. Bulk and tail descriptions
follow the test and are not additional hypotheses.

## Preprocessing and null

Fit label-independent sample factors on a declared normalization universe after
initial QC. RLE uses genes positive in every fit sample, their geometric means,
and column medians of log ratios, centered to geometric mean one. Explicit
validated reference IDs may restrict that universe. No automatic control
discovery or noise is applied. The expression is

\[x_{gj}=\log_2(c_{gj}/s_j+1).\]

Fix the later test-gene mask without refitting factors. Under the null, labels
must be exchangeable for the preprocessed data conditional on any strata.
A label-invariant procedure is necessary but cannot create exchangeability
when depth, technical effects, dependence or sample selection violates it.

## Exact distance

For empirical group CDFs, the statistic is

\[W_g=\int |F_{1g}(x)-F_{0g}(x)|\,dx
       =\int_0^1 |Q_{1g}(p)-Q_{0g}(p)|\,dp.\]

Sort all N observations. Let d_k be the gap between sorted observations k and
k+1, and a_k the cumulative number of cases through k. With n1 cases and n0
controls, calculate

\[W_g=\frac{1}{n_1n_0}\sum_{k=1}^{N-1}|N a_k-n_1 k|d_k.\]

Each group has its own empirical probability masses. No common coarse quantile
grid, smoothing, endpoint trimming, or interpolation is used for inference.
Ties have zero gaps. Absolute area prevents opposing changes from canceling.
An exact distance does not mean an exhaustively enumerated permutation test.

## Resampling

Generate independent label shuffles with replacement, retaining group sizes
and stratum counts. Use the same schedule for all genes and chunks. Sort each
gene once and accumulate exact W1 for each permitted assignment.

The inclusive comparison uses the observed numerator times
`1 - 100 * numpy.finfo(float).eps`, a small conservative tolerance for numerical
ties. The observed and null sums use the same sequential order. Native Rust and
readable NumPy implementations are checked against independent SciPy distances.

Use a prespecified Besag–Clifford stopping count h=20 and cap B=499,999 by
default. If the h-th exceedance arrives at draw T, p=h/T; otherwise finish B
and use p=(K+1)/(B+1). The actual crossing within a batch is the stopping time.
For B<20 use h=B. The rule is justified under exchangeability, with inclusive
ties conservative; arbitrary early stopping of a counted p-value is not valid.
See [Besag's treatment, §2.2.2](https://imsarchives.nus.edu.sg/oldwww/Programs/mcmc/files/besag_tl.pdf)
and the [modern sequential Monte Carlo treatment, §2.1](https://academic.oup.com/jrsssb/article/87/4/1200/8106328).

A useful independent finite check integrates over a continuous uniform null
rank. Stopping at T=t has mass h/[t(t+1)] for t=h,...,B. Each capped K<h
outcome has mass 1/(B+1). The resulting p-value CDF lies at or below the
identity at every attainable threshold. Tests verify this property as well as
inclusive ties, exact crossing, cap exhaustion and invalid partial records.

Apply BH once over every retained gene. Excluded genes have no p-value or q-value.
Separate chunks do not create separate correction families. BH is not a general
guarantee under arbitrary gene dependence; the empirical confirmation includes
correlated modules and reports replicate-level uncertainty.

## Descriptive effects

Define the bulk shift as median(case log expression) minus median(control log
expression), using the ordinary average of the two central observations at
even group sizes. Define upper/lower contrasts by averaging each empirical
quantile function over [0.9,1] or [0,0.1], taking case minus control, and then
subtracting the bulk shift. Fractional quantile boundaries are integrated
exactly. These fixed descriptions introduce no new inferential fit.

A constant log-expression translation yields that bulk shift and zero residual
decile differences. A bulk decrease with an upper tail may have opposite-signed
bulk and upper descriptions. Neither descriptor receives a separate p-value,
confidence interval, biological prevalence estimate or sample membership call.

## Implementation and limits

The engine retains log expression and processes bounded gene chunks and
shuffle batches. It does not materialize a gene-by-permutation null matrix.
Sorting is O(G N log N); subsequent work scales with completed gene-permutation
sample evaluations. Genes with strong differences often reach the full cap.
A lower cap improves runtime at a documented resolution cost.

The production migration removes signed-mean discovery, fitted-fold/thinning
subset nulls, GPD extrapolation, separate stage q-values and their old result
schema. Historical sources and tests are preserved in the pre-migration archive.
No backwards compatibility aliases disguise the changed inferential target.

Conditional label tests do not answer how unusual a 10-log2 event is across
all genes. Few controls can limit recurrent-tail power despite extreme
amplitude. Those limitations are measured in the
[transcriptome confirmation](../archive/development-20260923/docs/transcriptome-workflow-validation.md). Predictive
rare-event inference is explicitly the next research phase after production
and release validation, not an additional claim of the current test.
