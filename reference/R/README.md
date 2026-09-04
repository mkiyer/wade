# `reference/R/` — the original R implementation, frozen

**The R implementation is retired.** The port is complete, this code is no
longer developed, and nothing in WADE depends on it at run time or at test
time: the suite reads the golden fixtures in `tests/fixtures/*.json` and never
invokes R.

It is kept, frozen and checksummed, for one reason. Those fixtures were
*generated* from this code, so it is the only way to check a future change
against the original rather than against WADE's own output. Regenerating them
(`tools/r/generate_fixtures.R`) is the one task that still needs a working R,
and it is rare. `tests/test_reference_is_frozen.py` fails if any file here
changes, because evidence that changes is not evidence — a patch belongs in
the harness, never here.

It is reference material, not a package: there is no `DESCRIPTION`, no
`NAMESPACE`, and nothing here is intended to be installed or maintained.

`wade.R` is a byte-identical copy of the cfRNA original and must stay that way.
Verify with `shasum -a 256 -c sha256sums.txt` from this directory (8 files, all
must report `OK`); source paths, the cfRNA git SHA, and what is *not* a verbatim
copy are recorded in `CHECKSUMS.txt` and `../PROVENANCE.md`.

## Running it

Three things are required, and all three are easy to get wrong:

```bash
export PATH="/usr/local/bin:$PATH"                 # 1. R 4.6.1 is not on the default PATH
cd <repo>/reference/R                              # 2. run FROM this directory
RENV_CONFIG_SANDBOX_ENABLED=FALSE \
RENV_PATHS_CACHE="$HOME/Library/Caches/org.R-project.R/R/renv/cache" \
  Rscript validation_sims_v7.R                     # 3. see the two env vars below
```

1. **`PATH`.** R 4.6.1 lives at `/usr/local/bin/R` and is not on the default
   `PATH`.
2. **Run from this directory.** The renv project root is `reference/R/`, not the
   repo root — that is deliberate, so activating this sandbox cannot put R's
   library path anywhere near the rest of the repo. `.Rprofile` here sources
   `renv/activate.R`, which only happens if R starts with this as its working
   directory. `validation_sims_v7.R` also does a plain `source("wade.R")`.
3. **`RENV_CONFIG_SANDBOX_ENABLED=FALSE`** skips renv's base-library sandbox
   copy, which otherwise costs about 13 s on every R start and emits a warning
   when it exceeds its own time budget. It has nothing to do with the security
   sandbox; it is purely a startup cost.
4. **`RENV_PATHS_CACHE`** is the one that is genuinely load-bearing — see below.

## The offline-cache constraint

There are **no R 4.6 arm64 binaries on CRAN**, and the two Posit mirrors that
would serve them are unreachable from the environment this was staged in. So
nothing here can be installed, built, or upgraded from a repository. The only
supply of packages is a **pre-existing, read-only shared renv cache** at
`~/Library/Caches/org.R-project.R/R/renv/cache`, which already contains the 31
packages this subtree needs at exactly the versions cfRNA pinned.

`renv.lock` therefore records only that closure — 31 of the 288 entries in the
cfRNA lockfile — with the version and hash of every entry copied out verbatim so
the cache lookups hit. `renv::restore()` **links** each package from the cache
into `renv/library/…`; it does not copy or compile. Every package in the project
library is a symlink into the shared cache, which is why the library takes no
meaningful disk space and why the cache must never be written to.

Two consequences worth knowing before you touch the lockfile:

- **`RENV_PATHS_CACHE` must be set.** In this environment `XDG_CACHE_HOME` is
  redirected to a workspace-local `./.cache`, so renv resolves its cache root to
  `./.cache/R/renv` — an empty directory — and concludes it must download
  everything. Pointing `RENV_PATHS_CACHE` at the real shared cache is what makes
  restore an offline link operation. Set it to the cache root
  (`…/org.R-project.R/R/renv/cache`), **not** to the platform subdirectory
  underneath it; renv appends `v5/macos/R-4.6/<arch>` itself, and pointing at the
  deeper path makes it try to create that suffix twice and fail.
- **Adding a package means finding it already in the cache.** There is no
  install path. `renv::install()` will fail. If a package is not in the shared
  cache at a version compatible with the rest of the closure, it cannot be used
  here — write the code without it, or guard the code that needs it (as
  `validation_sims_v7.R` does for its plotting block).

### A known wrinkle in `renv::restore()`

On a **fresh, empty** project library, `renv::restore()` links all 31 packages
successfully and *then* fails with:

```
Error in renv_socket_server() : error creating socket server: couldn't find open port
```

This is not a package problem. renv's parallel installer opens a local socket
server to coordinate worker processes, and this environment does not permit
binding a listening socket (verified directly: `serverSocket()` fails on every
port tried, including an OS-assigned one). The failure happens *after* the
linking work is done, so the library is complete and correct.

**What to do:** run `renv::restore(prompt = FALSE)` once, ignore that error, then
run it again. The second call finds the library already populated, reports
`The library is already synchronized with the lockfile`, and exits 0. Confirmed
working: 31 of 31 packages present as symlinks into the shared cache, at the
locked versions.

`renv::status()` reports two remaining inconsistencies, both expected and
neither worth fixing:

- **`renv` itself** — installed (bootstrapped into the project library by
  `activate.R`) but not recorded in this lockfile. Normal for a project that
  pins only its analysis dependencies.
- **`ggrepel`** — recorded in the lockfile but not installed. It was referenced
  only by the cfRNA figure layer, which has since been removed (see
  `CHECKSUMS.txt`). Nothing that runs here needs it.

## What each file is

| file | what it is |
| --- | --- |
| `wade.R` | **The port's source of truth.** Byte-identical copy of the cfRNA original: normalization, the quantile-area statistic, the permutation loop with GPD tail refinement, the driver, the rank scores, and the per-gene detail function. Do not edit. |
| `validation_sims_v7.R` | The three method-validation simulations from the v7 notebook's §9.6, as a standalone runnable script — null calibration, subset-vs-bulk discrimination, and power against the combinatorial permutation floor. Writes a three-panel PNG and two CSVs. Runs in about 95 s on one core. |
| `CHECKSUMS.txt` | Byte-identity record: digests, source paths, cfRNA git SHA, and an explicit list of what is *not* a verbatim copy and why. |
| `sha256sums.txt` | The same digests, digest-only, for `shasum -a 256 -c`. |
| `renv.lock` | The 31-package dependency closure, versions and hashes copied verbatim from the cfRNA lockfile. |
| `.Rprofile`, `renv/activate.R`, `renv/settings.json`, `renv/.gitignore` | Copied verbatim from the cfRNA repo, per house convention. `renv/.gitignore` keeps `library/` and `staging/` untracked; `renv.lock` is committed. |
| `validation_sims_v7.png`, `validation_sims_v7_*.csv` | Run products, regenerated on every run. |

## What the validation script produced here

Recorded so a port has something concrete to diff against. These are **measured
outputs of this sandbox**, reproduced identically across three consecutive runs
(1000 permutations, seeds as in the notebook, 77 cases vs 18 controls, giving an
18-point quantile grid and a 2-point tail window):

**(a) Null calibration** — Kolmogorov-Smirnov test against Uniform(0,1):
*p* = 0.6554 on the bulk axis (`diff.mean`), *p* = 0.2327 on the subset axis
(`tail.mean`). Realised type-I error at the nominal 0.05 level: 0.0512 (bulk)
and 0.0488 (subset). Both axes are at nominal; the permutation p-values are not
inflated. Note that the KS test warns about ties, because permutation p-values
are discrete — expected, and its *p* is conservative rather than exact, which is
why the type-I error rates are reported alongside.

**(b) Subset-vs-bulk discrimination** — median `tail.mean` for the 25 planted
8%-subset genes is 27,537 against a median `diff.mean` of 3,046, a ratio of 8.9;
the 25 planted bulk-shift genes sit at 4,064 versus 1,348, a ratio of 3.3, and
the 600 null-background genes at 535 versus −46. The subset genes separate from
the bulk shifts on the tail axis, which is the reason the tail statistic exists.

**(c) Power vs subset fraction** — BH-adjusted (`q` = 0.10) detection power for
80 planted subset genes against a 600-gene background, with the exact
combinatorial floor `exp(lchoose(n1,k) - lchoose(nn,k))`:

| altered fraction | k | permutation floor | power |
| --- | --- | --- | --- |
| 3% | 2 | 6.55e-01 | 0 |
| 5% | 4 | 4.25e-01 | 0 |
| 8% | 6 | 2.73e-01 | 0 |
| 12% | 9 | 1.37e-01 | 0 |
| 20% | 15 | 3.20e-02 | 0 |
| 35% | 27 | 1.15e-03 | 1 |
| 50% | 38 | 2.79e-05 | 1 |

The 0→1 transition between 20% and 35% is **not** a simulation artefact and not
a limit of the permutation count. With 80 true positives among 680 genes, BH at
*q* = 0.10 can only declare p-values at or below 0.10 × 80/680 = 0.0118; the
floor column crosses that threshold exactly between those two rows. Raising
`nperms` cannot move it — the floor is a property of the design.

**Two discrepancies with the v7 notebook, both documented in the script's
header:**

- The notebook's figure caption and prose put this collapse "below ~5% of
  cases". As re-run here, power is 0 at every fraction through 20% and 1 from
  35% on, so the first fraction exceeding 0.5 power is 35% (~27 of 77 cases),
  not ~5%. The 35% figure is what this script computes and is fully explained by
  the BH arithmetic above.
- The notebook's control count is ambiguous. Its simulation chunk set
  `n0v <- n_noncancer` (the pooled non-cancer reference) while the figure
  caption says "18 controls", which internal evidence identifies as `n_ctrl`
  (the healthy controls alone); a code comment in the same section spans
  "18-33 controls". Both readings were run: they agree on every qualitative
  conclusion and give the identical 35% transition, so nothing above depends on
  resolving it. The script defaults to 18 and documents how to get 33.

A smoke test of `wade.R` itself was also run (30 genes, 6 cases vs 5 controls,
one gene planted high in 2 of 6 cases): the driver returns a 30 × 19 frame with
all p-values in [0, 1], the planted gene ranks first on both the bulk and subset
axes, `wade_score()`'s scores stay inside [−1, 1], and `wade_gene()`'s cumulative
signed area at *p* = 1 equals that gene's `diff.mean` to floating-point equality
— the identity that function is built to satisfy.
