"""Per-stage timing and peak memory of a wade() run.

NB counts (mean 50, dispersion 0.1), G genes by 2n samples, B permutations,
timed stage by stage through the same calls ``wade.wade()`` makes. The
timing and memory tables in ``docs/manual.md`` came from this script; rerun
it before and after any performance change.

Usage:
    python tools/bench_scaling.py --genes 1000 --n 2000 --nperms 200
    python tools/bench_scaling.py --genes 1000 --n 2000 --nperms 200 --max-probs 1000
    python tools/bench_scaling.py ... --gene-chunk 250   # the chunked driver
    python tools/bench_scaling.py ... --full             # one wade() call, total only
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return ru / 2**30 if sys.platform == "darwin" else ru / 2**20


def make_counts(g: int, n_per_group: int, seed: int = 0) -> np.ndarray:
    """NB counts, mean 50, dispersion 0.1 (gamma(10, 5)-Poisson)."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(10.0, 5.0, size=(g, 1))
    return rng.poisson(lam, size=(g, 2 * n_per_group)).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=1000)
    ap.add_argument("--n", type=int, default=2000, help="samples per group")
    ap.add_argument("--nperms", type=int, default=200)
    ap.add_argument("--max-probs", type=int, default=None,
                    help="cap the quantile grid (wade default applies when omitted)")
    ap.add_argument("--gene-chunk", type=int, default=None,
                    help="run the chunked driver with this many genes per chunk")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--stage1", default=None, choices=("grid", "gemm"),
                    help="stage-1 backend for the --full / --gene-chunk run")
    ap.add_argument("--fit-backend", default=None, choices=("numpy", "rust"))
    ap.add_argument("--full", action="store_true",
                    help="time one wade() call instead of the per-stage breakdown")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import wade
    from wade import normalize as _normalize
    from wade.permutation import draw_perms, null_statistics
    from wade.stats import wade_stats
    from wade.subset import subset_test
    from wade.thinning import fit_fold_change, one_count, thin_counts

    g, n, B = args.genes, args.n, args.nperms
    counts = make_counts(g, n)
    cond = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    normalizer = np.ones(g)

    wade_kwargs: dict = {}
    if args.max_probs is not None:
        wade_kwargs["max_probs"] = args.max_probs
    if args.gene_chunk is not None:
        wade_kwargs["gene_chunk"] = args.gene_chunk
    if args.n_boot:
        wade_kwargs["n_boot"] = args.n_boot
    if args.stage1 is not None:
        wade_kwargs["stage1"] = args.stage1
    if args.fit_backend is not None:
        wade_kwargs["fit_backend"] = args.fit_backend

    header = (f"G={g}  n={n} v {n}  B={B}"
              + (f"  max_probs={args.max_probs}" if args.max_probs is not None else "")
              + (f"  gene_chunk={args.gene_chunk}" if args.gene_chunk is not None else "")
              + (f"  n_boot={args.n_boot}" if args.n_boot else "")
              + (f"  stage1={args.stage1}" if args.stage1 else "")
              + (f"  fit={args.fit_backend}" if args.fit_backend else ""))
    print(header)

    if args.full or args.gene_chunk is not None:
        t0 = time.perf_counter()
        res = wade.wade(counts, normalizer, cond, nperms=B, seed=args.seed, **wade_kwargs)
        total = time.perf_counter() - t0
        print(f"  wade() total  {total:8.2f} s   m = {res.nprobs}   peak RSS {peak_rss_gb():.2f} GB")
        return

    # The per-stage breakdown, mirroring wade()'s pipeline call for call.
    times: dict[str, float] = {}

    def stage(name):
        class _T:
            def __enter__(self):
                self.t = time.perf_counter()
            def __exit__(self, *exc):
                times[name] = time.perf_counter() - self.t
        return _T()

    js, ps, ts, bs = np.random.SeedSequence(args.seed).spawn(4)
    jitter_rng, perm_rng, thin_rng, _ = (np.random.default_rng(s) for s in (js, ps, ts, bs))

    with stage("normalize"):
        jitter = _normalize.draw_jitter(counts.shape, rng=jitter_rng)
        lib = _normalize.library_sizes(counts, normalizer)
        tpm = _normalize.tpm_like(counts, normalizer, lib, jitter=jitter)

    stats_kwargs = {} if args.max_probs is None else {"max_probs": args.max_probs}
    with stage("observed"):
        obs = wade_stats(tpm, cond, **stats_kwargs)

    perms = draw_perms(cond, B, rng=perm_rng)
    with stage("mean null"):
        null_statistics(tpm, perms, **stats_kwargs)

    from wade.api import _fit_alpha
    with stage("f fit"):
        shift = fit_fold_change(counts, cond,
                                alpha=_fit_alpha(normalizer, lib, 1e6, counts.shape[1]),
                                seed=int(thin_rng.integers(2**31)))
    with stage("thin"):
        corrected = _normalize.tpm_like(thin_counts(counts, cond, shift, thin_rng),
                                        normalizer, lib, jitter=jitter)
    pc = one_count(normalizer, lib, counts.shape, norm_factor=1e6)
    with stage("subset"):
        subset_test(tpm, cond, perms, pseudocount=pc,
                    corrected=corrected, shift=shift, **stats_kwargs)

    total = sum(times.values())
    row = "  ".join(f"{k} {v:.2f}" for k, v in times.items())
    print(f"  {row}  |  total {total:.2f} s   m = {obs.nprobs}   peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
