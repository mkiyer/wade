"""Confirm the frozen 0.2.0 procedure on fresh seeds: protocol version 7.

research/protocols/confirmation-v7.json predeclares every check: its quantity, the acceptable and
unacceptable values, and the size that tells them apart with power 0.9, the 27 checks sharing a 5% chance
of any false failure. The protocol and this script were committed together before any version-7 data were
generated; a run records its git HEAD and refuses to start with uncommitted changes to the package, this
script, the protocol or the generators it imports.

Arms (--arm; default all, in this order):

  count_model   TCGA prostate tumours split at random per gene and thinned binomially to known depths and
                folds (research/count_model.py): per-gene null rates by expected reads
  real_labels   TCGA prostate and breast tumours under random labels, designs 1:10 to 10:1
  strata        stratified nulls (research/strata.py): simulated studies with batch effects, TCGA sites,
                the real contrast's five-site layout, TCGA and GTEx normals by study
  biology       the TCGA notebooks' recovery of ERG and ETV fusions and ERBB2 amplification (descriptive)
  simulated     transcriptome-scale matrices (wade_research.point_simulation) through the public wade():
                partial nulls (the false discovery proportion; power against a Wilcoxon comparator), then
                complete nulls (the chance of any discovery)

The per-gene arms give each gene its own labelling, which wade() cannot take: they run the functions wade()
runs, with its settings and every run finished, and a preflight checks them against wade() itself.
--evaluate scores the checks from a run's outputs.

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/confirm.py --output .release/0.2.0/confirmation-v7 [--arm ...] [--resume]
    ... python research/confirm.py --output .release/0.2.0/confirmation-v7 --evaluate
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
from scipy.stats import binom, mannwhitneyu, norm, t as student

import wade
from wade._means import area_values, mean_saddlepoint, mean_test
from wade._subsets import DRAWS, subset_test
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/confirmation-v7.json"
FROZEN = ("src/wade", "rust", "Cargo.toml", "Cargo.lock", "research/confirm.py", "research/protocols/confirmation-v7.json",
          "research/wade_research", "research/count_model.py", "research/strata.py", "tools/tcga_examples.py")
ARMS = ("count_model", "real_labels", "strata", "biology", "simulated")
CHUNK = 4096                         # genes tested at once; no p-value depends on it
TAGS = dict(preflight=199, count_model=101, real_labels=102, strata=103)
BINS = (8, 32, 128)                  # reads per sample: < 8, 8-32, 32-128, >= 128


# ---------------------------------------------------------------------------------------------
# The package's tests, as wade() runs them

def package_pvalues(counts, factors, cond, rows, seed, proc, code=None, area=True):
    """Both tests' p-values as wade() computes them with fdr=None, for genes `rows` (their row indices seed
    the jitter and thinning), each with its own samples (cases first) and size factors (one row per gene):
    the area test by its saddlepoint, shuffled where the saddlepoint's error estimate exceeds
    relative_error, and the subset test with its thinning draws. With strata (code), every stratum holds
    both groups. area=False skips the area test (NaN)."""
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    rows = np.asarray(rows)
    stop, maximum = proc["stop_exceedances"], proc["max_permutations"]
    area_p, subset_p = np.full(len(rows), np.nan), np.ones(len(rows))
    for at in range(0, len(rows), CHUNK):
        part = slice(at, at + CHUNK)
        c, f, r = counts[part], factors[part], rows[part]
        if area:
            z = area_values(c, r, f, seed, code)
            p, error = mean_saddlepoint(z, cond, direction="both", code=code)
            shuffle = error > proc["relative_error"]
            if shuffle.any():
                p[shuffle] = mean_test(z[shuffle], cond, direction="both", stop=stop, maximum=maximum, seed=seed, code=code)[0]
            area_p[part] = p
        subset_p[part] = subset_test(c, f, cond, r, direction="both", stop=stop, maximum=maximum, seed=seed,
                                     code=code, draws=DRAWS)[0]
    return area_p, subset_p


def lower_mean(counts, factors, cond):
    """Each gene's lower group mean of normalized counts (the package's validated-range measure)."""
    x = counts / factors
    return np.minimum(x[:, cond == 1].mean(axis=1), x[:, cond == 0].mean(axis=1))


def preflight(cfg, out):
    """package_pvalues against wade() itself, with and without strata, on a small simulated matrix."""
    proc = cfg["procedure"]
    sim = dict(cfg["arms"]["simulated"], seed_root=cfg["seed_root"], genes=600,
               generator=dict(cfg["arms"]["simulated"]["generator"], profile_copies=1))
    setting = dict(next(datasets(sim)), panel="mixed", seed=[cfg["seed_root"], TAGS["preflight"], 0])
    data = generate(sim, setting)
    counts, cond = data["counts"], data["cond"].astype(np.uint8)
    norm_ = wade.fit_normalization(counts)
    tested = counts.any(axis=1)
    rows = np.flatnonzero(tested)
    seed = [cfg["seed_root"], TAGS["preflight"], 1]
    code = np.arange(len(cond)) % 3                                   # three strata, each holding both groups
    result = {}
    for name, strata in (("unstratified", None), ("stratified", code)):
        res = wade.wade(counts, cond, normalization=norm_, test_genes=tested, direction="both", pvalues=proc["pvalues"],
                        relative_error=proc["relative_error"], max_permutations=proc["max_permutations"], fdr=None,
                        seed=seed, strata=strata)
        factors = np.broadcast_to(norm_.size_factors, counts.shape)[rows]
        area, subset = package_pvalues(counts[rows], np.ascontiguousarray(factors), cond, rows, seed, proc, code=strata)
        result[name] = dict(genes=len(rows), area_equal=bool(np.array_equal(area, res.area_pvalue[rows])),
                            subset_equal=bool(np.array_equal(subset, res.subset_pvalue[rows])))
    (out / "preflight.json").write_text(json.dumps(result, indent=1) + "\n")
    print("preflight:", json.dumps(result), flush=True)
    if not all(v["area_equal"] and v["subset_equal"] for v in result.values()):
        raise SystemExit("preflight failed: the per-gene arms do not reproduce wade()")


# ---------------------------------------------------------------------------------------------
# Per-gene arms

def save(folder, name, **arrays):
    np.savez_compressed(folder / f"{name}.npz", **arrays)


def arm_count_model(cfg, out, resume):
    from count_model import cohort_ratio, dataset, levels, shares
    arm, proc, root = cfg["arms"]["count_model"], cfg["procedure"], cfg["seed_root"]
    folder = out / "count_model"
    folder.mkdir(exist_ok=True)
    level, ratio = levels(), cohort_ratio("prostate")
    k = 0
    for design in arm["designs"]:
        for depth in arm["depths"]:
            for fold, exposure in arm["arms"]:
                exposure = ratio if exposure == "cohort" else exposure
                name = f"{design[0]}v{design[1]}-depth{depth:g}-fold{fold:g}-exposure{exposure:.3g}"
                k += 1
                if resume and (folder / f"{name}.npz").exists():
                    continue
                start = time.time()
                # dataset() seeds by rounded fold and exposure; a seed per data set keeps them all apart
                counts, factors, cond = dataset(tuple(design), depth, fold, exposure, root * 1000 + k)
                rows = np.flatnonzero(counts.any(axis=1))
                rows = rows[rows % arm["every"] == 0]
                p1, p0 = shares(fold, exposure)
                area, subset = package_pvalues(counts[rows], factors[rows], cond, rows, [root, TAGS["count_model"], k],
                                               proc, area=fold == 1)
                save(folder, name, rows=rows, reads=level[rows] * depth * min(p1, p0), area_p=area, subset_p=subset,
                     area_null=fold == 1, subset_null=True)
                print(f"count_model {name}: {len(rows):,} genes, {time.time() - start:.0f}s", flush=True)


def arm_real_labels(cfg, out, resume):
    from count_model import tcga
    arm, proc, root = cfg["arms"]["real_labels"], cfg["procedure"], cfg["seed_root"]
    folder = out / "real_labels"
    folder.mkdir(exist_ok=True)
    for ci, cohort in enumerate(arm["cohorts"]):
        d = tcga(cohort)
        tumour = np.flatnonzero(d["cond"] == 1)
        raw, sf = d["raw"][:, tumour].astype(np.int64), d["factors"][tumour]
        for di, (n1, n0) in enumerate(arm["designs"][cohort]):
            name = f"{cohort}-{n1}v{n0}"
            if resume and (folder / f"{name}.npz").exists():
                continue
            start = time.time()
            assert n1 + n0 == len(tumour)
            rng = np.random.default_rng([root, TAGS["real_labels"], ci, di])
            pick = np.argsort(rng.random(raw.shape), axis=1)
            counts, factors = np.take_along_axis(raw, pick, axis=1), sf[pick]
            cond = np.r_[np.ones(n1, np.uint8), np.zeros(n0, np.uint8)]
            rows = np.flatnonzero(counts.any(axis=1))
            area, subset = package_pvalues(counts[rows], factors[rows], cond, rows, [root, TAGS["real_labels"], ci, di, 97], proc)
            save(folder, name, rows=rows, reads=lower_mean(counts[rows], factors[rows], cond), area_p=area, subset_p=subset,
                 area_null=True, subset_null=True)
            print(f"real_labels {name}: {len(rows):,} genes, {time.time() - start:.0f}s", flush=True)


def arm_strata(cfg, out, resume):
    import strata as ST
    from count_model import tcga, tumours
    arm, proc, root = cfg["arms"]["strata"], cfg["procedure"], cfg["seed_root"]
    folder = out / "strata"
    folder.mkdir(exist_ok=True)
    per = arm["genes_per_set"]
    sets = []                                    # (name, maker) pairs, each maker returning one data set
    raw_t = tumours()
    expressed = np.flatnonzero((raw_t > 0).mean(axis=1) >= 0.5)
    for di, (label, (sizes, depths)) in enumerate(ST.MODEL_DESIGNS.items()):
        for batch_sd in (0.0, 0.5):
            for fold in (1.0, 1.5, 3.0):
                def make(k, sizes=sizes, depths=depths, batch_sd=batch_sd, fold=fold):
                    pick = np.sort(np.random.default_rng([root, TAGS["strata"], k]).choice(expressed, per, replace=False))
                    counts, factors, cond, code, lowest = ST.model_dataset(sizes, depths, batch_sd, fold, False,
                                                                           root * 1000 + k, pick)
                    return counts, factors, cond, code, pick, lowest, fold
                sets.append((f"studies{len(sizes)}-batch{batch_sd:g}-fold{fold:g}", make))
    d = tcga("prostate")
    raw = d["raw"].astype(np.int64)
    tumour = np.flatnonzero(d["cond"] == 1)
    site_all = np.array([s_.split("-")[1] for s_ in d["samples"]])
    names, counts_ = np.unique(site_all[tumour], return_counts=True)
    groups_sites = [tumour[site_all[tumour] == s_] for s_ in names[counts_ >= 4]]
    rank = np.empty(len(groups_sites), int)
    rank[np.argsort(-np.array([len(g) for g in groups_sites]))] = np.arange(len(groups_sites))
    real = {}
    for s_ in np.unique(site_all):
        here = site_all == s_
        n1, n0 = int(np.sum(d["cond"][here] == 1)), int(np.sum(d["cond"][here] == 0))
        if n1 and n0:
            real[s_] = (n1, n0)
    groups_layout = [tumour[site_all[tumour] == s_] for s_ in real]
    shares_layout = [max(1, min(len(g) - 1, round(len(g) * a / (a + b)))) / len(g) for g, (a, b) in zip(groups_layout, real.values())]
    eligible = np.flatnonzero((raw[:, tumour] / d["factors"][tumour]).mean(axis=1) >= 1)
    nulls = [("sites", raw, d["factors"], groups_sites, lambda s_: 0.25 if rank[s_] % 2 else 0.75, eligible, (1.0, 1.5)),
             ("layout", raw, d["factors"], groups_layout, lambda s_: shares_layout[s_], eligible, (1.0, 1.5))]
    normals, study, test_genes, _ = ST._normals()
    tested_n = np.flatnonzero(test_genes & (normals.sum(axis=1) > 0))
    sf_n = wade.fit_normalization(normals).size_factors
    eligible_n = tested_n[(normals[tested_n] / sf_n).mean(axis=1) >= 1]
    nulls.append(("gtex", normals.astype(np.int64), sf_n, [np.flatnonzero(study == "TCGA"), np.flatnonzero(study == "gtex")],
                  lambda s_: (0.75, 0.25)[s_], eligible_n, (1.0,)))
    for label, counts_all, factors_all, groups, share_of, pool, folds in nulls:
        for fold in folds:
            def make(k, counts_all=counts_all, factors_all=factors_all, groups=groups, share_of=share_of, pool=pool, fold=fold):
                rng = np.random.default_rng([root, TAGS["strata"], k])
                pick = np.sort(rng.choice(pool, min(per, len(pool)), replace=False))
                orders, cond, code = ST.relabelled(groups, share_of, rng, len(pick))
                c = np.take_along_axis(counts_all[pick], orders, axis=1).astype(np.int64)
                if fold != 1:                                  # every pseudo-case carries the fold
                    c = np.random.default_rng([root, TAGS["strata"], k, 1]).binomial(c, np.where(cond == 1, 1.0, 1 / fold))
                f = factors_all[orders]
                return c, f, cond, code, pick, lower_mean(c, f, cond), fold
            sets.append((f"{label}-fold{fold:g}", make))
    for k, (name, make) in enumerate(sets):
        if resume and (folder / f"{name}.npz").exists():
            continue
        start = time.time()
        counts, factors, cond, code, rows, reads, fold = make(k)
        area, subset = package_pvalues(counts, factors, cond, rows, [root, TAGS["strata"], k, 97], proc, code=code, area=fold == 1)
        save(folder, name, rows=rows, reads=reads, area_p=area, subset_p=subset, area_null=fold == 1, subset_null=True)
        print(f"strata {name}: {len(rows):,} genes, {len(np.unique(code))} strata, {time.time() - start:.0f}s", flush=True)


def arm_biology(cfg, out, resume):
    """The TCGA notebooks' recovery tables, copied with the state of the package that rendered them."""
    folder = out / "biology"
    folder.mkdir(exist_ok=True)
    results = ROOT / "notebooks/results"
    for rel in ("prostate/fusion-subsets.tsv", "prostate/primary-panel.tsv", "prostate/environment.json",
                "breast/erbb2-subset.tsv", "breast/primary-panel.tsv", "breast/environment.json"):
        target = folder / rel.replace("/", "-")
        shutil.copyfile(results / rel, target)
    changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "6ed1b56", "HEAD", "--", "src", "rust"],
                             capture_output=True, text=True).stdout.strip()
    note = dict(rendered="2026-10-25, notebooks/examples.qmd and examples-breast.qmd",
                package_changed_since_render=changed or "no changes to src/ or rust/ since 6ed1b56, the package that rendered them")
    (folder / "provenance.json").write_text(json.dumps(note, indent=1) + "\n")
    print("biology:", note["package_changed_since_render"], flush=True)


# ---------------------------------------------------------------------------------------------
# Simulated transcriptomes

def sided(q, sign):
    """A gene's q-value in the column of its call's side (up, down); NaN in the other."""
    return np.c_[np.where(sign > 0, q, np.nan), np.where(sign < 0, q, np.nan)]


def wilcoxon(x, cond, tested):
    """Two-sided rank-sum p-values, one BH family over the tested genes, q-values by side."""
    p, sign = np.full(len(x), np.nan), np.zeros(len(x))
    case, control = x[tested][:, cond == 1], x[tested][:, cond == 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        test = mannwhitneyu(case, control, alternative="two-sided", axis=1, method="asymptotic")
    p[tested] = np.nan_to_num(test.pvalue, nan=1.0)   # an all-tied gene carries no evidence
    sign[tested] = test.statistic - case.shape[1] * control.shape[1] / 2
    return p, sided(wade.bh_adjust(p), sign)


def simulated_registry(cfg):
    sim = dict(cfg["arms"]["simulated"], seed_root=cfg["seed_root"])
    rows = list(datasets(sim))
    return sim, [r for r in rows if r["panel"] == "mixed"] + [r for r in rows if r["panel"] == "complete_null"]


def arm_simulated(cfg, out, resume):
    proc = cfg["procedure"]
    sim, registry = simulated_registry(cfg)
    folder = out / "simulated"
    folder.mkdir(exist_ok=True)
    with (folder / "runs.jsonl").open("a") as log:
        for setting in registry:
            path = folder / f"{setting['id']}.npz"
            if resume and path.exists():
                continue
            data = generate(sim, setting)
            counts, cond = data["counts"], data["cond"]
            norm_ = wade.fit_normalization(counts)
            keep = wade.filter_counts(counts, min_count=3, min_samples=2)
            start = time.time()
            res = wade.wade(counts, cond, normalization=norm_, test_genes=keep, direction="both", pvalues=proc["pvalues"],
                            relative_error=proc["relative_error"], max_permutations=proc["max_permutations"], fdr=proc["fdr"],
                            seed=setting["permutation_seed"])
            seconds = time.time() - start
            wp, wq = wilcoxon(res.expression, cond, res.tested)
            in_range = res.tested & (np.minimum(res.case_mean, res.control_mean) >= proc["subset_validated_reads"])
            np.savez_compressed(path, profile=data["profile"], baseline_mean=data["baseline_mean"], altered=data["altered"],
                                tested=res.tested, in_range=in_range, area_q=res.area_padj, area_sign=np.sign(res.area),
                                subset_q=res.subset_padj, subset_side=res.subset_side, wilcoxon_q=wq)
            record = dict(id=setting["id"], seconds=round(seconds, 1), tested=int(res.tested.sum()),
                          shuffled_area=int((res.area_permutations > 0).sum()))
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)


# ---------------------------------------------------------------------------------------------
# Evaluation

def binomial_check(x, n, a, bad, level):
    """One-sided exact binomial test of rate <= a; the critical count, and power at bad, at size n."""
    p = float(binom.sf(x - 1, n, a)) if n else np.nan
    c = int(binom.isf(level, n, a)) if n else 0
    while c > 0 and binom.sf(c - 1, n, a) <= level:
        c -= 1
    while binom.sf(c - 1, n, a) > level:
        c += 1
    return p, float(binom.sf(c - 1, n, bad)) if n else 0.0


def mean_check(values, a, bad, level):
    """One-sided t test of mean <= a over matrices, and its power at bad (normal approximation)."""
    v = np.asarray(values, float)
    m, s, n = v.mean(), v.std(ddof=1), len(v)
    if s == 0:
        return (0.0 if m > a else 1.0), (1.0 if m < bad else 0.0)
    se = s / np.sqrt(n)
    return float(student.sf((m - a) / se, n - 1)), float(norm.cdf((bad - a) / se - norm.ppf(1 - level)))


def per_gene_tests(out, arm):
    """Every null p-value of an arm: (test, p, reads) arrays, the subset test's in its validated range."""
    area, subset, area_reads, subset_reads = [], [], [], []
    for path in sorted((out / arm).glob("*.npz")):
        z = np.load(path)
        if bool(z["area_null"]):
            area.append(z["area_p"])
            area_reads.append(z["reads"])
        subset.append(z["subset_p"])
        subset_reads.append(z["reads"])
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return dict(area=(cat(area), cat(area_reads)), subset=(cat(subset), cat(subset_reads)))


def simulated_tables(cfg, out):
    """Per matrix: any discovery (complete nulls) and the false discovery proportions (partial nulls)."""
    proc, sim = cfg["procedure"], cfg["arms"]["simulated"]
    no_subset = sim["no_subset"]
    rows = []
    for path in sorted((out / "simulated").glob("*.npz")):
        z = np.load(path)
        panel = "mixed" if path.name.startswith("mixed") else "complete_null"
        row = dict(id=path.stem, panel=panel)
        for level in proc["alpha"]:
            area = z["tested"] & (z["area_q"] <= level)
            subset = z["in_range"] & (z["subset_q"] <= level)
            false_subset = subset & np.isin(z["profile"], no_subset)
            row[f"area_calls_{level:g}"], row[f"area_false_{level:g}"] = int(area.sum()), int((area & ~z["altered"]).sum())
            row[f"subset_calls_{level:g}"], row[f"subset_false_{level:g}"] = int(subset.sum()), int(false_subset.sum())
            union = area | subset
            row[f"union_calls_{level:g}"], row[f"union_false_{level:g}"] = int(union.sum()), int((union & ~z["altered"]).sum())
        rows.append(row)
    return rows


def evaluate(cfg, out):
    checks, rule = cfg["checks"], cfg["rule"]
    level = rule["familywise_false_fail"] / len(checks)
    genes = {arm: per_gene_tests(out, arm) for arm in ("count_model", "real_labels", "strata") if (out / arm).exists()}
    sims = simulated_tables(cfg, out) if (out / "simulated").exists() else []
    verdicts = []
    for c in checks:
        a, bad = c["acceptable"], c["unacceptable"]
        if c["arm"] == "simulated":
            parts = c["id"].split("/")
            if parts[1] == "any":
                test, lvl = parts[2], float(parts[3])
                null = [r for r in sims if r["panel"] == "complete_null"]
                x, n = sum(r[f"{test}_calls_{lvl:g}"] > 0 for r in null), len(null)
                p, power = binomial_check(x, n, a, bad, level)
                estimate, size = x / n if n else np.nan, n
            else:
                test, lvl = ("union", 0.05) if parts[1] == "union" else (parts[2], float(parts[3]))
                mixed = [r for r in sims if r["panel"] == "mixed"]
                fdp = [r[f"{test}_false_{lvl:g}"] / max(r[f"{test}_calls_{lvl:g}"], 1) for r in mixed]
                p, power = mean_check(fdp, a, bad, level) if len(fdp) > 1 else (np.nan, 0.0)
                estimate, size = (float(np.mean(fdp)) if fdp else np.nan), len(fdp)
        else:
            arm, test, lvl = c["arm"], c["test"], c["level"]
            pv, reads = genes.get(arm, {}).get(test, (np.array([]), np.array([])))
            if test == "subset":
                pv = pv[reads >= cfg["procedure"]["subset_validated_reads"]]
            x, n = int(np.sum(pv <= lvl)), len(pv)
            p, power = binomial_check(x, n, a, bad, level)
            estimate, size = (x / n if n else np.nan), n
        verdict = "underpowered" if power < rule["power"] else ("fail" if p <= level else "pass")
        verdicts.append(dict(id=c["id"], estimate=estimate, acceptable=a, unacceptable=bad, size=size, p=p, power=power,
                             verdict=verdict))
    with (out / "checks.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(verdicts[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(verdicts)
    print(f"\n{len(checks)} checks, each one-sided at {level:.5f}; power {rule['power']} at the unacceptable value")
    for v in verdicts:
        print(f"  {v['id']:30s} {v['estimate']:9.5f}  (acceptable {v['acceptable']:g}, unacceptable {v['unacceptable']:g}; "
              f"n {v['size']:,}; p {v['p']:.3g}; power {v['power']:.3f})  {v['verdict'].upper()}")
    describe(cfg, out, genes, sims)


def describe(cfg, out, genes, sims):
    """Descriptive tables: per-gene null rates by reads, the simulated partial nulls' power, and the biology."""
    edges = (0,) + BINS + (np.inf,)
    print("\nPer-gene null rates at p <= 0.05 / 0.01 / 0.001, by reads per sample")
    for arm, tests in genes.items():
        for test, (pv, reads) in tests.items():
            cells = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (reads >= lo) & (reads < hi)
                cells.append(f"[{lo:g},{hi:g}) " + (" / ".join(f"{np.mean(pv[k] <= a):.4f}" for a in (0.05, 0.01, 0.001))
                                                     + f" of {k.sum():,}" if k.any() else "--"))
            print(f"  {arm:11s} {test:6s} " + " | ".join(cells))
    if sims:
        mixed = [r for r in sims if r["panel"] == "mixed"]
        print(f"\nSimulated: {len(mixed)} partial-null and {len(sims) - len(mixed)} complete-null matrices")
        power_tables(cfg, out)


def power_tables(cfg, out):
    sim, alpha = cfg["arms"]["simulated"], 0.05
    intended = {p: d for d, profiles in sim["intended_direction"].items() for p in profiles}
    tally = {}
    for path in sorted((out / "simulated").glob("mixed-*.npz")):
        z = np.load(path)
        design = path.stem.split("-")[1]
        up = dict(wade=(z["area_q"] <= alpha) & (z["area_sign"] > 0), subset=(z["subset_q"] <= alpha) & (z["subset_side"] == "up"),
                  wilcoxon=z["wilcoxon_q"][:, 0] <= alpha)
        down = dict(wade=(z["area_q"] <= alpha) & (z["area_sign"] < 0), subset=(z["subset_q"] <= alpha) & (z["subset_side"] == "down"),
                    wilcoxon=z["wilcoxon_q"][:, 1] <= alpha)
        for profile in sim["generator"]["profiles"]:
            genes = (z["profile"] == profile) & (z["baseline_mean"] >= 5)
            for method in up:
                t = tally.setdefault((design, profile, method), np.zeros(3))
                t += [genes.sum(), (up[method] & genes).sum(), (down[method] & genes).sum()]
    rows = []
    for (design, profile, method), (n, u, d) in tally.items():
        rows.append(dict(design=design, profile=profile, method=method, intended=intended[profile], genes=int(n),
                         up=u / n, down=d / n))
    with (out / "power.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("Intended-direction power at q <= 0.05, abundance >= 5 (other direction in brackets): see power.tsv")
    for design in dict.fromkeys(r["design"] for r in rows):
        print(f"  {design}")
        for profile in dict.fromkeys(r["profile"] for r in rows):
            cells = []
            for method in ("wade", "wilcoxon", "subset"):
                r = next(r for r in rows if (r["design"], r["profile"], r["method"]) == (design, profile, method))
                main, other = {"up": (r["up"], r["down"]), "down": (r["down"], r["up"])}.get(r["intended"], (max(r["up"], r["down"]), min(r["up"], r["down"])))
                cells.append(f"{method} {main:4.0%} [{other:3.0%}]")
            print(f"    {profile:34s} " + "   ".join(cells))


# ---------------------------------------------------------------------------------------------

def provenance(text):
    files = sorted(p for p in (ROOT / "src/wade").rglob("*") if p.suffix in (".py", ".so", ".pyd")) + [Path(__file__)]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return dict(wade_version=wade.__version__, git_head=head, protocol_sha256=hashlib.sha256(text.encode()).hexdigest(),
                sources={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", nargs="*", choices=ARMS, help="arms to run (default: all, in order)")
    parser.add_argument("--resume", action="store_true", help="skip data sets already saved in --output")
    parser.add_argument("--evaluate", action="store_true", help="score the checks from --output's results")
    args = parser.parse_args()
    text = PROTOCOL.read_text(encoding="utf-8")
    cfg = json.loads(text)
    if args.evaluate:
        evaluate(cfg, args.output)
        return
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *FROZEN], capture_output=True, text=True).stdout
    if dirty.strip():
        raise SystemExit(f"uncommitted changes to frozen files; commit them before a confirmation run:\n{dirty}")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} exists; pass --resume to continue it")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(text, encoding="utf-8")
    record = provenance(text)
    if (args.output / "provenance.json").exists():
        first = json.loads((args.output / "provenance.json").read_text())
        if first["git_head"] != record["git_head"] or first["sources"] != record["sources"]:
            raise SystemExit("--resume with a different package or script than the run began with")
    else:
        (args.output / "provenance.json").write_text(json.dumps(record, indent=1) + "\n")
    preflight(cfg, args.output)
    for arm in args.arm or ARMS:
        start = time.time()
        globals()[f"arm_{arm}"](cfg, args.output, args.resume)
        print(f"== arm {arm} done in {time.time() - start:.0f}s", flush=True)
    evaluate(cfg, args.output)


if __name__ == "__main__":
    main()
