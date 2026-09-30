"""Confirm the selected public procedure on fresh transcriptome-scale seeds.

Runs only the public API, wade.wade(direction="both"), plus a two-sided
Wilcoxon comparator on the same normalized counts, test mask and BH family.
Area calls are scored against each profile's intended direction, and subset
calls against the profiles that carry no subset.

The protocol's "call" records the procedure confirmed under protocol v6: the
mean test on unthinned counts, with a log2 option. This script runs the
present wade(), whose area test thins every sample to the shallowest depth
first; its settings (pvalues, relative_error, max_permutations, alpha) are the
protocol's.

    PYTHONPATH=src:research python research/confirm.py --output .release/0.2.0/confirmation
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import time

import numpy as np
from scipy.stats import mannwhitneyu

import wade
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/confirmation.json"
METHODS = ("wade", "wilcoxon")


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


def run_one(cfg, setting, folder):
    data = generate(cfg, setting)
    counts, cond = data["counts"], data["cond"]
    norm = wade.fit_normalization(counts)
    keep = wade.filter_counts(counts, min_count=3, min_samples=2)
    start = time.time()
    res = wade.wade(counts, cond, normalization=norm, test_genes=keep, direction="both",
                    pvalues=cfg["procedure"]["pvalues"], relative_error=cfg["procedure"]["relative_error"],
                    max_permutations=cfg["procedure"]["max_permutations"], seed=setting["permutation_seed"])
    seconds = time.time() - start
    wp, wq = wilcoxon(res.expression, cond, res.tested)
    np.savez_compressed(folder / f"{setting['id']}.npz", profile=data["profile"], baseline_mean=data["baseline_mean"],
                        altered=data["altered"], tested=res.tested, size_factors=norm.size_factors,
                        wade_area=res.area, wade_p=res.area_pvalue, wade_q=sided(res.area_padj, res.area),
                        wade_permutations=res.area_permutations, wade_subset=np.c_[res.subset_fraction, res.subset_log2fc],
                        wade_subset_q=sided(res.subset_padj, (res.subset_side == "up").astype(int)
                                            - (res.subset_side == "down")), wilcoxon_p=wp, wilcoxon_q=wq)
    return dict(id=setting["id"], seconds=round(seconds, 1), tested=int(res.tested.sum()),
                shuffled_area=int((res.area_permutations > 0).sum()))


def summarize(cfg, folder):
    """Calibration per matrix, intended-direction power per profile and abundance, and the subset test's."""
    alpha, gen = cfg["procedure"]["alpha"], cfg["generator"]
    calibration, power, subsets = [], {}, {}
    for setting in datasets(cfg):
        path = folder / f"{setting['id']}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        design = f"{setting['n_case']}:{setting['n_control']}"
        called = z["wade_subset_q"] <= alpha
        none = np.isin(z["profile"], cfg["no_subset"])
        calls, false = int(called.sum()), int(called[none].sum())
        calibration.append(dict(design=design, panel=setting["panel"], biology=setting["biology"], method="subset",
                                calls=calls, false=false, fdp=false / max(calls, 1)))
        for profile in gen["profiles"] if setting["panel"] == "mixed" else ():
            for mean in gen["profile_means"]:
                genes = (z["profile"] == profile) & (z["baseline_mean"] == mean)
                tally = subsets.setdefault((design, profile, mean), np.zeros(3, int))
                tally += [genes.sum(), called[genes, 0].sum(), called[genes, 1].sum()]
        for method in METHODS:
            called = z[f"{method}_q"] <= alpha
            calls, false = int(called.sum()), int(called[~z["altered"]].sum())
            calibration.append(dict(design=design, panel=setting["panel"], biology=setting["biology"], method=method,
                                    calls=calls, false=false, fdp=false / max(calls, 1)))
            if setting["panel"] != "mixed":
                continue
            for profile in gen["profiles"]:
                for mean in gen["profile_means"]:
                    genes = (z["profile"] == profile) & (z["baseline_mean"] == mean)
                    tally = power.setdefault((design, method, profile, mean), np.zeros(3, int))
                    tally += [genes.sum(), called[genes, 0].sum(), called[genes, 1].sum()]
    intended = {p: d for d, profiles in cfg["intended_direction"].items() for p in profiles}
    power_rows = [dict(design=d, method=m, profile=p, mean=mu, intended=intended[p], genes=int(n),
                       up=u / n, down=v / n) for (d, m, p, mu), (n, u, v) in power.items()]
    power_rows += [dict(design=d, method="subset", profile=p, mean=mu, intended=intended[p], genes=int(n),
                        up=u / n, down=v / n) for (d, p, mu), (n, u, v) in subsets.items()]
    for name, rows in (("calibration", calibration), ("power", power_rows)):
        if rows:
            with (folder / f"{name}.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
    return calibration, power_rows


def report(calibration, power_rows):
    print("\nCalibration (alpha 0.05; subset: calls on genes with no planted subset):")
    print(f"{'design':>8} {'method':>9} | {'null: any call':>15} | {'mixed: mean FDP':>15} {'mean calls':>10}")
    for design in dict.fromkeys(r["design"] for r in calibration):
        for method in METHODS + ("subset",):
            rows = [r for r in calibration if r["design"] == design and r["method"] == method]
            null = [r["calls"] > 0 for r in rows if r["panel"] == "complete_null"]
            mixed = [r for r in rows if r["panel"] == "mixed"]
            print(f"{design:>8} {method:>9} | {np.mean(null) if null else np.nan:>8.0%} of {len(null):>3} |"
                  f" {np.mean([r['fdp'] for r in mixed]) if mixed else np.nan:>15.1%}"
                  f" {np.mean([r['calls'] for r in mixed]) if mixed else np.nan:>10.1f}")
    print("\nIntended-direction power, abundance >= 5 (other-direction rate in brackets):")
    for design in dict.fromkeys(r["design"] for r in power_rows):
        print(f"  {design}")
        for profile in dict.fromkeys(r["profile"] for r in power_rows):
            cells = []
            for method in METHODS + ("subset",):
                rows = [r for r in power_rows if (r["design"], r["profile"], r["method"]) == (design, profile, method)
                        and r["mean"] >= 5]
                n = sum(r["genes"] for r in rows)
                up, down = (sum(r[k] * r["genes"] for r in rows) / n for k in ("up", "down"))
                main, other = {"up": (up, down), "down": (down, up)}.get(rows[0]["intended"], (max(up, down), min(up, down)))
                cells.append(f"{method} {main:4.0%} [{other:3.0%}]")
            print(f"    {profile:34s} " + "   ".join(cells))


def provenance(cfg_text):
    files = sorted(ROOT.glob("src/wade/**/*.py")) + [Path(__file__)]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return dict(wade_version=wade.__version__, git_head=head, protocol_sha256=hashlib.sha256(cfg_text.encode()).hexdigest(),
                sources={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="skip matrices already saved in --output")
    parser.add_argument("--only", nargs="*", help="dataset ids to run (default: all)")
    args = parser.parse_args()
    text = PROTOCOL.read_text()
    cfg = json.loads(text)
    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} exists; pass --resume to continue it")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(text)
    (args.output / "provenance.json").write_text(json.dumps(provenance(text), indent=1) + "\n")
    with (args.output / "runs.jsonl").open("a") as log:
        for setting in datasets(cfg):
            if (args.only and setting["id"] not in args.only) or (args.output / f"{setting['id']}.npz").exists():
                continue
            record = run_one(cfg, setting, args.output)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    report(*summarize(cfg, args.output))


if __name__ == "__main__":
    main()
