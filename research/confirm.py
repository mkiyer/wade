"""Confirm the selected public procedure on fresh transcriptome-scale seeds.

Runs only the public API, wade.wade(direction="both", scale="linear"), plus a
one-sided Wilcoxon comparator on the same normalized counts, test mask and
BH family. Calls are scored against each profile's intended direction.

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


def wilcoxon(x, cond, tested):
    """One-sided rank-sum p-values (columns up, down) and one joint BH family."""
    p = np.full((len(x), 2), np.nan)
    case, control = x[tested][:, cond == 1], x[tested][:, cond == 0]
    for d, alternative in enumerate(("greater", "less")):
        with np.errstate(invalid="ignore", divide="ignore"):
            p[tested, d] = mannwhitneyu(case, control, alternative=alternative, axis=1, method="asymptotic").pvalue
    p[tested] = np.nan_to_num(p[tested], nan=1.0)     # an all-tied gene carries no evidence
    return p, wade.bh_adjust(p.ravel()).reshape(p.shape)


def run_one(cfg, setting, folder):
    data = generate(cfg, setting)
    counts, cond = data["counts"], data["cond"]
    norm = wade.fit_normalization(counts)
    keep = wade.filter_counts(counts, min_count=3, min_samples=2)
    start = time.time()
    res = wade.wade(counts, cond, normalization=norm, test_genes=keep, direction="both", scale="linear",
                    max_permutations=cfg["procedure"]["max_permutations"], seed=setting["permutation_seed"])
    seconds = time.time() - start
    wp, wq = wilcoxon(res.expression, cond, res.tested)
    np.savez_compressed(folder / f"{setting['id']}.npz", profile=data["profile"], baseline_mean=data["baseline_mean"],
                        altered=data["altered"], tested=res.tested, size_factors=norm.size_factors,
                        wade_area=np.c_[res.area_up, res.area_down], wade_p=np.c_[res.pvalue_up, res.pvalue_down],
                        wade_q=np.c_[res.padj_up, res.padj_down], wade_permutations=res.n_permutations,
                        wilcoxon_p=wp, wilcoxon_q=wq)
    return dict(id=setting["id"], seconds=round(seconds, 1), tested=int(res.tested.sum()),
                capped=int((res.n_permutations == cfg["procedure"]["max_permutations"]).sum()))


def summarize(cfg, folder):
    """Calibration per matrix, and intended-direction power per profile and abundance."""
    alpha, gen = cfg["procedure"]["alpha"], cfg["generator"]
    calibration, power = [], {}
    for setting in datasets(cfg):
        path = folder / f"{setting['id']}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        design = f"{setting['n_case']}:{setting['n_control']}"
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
    for name, rows in (("calibration", calibration), ("power", power_rows)):
        if rows:
            with (folder / f"{name}.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
    return calibration, power_rows


def report(calibration, power_rows):
    print("\nCalibration (alpha 0.05, gene-direction calls):")
    print(f"{'design':>8} {'method':>9} | {'null: any call':>15} | {'mixed: mean FDP':>15} {'mean calls':>10}")
    for design in dict.fromkeys(r["design"] for r in calibration):
        for method in METHODS:
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
            for method in METHODS:
                rows = [r for r in power_rows if (r["design"], r["profile"], r["method"]) == (design, profile, method)
                        and r["mean"] >= 5]
                n = sum(r["genes"] for r in rows)
                up, down = (sum(r[k] * r["genes"] for r in rows) / n for k in ("up", "down"))
                main, other = {"up": (up, down), "down": (down, up)}.get(rows[0]["intended"], (max(up, down), min(up, down)))
                cells.append(f"{method} {main:4.0%} [{other:3.0%}]")
            print(f"    {profile:34s} " + "   ".join(cells))


def provenance(cfg_text):
    files = sorted(ROOT.glob("src/wade/**/*.py")) + [ROOT / "rust/src/lib.rs", Path(__file__)]
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
