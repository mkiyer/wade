"""Do the subsets recover the planted cases, on the side they were planted?

Regenerates the confirmation's mixed matrices from their frozen seeds (the generator is
deterministic) and scores every planted gene with baseline mean >= 5 counts, on the linear
scale (the default) and the log2 scale. A planted subset is scored by the side the split
chose, the share of cases selected, precision, recall and the subset test (999 shuffles).
A change planted in every case has no subset, and its test should stay near 5%. Unchanged
genes show what the split finds in noise.

    PYTHONPATH=src:research python research/subset_recovery.py
"""
from pathlib import Path
import json

import numpy as np

import wade
from wade._subsets import split, subset_test
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
SCALES = ("linear", "log2")


def planted(profile, enriched, perturbed):
    """The planted subset's cases and side, or None where every case carries the change."""
    if profile in ("bulk_up_1.5x", "bulk_down_two_thirds"):
        return None
    if profile in ("subset_down_10pct_quarter",):
        return perturbed, "down"
    if profile == "opposing_20pct":                        # two subsets; the split may take either
        return (enriched, perturbed & ~enriched), "either"
    return enriched, "up"


def main():
    cfg = json.loads((ROOT / "research/protocols/confirmation.json").read_text())
    rows, noise = [], []
    for setting in datasets(cfg):
        if setting["panel"] != "mixed":
            continue
        data = generate(cfg, setting)
        counts, cond = data["counts"], data["cond"]
        design = f"{setting['n_case']}:{setting['n_control']}"
        linear = counts / wade.fit_normalization(counts).size_factors
        steady = data["baseline_mean"] >= 5
        genes = np.flatnonzero(data["altered"] & steady & (data["profile"] != "same_mean_wider"))
        background = np.flatnonzero((data["profile"] == "background_null") & steady)[:500]
        for scale in SCALES:
            x = linear if scale == "linear" else np.log2(linear + 1)
            cases, fold = split(x[genes], cond, scale)
            p, _, _ = subset_test(x[genes], cond, scale, direction="both", stop=25, maximum=999, seed=1)
            for j, g in enumerate(genes):
                truth = planted(data["profile"][g], data["enriched"][g], data["perturbed"][g])
                case = x[g, cond == 1]
                order = np.argsort(-case if fold[j] > 0 else case, kind="stable")
                chosen = np.zeros(len(case), bool)
                chosen[order[:cases[j]]] = True
                row = dict(design=design, scale=scale, profile=data["profile"][g], share=chosen.mean(),
                           side="none" if cases[j] == 0 else "up" if fold[j] > 0 else "down", significant=p[j] <= .05)
                if truth is not None:
                    members, side = truth
                    if side == "either":                    # score against the planted subset on the chosen side
                        members, side = (members[0], "up") if row["side"] == "up" else (members[1], "down")
                    row.update(planted_side=side, right_side=row["side"] == side,
                               precision=(chosen & members).sum() / chosen.sum() if chosen.any() else np.nan,
                               recall=(chosen & members).sum() / members.sum())
                rows.append(row)
            nc, _ = split(x[background], cond, scale)
            pn, _, _ = subset_test(x[background], cond, scale, direction="both", stop=25, maximum=999, seed=1)
            noise.append((design, scale, nc / cond.sum(), pn))
    report(rows, noise)


def report(rows, noise):
    for design in dict.fromkeys(r["design"] for r in rows):
        print(f"\n{design}, baseline mean >= 5: side right / share selected / precision / recall / subset p <= .05")
        print(f"  {'profile':34s} " + "".join(f"{s:>36s}" for s in SCALES))
        for profile in dict.fromkeys(r["profile"] for r in rows if r["design"] == design):
            cells = []
            for scale in SCALES:
                g = [r for r in rows if (r["design"], r["scale"], r["profile"]) == (design, scale, profile)]
                if "precision" in g[0]:
                    cells.append(f"{np.mean([r['right_side'] for r in g]):4.0%} {np.mean([r['share'] for r in g]):.2f} "
                                 f"{np.nanmean([r['precision'] for r in g]):.2f} {np.mean([r['recall'] for r in g]):.2f} "
                                 f"{np.mean([r['significant'] for r in g]):4.0%}")
                else:
                    cells.append(f"shared by every case: {np.mean([r['significant'] for r in g]):4.0%}")
            print(f"  {profile[:34]:34s} " + "".join(f"{c:>36s}" for c in cells))
    for design, scale in dict.fromkeys((d, s) for d, s, _, _ in noise):
        share = np.concatenate([n for d, s, n, _ in noise if (d, s) == (design, scale)])
        p = np.concatenate([q for d, s, _, q in noise if (d, s) == (design, scale)])
        print(f"\n{design} {scale}, unchanged genes (n={len(share)}): split takes {np.mean(share):.2f} of cases, "
              f"subset p <= .05 in {np.mean(p <= .05):.1%}")


if __name__ == "__main__":
    main()
