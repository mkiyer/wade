"""Do the subsets recover the planted cases, on the side they were planted?

Regenerates the confirmation's mixed matrices from their frozen seeds (the generator is
deterministic) and scores every planted gene with baseline mean >= 5 counts, on jittered
normalized counts as wade() runs it. A planted subset is scored by the side the split
chose, the share of cases selected, precision, recall and the subset test (999 shuffles).
A change planted in every case has no subset, and its test should stay near 5%. Unchanged
genes show what the split finds in noise.

    PYTHONPATH=src:research python research/subset_recovery.py
"""
from pathlib import Path
import json

import numpy as np

import wade
from wade._means import expression, jittered
from wade._subsets import split, subset_test
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]


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
        factors = wade.fit_normalization(counts).size_factors
        steady = data["baseline_mean"] >= 5
        genes = np.flatnonzero(data["altered"] & steady & (data["profile"] != "same_mean_wider"))
        background = np.flatnonzero((data["profile"] == "background_null") & steady)[:500]
        x = expression(jittered(counts[genes], genes, 1), factors)
        cases, sides, _, _, _ = split(x, cond)
        p, _, _ = subset_test(counts[genes], factors, cond, genes, direction="both", stop=25, maximum=999, seed=1)
        for j, g in enumerate(genes):
            truth = planted(data["profile"][g], data["enriched"][g], data["perturbed"][g])
            case = x[j, cond == 1]
            order = np.argsort(-case if sides[j] > 0 else case, kind="stable")
            chosen = np.zeros(len(case), bool)
            chosen[order[:cases[j]]] = True
            row = dict(design=design, profile=data["profile"][g], share=chosen.mean(), baseline=data["baseline_mean"][g],
                       side="none" if cases[j] == 0 else "up" if sides[j] > 0 else "down", significant=p[j] <= .05)
            if truth is not None:
                members, side = truth
                if side == "either":                        # score against the planted subset on the chosen side
                    members, side = (members[0], "up") if row["side"] == "up" else (members[1], "down")
                row.update(planted_side=side, right_side=row["side"] == side,
                           precision=(chosen & members).sum() / chosen.sum() if chosen.any() else np.nan,
                           recall=(chosen & members).sum() / members.sum())
            rows.append(row)
        xb = expression(jittered(counts[background], background, 1), factors)
        nc = split(xb, cond)[0]
        pn, _, _ = subset_test(counts[background], factors, cond, background, direction="both", stop=25,
                               maximum=999, seed=1)
        noise.append((design, nc / cond.sum(), pn))
    report(rows, noise)


def report(rows, noise):
    for design in dict.fromkeys(r["design"] for r in rows):
        print(f"\n{design}, baseline mean >= 5: side right / share selected / precision / recall / subset p <= .05")
        for profile in dict.fromkeys(r["profile"] for r in rows if r["design"] == design):
            g = [r for r in rows if (r["design"], r["profile"]) == (design, profile)]
            if "precision" in g[0]:
                cell = (f"{np.mean([r['right_side'] for r in g]):4.0%} {np.mean([r['share'] for r in g]):.2f} "
                        f"{np.nanmean([r['precision'] for r in g]):.2f} {np.mean([r['recall'] for r in g]):.2f} "
                        f"{np.mean([r['significant'] for r in g]):4.0%}")
            else:
                levels = sorted({r["baseline"] for r in g})
                by_level = ", ".join(f"{level:g}: {np.mean([r['significant'] for r in g if r['baseline'] == level]):.0%}"
                                     for level in levels)
                cell = f"shared by every case: {np.mean([r['significant'] for r in g]):4.0%} (by baseline mean {by_level})"
            print(f"  {profile[:34]:34s} {cell}")
    for design in dict.fromkeys(d for d, _, _ in noise):
        share = np.concatenate([n for d, n, _ in noise if d == design])
        p = np.concatenate([q for d, _, q in noise if d == design])
        print(f"\n{design}, unchanged genes (n={len(share)}): split takes {np.mean(share):.2f} of cases, "
              f"subset p <= .05 in {np.mean(p <= .05):.1%}")


if __name__ == "__main__":
    main()
