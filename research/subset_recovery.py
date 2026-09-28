"""How well do learned subsets recover the planted cases?

Regenerates the confirmation's mixed matrices from their frozen seeds (the
generator is deterministic) and scores, for every planted gene, the cases each
walk places in the subset of the planted direction against the planted cases:

  standardized : WADE's walk (the running sum of cases over controls, divided
                 by its random-labelling standard deviation; peak = boundary)
  running_sum  : the same walk without standardization (GSEA's leading edge)
  area         : the cumulative quantile-area walk, argmax_q of the integral
                 from q to 1 of (Q_case - Q_control)

Scores are reported for all planted genes and for those WADE called in the
planted direction (the confirmation's saved q <= 0.05).

    PYTHONPATH=src:research python research/subset_recovery.py \
        --confirmation .release/0.2.0/confirmation-20260927
"""
from pathlib import Path
import argparse
import json

import numpy as np

import wade
from wade._descriptions import subsets
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
UP = {"bulk_up_1.5x", "subset_up_5pct_4x", "subset_up_10pct_4x", "subset_up_20pct_4x", "subset_up_10pct_16x",
      "mean_preserving_subset_10pct_4x", "singleton_up_16x"}
DOWN = {"bulk_down_two_thirds", "subset_down_10pct_quarter"}


def planted(profile, enriched, perturbed):
    """Planted cases per direction; bulk profiles move every case."""
    if profile.startswith("bulk_half_subset_up"):
        return {"up": enriched, "down": perturbed & ~enriched}
    if profile == "mean_preserving_subset_10pct_4x":      # the rest of the cases are lowered slightly
        return {"up": enriched}
    if profile in UP:
        return {"up": perturbed}
    if profile in DOWN:
        return {"down": perturbed}
    if profile == "opposing_20pct":
        return {"up": enriched, "down": perturbed & ~enriched}
    return {}


def running_sum(x, cond, direction):
    order = np.argsort(-x if direction == "up" else x, kind="stable")
    v, c = x[order], cond[order]
    excess = np.cumsum(c) / c.sum() - np.cumsum(1 - c) / (1 - c).sum()
    distinct = np.r_[v[1:] != v[:-1], True]
    k = int(np.argmax(np.where(distinct, excess, -np.inf)))
    return (x >= v[k]) if direction == "up" else (x <= v[k])


def area_walk(x, cond, direction):
    a, b = np.sort(x[cond == 1]), np.sort(x[cond == 0])
    if direction == "down":
        a, b = -a[::-1], -b[::-1]
    edges = np.unique(np.r_[np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b)])
    mid = (edges[:-1] + edges[1:]) / 2
    d = a[np.minimum((mid * len(a)).astype(int), len(a) - 1)] - b[np.minimum((mid * len(b)).astype(int), len(b) - 1)]
    tail = np.r_[np.cumsum((d * np.diff(edges))[::-1])[::-1], 0]
    q = edges[int(np.argmax(tail))]
    cut = a[min(int(np.ceil(q * len(a))), len(a) - 1)] if q < 1 else np.inf
    return (x >= cut) if direction == "up" else (x <= -cut)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirmation", type=Path, required=True, help="saved confirmation run (for its q-values)")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "research/protocols/confirmation.json").read_text())
    rows, null = [], []
    for setting in datasets(cfg):
        if setting["panel"] != "mixed":
            continue
        data = generate(cfg, setting)
        saved = np.load(args.confirmation / f"{setting['id']}.npz")
        counts, cond = data["counts"], data["cond"]
        x = np.log2(counts / wade.fit_normalization(counts).size_factors + 1)
        # Unchanged genes: what a walk over noise selects.
        background = np.flatnonzero((data["profile"] == "background_null") & (data["baseline_mean"] >= 5)
                                    & saved["tested"])[:500]
        _, bu, _, _, bd, _, _ = subsets(x[background], cond)
        null.append((f"{setting['n_case']}:{setting['n_control']}", bu / cond.sum(), bd / cond.sum()))
        genes = np.flatnonzero(data["altered"])
        up_cut, up_n, _, down_cut, down_n, _, values = subsets(x[genes], cond)
        for j, g in enumerate(genes):
            case = x[g, cond == 1]
            truth = planted(data["profile"][g], data["enriched"][g], data["perturbed"][g])
            for direction, members in truth.items():
                cut = (values[j, up_cut[j]] if direction == "up" else values[j, down_cut[j] - 1]) \
                    if (up_n if direction == "up" else down_n)[j] else None
                ours = np.zeros(len(case), bool) if cut is None else (case >= cut if direction == "up" else case <= cut)
                calls = dict(standardized=ours, running_sum=running_sum(x[g], cond, direction)[cond == 1],
                             area=area_walk(x[g], cond, direction)[cond == 1])
                called = bool(saved["wade_q"][g, 0 if direction == "up" else 1] <= cfg["procedure"]["alpha"])
                for method, sel in calls.items():
                    rows.append(dict(design=f"{setting['n_case']}:{setting['n_control']}", profile=data["profile"][g],
                                     mean=float(data["baseline_mean"][g]), direction=direction, method=method,
                                     called=called, share=sel.mean(), planted=members.mean(),
                                     precision=(sel & members).sum() / sel.sum() if sel.any() else np.nan,
                                     recall=(sel & members).sum() / members.sum()))
    report(rows)
    for design in dict.fromkeys(d for d, _, _ in null):
        up = np.concatenate([u for d, u, _ in null if d == design])
        down = np.concatenate([v for d, _, v in null if d == design])
        print(f"\n{design}: unchanged genes (n={len(up)}): mean share of cases in the up subset {up.mean():.2f}, "
              f"down {down.mean():.2f}; empty up {np.mean(up == 0):.0%}, empty down {np.mean(down == 0):.0%}")


def report(rows):
    for design in dict.fromkeys(r["design"] for r in rows):
        print(f"\n{design}: share of cases selected / precision / recall, baseline mean >= 5, "
              "all planted genes | genes called in that direction")
        for key in dict.fromkeys((r["profile"], r["direction"]) for r in rows if r["design"] == design):
            cells = []
            for method in ("standardized", "running_sum", "area"):
                for called_only in (False, True):
                    g = [r for r in rows if (r["design"], r["profile"], r["direction"], r["method"]) == (design, *key, method)
                         and r["mean"] >= 5 and (r["called"] or not called_only)]
                    cells.append("  —  " if not g else
                                 f"{np.mean([r['share'] for r in g]):.2f}/{np.nanmean([r['precision'] for r in g]):.2f}"
                                 f"/{np.mean([r['recall'] for r in g]):.2f} (n={len(g)})")
            print(f"  {key[0][:32]:32s} {key[1]:>4s} planted {np.mean([r['planted'] for r in rows if (r['design'], r['profile'], r['direction']) == (design, *key)]):.2f} | "
                  + " | ".join(cells))


if __name__ == "__main__":
    main()
