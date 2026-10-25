"""Rarity: how unusual each gene's subset is among the genes, and which samples carry the rarest subsets.

A gene's subset sits at a cut, with k cases below it and n1 - k above. A Lorenz curve lies on or
below the diagonal, so the gap at a cut can be no larger than k/n1, the share of the cases below
it: a lowered subset's own share, or a raised subset's remainder. Gaps of subsets of different
sizes and sides do not compare. Rarity compares like with like. For each gene it
takes the share of genes whose own gap at the same cut, with a subset of the same size and side, is
at least as large. Each gene was scored at its own best cut, so those shares are then ranked across
the genes. The result reads as the share of genes whose subset is at least as rare: under
exchangeable genes, a conformal rank, uniform over the genes (Bates, Candes, Lei, Romano and
Sesia, Ann. Statist. 2023; Vovk, Mach. Learn. 2013, for ranks within categories).

Genes are ranked within groups: the genes in the subset test's validated range, and the genes below
it, so that low-count genes, whose gaps run larger, are compared only with each other. A gene
without a subset has rarity 1. Rarity describes and ranks; it is not a test, and selecting genes by
it before BH would invalidate the BH family.
"""
import numpy as np

from . import _kernel
from ._subsets import _sorted

GENE_BLOCK = 4096                       # genes compared at a time against a chunk of curves
CURVE_BYTES = 32 * 2**20                # gap curves held at a time


def cuts(size, side, n1):
    """Each gene's cut: the number of cases below it (0 for a gene without a subset)."""
    return np.where(size == 0, 0, np.where(side > 0, n1 - size, size))


def gap_curves(x, cond):
    """Each gene's Lorenz gap at every case boundary k = 1..n1-1 (-inf where ties forbid one)."""
    case, control = _sorted(x, cond)
    return _kernel.gaps(case, control).reshape(len(x), -1)


def rarity(walked, cond, size, side, groups):
    """Each gene's rarity within its group: groups holds one group number per gene, -1 for none
    (NaN rarity). walked(rows) gives the values the subset walk reads for those genes (rows)."""
    cond = np.asarray(cond)
    n1 = int(cond.sum())
    chunk = max(1, CURVE_BYTES // (8 * max(n1 - 1, 1)))
    k = cuts(size, side, n1)
    genes = len(size)
    rare, share = np.full(genes, np.nan), np.full(genes, np.nan)
    for group in np.unique(groups[groups >= 0]):
        members = np.flatnonzero(groups == group)
        with_subset = members[size[members] > 0]
        own = np.empty(genes)                                 # each such gene's gap at its own cut
        for at in range(0, len(with_subset), chunk):
            part = with_subset[at:at + chunk]
            own[part] = gap_curves(walked(part), cond)[np.arange(len(part)), k[part] - 1]
        counts = np.zeros(genes, np.int64)
        for at in range(0, len(members), chunk):
            curves = gap_curves(walked(members[at:at + chunk]), cond)
            for b in range(0, len(with_subset), GENE_BLOCK):
                rows = with_subset[b:b + GENE_BLOCK]
                counts[rows] += np.sum(curves[:, k[rows] - 1] >= own[rows], axis=0)
        share[members] = 1.0
        share[with_subset] = counts[with_subset] / len(members)
        ordered = np.sort(share[members])
        rare[members] = np.searchsorted(ordered, share[members], side="right") / len(members)
    return rare


def memberships(walked, cond, size, side, genes, block=4096):
    """For each of genes, the case positions (among the cases the walk reads) in its subset, furthest
    out first. walked(rows) gives the values the subset walk reads for those genes."""
    cases = np.flatnonzero(np.asarray(cond) == 1)
    genes = np.asarray(genes, dtype=np.int64)
    out = []
    for at in range(0, len(genes), block):
        part = genes[at:at + block]
        values = walked(part)[:, cases]
        for row, g in zip(values, part):
            out.append(np.argsort(-row if side[g] > 0 else row, kind="stable")[:size[g]])
    return out


def sample_load(walked, cond, size, side, rare, gene_names, *, drivers):
    """Each case's share of the rare subsets, over what random membership would give it.

    Every gene with a subset spreads its weight, -log(rarity), over its members: a singleton gives
    all of it to one case, a subset of 40% of the cases a little to each. A case's load is the weight
    it receives, divided by what it would receive if each subset's members were drawn at random
    (sum of the weights over n1); the loads average exactly 1. Returns the cases' load, the number of
    subsets each belongs to, and each case's top driving genes, most weight first.
    """
    cond = np.asarray(cond)
    n1 = int(cond.sum())
    genes = np.flatnonzero((size > 0) & np.isfinite(rare))
    weight = -np.log(rare[genes])
    load, joined = np.zeros(n1), np.zeros(n1, np.int64)
    entries = []
    for g, w, members in zip(genes, weight, memberships(walked, cond, size, side, genes)):
        load[members] += w / size[g]
        joined[members] += 1
        entries.append((members, np.full(len(members), g), np.full(len(members), w / size[g])))
    expected = weight.sum() / n1 if len(genes) else 1.0
    top = [[] for _ in range(n1)]
    if entries:
        case, gene, part = (np.concatenate(e) for e in zip(*entries))
        order = np.lexsort((-part, case))                 # by case, most weight first
        case, gene = case[order], gene[order]
        starts = np.searchsorted(case, np.arange(n1))
        ends = np.searchsorted(case, np.arange(n1), side="right")
        top = [list(gene_names[gene[a:min(e, a + drivers)]]) for a, e in zip(starts, ends)]
    return load / expected if expected > 0 else load, joined, top


def rare_subsets(walked, cond, size, side, rare, groups, group, gene_names, sample_names, *, n):
    """The n rarest subsets of one group, each with its members and the listed genes it shares members with."""
    cond = np.asarray(cond)
    candidates = np.flatnonzero((groups == group) & (size > 0) & np.isfinite(rare))
    genes = candidates[np.argsort(rare[candidates], kind="stable")[:n]]
    members = memberships(walked, cond, size, side, genes)
    cases = np.asarray(sample_names)[cond == 1]
    sets = [set(m.tolist()) for m in members]
    shared = []
    for i, mine in enumerate(sets):
        overlap = [(len(mine & other), j) for j, other in enumerate(sets) if j != i and mine & other]
        overlap.sort(key=lambda t: (-t[0], t[1]))
        shared.append([(gene_names[genes[j]], count) for count, j in overlap])
    return dict(gene=gene_names[genes], rarity=rare[genes], side=np.where(side[genes] > 0, "up", "down"),
                size=size[genes], members=[cases[m] for m in members], shared=shared)
