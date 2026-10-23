"""Label exchange preserving group sizes and prespecified strata."""
from math import comb, log10
import numpy as np

# The run's shuffles come in blocks of this many, each from its own generator, so that any
# shuffle can be regenerated from its position in the stream alone.
STREAM_BLOCK = 1024


def stream_block(cond, block, seed):
    """Block `block` of the run's one stream of label shuffles: shuffles block * STREAM_BLOCK
    onward, as rows of labels.

    Each block is drawn by its own generator, seeded by the run's seed and the block's index
    (spawn key (block, 2); the jitter uses (gene,) and the thinning (gene, 1)). Any shuffle can
    therefore be regenerated from its position alone, so a gene can pause and later resume exactly
    where it stopped, whatever the chunking or the order in which genes are run. Every gene, in
    both tests, sees the same stream.
    """
    cond = np.asarray(cond, dtype=np.uint8)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block, 2)))
    rows = np.tile(cond, (STREAM_BLOCK, 1))
    return rng.permuted(rows, axis=1, out=rows)


def validate_labels(cond, samples):
    labels = np.asarray(cond)
    if labels.shape != (samples,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("cond must contain one binary 0/1 label per sample")
    if not 0 < labels.sum() < samples:
        raise ValueError("both case and control groups must be nonempty")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def strata_indices(strata, n_samples: int) -> list[np.ndarray]:
    """Sample indices grouped by stratum label, in order of first appearance.

    Any hashable labels are accepted (study names, batch ids, integers), and
    the grouping is by equality. The order is deterministic — first
    appearance, not sorted — so a run is reproducible whatever the label type.
    """
    strata = np.asarray(strata)
    if strata.ndim != 1:
        raise ValueError(f"strata must be a 1-D per-sample vector, got shape {strata.shape}")
    if strata.shape[0] != n_samples:
        raise ValueError(
            f"strata must have one entry per sample: expected {n_samples}, "
            f"got {strata.shape[0]}"
        )
    seen: dict = {}
    for j, lab in enumerate(strata.tolist()):
        if lab is None or (isinstance(lab, float) and not np.isfinite(lab)):
            raise ValueError("strata cannot contain missing or non-finite labels")
        seen.setdefault(lab, []).append(j)
    return [np.asarray(v, dtype=np.intp) for v in seen.values()]


def permutation_space(cond: np.ndarray, strata=None) -> dict:
    """How many distinct label assignments the design admits, and the p-value
    floor that implies.

    Unrestricted, the count is ``C(n, n1)``. Restricted to strata it is
    ``prod_s C(n_s, k_s)``, and any stratum containing only one class
    contributes a factor of 1 — no freedom at all — so a design stratified
    into many small studies can have a permutation space too small to
    support the p-values it is asked for.

    Returns ``n_strata``, ``log10_space``, ``p_floor`` (the smallest
    attainable p-value, ``1 / space``), and ``uninformative_strata`` (those
    with only one class present).
    """
    cond = np.asarray(cond)
    cond = validate_labels(cond, len(cond))
    if strata is None:
        groups = [np.arange(cond.shape[0], dtype=np.intp)]
    else:
        groups = strata_indices(strata, cond.shape[0])
    log10_space = 0.0
    uninformative = 0
    for idx in groups:
        c = cond[idx]
        n_s = int(c.size)
        k_s = int(np.sum(c == 1))
        if k_s == 0 or k_s == n_s:
            uninformative += 1
            continue
        log10_space += log10(comb(n_s, k_s))
    return {
        "n_strata": len(groups),
        "log10_space": log10_space,
        "p_floor": 10.0 ** (-log10_space),
        "uninformative_strata": uninformative,
    }


def draw_perms(
    cond: np.ndarray,
    n_perms: int,
    seed: int | None = 1,
    rng: np.random.Generator | None = None,
    strata=None,
) -> np.ndarray:
    """Draw ``n_perms`` label permutations. Returns ``(n_perms, n_samples)``.

    Each row is a permutation of ``cond``'s entries (labels, not index
    vectors), so the group sizes are preserved.

    ``strata`` restricts the shuffle: labels are permuted **within** each
    stratum, so every stratum keeps its own case/control counts. This is the
    restricted permutation a batch-structured cohort needs; a stratum with
    only one class present contributes no freedom, and
    :func:`permutation_space` counts what is left.
    """
    cond = np.asarray(cond)
    cond = validate_labels(cond, len(cond))
    if isinstance(n_perms, bool) or not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")
    if rng is None:
        rng = np.random.default_rng(seed)
    out = np.empty((n_perms, cond.shape[0]), dtype=cond.dtype)
    if strata is None:
        for b in range(n_perms):
            out[b] = rng.permutation(cond)
        return out
    groups = strata_indices(strata, cond.shape[0])
    for b in range(n_perms):
        row = out[b]
        for idx in groups:
            row[idx] = rng.permutation(cond[idx])
    return out


def validate_perms(perms: np.ndarray, cond: np.ndarray, n_perms: int,
                   strata=None) -> np.ndarray:
    """Check a supplied permutation matrix is what it claims to be.

    With ``strata``, the check tightens: permuting within strata preserves the
    global label multiset *and* every stratum's own counts, so a row that
    moved a label across strata passes the global test and is still wrong.
    Both are checked.
    """
    perms = np.asarray(perms)
    cond = np.asarray(cond)
    if perms.ndim != 2:
        raise ValueError(
            f"perms must be 2-D, shape (n_perms, n_samples); got {perms.shape}"
        )
    if perms.shape != (n_perms, cond.shape[0]):
        raise ValueError(
            f"perms must have shape (n_perms, n_samples) = "
            f"({n_perms}, {cond.shape[0]}); got {perms.shape}. Note the row axis is "
            f"the permutation axis."
        )
    ok = np.all(np.sort(perms, axis=1) == np.sort(cond)[None, :], axis=1)
    if not ok.all():
        bad = int(np.argmin(ok))
        raise ValueError(
            f"every row of perms must be a permutation of cond (group sizes are "
            f"preserved under label exchange); row {bad} is not."
        )
    if strata is not None:
        for idx in strata_indices(strata, cond.shape[0]):
            want_s = int(np.sum(cond[idx] == 1))
            got_s = np.sum(perms[:, idx] == 1, axis=1)
            if not np.all(got_s == want_s):
                bad = int(np.flatnonzero(got_s != want_s)[0])
                raise ValueError(
                    f"restricted permutation must preserve each stratum's case count: "
                    f"row {bad} has {int(got_s[bad])} cases in a stratum that has "
                    f"{want_s}. Labels were moved across strata."
                )
    return perms
