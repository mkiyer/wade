"""Label exchange preserving group sizes and prespecified strata."""
from math import comb, log10

import numpy as np

from .io import _missing

# The run's shuffles come in blocks of this many, each from its own generator, so that any
# shuffle can be regenerated from its position in the stream alone.
STREAM_BLOCK = 1024


def stream_block(cond, block, seed, code=None):
    """Block `block` of the run's one stream of label shuffles: shuffles block * STREAM_BLOCK
    onward, as rows of labels.

    Each block is drawn by its own generator, seeded by the run's seed and the block's index
    (spawn key (block, 2); _means.stream lists the other keys). Any shuffle can
    therefore be regenerated from its position alone, so a gene can pause and later resume exactly
    where it stopped, whatever the chunking or the order in which genes are run. Every gene, in
    both tests, sees the same stream.

    With strata (code: each sample's stratum, 0 to S - 1), a shuffle exchanges labels only within
    strata: the block's generator permutes each stratum's labels in turn, in the order of the
    strata. A stratum with only one group cannot move and draws nothing. One stratum is the
    unstratified stream exactly.
    """
    cond = np.asarray(cond, dtype=np.uint8)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block, 2)))
    rows = np.tile(cond, (STREAM_BLOCK, 1))
    if code is None:
        return rng.permuted(rows, axis=1, out=rows)
    for idx in stratum_columns(code):
        if 0 < cond[idx].sum() < len(idx):
            rows[:, idx] = rng.permuted(rows[:, idx], axis=1)
    return rows


def stratum_columns(code):
    """Each stratum's sample positions, strata in order of their code (0 to S - 1)."""
    code = np.asarray(code)
    order = np.argsort(code, kind="stable")
    bounds = np.flatnonzero(np.diff(code[order])) + 1
    return np.split(order, bounds)


def strata_codes(strata, samples):
    """Each sample's stratum as 0 to S - 1, numbered in order of first appearance. strata is one label
    per sample, or a list of factors (one label per sample each), whose combinations are the strata.
    Missing or non-finite labels are an error."""
    if isinstance(strata, (list, tuple)) and len(strata) and all(np.ndim(f) == 1 and len(f) == samples
                                                                 for f in strata):
        keys = list(zip(*[np.asarray(f, dtype=object).tolist() for f in strata]))
    else:
        keys = np.asarray(strata, dtype=object).tolist()
        if np.ndim(np.asarray(strata, dtype=object)) != 1 or len(keys) != samples:
            raise ValueError(f"strata must be one label per sample ({samples}), or a list of such factors")
    seen = {}
    out = np.empty(len(keys), np.int64)
    for j, key in enumerate(keys):
        for label in (key if isinstance(key, tuple) else (key,)):
            if _missing(label):
                raise ValueError("strata cannot contain missing or non-finite labels")
        out[j] = seen.setdefault(key, len(seen))
    return out


def validate_labels(cond, samples):
    labels = np.asarray(cond)
    if labels.shape != (samples,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("cond must contain one binary 0/1 label per sample")
    if not 0 < labels.sum() < samples:
        raise ValueError("both case and control groups must be nonempty")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def permutation_space(cond, strata=None) -> dict:
    """How many distinct label assignments the design admits, and the p-value
    floor that implies.

    Unrestricted, the count is ``C(n, n1)``. Restricted to strata (as wade()
    takes them) it is ``prod_s C(n_s, k_s)``, and any stratum containing only
    one class contributes a factor of 1 — no freedom at all — so a design
    stratified into many small studies can have a permutation space too small
    to support the p-values it is asked for.

    Returns ``n_strata``, ``log10_space``, ``p_floor`` (the smallest
    attainable p-value, ``1 / space``), and ``uninformative_strata`` (those
    with only one class present). wade() records it in params["permutation_space"].
    """
    cond = np.asarray(cond)
    cond = validate_labels(cond, len(cond))
    if strata is None:
        groups = [np.arange(cond.shape[0], dtype=np.intp)]
    else:
        groups = stratum_columns(strata_codes(strata, cond.shape[0]))
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
