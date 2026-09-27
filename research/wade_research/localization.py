"""Descriptive attribution of upward area and a control-relative tail cutoff.

Neither the effective contributor fraction nor the case mass above the cutoff
is a latent mixture prevalence or an individual significance probability.
"""
from dataclasses import dataclass

import numpy as np

from .assessment import log2_ratio, paired_quantiles


@dataclass
class Localization:
    summary: dict
    membership: np.ndarray       # case kernel mass above the common abundance cutoff
    contributions: np.ndarray    # mean normalized upward displacement per case


def localize(distribution, labels):
    """Attribute upward quantile transport, splitting tied mass symmetrically.

    For a case abundance a with quantile interval I(a), average
    [a-Q_control(u)]+ over I(a). Integrating this displacement over patient i's
    kernel gives r_i. Then mean_i(r_i) equals the upward Wasserstein area.

    A separate descriptive cutoff maximizes F_control(a)-F_case(a), with a
    strict >a tail and the highest abundance on exact ties. Its maximum is
    excess tail mass, not the probability that a patient was biologically altered.
    """
    labels = np.asarray(labels)
    if (labels.shape != distribution.means.shape or not np.isin(labels, [0, 1]).all()
            or labels.sum() in (0, len(labels))):
        raise ValueError('Require aligned binary labels and both groups')
    case = labels.astype(bool)
    nodes = distribution.nodes[case]
    control = distribution.nodes[~case]
    edges, qa, qb = paired_quantiles(nodes, control)
    widths = np.diff(edges)
    displacements = np.maximum(qa-qb, 0)
    values, inverse = np.unique(qa, return_inverse=True)
    mass = np.bincount(inverse, weights=widths)
    area_by_value = np.bincount(inverse, weights=widths*displacements)
    per_mass = area_by_value/mass
    node_displacement = per_mass[np.searchsorted(values, nodes)]
    contributions = node_displacement.mean(axis=1)
    area = float(np.sum(widths*displacements))
    np.testing.assert_allclose(contributions.mean(), area, rtol=2e-12, atol=1e-12)
    # Scaling first avoids squaring large abundance units.
    relative = contributions/contributions.max() if contributions.max() > 0 else contributions
    effective = float(relative.mean()**2/np.mean(relative**2)) if area > 0 else 0.

    a, b = np.sort(nodes.ravel()), np.sort(control.ravel())
    support = np.unique(np.r_[a, b])
    na, nb = np.searchsorted(a, support, side='right'), np.searchsorted(b, support, side='right')
    # Integer masses make ties exact and independent of input/sample order.
    numerator = len(a)*nb-len(b)*na
    maximum = max(0, int(numerator.max()))
    membership = np.zeros(len(nodes))
    cutoff, case_tail, control_tail, tail_fold, area_share = np.nan, 0., 0., np.nan, 0.
    if maximum:
        index = np.flatnonzero(numerator == maximum)[-1]
        cutoff = float(support[index])
        above = nodes > cutoff
        membership = above.mean(axis=1)
        case_tail, control_tail = float(above.mean()), float(np.mean(control > cutoff))
        tail_fold = log2_ratio(nodes[above].mean(), distribution.means[~case].mean())
        area_share = float(np.mean(node_displacement*above)/area) if area > 0 else 0.
    return Localization(dict(area=area, effective_contributor_fraction=effective,
        cutoff=cutoff, case_tail_mass=case_tail, control_tail_mass=control_tail,
        excess_tail_mass=maximum/(len(a)*len(b)), tail_vs_control_log2=tail_fold,
        tail_area_share=area_share), membership, contributions)
