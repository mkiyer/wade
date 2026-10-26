"""Frozen biological truth for the replicated direction/scale comparison."""
import numpy as np


def datasets(protocol):
    """The registry of matrices: replicates per allocation and panel (protocol version 7 gives each panel
    its own count, "replicates"; earlier versions one, "replicates_per_cell")."""
    for allocation, (nc, n0) in enumerate(protocol["allocations"]):
        for panel_index, panel in enumerate(protocol["panels"]):
            count = protocol["replicates"][panel] if "replicates" in protocol else protocol["replicates_per_cell"]
            for replicate in range(count):
                yield dict(id=f"{panel}-{nc}v{n0}-r{replicate:02d}", n_case=nc, n_control=n0,
                           panel=panel, replicate=replicate, biology=protocol["biology"][replicate % 2],
                           seed=[protocol["seed_root"], allocation, panel_index, replicate],
                           permutation_seed=[protocol["seed_root"], allocation, panel_index, replicate, 97])


def generate(protocol, settings):
    cfg = protocol["generator"]
    rng = np.random.default_rng(settings["seed"])
    g, nc, n0 = protocol["genes"], settings["n_case"], settings["n_control"]
    n = nc+n0
    cond = np.r_[np.ones(nc, int), np.zeros(n0, int)]
    exposure = np.exp(rng.normal(0, cfg["exposure_log_sd"], n))
    exposure /= np.exp(np.log(exposure).mean())
    means = np.exp(rng.uniform(*np.log(cfg["background_mean_range"]), g))
    cv2 = rng.uniform(*cfg["biological_cv2_range"], g)
    first = cfg["reference_controls"]
    means[:first] = 100
    copies = cfg.get("profile_copies", 1)                 # the profile panel repeated (version 7: 16)
    profile_means = np.tile(np.repeat(cfg["profile_means"], len(cfg["profiles"])), copies)
    profile_indices = np.arange(first, first+len(profile_means))
    means[profile_indices] = profile_means
    profile = np.full(g, "background_null", dtype="<U52")
    if settings["panel"] == "mixed":
        profile[profile_indices] = np.tile(cfg["profiles"], len(cfg["profile_means"]) * copies)
    shared = np.arange(profile_indices[-1]+1, profile_indices[-1]+1+cfg["shared_tail_null_profiles"] * copies)
    means[shared] = np.tile(cfg["profile_means"], copies)
    profile[shared] = "shared_tail_null"
    module = np.arange(g) % cfg["lognormal_modules"]
    rho = cfg["within_module_log_correlation"]
    if settings["biology"] == "gamma":
        z = None
        latent = rng.gamma(1/cv2[:, None], means[:, None]*cv2[:, None], (g, n))
    elif settings["biology"] == "correlated_lognormal":
        z = (np.sqrt(rho)*rng.normal(size=(cfg["lognormal_modules"], n))[module]
             + np.sqrt(1-rho)*rng.normal(size=(g, n)))
        variance = np.log1p(cv2)
        latent = means[:, None]*np.exp(np.sqrt(variance[:, None])*z-variance[:, None]/2)
    else:
        raise ValueError("Unknown biology")
    altered = ~np.isin(profile, ["background_null", "shared_tail_null"])
    population_ratio = np.ones(g)
    ordered_direction = np.zeros(g, int)
    perturbed = np.zeros((g, nc), bool)
    enriched = np.zeros((g, nc), bool)
    for gene in np.flatnonzero(altered):
        pattern = profile[gene]
        permutation = rng.permutation(nc)
        fraction = .05 if "5pct" in pattern else .2 if "20pct" in pattern else .1
        k = max(1, round(nc*fraction))
        member = permutation[:k]
        multiplier = np.ones(nc)
        if pattern == "bulk_up_1.5x":
            multiplier[:] = 1.5
            ordered_direction[gene] = 1
        elif pattern == "bulk_down_two_thirds":
            multiplier[:] = 2/3
            ordered_direction[gene] = -1
        elif pattern in ("subset_up_5pct_4x", "subset_up_10pct_4x", "subset_up_20pct_4x"):
            multiplier[member] = 4
        elif pattern == "subset_up_10pct_16x":
            multiplier[member] = 16
        elif pattern == "subset_down_10pct_quarter":
            multiplier[member] = .25
        elif pattern.startswith("bulk_half_subset_up_10pct_"):
            multiplier[:] = .5
            multiplier[member] = float(pattern.rsplit("_", 1)[1].rstrip("x"))
        elif pattern == "same_mean_wider":
            if z is None:
                latent[gene, :nc] = rng.gamma(1/(4*cv2[gene]), means[gene]*4*cv2[gene], nc)
            else:
                variance = np.log1p(4*cv2[gene])
                latent[gene, :nc] = means[gene]*np.exp(np.sqrt(variance)*z[gene, :nc]-variance/2)
            perturbed[gene] = True
        elif pattern == "mean_preserving_subset_10pct_4x":
            multiplier[:] = (nc-4*k)/(nc-k)
            multiplier[member] = 4
        elif pattern == "opposing_20pct":
            multiplier[permutation[:k]] = 4
            multiplier[permutation[k:2*k]] = .25
        elif pattern == "singleton_up_16x":
            multiplier[permutation[0]] = 16
        else:
            raise ValueError(f"Unknown profile {pattern}")
        latent[gene, :nc] *= multiplier
        perturbed[gene] |= multiplier != 1
        enriched[gene] = multiplier > 1
        population_ratio[gene] = multiplier.mean()
    # Identical random tail mixtures in both groups are genuine distribution nulls.
    for gene in shared:
        latent[gene] *= np.where(rng.random(n) < .1, 4, 1)
    latent[-1] = 0
    means[-1] = 0
    counts = rng.poisson(latent*exposure)
    return dict(counts=counts, latent=latent, exposure=exposure, cond=cond,
                profile=profile, altered=altered, mean_altered=~np.isclose(population_ratio, 1),
                population_log2_ratio=np.log2(population_ratio), ordered_direction=ordered_direction,
                perturbed=perturbed, enriched=enriched, baseline_mean=means, cv2=cv2,
                profile_indices=np.r_[profile_indices, shared])
