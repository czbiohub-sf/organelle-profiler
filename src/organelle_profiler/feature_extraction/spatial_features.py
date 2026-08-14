"""
Spatial-relationship features for OPS feature extraction.

Two cheap, cell-level feature families ported from infer-subc, designed to add
negligible compute on top of the existing morphology/localization/network work:

1. Inter-organelle CONTACTS (``compute_contact_features``)
   Pairwise voxel overlap between every pair of organelle masks already loaded
   for the cell. For each pair with any overlap we emit overlap area, the
   overlap fraction relative to each organelle, and the number of distinct
   contact sites (connected components of the overlap). Cost is one boolean
   AND per pair over the (small) per-cell crop — dominated by, and far below,
   the regionprops pass that already runs.

2. Radial DISTRIBUTION profile (``compute_radial_distribution_features``)
   Per organelle, the fraction of objects in each concentric shell from the
   nucleus (0) to the cell edge (1), plus an angular-anisotropy scalar. This
   reuses the existing KDTree localization (``compute_localization_kdtree`` —
   the ~175x-faster-than-distance-transform path) to get per-object radial
   position and angle, then just histograms them. No per-pixel distance map is
   built (that would be the expensive trap).

Both return flat ``{column_name: value}`` dicts that merge straight into the
per-cell ``cell_features`` dict. Naming is parsed by fe_anndata to assign
category/metric/organelle metadata; keep the suffixes in sync with
``_CONTACT_METRICS`` / ``_DISTRIBUTION_METRIC_PREFIXES`` there.

Compartment masks (the whole-cell ``cell_membrane``) are excluded from contact
pairs since "organelle ∩ whole-cell" is just the organelle's own area.
"""

from itertools import combinations

import numpy as np
from scipy import ndimage as scipy_ndi


# Masks that are compartments, not organelles — excluded from contact pairing.
_COMPARTMENT_KEYS = {"cell_membrane", "cell", "cp_cell_membrane", "cp_cell"}

# Contact metric suffixes (kept in sync with fe_anndata registration).
_CONTACT_METRICS = ("overlap_area", "overlap_frac_a", "overlap_frac_b", "n_contacts")

# Segmentation groups whose names contain these tokens are derived from the SAME
# physical structures imaged in different label-free reconstructions (OPS phase:
# focus3d and phase2d are two reconstructions of the same light path). An overlap
# between a focus3d-derived and a phase2d-derived mask is a channel/registration
# artifact, not a biological contact, so those cross-channel pairs are skipped.
# Within-channel pairs (focus3d_tubular vs focus3d_vesicular) and pairs touching a
# distinct marker (gfp, nuclei, ...) are kept. Extend this tuple to add channels.
_REDUNDANT_CHANNEL_TOKENS = ("focus3d", "phase2d")


def _channel_token(name):
    """Return the redundant-imaging channel token in ``name``, or None."""
    for tok in _REDUNDANT_CHANNEL_TOKENS:
        if tok in name:
            return tok
    return None


def _is_redundant_cross_channel(a, b):
    """True if a and b are the same structures in two redundant phase channels."""
    ta, tb = _channel_token(a), _channel_token(b)
    return ta is not None and tb is not None and ta != tb


def _real_organelles(organelles_with_masks):
    """Names of non-empty organelle masks, excluding whole-cell compartments."""
    out = []
    for name, mask in organelles_with_masks.items():
        if name in _COMPARTMENT_KEYS:
            continue
        if mask is not None and np.any(mask > 0):
            out.append(name)
    return out


def compute_contact_features(organelles_with_masks, spacing):
    """Pairwise inter-organelle contact (mask-overlap) features.

    Parameters
    ----------
    organelles_with_masks : dict
        ``{organelle_name: labeled_mask}`` already cropped to the cell. Whole-cell
        compartment masks are ignored.
    spacing : tuple
        Pixel spacing ``(y_um, x_um)`` used to convert overlap pixels to µm².

    Returns
    -------
    dict
        ``{"contact_{A}__{B}_{metric}": value}`` for every organelle pair with
        non-zero overlap. Pairs are emitted in sorted-name order so the A/B
        assignment (and thus ``overlap_frac_a`` vs ``_frac_b``) is deterministic.
        Empty dict if fewer than two non-empty organelles are present.
    """
    names = sorted(_real_organelles(organelles_with_masks))
    if len(names) < 2:
        return {}

    pixel_area = float(spacing[0]) * float(spacing[1])
    binar = {n: (organelles_with_masks[n] > 0) for n in names}
    areas = {n: int(binar[n].sum()) for n in names}

    out = {}
    for a, b in combinations(names, 2):
        if _is_redundant_cross_channel(a, b):
            continue  # same structure in two phase reconstructions — not a real contact
        inter = binar[a] & binar[b]
        overlap_px = int(inter.sum())
        if overlap_px == 0:
            continue  # keep output sparse; most pairs don't touch
        key = f"contact_{a}__{b}"
        out[f"{key}_overlap_area"] = overlap_px * pixel_area
        out[f"{key}_overlap_frac_a"] = overlap_px / areas[a] if areas[a] else 0.0
        out[f"{key}_overlap_frac_b"] = overlap_px / areas[b] if areas[b] else 0.0
        # Number of distinct contact sites = connected components of the (small)
        # overlap region. Cheap: only runs on non-empty, typically tiny overlaps.
        _, n_sites = scipy_ndi.label(inter)
        out[f"{key}_n_contacts"] = int(n_sites)
    return out


def compute_radial_distribution_features(
    organelles_with_masks,
    tree_cache,
    spacing,
    n_bins=5,
    n_wedges=8,
):
    """Per-organelle radial distribution profile + angular anisotropy.

    Reuses the precomputed boundary KDTrees and the fast per-object localization
    (``compute_localization_kdtree``) rather than building a per-pixel distance
    map. Requires a nucleus in ``tree_cache`` to define the radial coordinate;
    if absent, returns an empty dict (radial position is undefined).

    Parameters
    ----------
    organelles_with_masks : dict
        ``{organelle_name: labeled_mask}`` cropped to the cell.
    tree_cache : dict
        Output of ``precompute_boundary_kdtrees`` (cell + nuclear boundary trees,
        nucleus centroid).
    spacing : tuple
        Pixel spacing ``(y_um, x_um)``.
    n_bins : int
        Number of concentric shells from nucleus (0) to cell edge (1).
    n_wedges : int
        Number of angular sectors for the anisotropy (coefficient-of-variation)
        of object counts.

    Returns
    -------
    dict
        ``{"{organelle}_radial_frac_bin{i}": frac, "{organelle}_radial_anisotropy": cv}``
    """
    from organelle_profiler.feature_extraction.localization_features import (
        compute_localization_kdtree,
    )

    if tree_cache is None or tree_cache.get("nuclear_boundary_tree") is None:
        return {}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for name in _real_organelles(organelles_with_masks):
        loc_df = compute_localization_kdtree(
            organelle_mask=organelles_with_masks[name],
            tree_cache=tree_cache,
            spacing=spacing,
        )
        if loc_df.empty or "normalized_radial_position" not in loc_df.columns:
            continue
        radial = loc_df["normalized_radial_position"].to_numpy()
        radial = radial[np.isfinite(radial)]
        if radial.size == 0:
            continue

        # Radial shells: fraction of objects per shell (sums to 1).
        counts, _ = np.histogram(np.clip(radial, 0.0, 1.0), bins=bin_edges)
        total = counts.sum()
        if total > 0:
            for i, c in enumerate(counts):
                out[f"{name}_radial_frac_bin{i}"] = float(c) / float(total)

        # Angular anisotropy: CV of object counts across wedges. 0 = isotropic.
        if "angle_from_nucleus" in loc_df.columns:
            ang = loc_df["angle_from_nucleus"].to_numpy()
            ang = ang[np.isfinite(ang)]
            if ang.size > 0:
                wedge_edges = np.linspace(-np.pi, np.pi, n_wedges + 1)
                wcounts, _ = np.histogram(ang, bins=wedge_edges)
                mean_w = wcounts.mean()
                out[f"{name}_radial_anisotropy"] = (
                    float(wcounts.std() / mean_w) if mean_w > 0 else 0.0
                )
    return out


def compute_spatial_features(organelles_with_masks, spacing):
    """Inter-organelle contact (colocalization) features for one cell.

    Radial distribution is NOT computed here — it is emitted for free by the
    localization summary (``compute_cell_level_localization_summary``), which
    reuses the per-object localization already computed. This function is just
    the cheap pairwise mask-overlap pass. Returns {} for <2 organelles.
    """
    return compute_contact_features(organelles_with_masks, spacing)
