"""
Subcellular Localization Features for OPS Feature Extraction.

This module computes features that describe the spatial relationship between
organelle objects and cellular landmarks (nucleus, cell boundary).

These features are biologically informative for detecting phenotypes like:
- ER stress (ER clustering near nucleus)
- Mitochondrial fragmentation/perinuclear clustering
- Golgi dispersal
- Lysosome/autophagosome trafficking defects

Features computed per organelle object:
--------------------------------------
- distance_from_cell_edge: Distance from object centroid to nearest cell boundary pixel
- distance_from_nucleus: Distance from object centroid to nearest nuclear boundary pixel
- distance_from_nucleus_centroid: Euclidean distance from object centroid to nucleus centroid
- normalized_radial_position: 0 = at nucleus, 1 = at cell edge (normalized position)

Performance:
------------
Uses KDTree-based queries for ~175x speedup over distance transform approach:
- Distance transform: O(H*W) = 90,000 operations per mask per organelle
- KDTree query: O(n * log(m)) where n=100 objects, m=~2000 boundary points

Aggregation to cell level:
--------------------------
For each feature, the standard 7 aggregation functions are applied:
sum, mean, median, std, min, max, count

This reveals organelle distribution patterns:
- Low std + high mean distance = uniformly peripheral
- High std = heterogeneous distribution
- Low mean distance = perinuclear clustering
"""

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries


# =============================================================================
# Fast KDTree-based Localization (175x faster than distance transforms)
# =============================================================================

def precompute_boundary_kdtrees(
    cell_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    spacing: tuple,
) -> dict:
    """Build KDTrees from cell/nuclear boundaries for fast distance queries.

    This is ~175x faster than distance transforms for localization because:
    - Distance transform: O(H*W) = 90,000 operations per mask
    - KDTree query: O(n * log(m)) where n=100 objects, m=~2000 boundary points

    Parameters
    ----------
    cell_mask : np.ndarray
        Cell segmentation mask.
    nuclear_mask : np.ndarray
        Nuclear segmentation mask (can be None).
    spacing : tuple
        Pixel spacing (y, x) for distance calculations.

    Returns
    -------
    dict
        Dictionary containing:
        - cell_boundary_tree: KDTree of cell boundary points
        - nuclear_boundary_tree: KDTree of nuclear boundary points (if available)
        - nucleus_centroid: (y, x) centroid of nucleus in physical units

    Examples
    --------
    >>> tree_cache = precompute_boundary_kdtrees(cell_mask, nuclear_mask, (0.65, 0.65))
    >>> loc_df = compute_localization_kdtree(organelle_mask, tree_cache, (0.65, 0.65))
    """
    cache = {}
    spacing_arr = np.array(spacing)

    # Cell boundary KDTree
    cell_binary = (cell_mask > 0).astype(np.uint8)
    cell_boundary = find_boundaries(cell_binary, mode='inner')
    boundary_coords = np.argwhere(cell_boundary)

    if len(boundary_coords) > 0:
        # Convert to physical coordinates
        boundary_physical = boundary_coords * spacing_arr
        cache["cell_boundary_tree"] = cKDTree(boundary_physical)
    else:
        cache["cell_boundary_tree"] = None

    # Nuclear boundary KDTree (if available)
    if nuclear_mask is not None and np.any(nuclear_mask > 0):
        nuclear_binary = (nuclear_mask > 0).astype(np.uint8)
        nuc_boundary = find_boundaries(nuclear_binary, mode='inner')
        nuc_coords = np.argwhere(nuc_boundary)

        if len(nuc_coords) > 0:
            nuc_physical = nuc_coords * spacing_arr
            cache["nuclear_boundary_tree"] = cKDTree(nuc_physical)
        else:
            cache["nuclear_boundary_tree"] = None

        # Nucleus centroid in physical units
        nuc_ys, nuc_xs = np.where(nuclear_mask > 0)
        if len(nuc_ys) > 0:
            cache["nucleus_centroid"] = (
                nuc_ys.mean() * spacing[0],
                nuc_xs.mean() * spacing[1]
            )

    return cache


def compute_localization_kdtree(
    organelle_mask: np.ndarray,
    tree_cache: dict,
    spacing: tuple = (1.0, 1.0),
) -> pd.DataFrame:
    """Compute localization features using KDTree queries (vectorized).

    ~175x faster than distance transform approach:
    - Uses scipy.ndimage.center_of_mass for fast centroid extraction
    - Batch queries all centroids at once via KDTree

    Parameters
    ----------
    organelle_mask : np.ndarray
        Labeled organelle segmentation mask.
    tree_cache : dict
        Pre-computed KDTrees from precompute_boundary_kdtrees().
    spacing : tuple
        Pixel spacing (y, x).

    Returns
    -------
    pd.DataFrame
        Localization features per organelle object with columns:
        - label: Object label ID
        - dist_to_cell_edge: Distance to nearest cell boundary
        - dist_to_nucleus: Distance to nearest nuclear boundary
        - dist_to_nucleus_centroid: Distance to nucleus centroid
        - radial_position: 0-1 normalized (0=nucleus, 1=cell edge)

    Examples
    --------
    >>> tree_cache = precompute_boundary_kdtrees(cell_mask, nuclear_mask, spacing)
    >>> loc_df = compute_localization_kdtree(organelle_mask, tree_cache, spacing)
    """
    if not np.any(organelle_mask > 0):
        return pd.DataFrame()

    # Get unique labels
    labels = np.unique(organelle_mask)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return pd.DataFrame()

    # Fast vectorized centroid computation using scipy.ndimage
    # This is much faster than regionprops_table for just centroids
    centroids = center_of_mass(organelle_mask > 0, organelle_mask, labels)
    centroids = np.array(centroids)  # Shape: (n_objects, 2) in (y, x) format

    # Convert to physical coordinates
    spacing_arr = np.array(spacing)
    centroids_physical = centroids * spacing_arr

    # Initialize results
    results = {
        "label": labels,
        "distance_from_cell_edge": np.full(len(labels), np.nan),
    }

    # Batch query cell boundary distances
    cell_tree = tree_cache.get("cell_boundary_tree")
    if cell_tree is not None and len(centroids_physical) > 0:
        cell_distances, _ = cell_tree.query(centroids_physical)
        results["distance_from_cell_edge"] = cell_distances

    # Batch query nuclear boundary distances (if available)
    nuc_tree = tree_cache.get("nuclear_boundary_tree")
    if nuc_tree is not None and len(centroids_physical) > 0:
        nuc_distances, _ = nuc_tree.query(centroids_physical)
        results["distance_from_nucleus"] = nuc_distances

        # Distance to nucleus centroid
        if "nucleus_centroid" in tree_cache:
            nuc_cy, nuc_cx = tree_cache["nucleus_centroid"]
            results["distance_from_nucleus_centroid"] = np.sqrt(
                (centroids_physical[:, 0] - nuc_cy)**2 +
                (centroids_physical[:, 1] - nuc_cx)**2
            )
            # Angle of each object about the nucleus centroid (radians, -pi..pi).
            # Reused by the radial-distribution anisotropy feature; not aggregated
            # by compute_cell_level_localization_summary (fixed feature list).
            results["angle_from_nucleus"] = np.arctan2(
                centroids_physical[:, 0] - nuc_cy,
                centroids_physical[:, 1] - nuc_cx,
            )

        # Radial position (0 = at nucleus, 1 = at cell edge)
        if cell_tree is not None:
            total_dist = results["distance_from_cell_edge"] + results["distance_from_nucleus"]
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                results["normalized_radial_position"] = np.where(
                    total_dist > 0,
                    results["distance_from_nucleus"] / total_dist,
                    0.5
                )

    return pd.DataFrame(results)


def compute_cell_level_localization_summary(
    localization_df: pd.DataFrame,
    organelle_name: str,
) -> dict:
    """Aggregate per-object localization features to cell level.

    Applies the standard 7 aggregation functions to each localization feature:
    sum, mean, median, std, min, max, count

    Parameters
    ----------
    localization_df : pd.DataFrame
        DataFrame from compute_localization_kdtree() with per-object features.
    organelle_name : str
        Name of the organelle (for prefixing feature names).

    Returns
    -------
    dict
        Dictionary of aggregated features with keys like:
        "{organelle}_dist_to_cell_edge_mean"
    """
    if localization_df.empty:
        return {}

    agg_funcs = ["sum", "mean", "median", "std", "min", "max", "count"]
    feature_cols = [
        "distance_from_cell_edge",
        "distance_from_nucleus",
        "distance_from_nucleus_centroid",
        "normalized_radial_position",
    ]

    result = {}
    for col in feature_cols:
        if col not in localization_df.columns:
            continue

        values = localization_df[col].dropna()
        if len(values) == 0:
            continue

        for agg_func in agg_funcs:
            if agg_func == "sum":
                val = values.sum()
            elif agg_func == "mean":
                val = values.mean()
            elif agg_func == "median":
                val = values.median()
            elif agg_func == "std":
                val = values.std() if len(values) > 1 else 0.0
            elif agg_func == "min":
                val = values.min()
            elif agg_func == "max":
                val = values.max()
            elif agg_func == "count":
                val = len(values)

            result[f"{organelle_name}_{col}_{agg_func}"] = val

    # Radial distribution profile — computed for free from the per-object
    # localization already in this dataframe (no extra segmentation/KDTree work):
    #   {organelle}_radial_frac_bin{i} : fraction of objects per concentric shell
    #                                    (nucleus=0 .. cell edge=1), sums to 1
    #   {organelle}_radial_anisotropy  : CV of object counts across angular wedges
    result.update(_radial_distribution_from_df(localization_df, organelle_name))

    return result


def _radial_distribution_from_df(localization_df, organelle_name, n_bins=5, n_wedges=8):
    """Radial-distribution features from an existing localization dataframe.

    Pure histogramming of `normalized_radial_position` (+ `angle_from_nucleus`
    for the anisotropy) — no recomputation. Returns {} if radial position is
    unavailable (no nucleus).
    """
    if "normalized_radial_position" not in localization_df.columns:
        return {}
    radial = localization_df["normalized_radial_position"].to_numpy()
    radial = radial[np.isfinite(radial)]
    if radial.size == 0:
        return {}

    out = {}
    counts, _ = np.histogram(np.clip(radial, 0.0, 1.0), bins=np.linspace(0.0, 1.0, n_bins + 1))
    total = counts.sum()
    if total > 0:
        for i, c in enumerate(counts):
            out[f"{organelle_name}_radial_frac_bin{i}"] = float(c) / float(total)

    if "angle_from_nucleus" in localization_df.columns:
        ang = localization_df["angle_from_nucleus"].to_numpy()
        ang = ang[np.isfinite(ang)]
        if ang.size > 0:
            wcounts, _ = np.histogram(ang, bins=np.linspace(-np.pi, np.pi, n_wedges + 1))
            mean_w = wcounts.mean()
            out[f"{organelle_name}_radial_anisotropy"] = (
                float(wcounts.std() / mean_w) if mean_w > 0 else 0.0
            )
    return out


def compute_localization_features(
    organelle_mask: np.ndarray,
    cell_mask: np.ndarray,
    nuclear_mask: np.ndarray = None,
    spacing: tuple = (1.0, 1.0),
) -> pd.DataFrame:
    """
    Compute localization features for organelles relative to cell/nucleus.

    This is a convenience function for visualization that wraps the KDTree-based
    localization computation.

    Parameters
    ----------
    organelle_mask : np.ndarray
        Labeled organelle segmentation mask.
    cell_mask : np.ndarray
        Binary cell mask.
    nuclear_mask : np.ndarray, optional
        Binary nuclear mask.
    spacing : tuple
        Pixel spacing (y, x).

    Returns
    -------
    pd.DataFrame
        Per-object localization features.
    """
    if not np.any(organelle_mask > 0):
        return pd.DataFrame()

    # Pre-compute boundary KDTrees
    tree_cache = precompute_boundary_kdtrees(
        cell_mask=cell_mask,
        nuclear_mask=nuclear_mask,
        spacing=spacing,
    )

    # Compute localization using KDTree queries
    return compute_localization_kdtree(
        organelle_mask=organelle_mask,
        tree_cache=tree_cache,
        spacing=spacing,
    )


# List of localization feature names for documentation and aggregation
LOCALIZATION_FEATURES = [
    "distance_from_cell_edge",
    "distance_from_nucleus",
    "distance_from_nucleus_centroid",
    "normalized_radial_position",
]
