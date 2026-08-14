"""
Feature Extraction Constants.

Minimal shared constants. Feature metadata flows through the pipeline,
not reverse-engineered from names.

AnnData var DataFrame columns stored during feature extraction:
- organelle: which organelle (e.g., nuclei, phase2d_tubular, cell, cp_cell)
- metric: base measurement (e.g., area, perimeter, branch_length)
- category: feature type (morphology, network, localization, intensity)
- aggregation: how object-level was aggregated to cell-level (mean, sum, etc.)
- unit: measurement unit (um, um^2, ratio, count, AU)
"""

# Aggregation functions used throughout the pipeline
AGGREGATION_FUNCTIONS = ["mean", "median", "std", "sum", "min", "max", "count"]

# Feature categories (types)
CATEGORY_CELL_MORPHOLOGY = "cell_morphology"
CATEGORY_MORPHOLOGY = "morphology"
CATEGORY_INTENSITY = "intensity"
CATEGORY_LOCALIZATION = "localization"
CATEGORY_NETWORK = "network"
CATEGORY_NETWORK_OBJECT = "network_object"  # Per-object network stats
CATEGORY_CONTACT = "contact"  # Inter-organelle pairwise overlap
CATEGORY_DISTRIBUTION = "distribution"  # Per-organelle radial distribution profile

# Units for common metrics
METRIC_UNITS = {
    # Morphology (scaled by spacing)
    "area": "um^2",
    "perimeter": "um",
    "perimeter_crofton": "um",
    "axis_major_length": "um",
    "axis_minor_length": "um",
    "equivalent_diameter_area": "um",
    "area_convex": "um^2",
    "area_filled": "um^2",
    # Morphology (unitless ratios)
    "solidity": "ratio",
    "extent": "ratio",
    "eccentricity": "ratio",
    "aspect_ratio": "ratio",
    "circularity": "ratio",
    "orientation": "radians",
    "area_fraction": "ratio",
    "euler_number": "count",
    "count": "count",
    # Centroid coordinates (scaled by spacing)
    "centroid_y": "um",
    "centroid_x": "um",
    "centroid_weighted_y": "um",
    "centroid_weighted_x": "um",
    # Intensity (arbitrary units from image)
    "intensity_mean": "AU",
    "intensity_max": "AU",
    "intensity_min": "AU",
    "intensity_std": "AU",
    "intensity_median": "AU",
    "intensity_q25": "AU",
    "intensity_q75": "AU",
    "intensity_iqr": "AU",
    "intensity_mad": "AU",
    "intensity_integrated": "AU",
    "intensity_range": "AU",
    "intensity_cv": "ratio",
    # Intensity-weighted moments (unitless)
    "moments_weighted_hu_0": "ratio",
    "moments_weighted_hu_1": "ratio",
    "moments_weighted_hu_2": "ratio",
    "moments_weighted_hu_3": "ratio",
    "moments_weighted_hu_4": "ratio",
    "moments_weighted_hu_5": "ratio",
    "moments_weighted_hu_6": "ratio",
    # Inertia tensor eigenvalues (unitless)
    "inertia_eigval_0": "ratio",
    "inertia_eigval_1": "ratio",
    # Localization (scaled by spacing)
    "distance_from_cell_edge": "um",
    "distance_from_nucleus": "um",
    "distance_from_nucleus_centroid": "um",
    "normalized_radial_position": "ratio",
    # Network (scaled by spacing)
    "branch_length": "um",
    "branch_thickness": "um",
    "total_branch_length": "um",
    "tortuosity": "ratio",
    "num_branches": "count",
    "num_nodes": "count",
    "num_endpoints": "count",
    "branch_count": "count",
    "average_degree": "ratio",
    "branching_density": "1/um^2",
    "network_length_density": "um/um^2",
    "euler_number": "count",
    "skeleton_pixel_count": "count",
    "num_skeleton_components": "count",
    # Inter-organelle contact (pairwise overlap)
    "overlap_area": "um^2",
    "overlap_frac_a": "ratio",
    "overlap_frac_b": "ratio",
    "n_contacts": "count",
    # Radial distribution profile (per-organelle, cell-level)
    "radial_anisotropy": "ratio",
}


def get_unit_for_metric(metric: str) -> str:
    """Get the unit for a given metric name."""
    if metric in METRIC_UNITS:
        return METRIC_UNITS[metric]
    # Radial distribution shell fractions: radial_frac_bin0, radial_frac_bin1, ...
    if metric.startswith("radial_frac_bin"):
        return "ratio"
    return "unknown"
