"""
Feature Validation Visualization for OPS Feature Extraction.

This module provides visualization tools for validating that extracted features
are measuring real biological structures. It generates grid canvases showing
sampled cells with feature heatmap overlays.

Output:
-------
For each organelle × feature combination, a PNG is generated showing:
- Left panel: Original intensity image with cell boundary
- Right panel: Feature heatmap overlay (objects colored by feature value)
- Shared colorbar across all cells in the grid

Usage:
------
This module is integrated into the feature extraction pipeline and called
automatically when --validation-samples > 0 (default 6).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table, label

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

# Import network analysis for on-demand per-object feature computation
try:
    from .network_analysis import calculate_network_features
    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False

try:
    from .localization_features import compute_localization_features
    LOCALIZATION_AVAILABLE = True
except ImportError:
    LOCALIZATION_AVAILABLE = False


def _expand_mask_to_intensity(
    mask: np.ndarray,
    intensity_shape: tuple,
    offset: tuple,
) -> np.ndarray:
    """
    Expand a mask/label array to match the intensity image size.

    When intensity is loaded with padding for visualization context,
    the masks/labels need to be placed at the correct offset within
    the larger canvas.

    Parameters
    ----------
    mask : np.ndarray
        2D mask or label array at original crop size.
    intensity_shape : tuple
        (H, W) shape of the expanded intensity image.
    offset : tuple
        (y_offset, x_offset) position of mask within expanded image.

    Returns
    -------
    np.ndarray
        Expanded mask/label array matching intensity_shape.
    """
    if mask is None:
        return None

    # Handle 3D masks (1, H, W) -> (H, W)
    if mask.ndim == 3:
        mask = mask[0]

    expanded = np.zeros(intensity_shape, dtype=mask.dtype)
    y_off, x_off = offset
    h, w = mask.shape

    # Calculate valid region (bounds checking)
    y_end = min(y_off + h, intensity_shape[0])
    x_end = min(x_off + w, intensity_shape[1])
    h_valid = y_end - y_off
    w_valid = x_end - x_off

    if h_valid > 0 and w_valid > 0:
        expanded[y_off:y_end, x_off:x_end] = mask[:h_valid, :w_valid]

    return expanded


def _compute_basic_features(label_array: np.ndarray) -> pd.DataFrame:
    """
    Compute basic morphological features from a label array.

    Lightweight feature computation for visualization only.
    Uses regionprops_table directly without spacing (pixel units).

    Parameters
    ----------
    label_array : np.ndarray
        2D label array where each unique value > 0 is an object.

    Returns
    -------
    pd.DataFrame
        DataFrame with label, area, perimeter, axis_major_length, eccentricity.
    """
    if not np.any(label_array > 0):
        return pd.DataFrame()

    props = regionprops_table(
        label_array,
        properties=["label", "area", "perimeter", "axis_major_length", "eccentricity"],
    )
    return pd.DataFrame(props)


def _compute_per_object_network_features(label_array: np.ndarray) -> pd.DataFrame:
    """
    Compute per-object network features from a label array.

    This runs network analysis on-demand for visualization purposes.
    Uses pixel units (spacing=1,1) for simplicity.

    Note: network analysis processes each connected component separately.
    The returned labels correspond to the original label_array IDs.

    When a single object's skeleton gets fragmented (multiple skeleton-ids),
    we aggregate the features: sum for counts (num_branches, num_nodes, etc.),
    mean for continuous metrics (branch_length, tortuosity, etc.).

    Parameters
    ----------
    label_array : np.ndarray
        2D label array where each unique value > 0 is an object.

    Returns
    -------
    pd.DataFrame
        DataFrame with label, num_branches, num_nodes, num_endpoints,
        average_degree, branch_length, branch_thickness, tortuosity.
    """
    if not NETWORK_ANALYSIS_AVAILABLE:
        return pd.DataFrame()

    if not np.any(label_array > 0):
        return pd.DataFrame()

    # Process each labeled object separately to preserve original label IDs
    # calculate_network_features re-labels the binary mask, so we process per-object
    all_object_features = []
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background

    for obj_label in unique_labels:
        # Create binary mask for this single object
        single_obj_mask = (label_array == obj_label).astype(np.uint8)

        try:
            _, network_summary, per_object_df = calculate_network_features(
                single_obj_mask,
                spacing=(1, 1),
                intensity_image=None,
                full_features=False,
            )

            # Use network_summary which has correct totals for the entire mask
            # per_object_df may have multiple rows if skeleton is fragmented
            if network_summary:
                # network_summary contains the aggregated values for the whole object
                all_object_features.append({
                    "label": obj_label,
                    "num_branches": network_summary.get("num_branches", 0),
                    "num_nodes": network_summary.get("num_nodes", 0),
                    "num_endpoints": network_summary.get("num_endpoints", 0),
                    "average_degree": network_summary.get("average_degree", 0),
                    # For branch metrics, use mean from per_object_df if available
                    "branch_length": per_object_df["branch_length"].mean() if not per_object_df.empty and "branch_length" in per_object_df.columns else 0,
                    "branch_thickness": per_object_df["branch_thickness"].mean() if not per_object_df.empty and "branch_thickness" in per_object_df.columns else 0,
                    "tortuosity": per_object_df["tortuosity"].mean() if not per_object_df.empty and "tortuosity" in per_object_df.columns else 1.0,
                })
        except Exception:
            continue

    if all_object_features:
        return pd.DataFrame(all_object_features)
    return pd.DataFrame()


def _compute_localization_features_for_viz(
    label_array: np.ndarray,
    cell_mask: np.ndarray,
    nuclear_mask: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute per-object localization features from a label array.

    This runs localization analysis on-demand for visualization purposes.
    Uses pixel units (spacing=1,1) for simplicity.

    Parameters
    ----------
    label_array : np.ndarray
        2D label array where each unique value > 0 is an object.
    cell_mask : np.ndarray
        2D binary mask of the cell.
    nuclear_mask : np.ndarray, optional
        2D binary mask of the nucleus.

    Returns
    -------
    pd.DataFrame
        DataFrame with label, distance_from_nucleus_centroid, distance_from_nuclear_boundary,
        distance_from_cell_edge, normalized_radial_position, angular_position.
    """
    if not LOCALIZATION_AVAILABLE:
        return pd.DataFrame()

    if not np.any(label_array > 0):
        return pd.DataFrame()

    try:
        return compute_localization_features(
            organelle_mask=label_array,
            cell_mask=cell_mask,
            nuclear_mask=nuclear_mask,
            spacing=(1.0, 1.0),
        )
    except Exception:
        return pd.DataFrame()


# Default features to visualize for validation
# Morphological features that are easy to visually verify:
# - area: larger objects = more pixels (obvious)
# - axis_major_length: major axis aligns with elongated structures
# - eccentricity: 0=circular, 1=elongated (shows tubular vs round)
# Per-object network features (only for network organelles):
# - num_branches: number of skeleton branches per object
# - num_nodes: number of junction points per object
# Localization features (relative to nucleus/cell boundary):
# - normalized_radial_position: 0=at nucleus, 1=at cell edge
# - distance_from_cell_edge: distance to nearest cell boundary
VALIDATION_FEATURES = [
    "area", "axis_major_length", "eccentricity",
    "num_branches", "num_nodes",
    "normalized_radial_position", "distance_from_cell_edge",
]

# Localization features that require special handling (need nuclear mask)
LOCALIZATION_FEATURES = {
    "distance_from_nucleus_centroid",
    "distance_from_nuclear_boundary",
    "distance_from_cell_edge",
    "normalized_radial_position",
    "angular_position",
}

# Network features that are per-object (computed from skeleton analysis)
# These are aggregated per connected component, not per-branch
PER_OBJECT_NETWORK_FEATURES = {
    "num_branches",
    "num_nodes",
    "num_endpoints",
    "average_degree",
    "branch_length",      # Mean branch length per object
    "branch_thickness",   # Mean branch thickness per object
    "tortuosity",         # Mean tortuosity per object
}

# Branch-level features (per-branch, used for skeleton coloring)
BRANCH_LEVEL_FEATURES = {"branch_type"}


def _get_channel_index_for_organelle(
    organelle_name: str,
    channel_names: list,
    organelle_channel_indices: dict,
) -> int:
    """
    Get the channel index for an organelle from the pre-computed mapping.

    Parameters
    ----------
    organelle_name : str
        Internal organelle name (e.g., "cp2_wga_plasma_membrane", "nuclei").
    channel_names : list
        List of channel names (for error message only).
    organelle_channel_indices : dict
        Pre-computed mapping from organelle name to channel index.

    Returns
    -------
    int
        Index of the matching channel.
        Raises ValueError if organelle not in mapping.
    """
    # Direct lookup
    if organelle_name in organelle_channel_indices:
        return organelle_channel_indices[organelle_name]

    # Case-insensitive lookup (feature names may have different case than mappings)
    organelle_lower = organelle_name.lower()
    if organelle_lower in organelle_channel_indices:
        return organelle_channel_indices[organelle_lower]

    # Try matching with case-insensitive key comparison
    for key, value in organelle_channel_indices.items():
        if key.lower() == organelle_lower:
            return value

    raise ValueError(
        f"No channel mapping found for organelle '{organelle_name}'. "
        f"Available mappings: {organelle_channel_indices}. "
        f"Channels: {channel_names}"
    )


def create_heatmap_rgba(
    organelle_labels: np.ndarray,
    features_df: pd.DataFrame,
    feature_name: str,
    cmap_name: str = "viridis",
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    """
    Create RGBA heatmap where each segmented object is colored by its feature value.

    Parameters
    ----------
    organelle_labels : np.ndarray
        2D label array where each unique value > 0 is an object.
    features_df : pd.DataFrame
        DataFrame with 'label' column and feature values.
    feature_name : str
        Column name in features_df to use for coloring.
    cmap_name : str
        Matplotlib colormap name.
    vmin, vmax : float
        Min/max for normalization. If None, uses data range.

    Returns
    -------
    np.ndarray
        RGBA image of shape (H, W, 4) with float values in [0, 1].
    """
    cmap = cm.get_cmap(cmap_name)

    # Create label -> feature value mapping
    if feature_name not in features_df.columns:
        # Return empty transparent image if feature not available
        return np.zeros((*organelle_labels.shape, 4), dtype=np.float32)

    label_to_value = dict(zip(features_df["label"], features_df[feature_name]))

    # Determine normalization range
    feature_values = features_df[feature_name].dropna().values
    if len(feature_values) == 0:
        return np.zeros((*organelle_labels.shape, 4), dtype=np.float32)

    if vmin is None:
        vmin = float(np.min(feature_values))
    if vmax is None:
        vmax = float(np.max(feature_values))

    # Create output RGBA image (transparent background)
    heatmap = np.zeros((*organelle_labels.shape, 4), dtype=np.float32)

    # Color each labeled object
    for label_id, value in label_to_value.items():
        if label_id == 0 or pd.isna(value):
            continue
        mask = organelle_labels == label_id

        # Normalize value to [0, 1]
        if vmax > vmin:
            norm_value = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        else:
            norm_value = 0.5

        # Map to colormap (returns RGBA)
        color = cmap(norm_value)

        # Apply to pixels
        heatmap[mask] = color

    return heatmap


def create_skeleton_heatmap_rgba(
    organelle_mask: np.ndarray,
    branch_df: pd.DataFrame,
    feature_name: str,
    cmap_name: str = "viridis",
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    """
    Create RGBA heatmap showing skeleton colored by branch features.

    Since branch features come from skeleton analysis (not regionprops),
    this method skeletonizes the mask and colors it uniformly based on
    the mean branch feature value.

    Parameters
    ----------
    organelle_mask : np.ndarray
        Binary mask of the organelle (label > 0).
    branch_df : pd.DataFrame
        DataFrame from calculate_network_features with branch-level features
        (branch_length, branch_type, tortuosity, branch_thickness).
    feature_name : str
        Column name in branch_df to use for coloring.
    cmap_name : str
        Matplotlib colormap name.
    vmin, vmax : float
        Min/max for normalization. If None, uses data range.

    Returns
    -------
    np.ndarray
        RGBA image of shape (H, W, 4) with skeleton colored by feature.
    """
    cmap = cm.get_cmap(cmap_name)

    # Create output RGBA image (transparent background)
    heatmap = np.zeros((*organelle_mask.shape, 4), dtype=np.float32)

    if branch_df.empty or feature_name not in branch_df.columns:
        return heatmap

    # Get skeleton
    binary_mask = organelle_mask > 0
    skeleton = skeletonize(binary_mask)

    if not np.any(skeleton):
        return heatmap

    # Get feature values
    feature_values = branch_df[feature_name].dropna().values
    if len(feature_values) == 0:
        return heatmap

    if vmin is None:
        vmin = float(np.min(feature_values))
    if vmax is None:
        vmax = float(np.max(feature_values))

    # For branch_type, use mode (most common type) for visualization
    # Branch types: 0=endpoint-endpoint, 1=endpoint-junction, 2=junction-junction
    if feature_name == "branch_type":
        # Color skeleton by the dominant branch type
        value = branch_df[feature_name].mode().iloc[0] if len(branch_df) > 0 else 1
    else:
        # Use mean for continuous features
        value = branch_df[feature_name].mean()

    # Normalize value to [0, 1]
    if vmax > vmin:
        norm_value = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    else:
        norm_value = 0.5

    # Map to colormap
    color = cmap(norm_value)

    # Color the skeleton pixels
    heatmap[skeleton] = color

    return heatmap


def create_skeleton_topology_rgba(
    organelle_mask: np.ndarray,
    edge_color: tuple = (1, 1, 1, 1),      # White for skeleton edges
    junction_color: tuple = (1, 0, 0, 1),   # Red for junctions (degree >= 3)
    endpoint_color: tuple = (0, 1, 0, 1),   # Green for endpoints (degree == 1)
    node_radius: int = 1,
) -> np.ndarray:
    """
    Create RGBA overlay showing skeleton topology: edges, junctions, and endpoints.

    This helps validate network features by showing exactly what the skeleton
    analysis detects - the paths (edges), branching points (junctions), and
    terminal points (endpoints).

    Parameters
    ----------
    organelle_mask : np.ndarray
        Binary or label mask of the organelle.
    edge_color : tuple
        RGBA color for skeleton edges (default: white).
    junction_color : tuple
        RGBA color for junction points where 3+ branches meet (default: red).
    endpoint_color : tuple
        RGBA color for endpoint/terminal points (default: green).
    node_radius : int
        Radius in pixels for drawing junction/endpoint markers (default: 3).

    Returns
    -------
    np.ndarray
        RGBA image of shape (H, W, 4) with skeleton topology visualization.
    """
    from scipy.ndimage import convolve

    # Create output RGBA image (transparent background)
    overlay = np.zeros((*organelle_mask.shape, 4), dtype=np.float32)

    # Get binary mask and skeleton
    binary_mask = organelle_mask > 0
    if not np.any(binary_mask):
        return overlay

    skeleton = skeletonize(binary_mask)
    if not np.any(skeleton):
        return overlay

    # Draw skeleton edges first (will be partially covered by nodes)
    overlay[skeleton] = edge_color

    # Find nodes by counting neighbors for each skeleton pixel
    # A 3x3 kernel counts the number of 8-connected neighbors
    neighbor_kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],  # Don't count the pixel itself
        [1, 1, 1]
    ], dtype=np.uint8)

    # Count neighbors for each skeleton pixel
    neighbor_count = convolve(skeleton.astype(np.uint8), neighbor_kernel, mode='constant', cval=0)

    # Mask to only skeleton pixels
    neighbor_count = neighbor_count * skeleton

    # Classify nodes:
    # - Endpoints: exactly 1 neighbor (terminal points)
    # - Junctions: 3+ neighbors (branching points)
    # Note: degree 2 = normal path pixel, not a node
    endpoints = (neighbor_count == 1) & skeleton
    junctions = (neighbor_count >= 3) & skeleton

    # Draw nodes as filled circles
    # Create coordinate arrays for distance calculation
    y_coords, x_coords = np.ogrid[:organelle_mask.shape[0], :organelle_mask.shape[1]]

    # Draw junctions (red circles)
    junction_y, junction_x = np.where(junctions)
    for jy, jx in zip(junction_y, junction_x):
        dist_sq = (y_coords - jy)**2 + (x_coords - jx)**2
        circle_mask = dist_sq <= node_radius**2
        overlay[circle_mask] = junction_color

    # Draw endpoints (green circles) - draw after junctions so they're visible
    endpoint_y, endpoint_x = np.where(endpoints)
    for ey, ex in zip(endpoint_y, endpoint_x):
        dist_sq = (y_coords - ey)**2 + (x_coords - ex)**2
        circle_mask = dist_sq <= node_radius**2
        overlay[circle_mask] = endpoint_color

    return overlay


def blend_intensity_heatmap(
    intensity: np.ndarray,
    heatmap_rgba: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Blend grayscale intensity image with RGBA heatmap overlay.

    Parameters
    ----------
    intensity : np.ndarray
        2D grayscale intensity image.
    heatmap_rgba : np.ndarray
        RGBA heatmap from create_heatmap_rgba() or create_skeleton_heatmap_rgba().
    alpha : float
        Blend factor for heatmap (0 = all intensity, 1 = all heatmap).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) with blended result.
    """
    # Normalize intensity to [0, 1]
    intensity = intensity.astype(np.float32)
    if intensity.max() > intensity.min():
        vmin_int = np.percentile(intensity, 1)
        vmax_int = np.percentile(intensity, 99)
        intensity_norm = np.clip(intensity, vmin_int, vmax_int)
        intensity_norm = (intensity_norm - vmin_int) / (vmax_int - vmin_int + 1e-8)
    else:
        intensity_norm = np.zeros_like(intensity)

    # Create RGB from grayscale
    intensity_rgb = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=-1)

    # Blend with heatmap where heatmap has content (alpha > 0)
    mask_nonzero = heatmap_rgba[..., 3] > 0
    blended = intensity_rgb.copy()
    blended[mask_nonzero] = (
        (1 - alpha) * intensity_rgb[mask_nonzero]
        + alpha * heatmap_rgba[mask_nonzero, :3]
    )

    return blended


def generate_validation_canvas(
    viz_data_list: list,
    organelle_name: str,
    feature_name: str,
    output_path: Path,
    n_cols: int = 3,
    cmap: str = "viridis",
    is_network_organelle: bool = False,
):
    """
    Generate a single canvas PNG with multiple cells showing feature heatmaps.

    For non-network organelles:
        Each cell shows: [original intensity | feature heatmap overlay]

    For network organelles:
        Each cell shows: [original intensity | feature heatmap | skeleton topology]
        Where skeleton topology shows edges (white), junctions (red), endpoints (green)

    Shared colorbar on the right side.

    Parameters
    ----------
    viz_data_list : list
        List of viz_data dicts from sampled cells.
    organelle_name : str
        Organelle to visualize (e.g., "mitochondria", "nuclei").
    feature_name : str
        Feature to map to colors (e.g., "area", "perimeter", "branch_type").
    output_path : Path
        Output path for the PNG.
    n_cols : int
        Number of columns in the grid.
    cmap : str
        Colormap name.
    is_network_organelle : bool
        Whether this organelle is a network organelle (determined upstream).
        Network features are only computed for network organelles.
        If True, adds a 3rd panel showing skeleton topology.
    """
    # Determine feature type
    is_branch_level_feature = feature_name in BRANCH_LEVEL_FEATURES
    is_per_object_network_feature = feature_name in PER_OBJECT_NETWORK_FEATURES
    is_localization_feature = feature_name in LOCALIZATION_FEATURES
    is_network_feature = is_branch_level_feature or is_per_object_network_feature

    # Only show skeleton topology panel for network features on network organelles
    show_skeleton_topology = is_network_organelle and is_network_feature

    # Filter to cells that have this organelle's label array
    if is_branch_level_feature:
        # For branch-level features (e.g., branch_type), need pre-computed per_branch_features
        valid_viz_data = [
            vd for vd in viz_data_list
            if organelle_name in vd.get("organelle_labels", {})
            and organelle_name in vd.get("per_branch_features", {})
        ]
    else:
        # For object features (morphological or per-object network), just need label array
        valid_viz_data = [
            vd for vd in viz_data_list
            if organelle_name in vd.get("organelle_labels", {})
        ]

    if not valid_viz_data:
        print(f"    No cells with {organelle_name} labels found, skipping.")
        return

    # Determine channel index once for this organelle (same for all cells)
    # Use pre-computed mapping from feature extraction
    first_viz = valid_viz_data[0]
    channel_names = first_viz.get("channel_names", [])
    organelle_channel_indices = first_viz.get("organelle_channel_indices", {})

    try:
        channel_idx = _get_channel_index_for_organelle(organelle_name, channel_names, organelle_channel_indices)
        channel_label = channel_names[channel_idx] if channel_idx < len(channel_names) else f"ch{channel_idx}"
    except ValueError as e:
        print(f"    Warning: {e}")
        print(f"    Skipping {organelle_name} visualization.")
        return

    n_cells = len(valid_viz_data)
    n_rows = int(np.ceil(n_cells / n_cols))

    # Compute global min/max for consistent colorbar across all cells
    # For non-branch features, compute from label arrays on-demand
    all_feature_values = []
    computed_features_cache = {}  # Cache computed features for reuse in rendering

    for i, viz_data in enumerate(valid_viz_data):
        if is_branch_level_feature:
            df = viz_data.get("per_branch_features", {}).get(organelle_name)
        elif is_per_object_network_feature:
            # Only compute per-object network features if this is a network organelle
            # This decision is made upstream in feature_extraction.py
            if is_network_organelle:
                label_array = viz_data["organelle_labels"][organelle_name]
                df = _compute_per_object_network_features(label_array)
                computed_features_cache[i] = df
            else:
                df = pd.DataFrame()  # Non-network organelles don't have network features
        elif is_localization_feature:
            # Localization features - compute on-demand with cell/nuclear masks
            label_array = viz_data["organelle_labels"][organelle_name]
            cell_mask = viz_data.get("cell_mask")
            if cell_mask is not None and cell_mask.ndim == 3:
                cell_mask = cell_mask[0]  # (1, H, W) -> (H, W)
            # Get nuclear mask from organelle_labels if available
            nuclear_mask = None
            for nuc_key in ("nuclei", "nuclear_seg"):
                if nuc_key in viz_data.get("organelle_labels", {}):
                    nuc_arr = viz_data["organelle_labels"][nuc_key]
                    nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                    break
            df = _compute_localization_features_for_viz(label_array, cell_mask, nuclear_mask)
            computed_features_cache[i] = df
        else:
            # Morphological features - try pre-computed first, otherwise compute on-demand
            df = viz_data.get("per_object_features", {}).get(organelle_name)
            if df is None or df.empty:
                label_array = viz_data["organelle_labels"][organelle_name]
                df = _compute_basic_features(label_array)
                computed_features_cache[i] = df

        if df is not None and not df.empty and feature_name in df.columns:
            all_feature_values.extend(df[feature_name].dropna().values)

    if not all_feature_values:
        print(f"    No {feature_name} values found for {organelle_name}, skipping.")
        return

    # For branch_type, use discrete values 0, 1, 2
    if feature_name == "branch_type":
        vmin, vmax = 0, 2
    else:
        vmin, vmax = np.percentile(all_feature_values, [5, 95])

    # Create figure with grid layout
    # Network features on network organelles: 3 panels per cell (intensity | heatmap | skeleton topology)
    # Morphological features: 2 panels per cell (intensity | heatmap)
    panels_per_cell = 3 if show_skeleton_topology else 2
    fig = plt.figure(figsize=(4 * n_cols * panels_per_cell + 1, 4 * n_rows), dpi=300)
    gs = GridSpec(n_rows, n_cols * panels_per_cell + 1, width_ratios=[1] * (n_cols * panels_per_cell) + [0.15], wspace=0.05, hspace=0.15)

    for idx, viz_data in enumerate(valid_viz_data):
        row = idx // n_cols
        col = idx % n_cols

        organelle_labels = viz_data["organelle_labels"][organelle_name]
        metadata = viz_data["metadata"]

        # Use correct intensity and cell mask for this organelle:
        # CellPainting organelles (cp*) use cp_intensity and cp_cell_seg (different spatial location!)
        # Standard organelles use regular intensity and cell_seg
        is_cp_organelle = organelle_name.lower().startswith("cp")
        if is_cp_organelle and viz_data.get("cp_intensity") is not None:
            intensity = viz_data["cp_intensity"]
            cell_mask = viz_data.get("cp_cell_mask")
            cell_mask_name = "cp_cell_seg"
            viz_offset = viz_data.get("cp_viz_offset", (0, 0))
        else:
            intensity = viz_data["intensity"]
            cell_mask = viz_data["cell_mask"]
            cell_mask_name = "cell_seg"
            viz_offset = viz_data.get("viz_offset", (0, 0))

        # Get the appropriate features dataframe
        if is_branch_level_feature:
            features_df = viz_data.get("per_branch_features", {}).get(organelle_name, pd.DataFrame())
        elif is_per_object_network_feature or is_localization_feature:
            # Use cached computed features (network or localization)
            features_df = computed_features_cache.get(idx, pd.DataFrame())
        else:
            # Use cached computed morphological features, or try pre-computed
            features_df = computed_features_cache.get(idx)
            if features_df is None:
                features_df = viz_data.get("per_object_features", {}).get(organelle_name, pd.DataFrame())

        # Select the appropriate channel for this organelle (channel_idx determined once above)
        if intensity.ndim == 3:
            # intensity shape is (C, H, W)
            if channel_idx < intensity.shape[0]:
                intensity_2d = intensity[channel_idx]
            else:
                print(f"    Warning: channel_idx {channel_idx} out of bounds for intensity shape {intensity.shape}")
                continue
        else:
            intensity_2d = intensity

        # Expand organelle_labels and cell_mask to match expanded intensity size
        # Masks are smaller (original crop), intensity is larger (with padding for context)
        expanded_labels = _expand_mask_to_intensity(organelle_labels, intensity_2d.shape, viz_offset)
        expanded_cell_mask = _expand_mask_to_intensity(cell_mask, intensity_2d.shape, viz_offset)

        # Normalize intensity for display
        if intensity_2d.max() > intensity_2d.min():
            vmin_int = np.percentile(intensity_2d, 1)
            vmax_int = np.percentile(intensity_2d, 99)
            intensity_norm = np.clip(intensity_2d, vmin_int, vmax_int)
            intensity_norm = (intensity_norm - vmin_int) / (vmax_int - vmin_int + 1e-8)
        else:
            intensity_norm = np.zeros_like(intensity_2d, dtype=np.float32)

        # Prepare cell boundary overlay (using expanded cell mask)
        boundary_overlay = None
        if expanded_cell_mask is not None:
            boundary = find_boundaries(expanded_cell_mask > 0, mode="inner")
            boundary_overlay = np.zeros((*expanded_cell_mask.shape, 4))
            boundary_overlay[boundary] = [0, 1, 1, 1]  # Cyan

        # Panel 1: Original intensity with cell boundary
        ax_orig = fig.add_subplot(gs[row, col * panels_per_cell])
        ax_orig.imshow(intensity_norm, cmap="gray", vmin=0, vmax=1)
        if boundary_overlay is not None:
            ax_orig.imshow(boundary_overlay)
        ax_orig.axis("off")
        ax_orig.set_title("Intensity", fontsize=7)

        # Panel 2: Heatmap overlay
        ax_heat = fig.add_subplot(gs[row, col * panels_per_cell + 1])

        # Use appropriate heatmap function based on feature type (with expanded labels)
        if is_branch_level_feature:
            # Branch-level features use skeleton visualization
            heatmap_rgba = create_skeleton_heatmap_rgba(
                expanded_labels, features_df, feature_name, cmap, vmin, vmax
            )
        else:
            # Both morphological and per-object network features use object-level heatmap
            heatmap_rgba = create_heatmap_rgba(
                expanded_labels, features_df, feature_name, cmap, vmin, vmax
            )

        blended = blend_intensity_heatmap(intensity_2d, heatmap_rgba, alpha=0.7)
        ax_heat.imshow(blended)
        if boundary_overlay is not None:
            ax_heat.imshow(boundary_overlay)
        ax_heat.axis("off")
        ax_heat.set_title(f"{feature_name}", fontsize=7)

        # Panel 3 (network features only): Skeleton topology
        if show_skeleton_topology:
            ax_topo = fig.add_subplot(gs[row, col * panels_per_cell + 2])
            topology_rgba = create_skeleton_topology_rgba(expanded_labels)
            blended_topo = blend_intensity_heatmap(intensity_2d, topology_rgba, alpha=0.9)
            ax_topo.imshow(blended_topo)
            if boundary_overlay is not None:
                ax_topo.imshow(boundary_overlay)
            ax_topo.axis("off")

            # Add metadata to the last panel
            well = metadata.get("well", "?")
            gene = metadata.get("gene_name", "?")
            ax_topo.set_title(f"Skeleton\n{well} | {gene}", fontsize=6)
        else:
            # Add metadata to the heatmap panel for non-network organelles
            well = metadata.get("well", "?")
            gene = metadata.get("gene_name", "?")
            ax_heat.set_title(f"{feature_name}\n{well} | {gene}", fontsize=6)

    # Shared colorbar
    ax_cbar = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)

    # Custom colorbar label for branch_type
    if feature_name == "branch_type":
        cbar.set_label("branch_type\n(0=end-end, 1=end-junc, 2=junc-junc)", fontsize=9)
        cbar.set_ticks([0, 1, 2])
    else:
        cbar.set_label(feature_name, fontsize=10)

    # Add legend for skeleton topology if showing it
    if show_skeleton_topology:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='white', linewidth=2, label='Edges'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Junctions', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Endpoints', linestyle='None'),
        ]
        fig.legend(handles=legend_elements, loc='lower right', fontsize=7, frameon=True, facecolor='lightgray', bbox_to_anchor=(0.99, 0.01))

    # Main title (include channel name and cell mask for clarity)
    # Determine which cell mask is used for this organelle
    if organelle_name.lower().startswith("cp"):
        mask_label = "cp_cell_seg"
    else:
        mask_label = "cell_seg"
    fig.suptitle(f"{organelle_name} - {feature_name} [{channel_label}] (mask: {mask_label})", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_validation_visualizations(
    viz_data_list: list,
    features_to_plot: list,
    output_dir: Path,
    network_organelles: list = None,
    visualization_organelles: list = None,
):
    """
    Generate feature validation heatmap canvases for sampled cells.

    Creates one PNG per organelle × feature combination, each showing
    a grid of cells with side-by-side intensity and heatmap overlay.

    Parameters
    ----------
    viz_data_list : list
        List of viz_data dicts from sampled cells.
    features_to_plot : list
        Features to visualize (e.g., ["area", "perimeter", "branch_type"]).
    output_dir : Path
        Output directory for feature extraction results.
    network_organelles : list, optional
        List of organelle names that have network analysis. Network features
        (num_branches, num_nodes, etc.) will only be computed for these organelles.
        If None, network features will be skipped for all organelles.
    visualization_organelles : list, optional
        If provided, only generate visualizations for these specific organelles.
        Useful for simplifying output when many organelles are present (e.g., CP experiments).
        If None, visualizes all organelles found in viz_data_list.
    """
    import random
    
    network_organelles = network_organelles or []
    if not viz_data_list:
        print("No visualization data collected, skipping validation visualizations.")
        return

    validation_dir = output_dir / "feature_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Get all organelles present in viz data
    all_organelles = set()
    for viz_data in viz_data_list:
        all_organelles.update(viz_data.get("organelle_labels", {}).keys())

    # Filter to specified organelles if provided
    if visualization_organelles:
        organelles = set(visualization_organelles) & all_organelles
        skipped = set(visualization_organelles) - all_organelles
        if skipped:
            print(f"  Note: Requested organelles not found in data: {skipped}")
    else:
        organelles = all_organelles

    # RANDOMLY SELECT 3 ORGANELLES to sample feature space better
    # This prevents overwhelming output while maintaining good coverage
    organelles_list = sorted(list(organelles))
    if len(organelles_list) > 3:
        random.seed(42)  # Reproducible sampling
        organelles_list = random.sample(organelles_list, 3)
        print(f"  Randomly selected 3 organelles for visualization: {organelles_list}")
    else:
        print(f"  Visualizing all {len(organelles_list)} organelles (≤3 available)")

    # RANDOMLY SELECT 3 FEATURES to sample feature space better
    # Ensure we get a mix of feature types
    features_to_plot_list = list(features_to_plot)
    if len(features_to_plot_list) > 3:
        random.seed(42)  # Reproducible sampling
        features_to_plot_list = random.sample(features_to_plot_list, 3)
        print(f"  Randomly selected 3 features for visualization: {features_to_plot_list}")
    else:
        print(f"  Visualizing all {len(features_to_plot_list)} features (≤3 available)")

    print(f"Generating validation visualizations for {len(organelles_list)} organelles × {len(features_to_plot_list)} features...")

    # All network-related features (both branch-level and per-object)
    all_network_features = BRANCH_LEVEL_FEATURES | PER_OBJECT_NETWORK_FEATURES

    # Generate one canvas per organelle × feature
    for organelle in organelles_list:
        # Check if this organelle is a network organelle (determined upstream)
        is_network_organelle = organelle in network_organelles

        # Check if this organelle has branch/network data
        # For branch-level features, need pre-computed per_branch_features
        has_branch_data = any(
            organelle in vd.get("per_branch_features", {})
            for vd in viz_data_list
        )

        for feature in features_to_plot_list:
            # Skip all network features for non-network organelles
            # This decision is made upstream in feature_extraction.py
            if feature in all_network_features and not is_network_organelle:
                continue

            # Skip branch-level features for organelles without pre-computed branch data
            if feature in BRANCH_LEVEL_FEATURES and not has_branch_data:
                continue

            # Skip localization features for nuclei (doesn't make sense)
            if feature in LOCALIZATION_FEATURES and organelle in ("nuclei", "nuclear_seg"):
                continue

            output_path = validation_dir / f"{organelle}_{feature}.png"
            # Only show skeleton topology for network features on network organelles
            is_network_feature = feature in all_network_features
            show_skeleton = is_network_organelle and is_network_feature
            print(f"  -> {output_path.name}" + (" (with skeleton topology)" if show_skeleton else ""))
            generate_validation_canvas(
                viz_data_list=viz_data_list,
                organelle_name=organelle,
                feature_name=feature,
                output_path=output_path,
                is_network_organelle=is_network_organelle,
            )

    print(f"Validation visualizations saved to {validation_dir}")


def collect_validation_viz_data(
    morphology_path: Path,
    cells_df: pd.DataFrame,
    validation_indices: set,
    available_labels: dict,
    channel_names: list,
    organelle_channel_indices: dict,
    initial_yx_patch_size: tuple = (300, 300),
    viz_padding: int = 50,
    organelles_to_load: list = None,
) -> list:
    """
    Collect visualization data for validation cells.

    This function extracts the intensity images, masks, and organelle labels
    for a sample of cells to be used in feature validation visualizations.

    Parameters
    ----------
    morphology_path : Path
        Path to the zarr v3 store.
    cells_df : pd.DataFrame
        DataFrame with cell metadata including well, bbox, etc.
    validation_indices : set
        Set of DataFrame indices for cells to collect visualization data for.
    available_labels : dict
        Mapping of internal organelle names to zarr label names.
    channel_names : list
        List of channel names from the zarr store.
    organelle_channel_indices : dict
        Mapping of organelle name to channel index for visualization.
    initial_yx_patch_size : tuple
        Size of patches for cell crops.
    viz_padding : int
        Padding around cell bbox for visualization context. Default 50px.
    organelles_to_load : list, optional
        List of specific organelle names to load. If None, loads all organelles.
        Use this to speed up loading when you only need specific organelles.

    Returns
    -------
    list
        List of viz_data dicts for each validation cell.
    """
    from iohub import open_ome_zarr
    from cyclops_utils.data.bbox_utils import BaseDataset

    if not validation_indices:
        return []

    viz_data_list = []
    print(f"Collecting visualization data for {len(validation_indices)} validation cells...")

    pheno_store = open_ome_zarr(morphology_path, mode="r")
    stores = {"pheno_assembled_v3": pheno_store}

    # Create BaseDataset for validation cells only
    validation_cells_df = cells_df.loc[list(validation_indices)].reset_index(drop=True)
    base_dataset = BaseDataset(
        stores=stores,
        labels_df=validation_cells_df,
        initial_yx_patch_size=initial_yx_patch_size,
        final_yx_patch_size=initial_yx_patch_size,
        out_channels="all",
        mask_cell=False,
        use_original_crop_size=False,
    )

    # Debug counters for skipped cells
    skipped_empty_mask = 0
    skipped_no_well = 0
    skipped_no_labels_group = 0
    skipped_no_organelle_labels = 0

    for i in range(len(validation_cells_df)):
        try:
            batch = base_dataset[i]
            data = batch["data"].numpy() if hasattr(batch["data"], "numpy") else np.array(batch["data"])
            mask = batch["mask"].numpy() if hasattr(batch["mask"], "numpy") else np.array(batch["mask"])
            crop_info = batch["crop_info"]
            bbox = batch.get("bbox")
            well = crop_info.get("well")

            cell_specific_mask = mask[0].astype(np.uint8)
            if not np.any(cell_specific_mask):
                skipped_empty_mask += 1
                continue

            if not well or bbox is None:
                skipped_no_well += 1
                continue

            # Load organelle labels for this cell
            organelle_labels = {}

            position = pheno_store[well]
            y_min, x_min, y_max, x_max = bbox

            # Load expanded intensity for visualization (padding around cell)
            fov = position["0"]
            img_h, img_w = fov.shape[-2], fov.shape[-1]

            # Compute expanded bbox with bounds checking
            exp_y_min = max(0, y_min - viz_padding)
            exp_x_min = max(0, x_min - viz_padding)
            exp_y_max = min(img_h, y_max + viz_padding)
            exp_x_max = min(img_w, x_max + viz_padding)

            # Calculate offset of original crop within expanded region
            viz_offset_y = y_min - exp_y_min
            viz_offset_x = x_min - exp_x_min

            # Load expanded intensity region
            expanded_intensity = np.array(fov[0, :, 0, exp_y_min:exp_y_max, exp_x_min:exp_x_max])

            if "labels" not in position.zgroup:
                skipped_no_labels_group += 1
                continue

            labels_group = position.zgroup["labels"]

            # Handle CellPainting dual-bbox system
            cp_bbox = crop_info.get("cp_bbox")
            cp_seg_id = crop_info.get("cp_cell_seg_id")
            cp_cell_mask = None
            cp_intensity = None
            cp_viz_offset_y = 0
            cp_viz_offset_x = 0

            if cp_bbox is not None and "cp_cell_seg" in labels_group:
                try:
                    cp_y_min, cp_x_min, cp_y_max, cp_x_max = cp_bbox
                    cp_label_array = labels_group["cp_cell_seg"]["0"]

                    # Handle different array dimensions
                    if cp_label_array.ndim == 5:
                        cp_cell_mask_raw = np.array(cp_label_array[0, 0, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                    elif cp_label_array.ndim == 4:
                        cp_cell_mask_raw = np.array(cp_label_array[0, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                    elif cp_label_array.ndim == 3:
                        cp_cell_mask_raw = np.array(cp_label_array[0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                    else:
                        cp_cell_mask_raw = np.array(cp_label_array[cp_y_min:cp_y_max, cp_x_min:cp_x_max])

                    # Create binary mask for this specific CP cell
                    if cp_seg_id is not None:
                        cp_cell_mask = (cp_cell_mask_raw == cp_seg_id).astype(np.uint8)
                    else:
                        cp_cell_mask = (cp_cell_mask_raw > 0).astype(np.uint8)

                    # Load expanded intensity image at cp_bbox location
                    try:
                        cp_exp_y_min = max(0, cp_y_min - viz_padding)
                        cp_exp_x_min = max(0, cp_x_min - viz_padding)
                        cp_exp_y_max = min(img_h, cp_y_max + viz_padding)
                        cp_exp_x_max = min(img_w, cp_x_max + viz_padding)

                        cp_viz_offset_y = cp_y_min - cp_exp_y_min
                        cp_viz_offset_x = cp_x_min - cp_exp_x_min

                        cp_intensity = np.array(fov[0, :, 0, cp_exp_y_min:cp_exp_y_max, cp_exp_x_min:cp_exp_x_max])
                    except Exception:
                        cp_intensity = None
                        cp_viz_offset_y = 0
                        cp_viz_offset_x = 0

                except Exception:
                    cp_cell_mask = None

            # Load organelle labels
            for internal_name, zarr_label_name in available_labels.items():
                if internal_name == "cell_mask":
                    continue
                if zarr_label_name not in labels_group:
                    continue
                
                # Filter to only requested organelles if specified
                if organelles_to_load is not None and internal_name not in organelles_to_load:
                    continue

                try:
                    label_array = labels_group[zarr_label_name]["0"]

                    # For CP organelles, use cp_bbox if available
                    is_cp_org = internal_name.lower().startswith("cp")
                    if is_cp_org and cp_bbox is not None:
                        crop_y_min, crop_x_min, crop_y_max, crop_x_max = cp_bbox
                    else:
                        crop_y_min, crop_x_min, crop_y_max, crop_x_max = y_min, x_min, y_max, x_max

                    # Handle different array dimensions
                    if label_array.ndim == 5:
                        label_crop = label_array[0, 0, 0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    elif label_array.ndim == 4:
                        label_crop = label_array[0, 0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    elif label_array.ndim == 3:
                        label_crop = label_array[0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    else:
                        label_crop = label_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                    label_crop = np.array(label_crop)

                    # Determine which cell mask to use for this organelle
                    if is_cp_org:
                        if cp_cell_mask is None:
                            continue
                        mask_to_use = cp_cell_mask
                        # Ensure shapes match
                        if label_crop.shape != mask_to_use.shape:
                            matched_mask = np.zeros(label_crop.shape, dtype=mask_to_use.dtype)
                            h = min(mask_to_use.shape[0], matched_mask.shape[0])
                            w = min(mask_to_use.shape[1], matched_mask.shape[1])
                            matched_mask[:h, :w] = mask_to_use[:h, :w]
                            mask_to_use = matched_mask
                    else:
                        mask_to_use = cell_specific_mask
                        # Match shape if needed
                        if label_crop.shape != cell_specific_mask.shape:
                            matched_crop = np.zeros(cell_specific_mask.shape, dtype=label_crop.dtype)
                            h = min(label_crop.shape[0], matched_crop.shape[0])
                            w = min(label_crop.shape[1], matched_crop.shape[1])
                            matched_crop[:h, :w] = label_crop[:h, :w]
                            label_crop = matched_crop

                    # Skip if empty before masking
                    if np.sum(label_crop > 0) == 0:
                        continue

                    # Mask to cell boundary
                    label_crop = label_crop * (mask_to_use > 0).astype(label_crop.dtype)

                    if np.sum(label_crop > 0) > 0:
                        organelle_labels[internal_name] = label_crop
                except Exception:
                    pass

            if organelle_labels:
                viz_data_list.append({
                    "intensity": expanded_intensity,
                    "cp_intensity": cp_intensity,
                    "cell_mask": mask,
                    "cp_cell_mask": cp_cell_mask,
                    "organelle_labels": organelle_labels,
                    "per_object_features": {},
                    "per_branch_features": {},
                    "metadata": crop_info,
                    "channel_names": channel_names,
                    "organelle_channel_indices": organelle_channel_indices,
                    "viz_offset": (viz_offset_y, viz_offset_x),
                    "cp_viz_offset": (cp_viz_offset_y, cp_viz_offset_x) if cp_intensity is not None else (0, 0),
                })
            else:
                skipped_no_organelle_labels += 1

        except Exception as e:
            print(f"  Error processing validation cell {i}: {e}")
            continue

    pheno_store.close()
    print(f"Collected visualization data for {len(viz_data_list)} cells.")
    if skipped_empty_mask or skipped_no_well or skipped_no_labels_group or skipped_no_organelle_labels:
        print(f"  Skipped cells: empty_mask={skipped_empty_mask}, no_well={skipped_no_well}, "
              f"no_labels_group={skipped_no_labels_group}, no_organelle_labels={skipped_no_organelle_labels}")

    return viz_data_list
