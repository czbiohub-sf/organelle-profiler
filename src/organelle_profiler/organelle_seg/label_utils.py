"""
Label Utilities for Organelle Segmentation
===========================================

This module provides utilities for:
- Label naming conventions and standardization
- Coordinate and bounding box calculations
- Sigma conversion for Frangi filter
"""

import numpy as np
from pathlib import Path
import zarr

from .configs import SPECIAL_LABEL_MAP


# =============================================================================
# LABEL NAMING FUNCTIONS
# =============================================================================

def _parse_channel_label(label: str) -> tuple:
    """
    Parse organelle and marker from channel map label.

    The channel map label format is "{organelle}, {marker}" (e.g., "mitochondria, TOMM20").

    Args:
        label: Label string like "mitochondria, TOMM20" or "ER, SEC61B"

    Returns:
        Tuple of (organelle, marker). If no marker present, marker is None.
    """
    if ", " in label:
        parts = label.split(", ", 1)
        return parts[0].strip(), parts[1].strip()
    return label.strip(), None


def _get_label_name(organelle: str, marker: str = None, structure_type: str = None) -> str:
    """
    Generate standardized label name from organelle, marker, and structure type.

    Format: {organelle[:5]}_{marker[:6]}_{structure_type}_seg (all lowercase, no padding)
    If structure_type is None, uses original format without structure suffix.

    Args:
        organelle: Organelle type (e.g., "mitochondria", "ER", "lysosome")
        marker: Marker protein/dye (e.g., "TOMM20", "SEC61B", "LAMP1", "vs" for virtual staining)
        structure_type: "tubular" or "vesicular" (optional, for dual Frangi segmentation)

    Returns:
        Label name like "mitoc_tomm20_seg" or "mitoc_tomm20_tubular_seg"

    Examples:
        >>> _get_label_name("mitochondria", "TOMM20")
        'mitoc_tomm20_seg'
        >>> _get_label_name("mitochondria", "TOMM20", "tubular")
        'mitoc_tomm20_tubular_seg'
        >>> _get_label_name("ER", "SEC61B", "vesicular")
        'er_sec61b_vesicular_seg'
        >>> _get_label_name("nuclei", "vs")
        'nucle_vs_seg'
        >>> _get_label_name("Phase", "2D", "tubular")
        'phase_2d_tubular_seg'
    """
    org_part = organelle[:5].lower()
    marker_part = marker[:6].lower() if marker else ""

    # Build the label name based on what parts are present
    if marker_part and structure_type:
        return f"{org_part}_{marker_part}_{structure_type}_seg"
    elif marker_part:
        return f"{org_part}_{marker_part}_seg"
    elif structure_type:
        return f"{org_part}_{structure_type}_seg"
    else:
        return f"{org_part}_seg"


def _get_output_label_name(
    organelle_name: str, channel_name: str = None, structure_type: str = None
) -> str:
    """
    Get the standardized output label name for an organelle entry.

    This handles:
    - Special zarr channel names (Phase2D, Focus3D, nuclei_prediction, etc.)
    - Auto-mapped organelles (nuclei, cell_membrane, nucleoli)
    - Full label strings from channel map like "mitochondria, TOMM20"
    - "no label" entries - autofluorescence channels named as no_label_{channel}_seg
    - Structure type suffix for dual Frangi segmentation (tubular/vesicular)

    Args:
        organelle_name: Either a full label string, zarr channel name, or special case
        channel_name: The actual zarr channel name (e.g., "Phase2D", "mCherry", "GFP")
        structure_type: "tubular" or "vesicular" (optional, for dual Frangi segmentation)

    Returns:
        Standardized label name like "mitoc_tomm20_seg" or "nucle_vs_seg"
        With structure_type: "mitoc_tomm20_tubular_seg" or "phase_2d_vesicular_seg"
        For "no label" entries: "no_label_gfp_seg" or "no_label_gfp_tubular_seg"

    Examples:
        >>> _get_output_label_name("nuclei")
        'nucle_vs_seg'
        >>> _get_output_label_name("nucleoli", "Phase2D")  # organelle_name takes priority
        'nuclo_phase_seg'
        >>> _get_output_label_name("Phase", "Phase2D")
        'phase_2d_seg'
        >>> _get_output_label_name("Phase", "Phase2D", "tubular")
        'phase_2d_tubular_seg'
        >>> _get_output_label_name("mitochondria, TOMM20")
        'mitoc_tomm20_seg'
        >>> _get_output_label_name("mitochondria, TOMM20", None, "vesicular")
        'mitoc_tomm20_vesicular_seg'
        >>> _get_output_label_name("ER, SEC61B")
        'er_sec61b_seg'
        >>> _get_output_label_name("no label", "GFP")
        'no_label_gfp_seg'
        >>> _get_output_label_name("no label", "GFP", "tubular")
        'no_label_gfp_tubular_seg'
        >>> _get_output_label_name("no label", "mCherry")
        'no_label_mcherr_seg'
    """
    # Handle "no label" entries - autofluorescence channels
    if organelle_name == "no label":
        if channel_name:
            ch_part = channel_name[:6].lower()
            if structure_type:
                return f"no_label_{ch_part}_{structure_type}_seg"
            return f"no_label_{ch_part}_seg"
        else:
            if structure_type:
                return f"no_label_{structure_type}_seg"
            return "no_label_seg"

    # Check organelle_name FIRST in special cases (auto-mapped organelles like "nuclei", "cell_membrane", "nucleoli")
    # This takes priority over channel_name to ensure nucleoli gets "nuclo_phase_seg" not "phase_2d_seg"
    if organelle_name in SPECIAL_LABEL_MAP:
        mapping = SPECIAL_LABEL_MAP[organelle_name]
        organelle, marker = mapping
        return _get_label_name(organelle, marker, structure_type)

    # Fallback: check channel_name in SPECIAL_LABEL_MAP (e.g., "Phase2D" -> "phase_2d_seg")
    # This handles cases where organelle_name is generic but channel is specific
    if channel_name and channel_name in SPECIAL_LABEL_MAP:
        mapping = SPECIAL_LABEL_MAP[channel_name]
        organelle, marker = mapping
        return _get_label_name(organelle, marker, structure_type)

    # Parse full label string like "mitochondria, TOMM20"
    organelle, marker = _parse_channel_label(organelle_name)
    return _get_label_name(organelle, marker, structure_type)


# =============================================================================
# COORDINATE AND BOUNDING BOX UTILITIES
# =============================================================================

def _calculate_center_crop_bbox(full_shape: tuple, crop_fraction: float) -> tuple:
    """
    Calculate bounding box for center crop of an image.

    Args:
        full_shape: Full image shape (T, C, Z, Y, X) or (Y, X) or (Z, Y, X)
        crop_fraction: Fraction of image to keep (e.g., 0.01 for 1%, 0.1 for 10%)

    Returns:
        Tuple of (y_start, y_end, x_start, x_end) for the center crop region
    """
    if len(full_shape) == 5:  # (T, C, Z, Y, X)
        height, width = full_shape[3], full_shape[4]
    elif len(full_shape) == 3:  # (Z, Y, X)
        height, width = full_shape[1], full_shape[2]
    elif len(full_shape) == 2:  # (Y, X)
        height, width = full_shape[0], full_shape[1]
    else:
        raise ValueError(f"Unexpected shape: {full_shape}")

    # Calculate crop dimensions (linear scale, not area)
    # For 1% area, we want sqrt(0.01) = 0.1 = 10% linear
    linear_fraction = np.sqrt(crop_fraction)
    crop_height = int(height * linear_fraction)
    crop_width = int(width * linear_fraction)

    # Ensure minimum size
    crop_height = max(crop_height, 512)
    crop_width = max(crop_width, 512)

    # Calculate center
    center_y = height // 2
    center_x = width // 2

    y_start = center_y - crop_height // 2
    y_end = y_start + crop_height
    x_start = center_x - crop_width // 2
    x_end = x_start + crop_width

    # Clamp to image bounds
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(height, y_end)
    x_end = min(width, x_end)

    return (y_start, y_end, x_start, x_end)


def _bbox(im, xp):
    """
    Find the bounding box of a binary image.

    Args:
        im: Binary image (2D or 3D)
        xp: Array module (numpy or cupy)

    Returns:
        For 2D: (ymin, ymax, xmin, xmax) or (None, None, None, None) if empty
        For 3D: (zmin, zmax, ymin, ymax, xmin, xmax) or tuple of Nones if empty
    """
    if im.ndim == 3:
        coords = xp.where(im)
        if len(coords[0]) == 0:
            return None, None, None, None, None, None
        zmin, zmax = coords[0].min(), coords[0].max()
        ymin, ymax = coords[1].min(), coords[1].max()
        xmin, xmax = coords[2].min(), coords[2].max()
        return zmin, zmax, ymin, ymax, xmin, xmax
    else:  # 2D
        coords = xp.where(im)
        if len(coords[0]) == 0:
            return None, None, None, None
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()
        return ymin, ymax, xmin, xmax


def _get_position_shape(zarr_path: Path, pos_path: str) -> tuple:
    """Get the shape of the image array at a position in a v3 zarr store."""
    store = zarr.open(str(zarr_path), mode="r")
    return store[pos_path]["0"].shape
