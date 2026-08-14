"""
Label Naming Utilities for Organelle Segmentation
==================================================

Organelle-specific label naming functions (get_label_name, get_output_label_name).

Shared functions (parse_channel_label, build_channel_metadata, determine_marker_type,
get_channel_type) live in cyclops_utils.data.naming.
"""

from .configs import SPECIAL_LABEL_MAP
from cyclops_utils.data.naming import parse_channel_label


def get_label_name(organelle: str, marker: str = None, structure_type: str = None) -> str:
    """
    Generate standardized label name from organelle, marker, and structure type.

    Format: {organelle}__{marker}_{structure_type}_seg (all lowercase, underscores normalized)
    Uses full organelle and marker names (not truncated) for clarity.
    If structure_type is None, uses original format without structure suffix.

    Args:
        organelle: Organelle type (e.g., "mitochondria", "ER", "lysosome")
        marker: Marker protein/dye (e.g., "TOMM20", "SEC61B", "LAMP1", "vs" for virtual staining)
        structure_type: "tubular" or "vesicular" (optional, for dual Frangi segmentation)

    Returns:
        Label name like "mitochondria_tomm20_seg" or "mitochondria_tomm20_tubular_seg"

    Examples:
        >>> get_label_name("mitochondria", "TOMM20")
        'mitochondria_tomm20_seg'
        >>> get_label_name("mitochondria", "TOMM20", "tubular")
        'mitochondria_tomm20_tubular_seg'
        >>> get_label_name("ER", "SEC61B", "vesicular")
        'er_sec61b_vesicular_seg'
        >>> get_label_name("nuclei", "vs")
        'nuclei_vs_seg'
        >>> get_label_name("Phase", "2D", "tubular")
        'phase_2d_tubular_seg'
    """
    # Normalize: lowercase and replace spaces/dashes with underscores
    org_part = organelle.lower().replace(" ", "_").replace("-", "_")
    marker_part = marker.lower().replace(" ", "_").replace("-", "_") if marker else ""

    # Build the label name based on what parts are present
    if marker_part and structure_type:
        return f"{org_part}_{marker_part}_{structure_type}_seg"
    elif marker_part:
        return f"{org_part}_{marker_part}_seg"
    elif structure_type:
        return f"{org_part}_{structure_type}_seg"
    else:
        return f"{org_part}_seg"


def get_output_label_name(
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
        >>> get_output_label_name("nuclei")
        'nucle_vs_seg'
        >>> get_output_label_name("nucleoli", "Phase2D")
        'nuclo_phase_seg'
        >>> get_output_label_name("Phase", "Phase2D")
        'phase_2d_seg'
        >>> get_output_label_name("Phase", "Phase2D", "tubular")
        'phase_2d_tubular_seg'
        >>> get_output_label_name("mitochondria, TOMM20")
        'mitoc_tomm20_seg'
        >>> get_output_label_name("mitochondria, TOMM20", None, "vesicular")
        'mitoc_tomm20_vesicular_seg'
        >>> get_output_label_name("ER, SEC61B")
        'er_sec61b_seg'
        >>> get_output_label_name("no label", "GFP")
        'no_label_gfp_seg'
        >>> get_output_label_name("no label", "GFP", "tubular")
        'no_label_gfp_tubular_seg'
        >>> get_output_label_name("no label", "mCherry")
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
        return get_label_name(organelle, marker, structure_type)

    # Fallback: check channel_name in SPECIAL_LABEL_MAP (e.g., "Phase2D" -> "phase_2d_seg")
    # This handles cases where organelle_name is generic but channel is specific
    if channel_name and channel_name in SPECIAL_LABEL_MAP:
        mapping = SPECIAL_LABEL_MAP[channel_name]
        organelle, marker = mapping
        return get_label_name(organelle, marker, structure_type)

    # Parse full label string like "mitochondria, TOMM20"
    organelle, marker = parse_channel_label(organelle_name)
    return get_label_name(organelle, marker, structure_type)
