"""
Metadata Building for Organelle Segmentation
=============================================

This module provides functions for building comprehensive metadata
dictionaries for organelle segmentation labels and vesselness maps.

Metadata includes:
- Source channel information
- Biological annotations (organelle, marker, marker type)
- Segmentation/processing method details
- CLAHE preprocessing parameters
- Detection parameters (Frangi or blob)
- Postprocessing parameters
"""

import zarr
from iohub import open_ome_zarr

from cyclops_utils.data.experiment import OpsDataset

from cyclops_utils.data.naming import parse_channel_label
from .naming import get_output_label_name
from .configs import (
    DEFAULT_METHODS,
    SEGMENTATION_CONFIGS,
    CLAHE_CONFIGS,
    STRUCTURE_TYPES,
    get_segmentation_config,
    get_clahe_config,
    get_channel_type,
    load_experiment_configs,
)
from cyclops_utils.io.zarr_labels import _check_label_has_data


def get_channel_index(channel_names: list[str], channel: str) -> int:
    """
    Safely get channel index, returning -1 if not found.

    Args:
        channel_names: List of channel names
        channel: Channel to find

    Returns:
        Channel index, or -1 if not found

    Example:
        >>> get_channel_index(["GFP", "mCherry", "Phase2D"], "mCherry")
        1
        >>> get_channel_index(["GFP", "mCherry"], "DAPI")
        -1
    """
    try:
        return channel_names.index(channel)
    except ValueError:
        return -1


def _build_description(organelle: str, marker: str, channel_name: str, segmenter_type: str) -> str:
    """Build a human-readable description for the segmentation."""
    if organelle and marker:
        return f"{organelle.capitalize()} segmentation from {marker} ({channel_name}) using {segmenter_type}"
    elif organelle:
        return f"{organelle.capitalize()} segmentation from {channel_name} using {segmenter_type}"
    elif channel_name:
        return f"Segmentation from {channel_name} using {segmenter_type}"
    else:
        return f"Segmentation using {segmenter_type}"


def _build_segmentation_metadata(
    label_name: str,
    organelle_name: str,
    channel_name: str,
    channel_label: str,
    channel_index: int,
    segmenter_type: str,
    channel_names: list,
    structure_type: str = None,
    clahe_params: dict = None,
    detection_params: dict = None,
    postprocess_params: dict = None,
) -> dict:
    """
    Build comprehensive metadata for an organelle segmentation label.

    Args:
        label_name: Standardized label name (e.g., "mitoc_tomm20_seg")
        organelle_name: Original organelle key (e.g., "nuclei", "mitochondria, TOMM20")
        channel_name: Source channel name (e.g., "GFP", "mCherry", "nuclei_prediction")
        channel_label: Full label from ops_channel_maps.yaml (e.g., "mitochondria, TOMM20")
        channel_index: Index of source channel in zarr store
        segmenter_type: Segmentation method used ("frangi", "blob")
        channel_names: List of all channel names in the zarr store
        structure_type: Structure type ("tubular", "vesicular", "vesicular_dark", "nucleoli")
        clahe_params: CLAHE preprocessing parameters (clip_limit, kernel_size, post_smoothing)
        detection_params: Frangi or blob detection parameters
        postprocess_params: Postprocessing parameters (opening, closing, min_object_size, etc.)

    Returns:
        Dictionary with comprehensive metadata
    """
    # Parse organelle and marker from channel_label (YAML label like "mitochondria, TOMM20")
    organelle_type, marker = parse_channel_label(channel_label) if channel_label else (None, None)

    # Cell Painting channels: parse organelle and marker from channel_name format
    # Format: CP{N}_{organelle}_{marker} (e.g., "CP1_plasma_membrane_WGA")
    # This takes priority because channel_label for CP channels is just the channel name
    if channel_name:
        parts = channel_name.split("_")
        if len(parts) >= 3 and parts[0].upper().startswith("CP"):
            organelle_type = "_".join(parts[1:-1])  # e.g., "plasma_membrane"
            marker = parts[-1]  # e.g., "WGA"

    # If still no organelle, try to infer from label_name prefix
    if organelle_type is None and label_name:
        label_prefix = label_name.split("_")[0].lower()
        ORGANELLE_PREFIX_MAP = {
            "nuclo": "nucleoli",
            "nucleoli": "nucleoli",
            "nucle": "nuclei",
            "mito": "mitochondria",
            "er": "endoplasmic reticulum",
            "lyso": "lysosome",
            "golgi": "golgi",
            "peroxi": "peroxisome",
            "lipid": "lipid droplet",
        }
        organelle_type = ORGANELLE_PREFIX_MAP.get(label_prefix)

    # Determine channel type category: "fluorescent", "label-free", or "virtual_stain"
    channel_type = None
    channel_type_lower = channel_name.lower() if channel_name else ""
    if "prediction" in channel_type_lower:
        channel_type = "virtual_stain"
    elif "bf" in channel_type_lower or "phase" in channel_type_lower or "focus" in channel_type_lower:
        channel_type = "label-free"
    else:
        # GFP, mCherry, Cy5, CP channels, etc. are all fluorescent
        channel_type = "fluorescent"

    # Handle "no label" channels: fluorescent channels with no specific organelle target
    marker_type = None
    if organelle_type and organelle_type.lower().replace("_", " ") == "no label":
        marker = "no label"
        marker_type = "autofluorescence"

    # Determine marker_type from marker name and channel context
    # Cell Painting channels (CP{N}_...) use fixed fluorescent antibodies/stains
    is_cell_painting = channel_name and channel_name.split("_")[0].upper().startswith("CP")
    if marker_type is None and marker:
        if is_cell_painting:
            marker_type = "fluorescent_antibody"
        else:
            marker_lower = marker.lower()
            if any(dye in marker_lower for dye in ["dye", "tracker", "live", "spy", "bodipy", "phrodo", "cellrox", "cellevent"]):
                marker_type = "live_cell_dye"
            elif any(emission in marker_lower for emission in ["chromalive", "emission"]):
                marker_type = "live_cell_dye"
            elif marker_lower in ["vs", "2d", "3d"]:
                marker_type = "virtual_stain"
            else:
                marker_type = "endogenous_tag"

    # For label-free channels (Phase2D, Focus3D, BF), set marker to "label-free"
    if marker is None and channel_type == "label-free":
        marker = "label-free"
        marker_type = "label_free"

    # Build the metadata dictionary
    metadata = {
        # Core identification
        "label_name": label_name,
        "annotation_type": "organelle_segmentation",
        "is_ome_label": True,

        # Source channel information
        "source_channel": {
            "name": channel_name,
            "index": channel_index,
            "type": channel_type,
            "all_channels": channel_names,
        },

        # Biological annotation
        "biological_annotation": {
            "organelle": organelle_type,
            "marker": marker,
            "marker_type": marker_type,
            "full_label": channel_label,
        },

        # Segmentation method and structure type
        "segmentation": {
            "method": segmenter_type,
            "version": "v3.0-position-based",
            "structure_type": structure_type,
        },

        # Human-readable description
        "description": _build_description(organelle_type, marker, channel_name, segmenter_type),
    }

    # Add mask info for nucleoli segmentation (uses nuclear_seg as mask)
    if organelle_type == "nucleoli":
        metadata["mask"] = {
            "name": "nuclear_seg",
            "description": "Nucleoli segmented within nuclei using nuclear_seg mask",
        }
        # Update description to mention mask
        metadata["description"] = f"Nucleoli segmentation from {channel_name} within nuclear_seg mask using {segmenter_type}"

    # Add CLAHE preprocessing parameters if provided
    if clahe_params:
        metadata["preprocessing"] = {
            "clahe": clahe_params,
        }

    # Add detection parameters (Frangi or blob)
    if detection_params:
        metadata["detection_params"] = detection_params

    # Add postprocessing parameters if provided
    if postprocess_params:
        metadata["postprocessing"] = postprocess_params

    return metadata


def _determine_processing_params(
    organelle_key: str,
    source_channel: str,
    structure_type: str,
    ch_info: dict,
    frangi_params: dict = None,
    clahe_params: dict = None,
    post_clahe_smoothing_sigma: float = None,
) -> tuple:
    """
    Determine the CLAHE and detection parameters that will be used for segmentation.

    This mirrors the logic in segment_organelles_slurm to compute params before metadata is built.

    Returns:
        Tuple of (detection_params, clahe_metadata, actual_method)
        - detection_params: Dict of Frangi or blob parameters
        - clahe_metadata: Dict with clip_limit, kernel_size, post_smoothing
        - actual_method: String describing the actual method used ("frangi", "blob")
    """
    nucleoli_method = ch_info.get("nucleoli_method")
    is_nucleoli = organelle_key.startswith("nucleoli")

    if is_nucleoli:
        # Nucleoli segmentation - use nucleoli_method to determine blob vs frangi
        actual_method = nucleoli_method  # "frangi" or "blob"
        if frangi_params is not None:
            detection_params = frangi_params
        else:
            # Get nucleoli config based on method
            detection_params = get_segmentation_config("nucleoli", source_channel, actual_method)

        # Nucleoli always use nucleoli CLAHE config
        clahe_config = get_clahe_config(is_nucleoli=True)
        clahe_metadata = {
            "clip_limit": clahe_config["clip_limit"],
            "kernel_size": clahe_config["kernel_size"],
            "post_smoothing": clahe_config["post_smoothing"],
        }

    else:
        # General case (not nucleoli) - determine method from DEFAULT_METHODS
        channel_type = get_channel_type(source_channel)
        is_vesicular = structure_type in ("vesicular", "vesicular_dark")

        # Get default method for this structure+channel combination
        default_method = DEFAULT_METHODS.get((structure_type or "tubular", channel_type), "frangi")
        actual_method = default_method

        # Determine detection params
        if frangi_params is not None:
            detection_params = frangi_params
        elif structure_type is not None:
            detection_params = get_segmentation_config(structure_type, source_channel, actual_method)
        else:
            # Fallback for legacy code without structure_type
            detection_params = {
                "min_radius_um": 0.1,
                "max_radius_um": 1.5,
                "alpha": 4.0,
                "beta": 0.5,
            }

        # Determine CLAHE params
        if clahe_params is not None:
            clahe_metadata = {
                "clip_limit": clahe_params.get("clip_limit"),
                "kernel_size": clahe_params.get("kernel_size"),
                "post_smoothing": post_clahe_smoothing_sigma if post_clahe_smoothing_sigma is not None else 0.0,
            }
        elif structure_type is not None:
            clahe_config = get_clahe_config(is_nucleoli=False)
            clahe_metadata = {
                "clip_limit": clahe_config["clip_limit"],
                "kernel_size": clahe_config["kernel_size"],
                "post_smoothing": clahe_config["post_smoothing"],
            }
        else:
            clahe_config = get_clahe_config(is_nucleoli=False)
            clahe_metadata = {
                "clip_limit": 0.03,
                "kernel_size": clahe_config["kernel_size"],
                "post_smoothing": post_clahe_smoothing_sigma if post_clahe_smoothing_sigma is not None else 0.0,
            }

    return detection_params, clahe_metadata, actual_method


def _build_vesselness_metadata(
    label_name: str,
    organelle_name: str,
    channel_name: str,
    channel_label: str,
    channel_index: int,
    channel_names: list,
) -> dict:
    """
    Build comprehensive metadata for a Frangi vesselness map.

    This is similar to segmentation metadata but indicates this is a continuous
    response map rather than a discrete segmentation.

    Args:
        label_name: Standardized label name (e.g., "mitoc_tomm20_vesselness")
        organelle_name: Original organelle key (e.g., "nuclei", "mitochondria, TOMM20")
        channel_name: Source channel name (e.g., "GFP", "mCherry")
        channel_label: Full label from ops_channel_maps.yaml (e.g., "mitochondria, TOMM20")
        channel_index: Index of source channel in zarr store
        channel_names: List of all channel names in the zarr store

    Returns:
        Dictionary with comprehensive metadata
    """
    # Parse organelle and marker from channel_label
    organelle_type, marker = parse_channel_label(channel_label) if channel_label else (None, None)

    # Determine channel type (same logic as segmentation metadata)
    channel_type = None
    channel_type_lower = channel_name.lower() if channel_name else ""
    if "gfp" in channel_type_lower:
        channel_type = "GFP"
    elif "mcherry" in channel_type_lower or "cherry" in channel_type_lower:
        channel_type = "mCherry"
    elif "cy5" in channel_type_lower:
        channel_type = "Cy5"
    elif "farred" in channel_type_lower or "far_red" in channel_type_lower:
        channel_type = "FarRed"
    elif "bf" in channel_type_lower or "phase" in channel_type_lower:
        channel_type = "Brightfield"
    elif "prediction" in channel_type_lower:
        channel_type = "VirtualStain"
    elif "focus" in channel_type_lower:
        channel_type = "Focus"

    # Handle "no label" channels
    marker_type = None
    if organelle_type and organelle_type.lower().replace("_", " ") == "no label":
        marker = "no label"
        marker_type = "autofluorescence"

    # Determine marker type
    if marker_type is None and marker:
        marker_lower = marker.lower()
        if marker_lower.replace("_", " ") == "no label":
            marker_type = "autofluorescence"
        else:
            if any(dye in marker_lower for dye in ["dye", "tracker", "live", "spy", "bodipy", "phrodo", "cellrox", "cellevent"]):
                marker_type = "live_cell_dye"
            elif any(emission in marker_lower for emission in ["chromalive", "emission"]):
                marker_type = "live_cell_dye"
            elif marker_lower in ["vs", "2d", "3d"]:
                marker_type = "virtual_stain"
            else:
                marker_type = "endogenous_tag"

    # Build description
    if organelle_type and marker:
        description = f"Frangi vesselness map for {organelle_type} from {marker} ({channel_name})"
    elif organelle_type:
        description = f"Frangi vesselness map for {organelle_type} from {channel_name}"
    elif channel_name:
        description = f"Frangi vesselness map from {channel_name}"
    else:
        description = "Frangi vesselness map"

    # Build the metadata dictionary
    metadata = {
        # Core identification
        "label_name": label_name,
        "annotation_type": "vesselness_map",  # Different from "organelle_segmentation"
        "is_ome_label": False,  # Not strictly NGFF-compliant (float32 vs integer labels)

        # Source channel information
        "source_channel": {
            "name": channel_name,
            "index": channel_index,
            "type": channel_type,
            "all_channels": channel_names,
        },

        # Biological annotation
        "biological_annotation": {
            "organelle": organelle_type,
            "marker": marker,
            "marker_type": marker_type,
            "full_label": channel_label,
        },

        # Processing method
        "processing": {
            "method": "frangi",
            "output_type": "continuous_response",  # Float32 values, not discrete labels
            "version": "v3.0-tiled",
        },

        # Human-readable description
        "description": description,
    }

    return metadata


def detect_segmentation_status(experiment: str) -> dict:
    """
    Detect the status of all segmentation labels for an experiment.

    Returns detailed information about:
    - Labels with actual data (protected, won't overwrite)
    - Labels with empty folders (will be overwritten)
    - Labels that don't exist (will be created)

    Parameters
    ----------
    experiment : str
        The experiment name

    Returns
    -------
    dict
        Dictionary with keys:
        - positions: list of position paths
        - channel_names: list of zarr channel names
        - labels_with_data: dict mapping label_name -> description (protected)
        - labels_empty: dict mapping label_name -> description (will overwrite)
        - labels_missing: dict mapping label_name -> description (will create)
        - channel_to_label: dict mapping channel_key -> expected_label_name
    """
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths["pheno_assembled_v3"]

    if not source_path.exists():
        raise FileNotFoundError(f"v3 phenotyping store not found at {source_path}")

    with open_ome_zarr(source_path, mode="r") as store:
        position_list = [p for p, _ in store.positions()]
        channel_names = store.channel_names

    store_zarr = zarr.open(str(source_path), mode="r")

    # Protected labels from convert_v3 that we should never overwrite
    PROTECTED_LABELS = {"seg", "nuclear_seg", "grid_edges", "grid_props", "iss_points", "iss_points_props"}

    # Build mapping of channels to their expected output labels
    # Skip channels that already have segmentation from convert_v3:
    # - membrane_prediction: already have "seg" (cell segmentation)
    # - nuclei_prediction: already have "nuclear_seg"
    SKIP_CHANNELS = {"membrane_prediction", "nuclei_prediction"}

    # channel_to_label maps "channel_key" -> "expected_label_name"
    # For Frangi channels, we create TWO entries per channel (tubular + vesicular)
    # The key format for dual labels is: "ChannelName_tubular" or "ChannelName_vesicular"
    channel_to_label = {}
    for ch_name in channel_names:
        ch_lower = ch_name.lower()

        # Skip prediction channels - we use existing seg/nuclear_seg from convert_v3
        if ch_name in SKIP_CHANNELS:
            continue

        # All other channels (including Phase2D, Focus3D, fluorescent markers) use Frangi
        # For label-free channels: create all 3 structure types (tubular, vesicular, vesicular_dark)
        # For fluorescent channels: check config for structure_type override, else default to tubular
        channel_type = get_channel_type(ch_name)
        if channel_type == "fluorescent":
            # Check if experiment config specifies a structure_type for this channel
            exp_configs = load_experiment_configs(experiment)
            channel_config = exp_configs.get(ch_name, {})
            config_structure_type = channel_config.get("structure_type")

            if config_structure_type:
                # Use structure_type from config (e.g., "vesicular" for lysosomes)
                structure_types_for_channel = [config_structure_type]
            else:
                # Default: fluorescent channels use tubular (organelles ARE the signal)
                structure_types_for_channel = ["tubular"]
        else:
            structure_types_for_channel = STRUCTURE_TYPES  # All 3 types for labelfree

        for structure_type in structure_types_for_channel:
            # Key format:
            # - Label-free: "Phase2D_tubular" (suffix needed for dual segmentation)
            # - Fluorescent: "GFP" (no suffix - structure_type is just metadata)
            if channel_type == "labelfree":
                # Dual segmentation: append structure_type to create multiple variants
                channel_key = f"{ch_name}_{structure_type}"
            else:
                # Single segmentation: channel_key is just the channel name
                channel_key = ch_name
            label_name = get_output_label_name(ch_name, ch_name, structure_type)
            channel_to_label[channel_key] = label_name

    # Add nucleoli segmentation for both Phase2D and Focus3D channels (uses nuclear_seg as mask)
    # nucleoli is NOT a channel - it's derived from Phase2D/Focus3D + nuclear_seg mask
    phase_channel = next((ch for ch in channel_names if "phase" in ch.lower()), None)
    focus_channel = next((ch for ch in channel_names if "focus" in ch.lower()), None)

    if phase_channel:
        # Nucleoli from Phase2D -> nucleoli_phase2d_seg
        channel_to_label["nucleoli_phase2d"] = "nucleoli_phase2d_seg"

    if focus_channel:
        # Nucleoli from Focus3D -> nucleoli_focus3d_seg
        channel_to_label["nucleoli_focus3d"] = "nucleoli_focus3d_seg"

    # Check status of each label across all positions
    labels_with_data = {}
    labels_empty = {}
    labels_missing = {}

    # Get all expected labels
    expected_labels = set(channel_to_label.values())

    # Labels to skip when scanning the store (not organelle segmentations)
    SKIP_LABELS = {"cell_seg", "cp_cell_seg", "cp_cell_seg_unstitched",
                   "grid_overlay", "iss_gene_image", "iss_guide_image"}

    # Check first position for label status (assume consistent across positions)
    if position_list:
        first_pos = position_list[0]
        labels_group = store_zarr[first_pos].get("labels", None)

        for label_name in expected_labels:
            # Check if this label exists and has data
            if labels_group is not None and label_name in labels_group:
                if _check_label_has_data(labels_group, label_name):
                    labels_with_data[label_name] = "has data"
                else:
                    labels_empty[label_name] = "empty folder"
            else:
                labels_missing[label_name] = "does not exist"

        # Also check protected labels
        if labels_group is not None:
            for label_name in PROTECTED_LABELS:
                if label_name in labels_group and _check_label_has_data(labels_group, label_name):
                    if label_name not in labels_with_data:
                        labels_with_data[label_name] = "protected (from convert_v3)"

        # Bottom-up scan: discover labels in the store that weren't predicted
        # This catches labels created with different naming conventions (e.g., "gfp_seg"
        # when config predicts "gfp_vesicular_seg" because structure_type wasn't used
        # in the label name for fluorescent channels)
        if labels_group is not None:
            mapped_labels = set(channel_to_label.values())
            for actual_label in labels_group.group_keys():
                if actual_label in PROTECTED_LABELS or actual_label in SKIP_LABELS:
                    continue
                if not actual_label.endswith("_seg"):
                    continue
                if actual_label in mapped_labels:
                    continue  # Already tracked

                # Parse label: remove _seg, then check for structure_type suffix
                base = actual_label[:-4]
                st_found = None
                for st in STRUCTURE_TYPES:
                    if base.endswith(f"_{st}"):
                        st_found = st
                        base = base[:-len(f"_{st}")]
                        break

                # Try to match base against channel names (case-insensitive)
                matched_ch = None
                for ch in channel_names:
                    if ch.lower() == base.lower():
                        matched_ch = ch
                        break

                # Also check nucleoli variants
                if matched_ch is None and base.startswith("nucleoli"):
                    matched_ch = base  # nucleoli_phase2d, nucleoli_focus3d

                if matched_ch is not None:
                    # Build channel_key matching existing convention
                    ch_type = get_channel_type(matched_ch) if matched_ch in channel_names else "fluorescent"
                    if ch_type == "labelfree" and st_found:
                        channel_key = f"{matched_ch}_{st_found}"
                    else:
                        channel_key = matched_ch

                    # Add to channel_to_label if not already mapped, or if the
                    # predicted label doesn't exist in the store (e.g., config
                    # predicts "gfp_vesicular_seg" but store has "gfp_seg")
                    existing_label = channel_to_label.get(channel_key)
                    if existing_label is None or existing_label not in labels_group:
                        channel_to_label[channel_key] = actual_label

                    # Track data status
                    if _check_label_has_data(labels_group, actual_label):
                        labels_with_data[actual_label] = "has data"
                    else:
                        labels_empty[actual_label] = "empty folder"

    return {
        "positions": position_list,
        "channel_names": channel_names,
        "labels_with_data": labels_with_data,
        "labels_empty": labels_empty,
        "labels_missing": labels_missing,
        "channel_to_label": channel_to_label,
        "protected_labels": PROTECTED_LABELS,
    }


def get_available_channels(experiment: str, skip_existing: bool = True, interactive: bool = False) -> dict:
    """
    Get available channels for segmentation from an experiment's v3 store.

    Parameters
    ----------
    experiment : str
        The experiment name
    skip_existing : bool
        If True, skip channels that already have their segmentation labels with actual data.
        Default is True.
    interactive : bool
        If True, show status and prompt user for confirmation before proceeding.
        Default is False.

    Returns
    -------
    dict
        Dictionary with keys:
        - positions: list of position paths
        - channels: list of available channel keys (e.g., "Phase2D_tubular", "GFP_vesicular", "nucleoli")
        - channel_methods: dict mapping channel keys to their segmentation method
        - channel_structure_types: dict mapping channel keys to structure type ("tubular" or "vesicular")
        - skipped_channels: list of channels skipped because segmentation already exists
        - labels_to_overwrite: list of labels that will be overwritten (empty folders)
        - labels_to_create: list of labels that will be created (don't exist)
    """
    # Get detailed status
    status = detect_segmentation_status(experiment)

    position_list = status["positions"]
    channel_names = status["channel_names"]
    channel_to_label = status["channel_to_label"]
    labels_with_data = status["labels_with_data"]
    labels_empty = status["labels_empty"]
    labels_missing = status["labels_missing"]

    # Protected labels that should never be overwritten
    # Map legacy names to new names for skip checking
    protected_equivalents = {
        "seg": ["membr_vs_seg", "cell_vs_seg"],
        "nuclear_seg": ["nucle_vs_seg"],
    }

    # Build effective "has data" set including legacy equivalents
    effective_has_data = set(labels_with_data.keys())
    for legacy, new_names in protected_equivalents.items():
        if legacy in labels_with_data:
            effective_has_data.update(new_names)

    # Categorize channels
    channel_methods = {}
    available_channels = []
    skipped_channels = []
    labels_to_overwrite = []
    labels_to_create = []

    # Also track structure_type for each channel key
    channel_structure_types = {}

    for ch_name, expected_label in channel_to_label.items():
        # Determine method: nucleoli uses frangi by default, everything else uses frangi
        if "nucleoli" in ch_name:
            # Nucleoli uses frangi by default
            method = "frangi"
            # Preserve full channel key (nucleoli_phase2d, nucleoli_focus3d, or nucleoli)
            actual_ch_key = ch_name
            structure_type = None  # Nucleoli method doesn't use structure_type (has its own params)
        else:
            method = "frangi"
            actual_ch_key = ch_name
            # Parse structure_type from channel key (e.g., "Phase2D_tubular" -> "tubular")
            # Check vesicular_dark BEFORE vesicular to avoid partial match
            if "_tubular" in ch_name:
                structure_type = "tubular"
            elif "_vesicular_dark" in ch_name:
                structure_type = "vesicular_dark"
            elif "_vesicular" in ch_name:
                structure_type = "vesicular"
            else:
                structure_type = None  # Legacy format without structure type

        # Check status
        if skip_existing and expected_label in effective_has_data:
            skipped_channels.append(ch_name)
        else:
            available_channels.append(actual_ch_key)
            channel_methods[actual_ch_key] = method
            if structure_type:
                channel_structure_types[actual_ch_key] = structure_type

            if expected_label in labels_empty:
                labels_to_overwrite.append(expected_label)
            elif expected_label in labels_missing:
                labels_to_create.append(expected_label)

    # Interactive mode: show status and prompt
    if interactive and (available_channels or skipped_channels):
        print(f"\n{'='*60}")
        print(f"Segmentation Status for {experiment}")
        print(f"{'='*60}\n")

        if skipped_channels:
            print("WILL SKIP (already have data):")
            for ch in skipped_channels:
                label = channel_to_label.get(ch, "?")
                print(f"  ✓ {ch} -> {label}")
            print()

        if labels_to_create:
            print("WILL CREATE (new labels):")
            for label in labels_to_create:
                ch = [k for k, v in channel_to_label.items() if v == label]
                ch_str = ch[0] if ch else "?"
                print(f"  + {ch_str} -> {label}")
            print()

        if labels_to_overwrite:
            print("WILL OVERWRITE (empty folders from failed runs):")
            for label in labels_to_overwrite:
                ch = [k for k, v in channel_to_label.items() if v == label]
                ch_str = ch[0] if ch else "?"
                print(f"  ⚠ {ch_str} -> {label}")
            print()

        # Show protected labels
        protected_with_data = [l for l in status["protected_labels"] if l in labels_with_data]
        if protected_with_data:
            print("PROTECTED (from convert_v3, never modified):")
            for label in protected_with_data:
                print(f"  🔒 {label}")
            print()

        print(f"{'='*60}")
        print(f"Summary: {len(available_channels)} to process, {len(skipped_channels)} to skip")
        print(f"{'='*60}\n")

        if available_channels:
            try:
                response = input("Proceed with segmentation? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user.\n")
                    return {
                        "positions": position_list,
                        "channels": [],
                        "channel_methods": {},
                        "channel_structure_types": {},
                        "skipped_channels": skipped_channels,
                        "labels_to_overwrite": labels_to_overwrite,
                        "labels_to_create": labels_to_create,
                        "cancelled": True,
                    }
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user.\n")
                return {
                    "positions": position_list,
                    "channels": [],
                    "channel_methods": {},
                    "channel_structure_types": {},
                    "skipped_channels": skipped_channels,
                    "labels_to_overwrite": labels_to_overwrite,
                    "labels_to_create": labels_to_create,
                    "cancelled": True,
                }

    return {
        "positions": position_list,
        "channels": available_channels,
        "channel_methods": channel_methods,
        "channel_structure_types": channel_structure_types,
        "skipped_channels": skipped_channels,
        "labels_to_overwrite": labels_to_overwrite,
        "labels_to_create": labels_to_create,
    }


def build_and_init_labels(
    source_path,
    position_list: list[str] | str,
    organelle_key: str,
    source_channel: str,
    channel_key: str,
    channel_names: list[str],
    ch_info: dict,
    structure_type: str = None,
    frangi_params: dict = None,
    clahe_params: dict = None,
    post_clahe_smoothing_sigma: float = None,
    frangi_postprocess: bool = False,
    experiment: str = None,
    skip_init: bool = False,
) -> tuple[dict, str]:
    """
    Build metadata and initialize label arrays (consolidated helper).

    This function consolidates multiple steps that were previously scattered:
    1. Get channel index
    2. Determine processing parameters
    3. Build segmentation metadata
    4. Get output label name
    5. Initialize label arrays (optional)
    6. Update labels metadata

    Args:
        source_path: Path to zarr store
        position_list: List of positions (batch) or single position string
        organelle_key: Organelle identifier (e.g., "GFP", "nucleoli_phase2d")
        source_channel: Actual channel name in zarr (may differ for nucleoli)
        channel_key: Channel key for labeling/display
        channel_names: List of all channel names in dataset
        ch_info: Channel info dict (from channel_processor)
        structure_type: Optional structure type override
        frangi_params: Optional Frangi parameter overrides
        clahe_params: Optional CLAHE parameter overrides
        post_clahe_smoothing_sigma: Optional smoothing sigma override
        frangi_postprocess: Whether postprocessing is enabled
        experiment: Experiment name for YAML config loading
        skip_init: If True, skip array initialization (for debug mode)

    Returns:
        tuple: (seg_metadata, objects_name)
            - seg_metadata: Complete segmentation metadata dict
            - objects_name: Output label name

    Example:
        >>> seg_metadata, objects_name = build_and_init_labels(
        ...     source_path, ["A/1/0", "A/2/0"], "GFP", "GFP", "GFP",
        ...     channel_names, ch_info, structure_type="tubular"
        ... )
    """
    from cyclops_utils.io.zarr_labels import get_position_shape, _init_organelle_label_array, _update_labels_metadata
    import numpy as np

    # Normalize position_list to list
    if isinstance(position_list, str):
        position_list = [position_list]

    # Get channel index for metadata
    channel_index = get_channel_index(channel_names, source_channel)

    # Determine processing parameters
    detection_params, clahe_metadata, actual_method = _determine_processing_params(
        organelle_key=organelle_key,
        source_channel=source_channel,
        structure_type=structure_type,
        ch_info=ch_info,
        frangi_params=frangi_params,
        clahe_params=clahe_params,
        post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
    )

    # Get output label name
    objects_name = get_output_label_name(organelle_key, source_channel, structure_type)

    # Use channel_label from ch_info (populated from ops_channel_maps.yaml)
    # Falls back to channel_key if no YAML label available
    channel_label = ch_info.get("channel_label", channel_key)

    # Build comprehensive metadata
    seg_metadata = _build_segmentation_metadata(
        label_name=objects_name,
        organelle_name=organelle_key,
        channel_name=source_channel,
        channel_label=channel_label,
        channel_index=channel_index,
        segmenter_type=actual_method,
        channel_names=channel_names,
        structure_type=structure_type,
        clahe_params=clahe_metadata,
        detection_params=detection_params,
        postprocess_params=frangi_postprocess,
    )

    # Initialize arrays for all positions (unless skipped)
    if not skip_init:
        for pos in position_list:
            pos_shape = get_position_shape(source_path, pos)
            _init_organelle_label_array(source_path, pos, objects_name, pos_shape, dtype=np.int32)
            _update_labels_metadata(source_path, pos, objects_name, metadata=seg_metadata)

    return seg_metadata, objects_name
