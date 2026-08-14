"""
Channel Processing Configuration Module
========================================

Provides unified channel resolution and configuration logic for organelle segmentation.

This module consolidates all channel-related logic that was previously duplicated
across run_organelle_segmentation() and segment_single_position_channel().

Main functions:
- build_channel_processing_map(): Configure all channels for batch processing
- resolve_single_channel_info(): Resolve single channel for targeted processing
- should_skip_channel(): Check if channel should be skipped
"""

from .configs import (
    NUCLEOLI_VARIANTS,
    SKIP_CHANNEL_PATTERNS,
    DEFAULT_METHODS,
    LABELFREE_CHANNELS,
    load_channel_labels,
)


def should_skip_channel(channel_name: str) -> tuple[bool, str]:
    """
    Check if a channel should be skipped during segmentation.

    Args:
        channel_name: Channel name to check

    Returns:
        tuple: (should_skip: bool, reason: str)
            - should_skip: True if channel should be skipped
            - reason: Human-readable skip reason

    Examples:
        >>> should_skip_channel("Focus3D")
        (True, "autofocus channel - not segmentable")
        >>> should_skip_channel("nuclei_prediction")
        (True, "virtual staining - use convert_v3.py labels")
        >>> should_skip_channel("GFP")
        (False, "")
    """
    ch_lower = channel_name.lower()

    for pattern, reason in SKIP_CHANNEL_PATTERNS.items():
        if pattern in ch_lower:
            return True, reason

    return False, ""


def _resolve_nucleoli_info(channel_key: str, channel_names: list[str]) -> dict | None:
    """
    Resolve nucleoli variant configuration.

    Args:
        channel_key: Nucleoli variant key (e.g., "nucleoli", "nucleoli_phase2d")
        channel_names: List of available channel names

    Returns:
        dict with ch_info or None if source channel not found

    Keys in returned dict:
        - method: "frangi" or "blob"
        - organelle_key: normalized organelle key
        - source_channel: actual channel name (e.g., "Phase2D")
        - requires_mask: "nuclei" (nucleoli needs nuclei mask)
        - nucleoli_method: method used (same as method)
        - nucleoli_source: "phase2d" or "focus3d"
    """
    # Check if this is a nucleoli variant
    if not channel_key.startswith("nucleoli"):
        return None

    # Look up variant config
    variant_config = NUCLEOLI_VARIANTS.get(channel_key)
    if variant_config is None:
        # Unknown nucleoli variant
        return None

    # Determine source and method
    source_type = variant_config["source"]  # "phase2d" or "focus3d"
    channel_search = variant_config["channel_search"]  # "phase" or "focus"

    # Find actual source channel
    source_channel = next(
        (ch for ch in channel_names if channel_search in ch.lower()),
        None
    )

    if source_channel is None:
        return None  # Source channel not found

    # Determine detection method
    if "force_method" in variant_config:
        # Explicit method specified (nucleoli_frangi, nucleoli_blob)
        method = variant_config["force_method"]
    else:
        # Use default method from configs
        method = DEFAULT_METHODS.get(("nucleoli", source_type), "frangi")

    # Build organelle_key (normalized form)
    if channel_key in ["nucleoli", "nucleoli_frangi", "nucleoli_blob"]:
        # Legacy formats → map to phase2d
        organelle_key = "nucleoli_phase2d"
    else:
        # Already in correct format
        organelle_key = channel_key

    return {
        "method": method,
        "organelle_key": organelle_key,
        "source_channel": source_channel,
        "requires_mask": "nuclei",
        "nucleoli_method": method,  # Redundant but kept for compatibility
        "nucleoli_source": source_type,
    }


def resolve_single_channel_info(
    channel_key: str,
    channel_names: list[str],
    experiment: str = None,
) -> dict | None:
    """
    Resolve processing configuration for a single channel.

    Handles:
    - Nucleoli variants (nucleoli, nucleoli_phase2d, nucleoli_focus3d,
      nucleoli_frangi, nucleoli_blob)
    - Standard channels (GFP, mCherry, Phase2D, etc.)
    - Skip patterns (focus, prediction channels)

    Args:
        channel_key: Channel identifier to resolve
        channel_names: List of available channel names in the dataset
        experiment: Experiment name for loading channel labels from ops_channel_maps.yaml

    Returns:
        dict with channel info or None if channel should be skipped/not found

    Returned dict contains:
        - method: "frangi" or "blob"
        - organelle_key: Key for output naming
        - source_channel: Actual channel name (may differ for nucleoli)
        - channel_label: Full label from ops_channel_maps.yaml (e.g., "lysosome, LysoTracker live-cell dye")
        - requires_mask: Optional mask name (e.g., "nuclei" for nucleoli)

    Examples:
        >>> resolve_single_channel_info("GFP", ["GFP", "Phase2D"], experiment="ops0113")
        {"method": "frangi", "organelle_key": "GFP", "channel_label": "lysosome, LysoTracker live-cell dye"}

        >>> resolve_single_channel_info("nucleoli_phase2d", ["Phase2D", "GFP"])
        {"method": "frangi", "organelle_key": "nucleoli_phase2d",
         "source_channel": "Phase2D", "requires_mask": "nuclei", ...}

        >>> resolve_single_channel_info("Focus3D", ["Focus3D"])
        None  # Skip channel
    """
    # Load channel labels from YAML if experiment is provided
    channel_labels = load_channel_labels(experiment) if experiment else {}

    # Check for nucleoli variants first
    nucleoli_info = _resolve_nucleoli_info(channel_key, channel_names)
    if nucleoli_info is not None:
        # Nucleoli derives from Phase/Focus channels - look up the source channel's label
        source_ch = nucleoli_info.get("source_channel")
        if source_ch and source_ch in channel_labels:
            nucleoli_info["channel_label"] = channel_labels[source_ch]
        return nucleoli_info

    # Check if channel exists in dataset (case-insensitive)
    # Find the actual channel name that matches (preserving original case)
    matched_channel = None
    channel_key_lower = channel_key.lower()
    for ch in channel_names:
        if ch.lower() == channel_key_lower:
            matched_channel = ch
            break

    if matched_channel is None:
        return None  # Channel not found

    # Check skip patterns (use matched channel name)
    should_skip, skip_reason = should_skip_channel(matched_channel)
    if should_skip:
        return None  # Skip this channel

    # Standard channel (GFP, mCherry, Phase2D, etc.)
    # All use Frangi by default (can be overridden via config)
    # Use the matched channel name (with original case from dataset)
    info = {
        "method": "frangi",
        "organelle_key": matched_channel,
        "source_channel": matched_channel,
    }

    # Attach channel label from ops_channel_maps.yaml if available
    if matched_channel in channel_labels:
        info["channel_label"] = channel_labels[matched_channel]

    return info


def build_channel_processing_map(
    channel_names: list[str],
    include_nucleoli: bool = True,
    experiment: str = None,
) -> dict:
    """
    Build channel processing configuration for all segmentable channels.

    This function processes all channels in the dataset and builds a complete
    mapping of channels to their segmentation configuration.

    Args:
        channel_names: List of available channel names in the dataset
        include_nucleoli: If True, add nucleoli variants (default: True)
        experiment: Experiment name for loading channel labels from ops_channel_maps.yaml

    Returns:
        dict: {channel_key: ch_info}
            where ch_info contains:
                - method: "frangi" or "blob"
                - organelle_key: Key for output naming
                - source_channel: Actual channel name (optional, for nucleoli)
                - channel_label: Full label from ops_channel_maps.yaml (if available)
                - requires_mask: Optional mask name (e.g., "nuclei")

    Processing rules:
        1. Skip channels matching SKIP_CHANNEL_PATTERNS (focus, prediction)
        2. Add standard channels with method="frangi"
        3. If include_nucleoli=True, add nucleoli_phase2d and nucleoli_focus3d
           (if corresponding source channels exist)

    Example:
        >>> channel_names = ["GFP", "mCherry", "Phase2D", "Focus3D"]
        >>> map = build_channel_processing_map(channel_names, experiment="ops0113")
        >>> sorted(map.keys())
        ['GFP', 'mCherry', 'Phase2D', 'nucleoli_focus3d', 'nucleoli_phase2d']
    """
    # Load channel labels from YAML if experiment is provided
    channel_labels = load_channel_labels(experiment) if experiment else {}

    processing_map = {}

    # Process each standard channel
    for ch_name in channel_names:
        # Check skip patterns
        should_skip, skip_reason = should_skip_channel(ch_name)
        if should_skip:
            print(f"Skipping '{ch_name}' ({skip_reason})")
            continue

        # Add standard channel with Frangi
        ch_info = {
            "method": "frangi",
            "organelle_key": ch_name,
        }
        # Attach channel label from ops_channel_maps.yaml if available
        if ch_name in channel_labels:
            ch_info["channel_label"] = channel_labels[ch_name]
        processing_map[ch_name] = ch_info

    # Add nucleoli variants if requested
    if include_nucleoli:
        # Try to add nucleoli_phase2d
        phase_info = _resolve_nucleoli_info("nucleoli_phase2d", channel_names)
        if phase_info is not None:
            # Nucleoli derives from Phase channel - look up its label
            phase_ch = phase_info["source_channel"]
            if phase_ch in channel_labels:
                phase_info["channel_label"] = channel_labels[phase_ch]
            processing_map["nucleoli_phase2d"] = phase_info
            phase_method = phase_info["method"]
            print(f"Added 'nucleoli_phase2d' segmentation ({phase_ch} + nuclei mask, method={phase_method})")

        # Try to add nucleoli_focus3d
        focus_info = _resolve_nucleoli_info("nucleoli_focus3d", channel_names)
        if focus_info is not None:
            focus_ch = focus_info["source_channel"]
            if focus_ch in channel_labels:
                focus_info["channel_label"] = channel_labels[focus_ch]
            processing_map["nucleoli_focus3d"] = focus_info
            focus_method = focus_info["method"]
            print(f"Added 'nucleoli_focus3d' segmentation ({focus_ch} + nuclei mask, method={focus_method})")

    return processing_map
