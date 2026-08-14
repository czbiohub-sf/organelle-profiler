"""
Organelle Segmentation Configuration
====================================

Configuration is organized by use case: (structure_type, channel_type, method).

SEGMENTATION_CONFIGS Dictionary:
    Complete configurations for all supported combinations.
    Keys: (structure_type, channel_type, method)
    - structure_type: "tubular", "vesicular", "vesicular_dark", "nucleoli"
    - channel_type: "labelfree", "fluorescent", "phase2d", "focus3d"
    - method: "frangi" or "blob"

    Example: SEGMENTATION_CONFIGS[("tubular", "labelfree", "frangi")]
    Returns: Complete dict with all Frangi parameters for this use case.

DEFAULT_METHODS Dictionary:
    Default detection method for each (structure_type, channel_type) combination.
    Keys: (structure_type, channel_type)
    Returns: "frangi" or "blob"

CLAHE_CONFIGS Dictionary:
    CLAHE preprocessing parameters.
    Keys: "default" or "nucleoli"

Main Functions:
    - get_segmentation_config(structure, channel, method) - Get base config
    - get_clahe_config(is_nucleoli) - Get CLAHE config
    - get_full_segmentation_config(...) - Get config with all overrides applied

Override Priority (highest priority last):
    1. Base config from SEGMENTATION_CONFIGS
    2. Experiment YAML overrides (ops_channel_maps.yaml)
    3. Runtime function parameter overrides
"""

import numpy as np

# =============================================================================
# STRUCTURE AND CHANNEL TYPE DEFINITIONS
# =============================================================================

# Structure types: tubular (ER, mitochondria), vesicular (lysosomes, vesicles),
# vesicular_dark (lipid droplets in phase), nucleoli (in nuclei)
STRUCTURE_TYPES = ["tubular", "vesicular", "vesicular_dark"]

# Label-free microscopy channels (canonical in cyclops_utils.data.naming)
from cyclops_utils.data.naming import LABELFREE_CHANNELS

# =============================================================================
# ORGANELLE MASK CONFIGURATION
# =============================================================================
# Maps organelle_key -> input_mask_name for mask-constrained segmentation
ORGANELLE_MASK_CONFIG = {
    "nucleoli_phase2d": "nuclear_seg",
    "nucleoli_focus3d": "nuclear_seg",
    "nucleoli": "nuclear_seg",
    "nucleoli_frangi": "nuclear_seg",
    "nucleoli_blob": "nuclear_seg",
}

# =============================================================================
# SEGMENTATION CONFIGURATIONS - ORGANIZED BY USE CASE
# =============================================================================

SEGMENTATION_CONFIGS = {
    # -------------------------------------------------------------------------
    # TUBULAR STRUCTURES (mitochondria, ER, etc.) - Frangi only
    # -------------------------------------------------------------------------
    ("tubular", "labelfree", "frangi"): {
        # Complete config - all Frangi params for labelfree tubular
        "pixel_size_um": 0.325,  # Coarser sampling for labelfree
        "alpha": 0.5,  # Balanced elongation sensitivity
        "beta": 0.5,  # Higher beta suppresses blob-like structures (good for tubes)
        "min_radius_um": 0.2,
        "max_radius_um": 1.5,
        "threshold": 0.01,  # Fixed threshold (set None for dynamic)
        "threshold_mult": None,
        "min_object_size": 5,  # Smaller for labelfree
        "postprocess": True,
        "postprocess_opening": False,
        "postprocess_opening_size": 2,  # Labelfree tubular needs morphological opening
        "postprocess_fill_holes": False,
        "structure_type": "tubular",
    },

    ("tubular", "fluorescent", "frangi"): {
        # Complete config - all Frangi params for fluorescent tubular
        "pixel_size_um": 0.185,  # Finer sampling for fluorescent
        "alpha": 0.5,
        "beta": 0.5,  # Suppress blobs
        "min_radius_um": 0.1,  # Smaller min for fluorescent
        "max_radius_um": 1.5,
        "threshold": 0.01,
        "threshold_mult": None,
        "min_object_size": 8,  # Larger for fluorescent (less noise)
        "postprocess": True,
        "postprocess_opening": False,
        "postprocess_opening_size": 0,  # No opening for fluorescent
        "postprocess_fill_holes": False,
        "structure_type": "tubular",
    },

    # -------------------------------------------------------------------------
    # VESICULAR STRUCTURES (lysosomes, vesicles, bright spots)
    # -------------------------------------------------------------------------
    ("vesicular", "labelfree", "frangi"): {
        # Complete config - Frangi params for labelfree vesicular
        "pixel_size_um": 0.1625,  # Half of tubular labelfree (finer sampling for small vesicles)
        "alpha": 0.5,
        "beta": 0.1,  # Lower beta allows blob-like structures (good for vesicles)
        "min_radius_um": 0.2,
        "max_radius_um": 1.5,
        "threshold": 0.01,
        "threshold_mult": 0.01,  # Vesicular uses dynamic thresholding
        "min_object_size": 5,  # Labelfree override
        "gamma": None,
        "black_ridges": False,
        "postprocess": False,
        "watershed": False,
        "watershed_min_distance": 1,
        "watershed_compactness": 0.5,
        "watershed_erosion": 0,
        "watershed_min_peak": 1.0,
        "watershed_h_maxima": 0.0,
        "structure_type": "vesicular",
    },

    ("vesicular", "labelfree", "blob"): {
        # Complete config - LoG blob detection for labelfree vesicular
        "min_radius_um": 0.2,
        "max_radius_um": 0.4,
        "num_sigma": 4,
        "threshold": 0.05,
        "overlap": 0.3,
        "exclude_border": False,
    },

    ("vesicular", "fluorescent", "frangi"): {
        # Complete config - Frangi params for fluorescent vesicular
        "pixel_size_um": 0.185,
        "alpha": 0.5,
        "beta": 0.1,  # Allow blobs
        "min_radius_um": 0.1,  # Smaller for fluorescent
        "max_radius_um": 1.5,
        "threshold": 0.01,
        "threshold_mult": 0.01,
        "min_object_size": 4,  # Base vesicular size
        "gamma": None,
        "black_ridges": False,
        "postprocess": False,
        "watershed": False,
        "watershed_min_distance": 1,
        "watershed_compactness": 0.5,
        "watershed_erosion": 0,
        "watershed_min_peak": 1.0,
        "watershed_h_maxima": 0.0,
        "structure_type": "vesicular",
    },

    ("vesicular", "fluorescent", "blob"): {
        # Complete config - LoG blob detection for fluorescent vesicular
        "min_radius_um": 0.1,
        "max_radius_um": 0.6,
        "num_sigma": 8,
        "threshold": 0.03,
        "overlap": 0.3,
        "exclude_border": False,
    },

    # -------------------------------------------------------------------------
    # VESICULAR DARK (lipid droplets, vacuoles - dark spots in phase)
    # -------------------------------------------------------------------------
    ("vesicular_dark", "labelfree", "frangi"): {
        # Dark vesicular structures (lipid droplets, vacuoles in phase contrast)
        "pixel_size_um": 0.1625,  # Unified pixel size
        "alpha": 0.5,  # Unified alpha
        "beta": 0.1,  # Unified beta (higher than bright vesicular to suppress tube-like)
        "gamma": None,  # Auto-scale based on max Hessian norm
        "min_radius_um": 0.2,  # Labelfree min radius
        "max_radius_um": 1.5,  # Standard max
        "threshold": 0.01,  # Fixed threshold (set to None for dynamic)
        "threshold_mult": 0.01,  # Multiplier for dynamic threshold
        "black_ridges": True,  # KEY DIFFERENCE: detect dark structures
        "postprocess": False,  # Passthrough (watershed handles separation)
        "watershed": False,  # Toggle watershed on/off
        "watershed_min_distance": 1,  # Aggressive separation
        "watershed_compactness": 0.5,  # Moderate compactness
        "watershed_erosion": 0,  # No erosion - vesicles are too small
        "watershed_min_peak": 1.0,  # Accept all peaks
        "watershed_h_maxima": 0.0,  # Disable h_maxima
        "min_object_size": 4,  # Remove objects < 4 pixels
        "structure_type": "vesicular_dark",
    },

    ("vesicular_dark", "labelfree", "blob"): {
        # LoG blob detection for dark labelfree vesicular (lipid droplets, vacuoles)
        "min_radius_um": 0.2,  # Keep small vesicles
        "max_radius_um": 0.4,  # Tighter max to exclude large circles
        "num_sigma": 4,  # Narrow scale range for small vesicles only
        "threshold": 0.08,  # Higher threshold for cleaner detection
        "black_ridges": True,  # Detect dark structures
        "overlap": 0.3,  # Lower overlap = fewer merged blobs
        "exclude_border": False,  # Include border blobs
    },

    ("vesicular_dark", "fluorescent", "frangi"): {
        # Dark vesicular structures in fluorescent imaging
        "pixel_size_um": 0.185,  # Finer sampling for fluorescent
        "alpha": 0.5,  # Unified alpha
        "beta": 0.1,  # Unified beta (higher than bright vesicular)
        "gamma": None,  # Auto-scale based on max Hessian norm
        "min_radius_um": 0.1,  # Smaller min for fluorescent
        "max_radius_um": 1.5,  # Standard max
        "threshold": 0.01,  # Fixed threshold
        "threshold_mult": 0.01,  # Multiplier for dynamic threshold
        "black_ridges": True,  # KEY DIFFERENCE: detect dark structures
        "postprocess": False,  # Passthrough
        "watershed": False,  # Toggle watershed on/off
        "watershed_min_distance": 1,  # Aggressive separation
        "watershed_compactness": 0.5,  # Moderate compactness
        "watershed_erosion": 0,  # No erosion
        "watershed_min_peak": 1.0,  # Accept all peaks
        "watershed_h_maxima": 0.0,  # Disable h_maxima
        "min_object_size": 4,  # Remove small objects
        "structure_type": "vesicular_dark",
    },

    ("vesicular_dark", "fluorescent", "blob"): {
        # LoG blob detection for dark fluorescent vesicular (lipid droplets)
        "min_radius_um": 0.2,  # Keep small vesicles
        "max_radius_um": 0.4,  # Tighter max to exclude large circles
        "num_sigma": 4,  # Narrow scale range for small vesicles only
        "threshold": 0.08,  # Higher threshold for cleaner detection
        "black_ridges": True,  # Detect dark structures
        "overlap": 0.3,  # Lower overlap = fewer merged blobs
        "exclude_border": False,  # Include border blobs
    },

    # -------------------------------------------------------------------------
    # FOCUS3D-SPECIFIC vesicular configs (split from labelfree so Phase2D
    # tuning stays independent). Frangi clones labelfree; blob raises threshold
    # to discriminate against tiny noise.
    # -------------------------------------------------------------------------
    ("vesicular", "focus3d", "frangi"): {
        "pixel_size_um": 0.1625,
        "alpha": 0.5,
        "beta": 0.1,
        "min_radius_um": 0.2,
        "max_radius_um": 1.5,
        "threshold": 0.01,
        "threshold_mult": 0.01,
        "min_object_size": 5,
        "gamma": None,
        "black_ridges": False,
        "postprocess": False,
        "watershed": False,
        "watershed_min_distance": 1,
        "watershed_compactness": 0.5,
        "watershed_erosion": 0,
        "watershed_min_peak": 1.0,
        "watershed_h_maxima": 0.0,
        "structure_type": "vesicular",
    },

    ("vesicular", "focus3d", "blob"): {
        "min_radius_um": 0.2,
        "max_radius_um": 0.4,
        "num_sigma": 4,
        "threshold": 0.10,  # Raised from labelfree 0.05 — Focus3D picks up too much noise
        "overlap": 0.3,
        "exclude_border": False,
    },

    ("vesicular_dark", "focus3d", "frangi"): {
        "pixel_size_um": 0.1625,
        "alpha": 0.5,
        "beta": 0.1,
        "gamma": None,
        "min_radius_um": 0.2,
        "max_radius_um": 1.5,
        "threshold": 0.01,
        "threshold_mult": 0.01,
        "black_ridges": True,
        "postprocess": False,
        "watershed": False,
        "watershed_min_distance": 1,
        "watershed_compactness": 0.5,
        "watershed_erosion": 0,
        "watershed_min_peak": 1.0,
        "watershed_h_maxima": 0.0,
        "min_object_size": 4,
        "structure_type": "vesicular_dark",
    },

    ("vesicular_dark", "focus3d", "blob"): {
        "min_radius_um": 0.2,
        "max_radius_um": 0.4,
        "num_sigma": 4,
        "threshold": 0.15,  # Raised from labelfree 0.08 — Focus3D picks up too much noise
        "black_ridges": True,
        "overlap": 0.3,
        "exclude_border": False,
    },

    # -------------------------------------------------------------------------
    # NUCLEOLI (special case - large round structures within nuclei)
    # -------------------------------------------------------------------------
    ("nucleoli", "phase2d", "frangi"): {
        # Complete Frangi config for nucleoli (Phase2D channel)
        # Optimized for LARGE ROUND structures with radius range 0.5-3.0um
        "pixel_size_um": 0.65,  # Larger pixels for nucleoli scale
        "alpha": 0.1,  # Low alpha for rounder structures
        "beta": 0.5,
        "min_radius_um": 0.5,  # Nucleoli are large (0.5-3um)
        "max_radius_um": 3.0,
        "threshold": 0.01,  # Fixed threshold
        "threshold_mult": 0.01,
        # Postprocessing controls
        "postprocess": True,
        "postprocess_opening": False,
        "postprocess_opening_radius": 1,
        "postprocess_closing": True,  # Closing fills internal gaps
        "postprocess_closing_radius": 1,
        "min_object_size": 10,
        # Watershed controls (for separating touching nucleoli)
        "watershed": False,
        "watershed_min_distance": 2,
        "watershed_compactness": 1.0,
        "watershed_erosion": 2,
        "watershed_min_peak": 2.0,
    },

    ("nucleoli", "phase2d", "blob"): {
        # LoG blob detection for nucleoli (Phase2D channel)
        "min_radius_um": 0.5,
        "max_radius_um": 3.0,
        "num_sigma": 12,
        "threshold": 0.01,
        "overlap": 0.5,  # Higher overlap for nucleoli
        "exclude_border": False,
    },

    ("nucleoli", "focus3d", "frangi"): {
        # Focus3D uses same Frangi params as Phase2D
        "pixel_size_um": 0.65,
        "alpha": 0.1,
        "beta": 0.5,
        "min_radius_um": 0.5,
        "max_radius_um": 3.0,
        "threshold": 0.01,
        "threshold_mult": 0.01,
        "postprocess": True,
        "postprocess_opening": False,
        "postprocess_opening_radius": 1,
        "postprocess_closing": True,
        "postprocess_closing_radius": 1,
        "min_object_size": 10,
        "watershed": False,
        "watershed_min_distance": 2,
        "watershed_compactness": 1.0,
        "watershed_erosion": 2,
        "watershed_min_peak": 2.0,
    },

    ("nucleoli", "focus3d", "blob"): {
        # Focus3D uses same blob params as Phase2D
        "min_radius_um": 0.5,
        "max_radius_um": 3.0,
        "num_sigma": 12,
        "threshold": 0.01,
        "overlap": 0.5,
        "exclude_border": False,
    },
}

# =============================================================================
# INTENSITY-THRESHOLD METHOD DEFAULTS (infer-subc style)
# =============================================================================
# The "threshold" detection method segments the raw (CLAHE'd) intensity image
# directly instead of the Frangi vesselness map. It is opt-in: a channel only
# uses it when method="threshold" is set in org_seg_params.yaml (or passed as a
# runtime override). get_segmentation_config() synthesizes a config from these
# defaults for any (structure_type, channel_type, "threshold") key so the method
# is available for every organelle without enumerating combinations.
THRESHOLD_DEFAULTS = {
    "detection_method": "threshold",
    # threshold_method: otsu | triangle | li | li_log | median | ave | multiotsu | masked_object
    "threshold_method": "otsu",
    "threshold_factor": 1.0,  # scales scalar thresholds (subc lipid-droplet uses 0.8)
    # Masked-Object (MO) params (used when threshold_method="masked_object")
    "mo_global_method": "triangle",
    "mo_local_adjust": 0.98,
    "mo_object_min_area_px": 100,
    # Postprocessing
    "fill_holes": True,
    "thinning": False,  # topology-preserving thinning (subc Golgi recipe)
    "thin_min_thickness": 1.6,
    "thin_dist": 1,
    # Object-size filtering. Pixel filter always applies; physical-um filter
    # (min/max_object_size_um2) additionally applies when set (µm² 2D / µm³ 3D).
    "min_object_size": 4,
    "min_object_size_um2": None,
    "max_object_size_um2": None,
    # Watershed separation (off by default; CC labeling otherwise)
    "watershed": False,
    "watershed_min_distance": 3,
    "watershed_compactness": 0.0,
    "watershed_erosion": 0,
    "watershed_min_peak": 1.0,
    "watershed_h_maxima": 0.0,
}


# =============================================================================
# DEFAULT METHODS - Which method to use for each (structure, channel) combo
# =============================================================================

DEFAULT_METHODS = {
    ("tubular", "labelfree"): "frangi",
    ("tubular", "fluorescent"): "frangi",
    ("vesicular", "labelfree"): "blob",  # Blob works better for labelfree vesicles
    ("vesicular", "fluorescent"): "blob",  # Blob works better for fluorescent vesicles
    ("vesicular_dark", "labelfree"): "blob",  # Blob for dark labelfree vesicles
    ("vesicular_dark", "fluorescent"): "blob",  # Blob for dark fluorescent vesicles
    ("vesicular", "focus3d"): "blob",  # Focus3D split from labelfree for stricter threshold
    ("vesicular_dark", "focus3d"): "blob",
    ("nucleoli", "phase2d"): "frangi",  # Frangi is default for nucleoli
    ("nucleoli", "focus3d"): "frangi",
}

# =============================================================================
# CLAHE PREPROCESSING CONFIGS
# =============================================================================

CLAHE_CONFIGS = {
    "default": {
        "clip_limit": 0.01,
        "kernel_size": (256, 256),
        "post_smoothing": 0.0,
    },
    "nucleoli": {
        "clip_limit": 0.01,
        "kernel_size": (256, 256),
        "post_smoothing": 1.0,  # Only difference - nucleoli need post-smoothing
    },
}

# =============================================================================
# NUCLEOLI VARIANT SPECIFICATIONS
# =============================================================================
# Maps nucleoli variant names to their configuration
NUCLEOLI_VARIANTS = {
    "nucleoli": {
        "source": "phase2d",
        "channel_search": "phase",
    },
    "nucleoli_phase2d": {
        "source": "phase2d",
        "channel_search": "phase",
    },
    "nucleoli_focus3d": {
        "source": "focus3d",
        "channel_search": "focus",
    },
    "nucleoli_frangi": {
        "source": "phase2d",
        "channel_search": "phase",
        "force_method": "frangi",
    },
    "nucleoli_blob": {
        "source": "phase2d",
        "channel_search": "phase",
        "force_method": "blob",
    },
}

# =============================================================================
# CHANNEL SKIP PATTERNS
# =============================================================================
# Channels that should be skipped during segmentation
# Note: Focus3D is segmentable (tubular/vesicular structures)
# It's only used as a *source* for nucleoli_focus3d, but can be segmented independently
SKIP_CHANNEL_PATTERNS = {
    "prediction": "virtual staining - use convert_v3.py labels",
}

# =============================================================================
# SPECIAL LABEL MAPPINGS
# =============================================================================
# Maps special channel names to their output label names
# (These are legacy aliases for nucleoli detection)
SPECIAL_LABEL_MAP = {
    "nucleoli": "nucleoli_phase2d",  # Default to phase2d
    "nucleoli_frangi": "nucleoli_phase2d",
    "nucleoli_blob": "nucleoli_phase2d",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Re-export from cyclops_utils (canonical implementation)
from cyclops_utils.data.naming import get_channel_type


def get_segmentation_config(
    structure_type: str,
    channel_name: str,
    method: str = None,
) -> dict:
    """
    Get base segmentation config for a specific use case.

    Args:
        structure_type: "tubular", "vesicular", "vesicular_dark", or "nucleoli"
        channel_name: Channel name (e.g., "Phase2D", "GFP")
        method: Detection method ("frangi" or "blob"). If None, uses DEFAULT_METHODS.

    Returns:
        Complete configuration dictionary (copy, safe to modify)

    Example:
        >>> config = get_segmentation_config("tubular", "Phase2D")
        >>> # Returns full config for ("tubular", "labelfree", "frangi")
    """
    # Determine channel type
    channel_type = "labelfree" if channel_name in LABELFREE_CHANNELS else "fluorescent"

    # Focus3D has its own vesicular configs (less sensitive than Phase2D's labelfree)
    if channel_name == "Focus3D" and structure_type in ("vesicular", "vesicular_dark"):
        channel_type = "focus3d"

    # Normalize structure_type for nucleoli
    if "nucleoli" in structure_type.lower():
        structure_type = "nucleoli"
        # Map Phase2D/Focus3D channel names
        if channel_name in ["Phase2D", "Phase3D", "Raw"]:
            channel_type = "phase2d"
        elif channel_name == "Focus3D":
            channel_type = "focus3d"

    # Determine method
    if method is None:
        method = DEFAULT_METHODS.get((structure_type, channel_type), "frangi")

    # Lookup config
    config_key = (structure_type, channel_type, method)
    if config_key not in SEGMENTATION_CONFIGS:
        # The intensity-threshold method is structure-agnostic: synthesize a
        # config from THRESHOLD_DEFAULTS rather than enumerating every combo.
        if method == "threshold":
            cfg = THRESHOLD_DEFAULTS.copy()
            cfg["structure_type"] = structure_type
            cfg.setdefault(
                "pixel_size_um",
                0.185 if channel_type == "fluorescent" else 0.325,
            )
            return cfg
        # Fallback logic if exact key not found
        raise KeyError(
            f"No config for {config_key}. "
            f"Available: {sorted(SEGMENTATION_CONFIGS.keys())}"
        )

    return SEGMENTATION_CONFIGS[config_key].copy()


def get_clahe_config(is_nucleoli: bool = False) -> dict:
    """
    Get CLAHE config (only 2 variants: default vs nucleoli).

    Args:
        is_nucleoli: Whether this is for nucleoli segmentation

    Returns:
        CLAHE configuration dictionary
    """
    key = "nucleoli" if is_nucleoli else "default"
    return CLAHE_CONFIGS[key].copy()


def um_to_sigmas(min_radius_um: float, max_radius_um: float, pixel_size_um: float, num_sigmas: int = 5):
    """
    Convert radius range in microns to Frangi sigma array in pixels.

    This matches the sweep script behavior exactly - radius in microns is
    converted directly to sigma in pixels (no division by 2 or 3).

    Args:
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        pixel_size_um: Pixel size in microns (e.g., 0.1625)
        num_sigmas: Number of sigma values to generate

    Returns:
        numpy array of sigma values in pixels
    """
    min_sigma_px = min_radius_um / pixel_size_um
    max_sigma_px = max_radius_um / pixel_size_um
    return np.geomspace(min_sigma_px, max_sigma_px, num=num_sigmas)


# =============================================================================
# EXPERIMENT-SPECIFIC CONFIG LOADING
# =============================================================================

# Global cache for loaded experiment configs
_EXPERIMENT_CONFIGS_CACHE = {}

# Global cache for loaded channel labels
_CHANNEL_LABELS_CACHE = {}


def _extract_marker_from_label(label):
    """Derive the marker id from a channel label (mirrors _extract_seg_params.py).

    Rules:
    - 'Phase' / bare 'no label' / 'empty' -> None (not real reporters).
    - 'A, B' -> marker is B (post-last-comma token).
    - Post-comma 'no label' -> fall back to pre-comma (preserves
      'bleedthrough' / 'autofluorescence' dim-signal references).
    - single-word label -> use as-is.
    - Common dye/excitation suffixes stripped.
    - Whitespace inside the marker token is folded to '_'.
    """
    import re as _re
    if not isinstance(label, str):
        return None
    label = label.strip()
    if not label:
        return None
    low = label.lower()
    if low == "phase":
        return None
    if low in ("no label", "empty", "empty, no label"):
        return None
    if "," in label:
        pre, tail = label.rsplit(",", 1)
        pre = pre.strip()
        tail = tail.strip()
        if tail.lower() in ("no label", ""):
            tail = pre
    else:
        tail = label
    if tail.lower() == "bleedthough":
        tail = "bleedthrough"
    for suf in (
        " Live Cell Dye", " Live Cell dye", " live Cell dye", " live cell dye",
        " Live-Cell Dye", " live-cell dye", " Live Cell", " live cell", " excitation",
    ):
        if tail.lower().endswith(suf.lower()):
            tail = tail[: -len(suf)].strip()
    tail = _re.sub(r"\s+", "_", tail)
    m = _re.match(r"[A-Za-z0-9_\-]+", tail)
    if not m:
        return None
    marker = m.group(0).strip("_-")
    return marker or None


def _load_marker_seg_params(config_path: str = None) -> dict:
    """Load org_seg_params.yaml and return {marker: raw_block_list} (the list-of-dicts)."""
    from pathlib import Path
    import yaml

    if config_path is None:
        try:
            from cyclops_utils.data.experiment import OpsDataset
            dataset = OpsDataset("dummy")
            config_path = str(dataset.marker_seg_params)
        except Exception:
            return {}
    if not Path(config_path).exists():
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_marker_config(marker: str, experiment: str, marker_params: dict):
    """Return the segmentation_config for this (marker, experiment), or {} if none.

    Handles both `segmentation_config` and per-experiment `segmentation_config_variants`.
    Falls back to any variant if the specific experiment isn't listed.
    """
    block = marker_params.get(marker)
    if not block:
        return None
    seg_cfg = None
    variants = None
    for item in block if isinstance(block, list) else []:
        if not isinstance(item, dict):
            continue
        if "segmentation_config" in item:
            seg_cfg = item["segmentation_config"]
        if "segmentation_config_variants" in item:
            variants = item["segmentation_config_variants"] or {}
    if variants:
        from cyclops_utils.data.filesystem import extract_ops_key
        ops_key = extract_ops_key(experiment) or experiment
        # Try the experiment's own variant first, then the ops key, else any variant
        for k in (experiment, ops_key):
            if k and k in variants:
                return variants[k]
        # Fallback: lexicographically-latest experiment's variant
        if variants:
            latest = sorted(variants.keys())[-1]
            return variants[latest]
    return seg_cfg


def load_experiment_configs(experiment: str, config_path: str = None,
                             marker_params_path: str = None) -> dict:
    """
    Load experiment-specific segmentation config overrides from YAML.

    The primary source is org_seg_params.yaml (keyed by marker, shared across
    experiments). For each channel in ops_channel_maps.yaml, we extract the marker
    from its label and look up the config by marker. If the marker is missing from
    org_seg_params.yaml or the label has no extractable marker, we fall back to a
    channel-map-embedded `segmentation_config` (legacy path, to be removed once all
    configs are migrated).

    Args:
        experiment: Experiment name (e.g., "ops0033")
        config_path: Path to ops_channel_maps.yaml. If None, uses OpsDataset.channel_maps.
        marker_params_path: Path to org_seg_params.yaml. If None, uses
            OpsDataset.marker_seg_params.

    Returns:
        Dict mapping channel_name -> segmentation_config dict (may be empty = defaults).
"""
    # Check cache first
    cache_key = (experiment, config_path, marker_params_path)
    if cache_key in _EXPERIMENT_CONFIGS_CACHE:
        return _EXPERIMENT_CONFIGS_CACHE[cache_key]

    # Import here to avoid circular dependency
    from cyclops_utils.data.experiment import OpsDataset
    import yaml
    from pathlib import Path

    # Get config path
    if config_path is None:
        dataset = OpsDataset("dummy")  # Just to get config paths
        config_path = str(dataset.channel_maps)

    if not Path(config_path).exists():
        print(f"Config path {config_path} does not exist")
        return {}

    # Load YAML
    with open(config_path, 'r') as f:
        channel_maps = yaml.safe_load(f)

    # Load the marker-keyed params (new source of truth)
    marker_params = _load_marker_seg_params(marker_params_path)

    # Extract ops key (e.g., "ops0113" from "ops0113_20260108" or "ops0113")
    from cyclops_utils.data.filesystem import extract_ops_key
    ops_key = extract_ops_key(experiment)

    # Try various key formats (full name, ops key, lowercase)
    exp_config = None
    for key in [experiment, ops_key, experiment.lower()]:
        if key and key in channel_maps:
            exp_config = channel_maps[key]
            break

    if exp_config is None:
        exp_config = []

    # Parse config
    channel_configs = {}

    def _resolve(channel_name, label, legacy_embedded_cfg):
        """Marker lookup first; fall back to legacy per-channel embedded config."""
        marker = _extract_marker_from_label(label) if label else None
        if marker:
            cfg = _resolve_marker_config(marker, experiment, marker_params)
            if cfg is not None:
                return cfg
        # Legacy fallback (pre-migration): embedded segmentation_config in channel map
        if legacy_embedded_cfg:
            return legacy_embedded_cfg
        return {}

    if isinstance(exp_config, list):
        # Standard format: list of channel configs
        for channel_entry in exp_config:
            if not isinstance(channel_entry, dict):
                continue

            channel_name = channel_entry.get("channel_name")
            if channel_name:
                label = channel_entry.get("label", "")
                legacy = channel_entry.get("segmentation_config") or {}
                channel_configs[channel_name] = _resolve(channel_name, label, legacy)
            elif "cell_painting" in channel_entry:
                # Cell painting metadata section (nested format).
                # CP channel names encode the marker in the final underscore token.
                cell_painting_config = channel_entry.get("cell_painting", {})
                if cell_painting_config.get("enabled") and "channel_overrides" in cell_painting_config:
                    for ch_name, ch_cfg in cell_painting_config["channel_overrides"].items():
                        if not isinstance(ch_cfg, dict):
                            continue
                        cp_marker = ch_name.split("_")[-1] if "_" in ch_name else ch_name
                        resolved = None
                        if cp_marker in marker_params:
                            resolved = _resolve_marker_config(cp_marker, experiment, marker_params)
                        channel_configs[ch_name] = resolved if resolved is not None else ch_cfg

    elif isinstance(exp_config, dict):
        # Dict format (e.g., cell painting with metadata sections)
        # Look for segmentation_config in each channel entry (legacy path only)
        for key, value in exp_config.items():
            if isinstance(value, dict) and "segmentation_config" in value:
                channel_configs[key] = value["segmentation_config"]

    # Cache result
    _EXPERIMENT_CONFIGS_CACHE[cache_key] = channel_configs

    return channel_configs


def load_channel_labels(experiment: str, config_path: str = None) -> dict:
    """
    Load channel labels from ops_channel_maps.yaml.

    Returns a mapping of channel_name -> label string (e.g., "lysosome, LysoTracker live-cell dye").
    These labels contain the biological annotation in "{organelle}, {marker}" format
    that parse_channel_label() expects.

    Args:
        experiment: Experiment name (e.g., "ops0113")
        config_path: Path to ops_channel_maps.yaml. If None, uses OpsDataset.channel_maps.

    Returns:
        Dict mapping channel_name -> label string.
        Empty dict if no config found.

    Example:
        >>> load_channel_labels("ops0113")
        {"BF": "Phase", "GFP": "lysosome, LysoTracker live-cell dye", "mCherry": "endosome, VPS35"}
    """
    cache_key = (experiment, config_path)
    if cache_key in _CHANNEL_LABELS_CACHE:
        return _CHANNEL_LABELS_CACHE[cache_key]

    from cyclops_utils.data.experiment import OpsDataset
    import yaml
    from pathlib import Path

    if config_path is None:
        dataset = OpsDataset("dummy")
        config_path = str(dataset.channel_maps)

    if not Path(config_path).exists():
        return {}

    with open(config_path, 'r') as f:
        channel_maps = yaml.safe_load(f)

    from cyclops_utils.data.filesystem import extract_ops_key
    ops_key = extract_ops_key(experiment)

    exp_config = None
    for key in [experiment, ops_key, experiment.lower()]:
        if key and key in channel_maps:
            exp_config = channel_maps[key]
            break

    labels = {}
    if isinstance(exp_config, list):
        for entry in exp_config:
            if isinstance(entry, dict) and "channel_name" in entry and "label" in entry:
                labels[entry["channel_name"]] = entry["label"]

    _CHANNEL_LABELS_CACHE[cache_key] = labels
    return labels


def load_channel_configs_from_exp_config(channel_name: str, exp_config: dict) -> dict:
    """
    Load segmentation config for a specific channel from experiment config dict.

    Args:
        channel_name: Channel name
        exp_config: Experiment config dict (from load_experiment_configs)

    Returns:
        Config dict for this channel, or {} if not found
    """
    return exp_config.get(channel_name, {})


def get_channel_segmentation_config(
    experiment: str,
    channel_name: str,
    config_path: str = None,
) -> dict:
    """
    Get segmentation config overrides for a specific channel in an experiment.

    Args:
        experiment: Experiment name
        channel_name: Channel name
        config_path: Optional path to ops_channel_maps.yaml

    Returns:
        Config dict with overrides, or {} if no overrides exist
    """
    all_configs = load_experiment_configs(experiment, config_path)
    return all_configs.get(channel_name, {})


def get_full_segmentation_config(
    organelle_key: str,
    channel_name: str,
    structure_type: str = None,
    experiment: str = None,
    frangi_params_override: dict = None,
    clahe_params_override: dict = None,
    post_clahe_smoothing_override: float = None,
    marker_params_path: str = None,
    method: str = None,
) -> dict:
    """
    Get complete segmentation config with YAML + runtime overrides applied.

    Resolution order (highest priority last):
    1. Base config from SEGMENTATION_CONFIGS
    2. Experiment YAML overrides (ops_channel_maps.yaml)
    3. Runtime parameter overrides

    Args:
        organelle_key: Organelle identifier (e.g., "GFP", "nucleoli_phase2d")
        channel_name: Channel name
        structure_type: Optional structure type override
        experiment: Optional experiment name for loading YAML configs
        frangi_params_override: Optional runtime Frangi parameter overrides
        clahe_params_override: Optional runtime CLAHE parameter overrides
        post_clahe_smoothing_override: Optional post-CLAHE smoothing sigma

    Returns:
        Complete config dict with keys:
        - detection_params: Frangi or blob parameters
        - clahe_params: CLAHE parameters
        - post_smoothing: Post-CLAHE smoothing sigma
        - input_mask_name: Optional mask name for constrained segmentation
        - detection_method: "frangi" or "blob"
        - is_nucleoli: Boolean flag
    """
    # Load YAML overrides. marker_params_path lets a caller point at an
    # alternate org_seg_params.yaml (e.g. the compare-preview overlay) instead
    # of the experiment's production params.
    yaml_config = {}
    if experiment:
        all_configs = load_experiment_configs(experiment, marker_params_path=marker_params_path)
        yaml_config = all_configs.get(channel_name, {})

    # Determine structure type
    is_nucleoli = "nucleoli" in organelle_key.lower()
    if structure_type is None:
        structure_type = yaml_config.get("structure_type")
        if structure_type is None:
            structure_type = "nucleoli" if is_nucleoli else "tubular"

    # Determine method (YAML override or default from DEFAULT_METHODS)
    method = method or yaml_config.get("method")   # explicit override wins over YAML / DEFAULT_METHODS
    if method is None:
        # Use DEFAULT_METHODS to determine method based on structure + channel type
        channel_type = "labelfree" if channel_name in LABELFREE_CHANNELS else "fluorescent"
        if is_nucleoli:
            # For nucleoli, map to phase2d or focus3d
            if channel_name == "Focus3D":
                channel_type = "focus3d"
            else:
                channel_type = "phase2d"
            method = DEFAULT_METHODS.get(("nucleoli", channel_type), "frangi")
        else:
            # Focus3D has its own vesicular configs (Phase2D stays on labelfree)
            if channel_name == "Focus3D" and structure_type in ("vesicular", "vesicular_dark"):
                channel_type = "focus3d"
            method = DEFAULT_METHODS.get((structure_type, channel_type), "frangi")

    # Get base config
    detection_params = get_segmentation_config(structure_type, channel_name, method)

    # Apply YAML overrides (frangi / blob / threshold param blocks)
    yaml_detection_overrides = (
        yaml_config.get("frangi", {})
        or yaml_config.get("blob", {})
        or yaml_config.get("threshold", {})
    )
    detection_params.update(yaml_detection_overrides)

    # Apply runtime overrides
    if frangi_params_override:
        detection_params.update(frangi_params_override)

    # Build CLAHE params
    clahe_params = get_clahe_config(is_nucleoli=is_nucleoli)
    clahe_params.update(yaml_config.get("clahe", {}))
    if clahe_params_override:
        clahe_params.update(clahe_params_override)

    # Post-smoothing
    post_smoothing = clahe_params.pop("post_smoothing", 0.0)
    if post_clahe_smoothing_override is not None:
        post_smoothing = post_clahe_smoothing_override

    return {
        "detection_params": detection_params,
        "clahe_params": clahe_params,
        "post_smoothing": post_smoothing,
        "input_mask_name": ORGANELLE_MASK_CONFIG.get(organelle_key),
        "detection_method": method,  # Use the determined method, not from detection_params
        "is_nucleoli": is_nucleoli,
    }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_structure_type(structure_type: str | None) -> None:
    """
    Validate structure_type against STRUCTURE_TYPES.

    Args:
        structure_type: Structure type to validate

    Raises:
        ValueError: If structure_type is invalid
    """
    if structure_type is not None and structure_type not in STRUCTURE_TYPES:
        raise ValueError(
            f"Invalid structure_type '{structure_type}'. "
            f"Must be one of {STRUCTURE_TYPES}"
        )


def validate_crop_fraction(crop_fraction: float | None) -> None:
    """
    Validate crop_fraction is in (0, 1] if provided.

    Args:
        crop_fraction: Crop fraction to validate

    Raises:
        ValueError: If crop_fraction is out of range
    """
    if crop_fraction is not None and not 0 < crop_fraction <= 1:
        raise ValueError(
            f"crop_fraction must be between 0 and 1, got {crop_fraction}"
        )
