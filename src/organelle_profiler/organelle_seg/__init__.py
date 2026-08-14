"""
Organelle Segmentation Package
==============================

This package provides modular organelle segmentation functionality for
large microscopy images using Frangi vesselness filters and LoG blob detection.

Modules:
- configs: Configuration dictionaries and parameter accessors
- naming: Label naming conventions and standardization
- geometry: Geometric utilities for bounding boxes and coordinates
- zarr_io: Zarr I/O helpers for reading/writing labels
- clahe: Memory-efficient tiled CLAHE preprocessing
- metadata: Metadata building for segmentation results
- frangi: FrangiFilter class and compute_frangi_threshold for vesselness detection
- blob_detection: LoG blob detection for punctate structures
- postprocessing: Morphological postprocessing (watershed, opening, closing)
- visualizations: Debug image saving and combined canvas generation
- tiled_processing: Tiled segmentation pipeline
- channel_processor: Channel resolution and configuration
- debug_utils: Debug mode setup helpers
- result_handler: Result processing and error handling
- batch_processing: Dask cluster setup

Main entry points (from organelle_segmentation.py):
- run_organelle_segmentation: Run segmentation on all organelles
- segment_single_position_channel: Segment a single channel at one position
- get_available_channels: List segmentable channels for an experiment
- detect_segmentation_status: Check what has already been segmented
"""

# Configuration
from .configs import (
    STRUCTURE_TYPES,
    LABELFREE_CHANNELS,
    SPECIAL_LABEL_MAP,
    ORGANELLE_MASK_CONFIG,
    # New unified configs
    SEGMENTATION_CONFIGS,
    DEFAULT_METHODS,
    CLAHE_CONFIGS,
    # Simplified helpers
    get_segmentation_config,
    get_clahe_config,
    get_channel_type,
    um_to_sigmas,
    # Experiment-specific config loading
    load_experiment_configs,
    load_channel_configs_from_exp_config,
    get_channel_segmentation_config,
    get_full_segmentation_config,
)

# Label naming utilities (shared functions from cyclops_utils, local functions from .naming)
from cyclops_utils.data.naming import (
    parse_channel_label,
    determine_marker_type,
    build_channel_metadata,
)
from .naming import (
    get_label_name,
    get_output_label_name,
)

# Geometry utilities
from .geometry import (
    calculate_center_crop_bbox,
    get_bbox,
)

# Zarr position shape utility (included with other zarr_io imports below)

# CLAHE preprocessing
from .clahe import _tiled_clahe

# Metadata building and status detection
from .metadata import (
    _build_segmentation_metadata,
    _build_description,
    _determine_processing_params,
    _build_vesselness_metadata,
    detect_segmentation_status,
    get_available_channels,
)

# Zarr I/O
from cyclops_utils.io.zarr_labels import (
    get_position_shape,
    _init_organelle_label_array,
    _update_labels_metadata,
    _write_label_to_tile,
    _check_label_has_data,
)

# Visualizations
from .visualizations import (
    save_debug_image,
    save_debug_jpeg,
    save_debug_overlay,
    create_combined_canvas,
    _save_tiled_debug_images,
    save_segmentation_params_yaml,
)

# Postprocessing (morphological operations)
from .postprocessing import (
    watershed_label,
    postprocess_vesicular_mask,
    postprocess_nucleoli_mask,
    postprocess_tubular_mask,
)

# Import compute_frangi_threshold from frangi module
from .frangi import compute_frangi_threshold

# Frangi filter
from .frangi import FrangiFilter

# Blob detection
from .blob_detection import (
    _segment_blob_log,
    _segment_nucleoli_blob,
    _segment_nucleoli_frangi,
    _segment_nucleoli_in_tile,
)

# Main entry points (imported at runtime to avoid circular imports)
# These are available as:
#   from organelle_profiler.feature_extraction.organelle_seg import run_organelle_segmentation
# Or:
#   from organelle_profiler.organelle_seg.organelle_segmentation import run_organelle_segmentation

__all__ = [
    # Configs
    "STRUCTURE_TYPES",
    "LABELFREE_CHANNELS",
    "SPECIAL_LABEL_MAP",
    "ORGANELLE_MASK_CONFIG",
    # Unified config dictionaries
    "SEGMENTATION_CONFIGS",
    "DEFAULT_METHODS",
    "CLAHE_CONFIGS",
    # Simplified config helper functions
    "get_segmentation_config",
    "get_clahe_config",
    "get_channel_type",
    "um_to_sigmas",
    # Experiment-specific config loading
    "load_experiment_configs",
    "load_channel_configs_from_exp_config",
    "get_channel_segmentation_config",
    "get_full_segmentation_config",
    # Label naming utilities
    "parse_channel_label",
    "get_label_name",
    "get_output_label_name",
    "determine_marker_type",
    "build_channel_metadata",
    # Geometry utilities
    "calculate_center_crop_bbox",
    "get_bbox",
    # Zarr position shape
    "get_position_shape",
    # CLAHE
    "_tiled_clahe",
    # Metadata and status
    "_build_segmentation_metadata",
    "_build_description",
    "_determine_processing_params",
    "_build_vesselness_metadata",
    "detect_segmentation_status",
    "get_available_channels",
    # Zarr I/O
    "_init_organelle_label_array",
    "_update_labels_metadata",
    "_write_label_to_tile",
    "_check_label_has_data",
    # Visualizations
    "save_debug_image",
    "save_debug_jpeg",
    "save_debug_overlay",
    "create_combined_canvas",
    "_save_tiled_debug_images",
    "save_segmentation_params_yaml",
    # Postprocessing
    "compute_frangi_threshold",
    "watershed_label",
    "postprocess_vesicular_mask",
    "postprocess_nucleoli_mask",
    "postprocess_tubular_mask",
    # Frangi
    "FrangiFilter",
    # Blob detection
    "_segment_blob_log",
    "_segment_nucleoli_blob",
    "_segment_nucleoli_frangi",
    "_segment_nucleoli_in_tile",
]
