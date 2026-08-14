"""
Organelle Segmentation Module
=============================

This module provides functions for segmenting organelles (nucleoli,
mitochondria, ER, etc.) in large microscopy images using Frangi filters.

Quick Start - Single Position Segmentation
-------------------------------------------

# Run from the ops_process_main directory with the appropriate conda environment

# 1. Segment a single position with Frangi (e.g., mitochondria, ER, fibers)
python -c "
from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
segment_single_position_channel(
    experiment='ops0049_20250626',
    position='A/3/0',
    channel_key='GFP',  # or 'mCherry', 'Focus3D', etc.
    use_clahe=True,
    debug_crop_fraction=0.1,  # Optional: test on 10% center crop first
)
"

# 2. List available channels for an experiment
python -c "
from organelle_profiler.organelle_seg.organelle_segmentation import get_available_channels
channels = get_available_channels('ops0049_20250626')
print('Available channels:', channels)
"

SLURM Batch Processing
----------------------

# Submit all positions for an experiment (recommended for production)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \\
    --experiment ops0049_20250626

# Submit specific positions/channels
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \\
    --experiment ops0049_20250626 \\
    --positions A/3/0 A/4/0 \\
    --channels GFP mCherry

# Preview what would be submitted (dry run)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \\
    --experiment ops0049_20250626 --dry-run

Segmentation Methods
--------------------
- **Frangi**: Used for all organelle segmentation (tubular, vesicular, nucleoli)
- **LoG Blob**: Alternative for vesicular/nucleoli detection

The method uses tiled processing for memory efficiency on large images (104k x 104k pixels).
"""

import os
import time
from itertools import combinations_with_replacement
from pathlib import Path
import shutil
import sys
import ctypes
import tempfile
import logging
import tifffile

logging.getLogger("distributed.scheduler").setLevel(logging.WARNING)
logging.getLogger("distributed").setLevel(logging.WARNING)

# ----------------------------------------------------
import numpy as np
from dask.distributed import Client, LocalCluster, as_completed
from iohub import open_ome_zarr
from skimage.measure import label
from tqdm import tqdm
import zarr
import dask
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import notify_step, versioned_function

# Import from refactored modules
from .configs import (
    STRUCTURE_TYPES,
    DEFAULT_METHODS,
    # Unified config getter
    get_full_segmentation_config,
)
# NOTE: Imports from naming, geometry, and zarr_io are done locally where needed
from .metadata import (
    _build_segmentation_metadata,
    _determine_processing_params,
    detect_segmentation_status,
    get_available_channels,
)
from cyclops_utils.io.zarr_labels import (
    _init_organelle_label_array,
    _update_labels_metadata,
)
from .tiled_processing import (
    segment_position_frangi,
)


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================
# Note: All processing functions have been moved to separate modules:
#       - configs.py: Configuration dictionaries and validators
#       - label_utils.py: Label naming and coordinate utilities
#       - clahe.py: CLAHE preprocessing
#       - metadata.py: Metadata building and channel index utilities
#       - zarr_io.py: Zarr I/O helpers
#       - frangi.py: FrangiFilter class
#       - blob_detection.py: LoG blob detection
#       - tiled_processing.py: Tiled segmentation pipeline
#       - channel_processor.py: Channel resolution and configuration
#       - debug_utils.py: Debug mode setup helpers
#       - result_handler.py: Result processing and error handling
#       - batch_processing.py: Dask cluster setup
#       See imports at top of file.


@notify_step(
    step_message="Started position-level organelle segmentation",
    success_message="Finished position-level organelle segmentation",
)
@versioned_function("v3.0-position-based")
def run_organelle_segmentation(
    experiment: str,
    frangi_params: dict = None,
    debug_crop_fraction: float = None,
    frangi_postprocess: bool = False,
    use_clahe: bool = True,
    clahe_params: dict = {"clip_limit": 0.03, "kernel_size": (256, 256)},
    post_clahe_smoothing_sigma: float = 0.0,
):
    """
    Perform organelle segmentation at the position level (entire stitched image).

    Memory requirements (per position):
    - Input: ~41 GB (single channel, ~105k x 105k float32)
    - Output: ~41 GB (int32 segmentation mask)
    - Total: ~100-150 GB RAM minimum per position

    This function loads entire positions (~105k x 105k pixels) into memory and
    runs segmentation on the full image at once. Results are saved to the v3
    zarr store under labels/ with 512x512 chunking to match convert_v3.py.

    Parameters
    ----------
    experiment : str
        The experiment name
    frangi_params : dict, optional
        Parameters for the Frangi filter.
    debug_crop_fraction : float, optional
        If set, only process a center crop of each position for quick debugging.
        Value should be between 0 and 1 (e.g., 0.01 for 1% of image, 0.1 for 10%).
        The crop is taken from the center of each position.
        Example: 0.01 on a 100k x 100k image processes ~1000 x 1000 center region.
    frangi_postprocess : bool, optional
        If True, applies post-processing steps to the Frangi binary mask.
    use_clahe : bool, optional
        If True, applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to the raw image before Frangi filtering to enhance local contrast.
    post_clahe_smoothing_sigma : float, optional
        The sigma for Gaussian smoothing applied after CLAHE to reduce noise.
        Set to 0 to disable. Default is 0.0 (no smoothing, matching sweep script).
    """
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths["pheno_assembled_v3"]
    if not source_path.exists():
        raise FileNotFoundError(
            f"v3 phenotyping store not found at {source_path}. Ensure v3 conversion is complete."
        )

    # --- Get channels directly from the zarr v3 store ---
    with open_ome_zarr(source_path, mode="r") as store:
        position_list = [p for p, _ in store.positions()]
        channel_names = store.channel_names
    print(f"Found {len(position_list)} positions to process.")
    print(f"Available channels: {channel_names}")

    # No tile splitting - process entire positions at once
    # Memory requirement: ~100-150 GB RAM per position
    print("Processing entire positions at once (no tile splitting)")

    # --- Build channel processing map using unified channel processor ---
    from .channel_processor import build_channel_processing_map
    channel_processing_map = build_channel_processing_map(
        channel_names=channel_names,
        include_nucleoli=True,
        experiment=experiment,
    )

    # --- Debug Mode: Center Crop ---
    from .debug_utils import setup_batch_debug_cropping
    crop_bbox_per_position = setup_batch_debug_cropping(
        source_path=source_path,
        position_list=position_list,
        debug_crop_fraction=debug_crop_fraction,
    )

    if not channel_processing_map:
        print("\nWarning: No channels found for segmentation.")
        return

    print(f"Processing channels: {list(channel_processing_map.keys())}")

    # --- Dask Setup ---
    # For position-level processing, each position requires ~100-150 GB RAM
    # We use only 1 worker to avoid OOM errors, and process positions sequentially
    import dask
    dask.config.set({
        "distributed.diagnostics.progressbar": False,
        # Increase memory thresholds to prevent premature worker restarts
        "distributed.worker.memory.target": 0.90,  # Start spilling at 90%
        "distributed.worker.memory.spill": 0.95,   # Spill to disk at 95%
        "distributed.worker.memory.pause": 0.98,   # Pause at 98%
        "distributed.worker.memory.terminate": 0.99,  # Terminate at 99%
    })
    use_gpu = True
    # Force 1 worker for position-level processing - each position needs >100GB RAM
    # NumPy/BLAS parallelism is controlled by OMP_NUM_THREADS env var
    num_workers = 1

    client = None
    cluster = None
    try:
        from .batch_processing import setup_dask_cluster
        client, cluster = setup_dask_cluster(num_workers)

        nuclei_objects_name = None  # Will store the name of the nuclei segmentation output

        # --- Process each channel based on its configuration ---
        total_start_time = time.time()
        for i, (channel_key, ch_info) in enumerate(channel_processing_map.items(), start=1):
            channel_start_time = time.time()
            method = ch_info["method"]
            organelle_key = ch_info.get("organelle_key", channel_key)
            # For most channels, the source channel is the channel_key itself
            # But nucleoli uses Phase channel as source
            channel_name = ch_info.get("source_channel", channel_key)

            print(f"\n{'='*60}")
            print(f"[{i}/{len(channel_processing_map)}] Processing '{channel_key}' on channel '{channel_name}'")
            print(f"{'='*60}")

            futures = []
            segmenter_type = method

            # All channels use Frangi/blob-based segmentation
            print(f"Using {method} filter for '{channel_key}'")

            # Default frangi params
            local_frangi_params = frangi_params if frangi_params else {
                "min_radius_um": 0.1,
                "max_radius_um": 1.5,
                "alpha": 4.0,
                "beta": 0.5,
            }

            # Use default CLAHE settings for Frangi channels
            clahe_params = {"clip_limit": 0.03, "kernel_size": (256, 256)}

            print(f"Using Frangi parameters: {local_frangi_params}")
            print(f"Using CLAHE: {use_clahe}")
            print(f"Using post-CLAHE smoothing sigma: {post_clahe_smoothing_sigma}")

            # Submit one job per position using client.submit for explicit argument control
            futures = []
            for pos_path in position_list:
                crop_bbox = crop_bbox_per_position.get(pos_path)  # None if not in debug mode
                future = client.submit(
                    segment_position_frangi,
                    pos_path,
                    source_zarr_path=str(source_path),
                    channel_to_segment=channel_name,
                    organelle_name=organelle_key,
                    frangi_params=local_frangi_params,
                    use_gpu=True,
                    frangi_postprocess=frangi_postprocess,
                    use_clahe=use_clahe,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    clahe_params=clahe_params,
                    crop_bbox=crop_bbox,
                )
                futures.append(future)

            if not futures:
                print(f"No segmentation task was launched for '{channel_key}'. Skipping result collection.")
                continue

            # Build metadata and initialize label arrays for all positions
            from .metadata import build_and_init_labels
            seg_metadata, objects_name = build_and_init_labels(
                source_path=source_path,
                position_list=position_list,
                organelle_key=organelle_key,
                source_channel=channel_name,
                channel_key=channel_key,
                channel_names=channel_names,
                ch_info=ch_info,
                structure_type=None,  # Legacy batch path doesn't use structure_type
                frangi_params=local_frangi_params,
                clahe_params=clahe_params,
                post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                frangi_postprocess=frangi_postprocess,
                experiment=experiment,
            )

            if objects_name is None:
                print(f"Skipping '{channel_key}' - no label mapping")
                continue

            # Collect results and write to Zarr
            from .result_handler import write_segmentation_to_zarr
            progress_bar = tqdm(
                as_completed(futures),
                total=len(position_list),
                desc=f"Segmenting {channel_key}",
            )

            # Track which positions we've written to for each label
            written_positions = set()

            for future in progress_bar:
                pos_path, vesselness, binary, objects, scale, result_crop_bbox = future.result()

                if objects is not None:
                    # Write position result - either full or cropped region
                    write_segmentation_to_zarr(
                        store_path=source_path,
                        position=pos_path,
                        objects_name=objects_name,
                        objects=objects,
                        crop_bbox=result_crop_bbox,
                    )
                    written_positions.add(pos_path)

            channel_elapsed = time.time() - channel_start_time
            print(f"\n[TIMING] Channel '{channel_key}' complete: {len(written_positions)} positions in {channel_elapsed:.1f}s ({channel_elapsed/60:.1f}min)")
            if len(written_positions) > 0:
                print(f"[TIMING] Average per position: {channel_elapsed/len(written_positions):.1f}s ({channel_elapsed/len(written_positions)/60:.1f}min)")

        total_elapsed = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"[TIMING] All channels complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min, {total_elapsed/3600:.2f}h)")
        print(f"{'='*60}")

    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()

    print("\n--- Position-level organelle segmentation complete. ---")


def segment_single_position_channel(
    experiment: str,
    position: str,
    channel_key: str,
    structure_type: str = None,
    frangi_params: dict = None,
    debug_crop_fraction: float = None,
    frangi_postprocess: bool = False,
    use_clahe: bool = True,
    clahe_params: dict = None,
    post_clahe_smoothing_sigma: float = None,
    debug_only: bool = False,
    debug_size: int = 2048,
    no_stitch: bool = False,
    marker_params_path: str = None,
    method: str = None,
):
    """
    Segment a single position-channel combination.

    This function is designed to be called as a standalone SLURM job,
    allowing parallel processing of different position-channel combinations
    across separate SLURM jobs.

    Parameters
    ----------
    experiment : str
        The experiment name
    position : str
        Position path (e.g., "A/1/0", "A/2/0", "A/3/0")
    channel_key : str
        Channel key to segment. Can be:
        - A zarr channel name (e.g., "GFP", "mCherry", "Phase2D", "nuclei_prediction")
        - A special organelle key (e.g., "nucleoli" which uses Phase + nuclei mask)
    structure_type : str, optional
        For Frangi segmentation: "tubular" or "vesicular" (default: None)
        When set, uses structure-specific parameters via get_frangi_params().
        When None, defaults are determined by get_full_segmentation_config().
    frangi_params : dict, optional
        Parameters for the Frangi filter. If provided, overrides structure_type params.
    debug_crop_fraction : float, optional
        If set, only process a center crop (e.g., 0.01 for 1% area)
    frangi_postprocess : bool, optional
        If True, applies post-processing to Frangi binary mask
    use_clahe : bool, optional
        If True, applies CLAHE preprocessing
    clahe_params : dict, optional
        Parameters for CLAHE. If None, uses get_clahe_params() to determine defaults.
    post_clahe_smoothing_sigma : float, optional
        Sigma for Gaussian smoothing after CLAHE. If None, uses default from CLAHE config.
    debug_only : bool, optional
        If True, runs in preview mode: processes a small center crop and saves debug
        images without writing any data to zarr. Useful for testing parameters.
    debug_size : int, optional
        Size of center crop in pixels when debug_only=True (default: 2048).
        The crop will be debug_size x debug_size pixels from the image center.
    no_stitch : bool, optional
        If True, disables tile stitching in debug_only mode. Processes the entire
        crop as a single tile (faster but uses more memory). Default: False.

    Returns
    -------
    dict
        Result dictionary with keys:
        - success: bool
        - position: str
        - channel: str
        - output_label: str (name of the output label)
        - num_objects: int (number of segmented objects)
        - elapsed_time: float (processing time in seconds)
        - error: str (if success=False)
        - debug_dir: str (path to debug images, only if debug_only=True)
    """
    import time
    start_time = time.time()

    # Import error result helper
    from .result_handler import create_error_result

    # Validate structure_type if provided
    from .configs import validate_structure_type
    try:
        validate_structure_type(structure_type)
    except ValueError as e:
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=str(e)
        )

    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths["pheno_assembled_v3"]

    if not source_path.exists():
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=f"v3 phenotyping store not found at {source_path}"
        )

    # Get channel info from zarr store
    with open_ome_zarr(source_path, mode="r") as store:
        channel_names = store.channel_names
        position_list = [p for p, _ in store.positions()]

    if position not in position_list:
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=f"Position '{position}' not found in store. Available: {position_list}"
        )

    # Resolve channel processing info using unified channel processor
    from .channel_processor import resolve_single_channel_info
    ch_info = resolve_single_channel_info(channel_key, channel_names, experiment=experiment)

    if ch_info is None:
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=f"Channel '{channel_key}' not found or should be skipped. Available: {channel_names + ['nucleoli']}"
        )

    method = method or ch_info["method"]   # explicit override wins over the channel-map-resolved method
    organelle_key = ch_info.get("organelle_key", channel_key)
    source_channel = ch_info.get("source_channel", channel_key)

    # Debug-only mode header
    if debug_only:
        print(f"\n{'='*60}")
        print(f"DEBUG PREVIEW MODE - No data will be written to zarr")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")

    print(f"Processing position '{position}', channel '{channel_key}'")
    print(f"  Method: {method}")
    print(f"  Source channel: {source_channel}")
    print(f"  Organelle key: {organelle_key}")
    if structure_type:
        print(f"  Structure type: {structure_type}")
    if debug_only:
        print(f"  Debug size: {debug_size}x{debug_size} pixels")
    print(f"{'='*60}\n")

    # Setup debug crop configuration using unified debug utils
    from .debug_utils import setup_debug_crop_bbox

    # Tile settings - kept the same for consistent crop size (1920x1920)
    # The crop formula is: 2 * tile_size - overlap = 2 * 1024 - 128 = 1920
    debug_tile_size = 1024  # Tile size for debug preview
    debug_tile_overlap = 128  # Overlap between tiles

    try:
        crop_bbox, force_tiled, debug_info = setup_debug_crop_bbox(
            source_path=source_path,
            position=position,
            debug_only=debug_only,
            debug_size=debug_size,
            debug_crop_fraction=debug_crop_fraction,
            debug_tile_size=debug_tile_size,
            debug_tile_overlap=debug_tile_overlap,
        )
        # Override force_tiled when no_stitch is requested
        # This processes the 1920x1920 crop as a single chunk without tiling overhead
        if no_stitch:
            force_tiled = False
            print(f"NO-STITCH MODE: Processing 1920x1920 crop as single chunk (no tiling)")
    except ValueError as e:
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=str(e)
        )

    # Define output label name (include structure_type for dual Frangi segmentation)
    from .naming import get_output_label_name
    from .metadata import get_channel_index

    objects_name = get_output_label_name(organelle_key, source_channel, structure_type)
    print(f"Output label name: '{objects_name}'")

    # Get channel index for metadata
    channel_index = get_channel_index(channel_names, source_channel)

    # Determine processing parameters for metadata (mirrors the logic below)
    detection_params, clahe_metadata, actual_method = _determine_processing_params(
        organelle_key=organelle_key,
        source_channel=source_channel,
        structure_type=structure_type,
        ch_info=ch_info,
        frangi_params=frangi_params,
        clahe_params=clahe_params,
        post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
    )

    # Use channel_label from ch_info (populated from ops_channel_maps.yaml)
    # Falls back to channel_key if no YAML label available
    channel_label = ch_info.get("channel_label", channel_key)

    # Build metadata with all processing parameters
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

    # Initialize output array (skip in debug_only mode)
    if not debug_only:
        from cyclops_utils.io.zarr_labels import get_position_shape, _init_organelle_label_array, _update_labels_metadata
        pos_shape = get_position_shape(source_path, position)
        _init_organelle_label_array(source_path, position, objects_name, pos_shape, dtype=np.int32)
        _update_labels_metadata(source_path, position, objects_name, metadata=seg_metadata)

    # Run segmentation based on method
    result = None
    try:
        # Get unified segmentation config (handles nucleoli, vesicular blob, regular frangi)
        seg_config = get_full_segmentation_config(
            organelle_key=organelle_key,
            channel_name=source_channel,
            structure_type=structure_type,
            experiment=experiment,
            frangi_params_override=frangi_params,
            clahe_params_override=clahe_params,
            post_clahe_smoothing_override=post_clahe_smoothing_sigma,
            marker_params_path=marker_params_path,
            method=method,
        )

        local_frangi_params = seg_config["detection_params"]
        local_clahe_params = seg_config["clahe_params"]
        local_smoothing = seg_config["post_smoothing"]
        input_mask_name = seg_config.get("input_mask_name")  # None for regular channels
        detection_method = seg_config["detection_method"]   # "frangi" or "blob"
        is_nucleoli = seg_config.get("is_nucleoli", False)

        # Log configuration
        print(f"  Detection method: {detection_method}")
        print(f"  Detection params: {local_frangi_params}")
        print(f"  CLAHE params: {local_clahe_params}, smoothing={local_smoothing}")
        if input_mask_name:
            print(f"  Input mask: {input_mask_name}")

        # Single unified call to segment_position_frangi
        # The downstream function handles blob vs frangi routing based on the method params
        result = segment_position_frangi(
            pos_path=position,
            source_zarr_path=str(source_path),
            channel_to_segment=source_channel,
            organelle_name=organelle_key,
            frangi_params=local_frangi_params,
            use_gpu=True,
            frangi_postprocess=frangi_postprocess,
            use_clahe=True if is_nucleoli else use_clahe,  # Always use CLAHE for nucleoli
            post_clahe_smoothing_sigma=local_smoothing,
            debug_output_path=None,
            clahe_params=local_clahe_params,
            crop_bbox=crop_bbox,
            input_mask_name=input_mask_name,  # Unified mask passing
            structure_type=structure_type,
            force_tiled=force_tiled,
            tile_size=debug_tile_size if force_tiled else 4096,
            tile_overlap=debug_tile_overlap if force_tiled else 256,
            # Route to appropriate detection method
            nucleoli_method=detection_method if is_nucleoli else None,
            vesicular_method=detection_method if not is_nucleoli and detection_method == "blob" else None,
            save_vesselness=False,  # Don't save vesselness by default (even in debug_only mode)
            preview_mode=debug_only,
        )

        # Process result using unified result handler
        from .result_handler import process_segmentation_result
        return process_segmentation_result(
            result=result,
            source_path=source_path,
            position=position,
            objects_name=objects_name,
            source_channel=source_channel,
            channel_names=channel_names,
            channel_key=channel_key,
            debug_only=debug_only,
            force_tiled=force_tiled,
            start_time=start_time,
            local_frangi_params=local_frangi_params,
            local_clahe_params=local_clahe_params,
            structure_type=structure_type,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_result(
            position=position,
            channel=channel_key,
            error_message=str(e)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run organelle segmentation")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--position", "-p", type=str, help="Position path (e.g., A/1/0)")
    parser.add_argument("--channel", "-c", type=str, help="Channel key to segment")
    parser.add_argument("--debug-crop", type=float, default=None, help="Debug crop fraction (e.g., 0.01 for 1%%)")
    parser.add_argument("--list-channels", action="store_true", help="List available channels and exit")
    parser.add_argument("--status", action="store_true", help="Show detailed segmentation status and exit")

    args = parser.parse_args()

    if args.status:
        # Show detailed segmentation status
        status = detect_segmentation_status(args.experiment)
        print(f"\n{'='*60}")
        print(f"Segmentation Status: {args.experiment}")
        print(f"{'='*60}\n")

        print(f"Positions: {len(status['positions'])}")
        print(f"Channels: {status['channel_names']}\n")

        print("Channel -> Label mapping:")
        for ch, label in status['channel_to_label'].items():
            print(f"  {ch} -> {label}")
        print()

        if status['labels_with_data']:
            print("LABELS WITH DATA (will skip):")
            for label, desc in status['labels_with_data'].items():
                print(f"  ✓ {label}: {desc}")
            print()

        if status['labels_empty']:
            print("EMPTY LABELS (will overwrite):")
            for label, desc in status['labels_empty'].items():
                print(f"  ⚠ {label}: {desc}")
            print()

        if status['labels_missing']:
            print("MISSING LABELS (will create):")
            for label, desc in status['labels_missing'].items():
                print(f"  + {label}: {desc}")
            print()

        print(f"{'='*60}\n")

    elif args.list_channels:
        # List available channels (interactive mode shows status)
        info = get_available_channels(args.experiment, interactive=True)
        if info.get("cancelled"):
            pass  # Already printed cancellation message
        else:
            print(f"\nExperiment: {args.experiment}")
            print(f"\nPositions ({len(info['positions'])}):")
            for pos in info['positions']:
                print(f"  {pos}")
            print(f"\nChannels to process ({len(info['channels'])}):")
            for ch in info['channels']:
                method = info['channel_methods'].get(ch, "unknown")
                print(f"  {ch} ({method})")
            if info['skipped_channels']:
                print(f"\nSkipped channels ({len(info['skipped_channels'])}):")
                for ch in info['skipped_channels']:
                    print(f"  {ch} (already exists)")
    elif args.position and args.channel:
        # Run single position-channel segmentation
        result = segment_single_position_channel(
            experiment=args.experiment,
            position=args.position,
            channel_key=args.channel,
            debug_crop_fraction=args.debug_crop,
        )
        if result["success"]:
            print(f"\nSuccess! Segmented {result['num_objects']} objects")
        else:
            print(f"\nFailed: {result['error']}")
    else:
        # Run full segmentation on all positions/channels
        run_organelle_segmentation(
            experiment=args.experiment,
            debug_crop_fraction=args.debug_crop,
        )
