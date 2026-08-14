"""
Debug Utilities Module
======================

Provides debug mode configuration helpers for organelle segmentation.

This module consolidates debug crop setup logic that was previously duplicated
in the main segmentation functions.

Main functions:
- setup_debug_crop_bbox(): Configure debug crop parameters for testing
- setup_batch_debug_cropping(): Configure debug cropping for batch processing
- run_sweep_mode(): Run parameter sweep for a single channel
"""

import sys
from pathlib import Path
import numpy as np
from cyclops_utils.io.zarr_labels import get_position_shape
from .geometry import calculate_center_crop_bbox


def setup_debug_crop_bbox(
    source_path: Path,
    position: str,
    debug_only: bool = False,
    debug_size: int = 2048,
    debug_crop_fraction: float = None,
    debug_tile_size: int = 1024,
    debug_tile_overlap: int = 128,
    debug_offset_y: int = 28000,
    debug_offset_x: int = 28000,
) -> tuple[tuple | None, bool, dict]:
    """
    Setup debug crop configuration for testing segmentation.

    Supports two modes:
    1. debug_only mode: Process 2x2 tile grid with stitching test
    2. debug_crop_fraction mode: Legacy center crop based on fraction

    Args:
        source_path: Path to zarr store
        position: Position identifier (e.g., "A/1/0")
        debug_only: If True, use tile grid mode
        debug_size: Size of center crop in debug_only mode (pixels)
        debug_crop_fraction: Fraction of image to process (0 < f <= 1)
        debug_tile_size: Tile size for debug_only mode
        debug_tile_overlap: Overlap between tiles in debug_only mode
        debug_offset_y: Y offset from center for debug crop (default: 28000)
        debug_offset_x: X offset from center for debug crop (default: 28000)

    Returns:
        tuple: (crop_bbox, force_tiled, debug_info)
            - crop_bbox: (y_start, y_end, x_start, x_end) or None
            - force_tiled: True if tiling should be forced
            - debug_info: Dict with debug configuration details

    Examples:
        >>> # Debug-only mode with 2x2 tile grid
        >>> bbox, tiled, info = setup_debug_crop_bbox(
        ...     source_path, "A/1/0", debug_only=True, debug_size=2048
        ... )
        >>> info["tile_grid"]  # (2, 2)

        >>> # Legacy crop fraction mode
        >>> bbox, tiled, info = setup_debug_crop_bbox(
        ...     source_path, "A/1/0", debug_crop_fraction=0.1
        ... )
        >>> tiled  # False
    """
    crop_bbox = None
    force_tiled = False
    debug_info = {}

    # Get position shape for calculations
    pos_shape = get_position_shape(source_path, position)
    height, width = pos_shape[3], pos_shape[4]

    if debug_only:
        # Debug-only mode: 2x2 tile grid with stitching test
        # Total processing size = 2 * tile_size - overlap (to get exactly 2x2 tiles)
        debug_crop_size = 2 * debug_tile_size - debug_tile_overlap

        # Apply offset from center to test different regions
        # 113k/4 = ~28k offset to get halfway between center and edge
        y_center = height // 2 + debug_offset_y
        x_center = width // 2 + debug_offset_x
        half_size = debug_crop_size // 2

        y_start = max(0, y_center - half_size)
        y_end = min(height, y_center + half_size)
        x_start = max(0, x_center - half_size)
        x_end = min(width, x_center + half_size)
        crop_bbox = (y_start, y_end, x_start, x_end)

        # Force tiled processing to test stitching
        force_tiled = True

        # Calculate actual tile grid dimensions
        actual_crop_h = y_end - y_start
        actual_crop_w = x_end - x_start
        step = debug_tile_size - debug_tile_overlap
        n_tiles_y = max(1, (actual_crop_h - debug_tile_overlap + step - 1) // step)
        n_tiles_x = max(1, (actual_crop_w - debug_tile_overlap + step - 1) // step)

        debug_info = {
            "mode": "debug_only",
            "full_image_size": (height, width),
            "crop_bbox": crop_bbox,
            "crop_size": (actual_crop_h, actual_crop_w),
            "tile_size": debug_tile_size,
            "tile_overlap": debug_tile_overlap,
            "tile_grid": (n_tiles_y, n_tiles_x),
            "num_tiles": n_tiles_y * n_tiles_x,
        }

        print(f"DEBUG PREVIEW: Processing center crop with TILED STITCHING")
        print(f"  Full image: {height} x {width}")
        print(f"  Crop bbox: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        print(f"  Crop size: {actual_crop_h} x {actual_crop_w}")
        print(f"  Tile config: {debug_tile_size}x{debug_tile_size} with {debug_tile_overlap}px overlap")
        print(f"  Tile grid: {n_tiles_y} x {n_tiles_x} = {n_tiles_y * n_tiles_x} tiles")

    elif debug_crop_fraction is not None:
        # Legacy debug crop mode (fraction-based)
        from .configs import validate_crop_fraction
        validate_crop_fraction(debug_crop_fraction)

        crop_bbox = calculate_center_crop_bbox(pos_shape, debug_crop_fraction)
        y_start, y_end, x_start, x_end = crop_bbox

        debug_info = {
            "mode": "crop_fraction",
            "full_image_size": (height, width),
            "crop_fraction": debug_crop_fraction,
            "crop_bbox": crop_bbox,
            "crop_size": (y_end - y_start, x_end - x_start),
        }

        print(f"DEBUG MODE: Processing center crop")
        print(f"  Crop bbox: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")

    return crop_bbox, force_tiled, debug_info


def setup_batch_debug_cropping(
    source_path: Path,
    position_list: list[str],
    debug_crop_fraction: float = None,
) -> dict:
    """
    Setup debug crop configuration for all positions in batch processing.

    Args:
        source_path: Path to zarr store
        position_list: List of position identifiers
        debug_crop_fraction: Fraction of image to process (0 < f <= 1)

    Returns:
        dict: {position: (y_start, y_end, x_start, x_end)}
            Maps each position to its crop bbox

    Raises:
        ValueError: If debug_crop_fraction is out of range

    Example:
        >>> positions = ["A/1/0", "A/2/0", "A/3/0"]
        >>> crop_map = setup_batch_debug_cropping(
        ...     source_path, positions, debug_crop_fraction=0.1
        ... )
        >>> len(crop_map)  # 3
    """
    if debug_crop_fraction is None:
        return {}

    from .configs import validate_crop_fraction
    validate_crop_fraction(debug_crop_fraction)

    # Get shape from first position to calculate crop
    first_pos_shape = get_position_shape(source_path, position_list[0])
    crop_bbox = calculate_center_crop_bbox(first_pos_shape, debug_crop_fraction)
    y_start, y_end, x_start, x_end = crop_bbox
    crop_height = y_end - y_start
    crop_width = x_end - x_start

    print(f"DEBUG MODE: Processing center crop of each position")
    print(f"  Crop fraction: {debug_crop_fraction} ({debug_crop_fraction * 100:.2f}% area)")
    print(f"  Full image: {first_pos_shape[3]} x {first_pos_shape[4]} pixels")
    print(f"  Crop region: {crop_height} x {crop_width} pixels ({crop_height * crop_width / 1e6:.1f} MP)")
    print(f"  Crop bbox: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")

    # Store same crop bbox for all positions (assuming same size)
    crop_bbox_per_position = {}
    for pos_path in position_list:
        crop_bbox_per_position[pos_path] = crop_bbox

    return crop_bbox_per_position


def run_sweep_mode(
    experiment: str,
    positions: list[str],
    channel: str,
    sweep_var: str,
    sweep_min: float,
    sweep_max: float,
    sweep_n: int,
    sweep_log: bool = False,
    structure_type: str = None,
    no_stitch: bool = False,
) -> int:
    """
    Run parameter sweep for a single channel.

    Runs segmentation multiple times with one parameter varied across a range,
    creating comparison canvases showing how the parameter affects segmentation.

    Args:
        experiment: Resolved experiment name
        positions: List of positions to process
        channel: Single channel name to process
        sweep_var: Parameter to sweep. Options:
            - Frangi: pixel_size_um, alpha, beta, threshold, min_radius_um, max_radius_um, min_object_size
            - CLAHE: clip_limit
            - Blob: overlap (0-1, lower = fewer merged), num_sigma, threshold_mult
        sweep_min: Minimum value for sweep range
        sweep_max: Maximum value for sweep range
        sweep_n: Number of samples in sweep range
        sweep_log: If True, use logarithmic spacing; otherwise linear
        structure_type: Optional structure type ("tubular", "vesicular", or None for auto)
        no_stitch: If True, process entire crop as single tile (faster, more memory)

    Returns:
        Exit code: 0 if all sweeps succeeded, 1 otherwise

    Example:
        >>> exit_code = run_sweep_mode(
        ...     experiment="ops0033_20250429",
        ...     positions=["A/1/0"],
        ...     channel="GFP",
        ...     sweep_var="pixel_size_um",
        ...     sweep_min=0.1,
        ...     sweep_max=0.5,
        ...     sweep_n=5,
        ...     sweep_log=False,
        ... )
    """
    from organelle_profiler.organelle_seg.organelle_segmentation import (
        segment_single_position_channel,
    )
    from organelle_profiler.organelle_seg.visualizations import (
        create_sweep_canvas,
        save_segmentation_params_yaml,
    )
    from cyclops_utils.data.experiment import OpsDataset

    # Determine if sweep_var is CLAHE or Frangi param
    clahe_params_list = ["clip_limit"]
    is_clahe_param = sweep_var in clahe_params_list

    # Generate sweep values
    if sweep_log:
        sweep_values = np.logspace(np.log10(sweep_min), np.log10(sweep_max), num=sweep_n)
    else:
        sweep_values = np.linspace(sweep_min, sweep_max, num=sweep_n)

    print(f"\n{'='*60}")
    print(f"SWEEP MODE - Parameter sweep for {channel}")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Positions: {positions}")
    print(f"Channel: {channel}")
    print(f"Structure type: {structure_type or 'auto (from config)'}")
    print(f"Sweep variable: {sweep_var}")
    print(f"Sweep range: {sweep_min} to {sweep_max} ({sweep_n} samples)")
    print(f"Sweep spacing: {'logarithmic' if sweep_log else 'linear'}")
    print(f"Sweep values: {[f'{v:.4g}' for v in sweep_values]}")
    if no_stitch:
        print(f"Tile mode: SINGLE CHUNK (no tiling, 1920x1920px crop)")
    else:
        print(f"Tile grid: 2x2 (1024px tiles, 128px overlap) -> 1920x1920px crop")
    print(f"{'='*60}\n")

    all_results = []
    for pos in positions:
        for i, sweep_val in enumerate(sweep_values):
            val_str = f"{sweep_val:.4g}"
            print(f"\n--- [{i+1}/{sweep_n}] {pos} / {channel} / {sweep_var}={val_str} ---")

            # Build override params based on sweep variable
            frangi_override = None
            clahe_override = None

            if is_clahe_param:
                clahe_override = {sweep_var: sweep_val}
            else:
                frangi_override = {sweep_var: sweep_val}

            try:
                result = segment_single_position_channel(
                    experiment=experiment,
                    position=pos,
                    channel_key=channel,
                    structure_type=structure_type,
                    frangi_params=frangi_override,
                    clahe_params=clahe_override,
                    debug_only=True,
                    no_stitch=no_stitch,
                )
                # Store sweep value in result for canvas labeling
                result["sweep_var"] = sweep_var
                result["sweep_val"] = sweep_val
                all_results.append((pos, channel, sweep_var, sweep_val, result))

                if result.get("success"):
                    print(f"✓ {sweep_var}={val_str}: {result.get('num_objects', 0)} objects")
                else:
                    print(f"✗ Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"✗ Exception: {e}")
                import traceback
                traceback.print_exc()
                all_results.append((pos, channel, sweep_var, sweep_val, {"success": False, "error": str(e)}))

    # Create sweep canvas for each position
    print(f"\n{'='*60}")
    print(f"CREATING SWEEP CANVASES")
    print(f"{'='*60}")

    # Get output directory
    dataset = OpsDataset(experiment)
    source_path = dataset.store_paths["pheno_assembled_v3"]
    assembly_dir = source_path.parent
    canvas_dir = assembly_dir / "organelle_seg_debug" / f"sweep_{channel.lower()}"
    canvas_dir.mkdir(parents=True, exist_ok=True)

    for pos in positions:
        # Collect all successful results for this position
        pos_results = [
            (sweep_val, r) for p, ch, var, sweep_val, r in all_results
            if p == pos and r.get("success") and r.get("labels") is not None
        ]

        if not pos_results:
            print(f"  No successful results for {pos}, skipping canvas")
            continue

        # Build channel_params dict for YAML export
        channel_params = {}
        for sweep_val, r in pos_results:
            name = f"{channel}_{sweep_var}={sweep_val:.4g}"
            channel_params[name] = {
                'frangi_params': r.get('frangi_params', {}),
                'clahe_params': r.get('clahe_params', {}),
                'structure_type': r.get('structure_type'),
            }

        # Save params YAML
        yaml_path = canvas_dir / f"sweep_params_{pos.replace('/', '_')}_{channel}_{sweep_var}.yaml"
        save_segmentation_params_yaml(
            channel_params=channel_params,
            output_path=yaml_path,
            experiment=experiment,
            position=pos,
            preview_mode=True,
        )

        # Build images list for canvas - crop to 512x512 center region
        # Uses same crop configs as --preview-all mode
        canvas_crop_size = 512

        # Define crop regions: same 4 regions as preview-all
        crop_configs = [
            {"name": "region1", "x_offset": -250, "y_offset": 0, "suffix": ""},
            {"name": "region2", "x_offset": -250, "y_offset": 500, "suffix": "_region2"},
            {"name": "region3", "x_offset": 250, "y_offset": 0, "suffix": "_region3"},
            {"name": "region4", "x_offset": 250, "y_offset": 500, "suffix": "_region4"},
        ]

        for crop_cfg in crop_configs:
            y_extra_offset = crop_cfg["y_offset"]
            x_offset = crop_cfg["x_offset"]
            suffix = crop_cfg["suffix"]

            images = []
            for sweep_val, r in pos_results:
                name = f"{sweep_var}={sweep_val:.4g}"
                labels = r.get('labels')
                raw = r.get('raw')
                vesselness = r.get('vesselness')

                # Crop to 512x512 region with same offsets as preview-all
                if labels is not None:
                    h, w = labels.shape[:2]
                    cy = h // 2 + y_extra_offset
                    cx = w // 2 + x_offset
                    half = canvas_crop_size // 2
                    y1, y2 = max(0, cy - half), min(h, cy + half)
                    x1, x2 = max(0, cx - half), min(w, cx + half)
                    labels = labels[y1:y2, x1:x2]
                    if raw is not None:
                        raw = raw[y1:y2, x1:x2]
                    if vesselness is not None:
                        vesselness = vesselness[y1:y2, x1:x2]

                images.append({
                    'name': name,
                    'labels': labels,
                    'raw': raw,
                    'vesselness': vesselness,
                })

            # Create sweep canvas for this region
            canvas_path = canvas_dir / f"sweep_{pos.replace('/', '_')}_{channel}_{sweep_var}{suffix}.png"
            region_label = f" (Y+{y_extra_offset}px, X{x_offset:+d}px)" if y_extra_offset or x_offset else ""
            create_sweep_canvas(
                images=images,
                output_path=canvas_path,
                title=f"Sweep: {pos}{region_label}",
                sweep_var=sweep_var,
                channel_name=channel,
            )

    # Summary
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*60}")
    successes = sum(1 for _, _, _, _, r in all_results if r.get("success"))
    print(f"Processed: {len(all_results)} configurations")
    print(f"Successful: {successes}")

    if successes > 0:
        print(f"\nResults per {sweep_var} value:")
        for pos, ch, var, sweep_val, result in all_results:
            if result.get("success"):
                tiled_str = " [tiled]" if result.get("tiled") else ""
                print(f"  {pos}/{ch} {sweep_var}={sweep_val:.4g}: {result.get('num_objects', 0)} objects, {result.get('elapsed_time', 0):.1f}s{tiled_str}")
        print(f"\nSweep canvases saved to: {canvas_dir}")
    print(f"{'='*60}\n")

    return 0 if successes == len(all_results) else 1
