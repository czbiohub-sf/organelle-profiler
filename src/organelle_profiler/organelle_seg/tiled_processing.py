"""
Tiled Processing for Organelle Segmentation
============================================

This module provides functions for tiled processing of large microscopy images
using Frangi vesselness filter or LoG blob detection.

The tiled approach:
1. Divides large images into overlapping tiles
2. Processes tiles in parallel using joblib workers
3. Stitches results with Union-Find label merging

Key functions:
- _process_single_frangi_tile: Process a single tile (worker function)
- _stitch_tiled_labels_pass2: Stitch tiles with Union-Find merging
- segment_position_frangi_tiled: Full tiled segmentation pipeline
- segment_position_frangi: Entry point (wraps tiled pipeline)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.ndimage as scipy_ndi
from iohub import open_ome_zarr
from skimage.filters import frangi as skimage_frangi
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from cyclops_utils.hpc.resource_manager import get_optimal_workers
from organelle_profiler.organelle_seg.visualizations import (
    _save_tiled_debug_images,
)
from organelle_profiler.organelle_seg.frangi import (
    compute_frangi_threshold,
)
from organelle_profiler.organelle_seg.postprocessing import (
    postprocess_vesicular_mask,
    postprocess_nucleoli_mask,
    postprocess_tubular_mask,
    watershed_label,
)

# CuPy 14 needs CUDA_PATH set for runtime kernel compilation (NVRTC reads
# headers from <CUDA_PATH>/include). Pass 1 uses pre-built cucim/cupy
# kernels so it runs without it, but Pass 2's JIT-compiled elementwise
# kernels (e.g. `cp.where(int32, int32, int32)`) hit
# `assemble_cupy_compiler_options` which raises if CUDA_PATH is unset.
# SLURM workers don't inherit a shell-set CUDA_PATH and the system has
# no /usr/local/cuda, so point CuPy at the venv-bundled nvidia/cuda_nvcc
# wheel (it ships bin/, include/, nvvm/ — exactly what CuPy needs).
if "CUDA_PATH" not in os.environ:
    try:
        import importlib.util as _iu
        _spec = _iu.find_spec("nvidia.cuda_nvcc")
        if _spec is not None and _spec.submodule_search_locations:
            os.environ["CUDA_PATH"] = _spec.submodule_search_locations[0]
    except Exception:
        pass

# Optional GPU imports — loaded lazily so CPU-only hosts aren't affected.
# Imports succeeding is NOT enough: on the login node cupy imports fine
# but no CUDA device is present, so every cp.asarray() raises
# cudaErrorNoDevice. Probe the device count up front and force the CPU
# fallback when there's no usable GPU. Otherwise the GPU-pipelined path
# silently produces 0 segmentations.
try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cp_ndi
    from cucim.skimage.exposure import equalize_adapthist as _cu_equalize_adapthist
    from cucim.skimage.filters import frangi as _cu_frangi
    try:
        _device_count = _cp.cuda.runtime.getDeviceCount()
    except Exception:
        _device_count = 0
    if _device_count > 0:
        _GPU_AVAILABLE = True
    else:
        _cp = None
        _cp_ndi = None
        _cu_equalize_adapthist = None
        _cu_frangi = None
        _GPU_AVAILABLE = False
except Exception as _gpu_import_err:  # pragma: no cover
    _cp = None
    _cp_ndi = None
    _cu_equalize_adapthist = None
    _cu_frangi = None
    _GPU_AVAILABLE = False


# Simple accumulator used by the GPU worker to report per-phase wall time
# across all tiles. Cleared and printed around Pass 1 by the orchestrator.
_GPU_PHASE_TIMERS: dict[str, float] = {}


def _reset_gpu_phase_timers() -> None:
    _GPU_PHASE_TIMERS.clear()


def _bump_gpu_phase_timer(name: str, seconds: float) -> None:
    _GPU_PHASE_TIMERS[name] = _GPU_PHASE_TIMERS.get(name, 0.0) + seconds


# -----------------------------------------------------------------------------
# Detailed per-phase timing — opt-in via ORG_SEG_DETAILED_TIMING=1. Host-side
# timers measure Python/dispatch wall time in the main loop; CUDA events
# measure GPU-side time without blocking the host.
# -----------------------------------------------------------------------------
_DETAILED_HOST_TIMERS: dict[str, float] = {}
_DETAILED_GPU_MS: dict[str, float] = {}
_DETAILED_TILES_SEEN: int = 0


def _optimized_on() -> bool:
    """Master fast-path toggle. ORG_SEG_OPTIMIZED=1 (default) enables the
    full GPU-pipelined + in-memory + GPU-paint path. Set to 0 to fall
    back to the legacy defaults (single spawn worker, CPU paint, zarr
    unstitched handoff, etc.). Individual ORG_SEG_* flags still override
    the master on a case-by-case basis when explicitly set.
    """
    return os.environ.get("ORG_SEG_OPTIMIZED", "1") == "1"


def _fast_default(optimized_value: str, legacy_value: str = "0") -> str:
    """Resolve a default for a fast-path flag based on ORG_SEG_OPTIMIZED.
    Used as the second arg to ``os.environ.get``: explicit env vars still
    win via the normal get() precedence; this only shapes the fallback.
    """
    return optimized_value if _optimized_on() else legacy_value


def _detailed_timing_enabled() -> bool:
    return os.environ.get("ORG_SEG_DETAILED_TIMING", "0") == "1"


def _reset_detailed_timers() -> None:
    _DETAILED_HOST_TIMERS.clear()
    _DETAILED_GPU_MS.clear()
    global _DETAILED_TILES_SEEN
    _DETAILED_TILES_SEEN = 0


def _bump_detailed_host(name: str, seconds: float) -> None:
    _DETAILED_HOST_TIMERS[name] = _DETAILED_HOST_TIMERS.get(name, 0.0) + seconds


def _bump_detailed_gpu_ms(name: str, ms: float) -> None:
    _DETAILED_GPU_MS[name] = _DETAILED_GPU_MS.get(name, 0.0) + ms


def _record_detailed_tile() -> None:
    global _DETAILED_TILES_SEEN
    _DETAILED_TILES_SEEN += 1


def _print_detailed_timers(pass1_wall_sec: float) -> None:
    if not _DETAILED_HOST_TIMERS and not _DETAILED_GPU_MS:
        return
    n = max(1, _DETAILED_TILES_SEEN)
    pw_ms = pass1_wall_sec * 1000
    wall_per_tile_ms = pw_ms / n
    print(f"  [DETAILED TIMING] per-tile average over {n} tiles (wall {wall_per_tile_ms:.2f} ms/tile):")
    print(f"    Host-side (main thread in _run_pass1_gpu_pipelined):")
    for k, total in sorted(_DETAILED_HOST_TIMERS.items(), key=lambda kv: -kv[1]):
        per = (total * 1000) / n
        share = 100 * per / wall_per_tile_ms if wall_per_tile_ms > 0 else 0.0
        print(f"      {k:<24} {per:7.2f} ms/tile  ({total:6.1f}s total, {share:5.1f}% of wall)")
    print(f"    GPU-side (CUDA events inside _compute_tile_batch_on_gpu_streams):")
    for k, total_ms in sorted(_DETAILED_GPU_MS.items(), key=lambda kv: -kv[1]):
        per = total_ms / n
        share = 100 * per / wall_per_tile_ms if wall_per_tile_ms > 0 else 0.0
        print(f"      {k:<24} {per:7.2f} ms/tile  ({total_ms/1000:6.1f}s total, {share:5.1f}% of wall)")
    # Summary: GPU-side sum vs wall — if GPU<<wall, main thread is blocking on
    # something other than GPU (Python, IPC, pool contention).
    gpu_sum_per_tile = sum(_DETAILED_GPU_MS.values()) / n
    host_sum_per_tile = sum(_DETAILED_HOST_TIMERS.values()) * 1000 / n
    print(f"    Summary: GPU total {gpu_sum_per_tile:.2f} ms/tile, Host total {host_sum_per_tile:.2f} ms/tile, wall {wall_per_tile_ms:.2f} ms/tile")
    print(f"      → If host+GPU > wall, they overlap; if host ≈ wall and GPU < host, we're CPU-bound.")


def _print_gpu_phase_timers(total_wall_sec: float) -> None:
    if not _GPU_PHASE_TIMERS:
        return
    print("  [GPU timing] per-phase total across Pass 1 tiles:")
    for name, t in sorted(_GPU_PHASE_TIMERS.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * t / total_wall_sec if total_wall_sec > 0 else 0.0
        print(f"    {name:20s} {t:8.2f}s  ({pct:5.1f}% of Pass 1 wall)")

from .configs import (
    um_to_sigmas,
)
from .naming import (
    get_output_label_name,
)
from .metadata import (
    _build_vesselness_metadata,
    _build_segmentation_metadata,
)
from cyclops_utils.io.zarr_labels import (
    _update_labels_metadata,
)
from .blob_detection import (
    _segment_blob_log,
)


def _process_single_frangi_tile(
    tile_info: dict,
    source_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str = None,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    output_label_name: str = None,
    save_vesselness: bool = False,
    write_to_zarr: bool = False,
    tile_overlap: int = 256,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    output_zarr_path: str = None,
) -> dict:
    """
    Process a single tile with Frangi filter or blob detection.

    This is a standalone worker function designed to be called by joblib workers
    in parallel. Each worker loads the tile, runs Frangi filter (or LoG blob
    detection), and writes ONLY the core (non-overlapping) region to zarr.

    Core Region Strategy:
    - Each tile processes the full tile_size (with overlap for context)
    - But writes ONLY its core region = center part excluding overlap margins
    - First/last tiles in each dimension include the edge margin
    - This ensures each pixel is written by exactly ONE tile (no race condition)

    Args:
        tile_info: Dict with tile coordinates (tile_idx, ty, tx, y_start_tile, x_start_tile,
                   y_end_tile, x_end_tile, src_y_start, src_y_end, src_x_start, src_x_end)
        source_zarr_path: Path to source zarr
        pos_path: Position path like "A/1/0"
        channel_index: Index of channel to segment
        frangi_params: Parameters for Frangi filter (or blob params for LoG)
        pixel_resolution: Dict with Z, Y, X resolution in um
        use_clahe: Whether to apply CLAHE
        clahe_params: CLAHE parameters
        post_clahe_smoothing_sigma: Post-CLAHE smoothing sigma
        frangi_postprocess: Whether to apply postprocessing
        input_mask_name: Optional mask name to constrain segmentation
        nucleoli_method: For nucleoli: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicles: "blob" for LoG, "frangi" for Frangi
        output_label_name: Name for output labels (required if write_to_zarr=True)
        save_vesselness: Whether to also save vesselness map
        write_to_zarr: If True, write results directly to zarr with locking
        tile_overlap: Overlap between adjacent tiles in pixels
        n_tiles_y: Total number of tiles in Y dimension
        n_tiles_x: Total number of tiles in X dimension
        output_zarr_path: Path to write output (defaults to source_zarr_path if None).
            Used in preview mode to write to temp zarr instead of original.

    Returns:
        Dict with tile_info, vesselness, binary_mask, and labels arrays
        (or minimal dict if write_to_zarr=True to save memory)
    """
    try:
        tile_idx = tile_info["tile_idx"]
        ty, tx = tile_info["ty"], tile_info["tx"]
        src_y_start = tile_info["src_y_start"]
        src_y_end = tile_info["src_y_end"]
        src_x_start = tile_info["src_x_start"]
        src_x_end = tile_info["src_x_end"]

        # Load tile from zarr
        input_mask_tile = None
        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            source_pos = ds[pos_path]
            image_array = source_pos["0"]
            # Load tile: [T, C, Z, Y, X] -> squeeze to (Y, X) for 2D
            tile_data = np.squeeze(
                np.asarray(image_array[0, channel_index, :, src_y_start:src_y_end, src_x_start:src_x_end])
            )

            # Load input mask tile if specified (for nucleoli segmentation)
            if input_mask_name:
                labels_group = source_pos.zgroup.get("labels", None)
                if labels_group is not None and input_mask_name in labels_group:
                    mask_array = labels_group[input_mask_name]["0"]
                    input_mask_tile = np.squeeze(
                        np.asarray(mask_array[0, 0, :, src_y_start:src_y_end, src_x_start:src_x_end])
                    )

        # Apply input mask BEFORE other processing (for nucleoli segmentation)
        # Note: We do NOT erode the input mask here - erosion should be applied to
        # the final labels instead (Frangi detects edges, so eroding the mask doesn't help)
        if input_mask_tile is not None:
            # Ensure mask matches tile_data shape (edge tiles may have 1px difference)
            if input_mask_tile.shape != tile_data.shape:
                # Crop or pad mask to match tile_data
                mask_h, mask_w = input_mask_tile.shape
                tile_h, tile_w = tile_data.shape
                if mask_h != tile_h or mask_w != tile_w:
                    # Create new mask matching tile_data shape
                    new_mask = np.zeros(tile_data.shape, dtype=input_mask_tile.dtype)
                    copy_h = min(mask_h, tile_h)
                    copy_w = min(mask_w, tile_w)
                    new_mask[:copy_h, :copy_w] = input_mask_tile[:copy_h, :copy_w]
                    input_mask_tile = new_mask
            binary_mask_input = input_mask_tile > 0
            tile_data = tile_data.astype(np.float32)
            tile_data[~binary_mask_input] = 0.0

        # Apply CLAHE if needed (no tiling - tile is already small enough)
        # IMPORTANT: Match org_seg_old2.py EXACTLY - normalize each tile to [0,1], apply CLAHE, scale to uint16
        if use_clahe:
            if clahe_params is None:
                clahe_params = {"clip_limit": 0.03}

            clip_limit = clahe_params.get("clip_limit", 0.03)
            kernel_size = clahe_params.get("kernel_size", None)

            original_dtype = tile_data.dtype
            original_min, original_max = None, None

            if not np.issubdtype(original_dtype, np.integer):
                original_min, original_max = np.min(tile_data), np.max(tile_data)

            # Normalize to [0, 1] for CLAHE (matches sweep script exactly)
            tile_min, tile_max = tile_data.min(), tile_data.max()
            if tile_max > tile_min:
                tile_normalized = (tile_data - tile_min) / (tile_max - tile_min)
            else:
                tile_normalized = np.zeros_like(tile_data, dtype=np.float64)

            # Apply CLAHE directly (tile is small enough)
            # Keep as [0,1] float for Frangi (matching sweep script exactly)
            tile_data = equalize_adapthist(
                tile_normalized,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
            )

            # Optional Gaussian smoothing (on [0,1] data)
            if post_clahe_smoothing_sigma > 0:
                tile_data = scipy_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)

        # Choose segmentation method
        pixel_res_um = pixel_resolution.get("y", pixel_resolution.get("x", 0.108))

        detection_method_cfg = frangi_params.get("detection_method")
        if detection_method_cfg == "threshold":
            # Intensity-threshold segmentation (infer-subc style): threshold the
            # raw (CLAHE'd) intensity directly instead of a Frangi/LoG response.
            from .thresholding import apply_intensity_threshold
            from .postprocessing import (
                topology_preserving_thinning,
                filter_objects_by_physical_size,
            )

            thr_method = frangi_params.get("threshold_method", "otsu")
            binary_mask = apply_intensity_threshold(
                tile_data,
                method=thr_method,
                threshold_factor=frangi_params.get("threshold_factor", 1.0),
                mo_global_method=frangi_params.get("mo_global_method", "triangle"),
                mo_local_adjust=frangi_params.get("mo_local_adjust", 0.98),
                mo_object_min_area_px=frangi_params.get("mo_object_min_area_px", 100),
                multiotsu_classes=frangi_params.get("multiotsu_classes", 3),
                multiotsu_level=frangi_params.get("multiotsu_level", 0),
            )
            # Constrain to input mask if provided (e.g. nucleoli within nuclei)
            if input_mask_tile is not None:
                binary_mask = binary_mask & (input_mask_tile > 0)

            if frangi_params.get("fill_holes", True):
                binary_mask = scipy_ndi.binary_fill_holes(binary_mask)

            if frangi_params.get("thinning", False) and binary_mask.ndim == 2 and binary_mask.any():
                binary_mask = topology_preserving_thinning(
                    binary_mask,
                    min_thickness=frangi_params.get("thin_min_thickness", 1.6),
                    thin_dist=frangi_params.get("thin_dist", 1),
                )

            # Label: watershed (discrete round objects) or connected components
            min_object_size = frangi_params.get("min_object_size", 0)
            if frangi_params.get("watershed", False):
                labeled_mask = watershed_label(
                    binary_mask,
                    min_distance=frangi_params.get("watershed_min_distance", 3),
                    min_object_size=min_object_size,
                    compactness=frangi_params.get("watershed_compactness", 0.0),
                    erosion_iterations=frangi_params.get("watershed_erosion", 0),
                    min_peak_distance=frangi_params.get("watershed_min_peak", 1.0),
                    h_maxima=frangi_params.get("watershed_h_maxima", 0.0),
                )
            else:
                footprint = scipy_ndi.generate_binary_structure(binary_mask.ndim, 1)
                labeled_mask, _ = scipy_ndi.label(binary_mask, structure=footprint)
                if min_object_size > 0 and labeled_mask.max() > 0:
                    label_ids, counts = np.unique(labeled_mask, return_counts=True)
                    small_labels = label_ids[(label_ids > 0) & (counts < min_object_size)]
                    if len(small_labels) > 0:
                        labeled_mask[np.isin(labeled_mask, small_labels)] = 0
                        labeled_mask, _ = scipy_ndi.label(labeled_mask > 0, structure=footprint)
                labeled_mask = labeled_mask.astype(np.int32)

            # Optional physical-size (µm) filtering on top of the pixel filter
            min_um2 = frangi_params.get("min_object_size_um2")
            max_um2 = frangi_params.get("max_object_size_um2")
            if (min_um2 is not None) or (max_um2 is not None):
                labeled_mask = filter_objects_by_physical_size(
                    labeled_mask,
                    pixel_size_um=pixel_resolution.get("X", pixel_res_um),
                    min_size_um2=min_um2,
                    max_size_um2=max_um2,
                    z_size_um=pixel_resolution.get("Z"),
                )

            vesselness_map = np.zeros_like(tile_data, dtype=np.float32)
            binary_mask = (labeled_mask > 0).astype(np.uint8)
        elif nucleoli_method == "blob" and input_mask_tile is not None:
            # Use LoG blob detection for nucleoli (with nuclear mask)
            labeled_mask = _segment_blob_log(
                tile_data=tile_data,
                pixel_resolution_um=pixel_res_um,
                blob_params=frangi_params,  # frangi_params contains blob params when nucleoli_method="blob"
                mask=input_mask_tile,
            )
            # Create placeholder vesselness and binary for consistency
            vesselness_map = np.zeros_like(tile_data, dtype=np.float32)
            binary_mask = (labeled_mask > 0).astype(np.uint8)
        elif vesicular_method == "blob":
            # Use LoG blob detection for vesicles (no mask - detect everywhere)
            # Check if this is vesicular_dark (invert image for dark blobs)
            invert_image = frangi_params.get("black_ridges", False)
            labeled_mask = _segment_blob_log(
                tile_data=tile_data,
                pixel_resolution_um=pixel_res_um,
                blob_params=frangi_params,  # frangi_params contains blob params when vesicular_method="blob"
                mask=None,
                invert=invert_image,
            )
            # Create placeholder vesselness and binary for consistency
            vesselness_map = np.zeros_like(tile_data, dtype=np.float32)
            binary_mask = (labeled_mask > 0).astype(np.uint8)
        else:
            # Run Frangi filter using skimage.frangi (matching sweep script exactly)
            # tile_data is already [0,1] from CLAHE - use directly like sweep script
            # Get Frangi params - use um_to_sigmas matching sweep script exactly
            min_r = frangi_params.get("min_radius_um", 0.2)
            max_r = frangi_params.get("max_radius_um", 1.5)
            num_sigmas = frangi_params.get("num_sigma", 5)
            black_ridges = frangi_params.get("black_ridges", False)
            pixel_size_um = pixel_resolution.get("X", 0.1625)

            # Convert radius to sigmas using the sweep script formula (radius/pixel_size directly)
            sigmas = um_to_sigmas(min_r, max_r, pixel_size_um, num_sigmas=num_sigmas)

            # Use skimage.frangi directly (matching sweep script behavior exactly)
            # tile_data is already [0,1] from CLAHE output
            vesselness_map = skimage_frangi(
                tile_data,
                sigmas=sigmas,
                black_ridges=black_ridges
            )

            # Threshold and label using fixed or dynamic thresholding
            if np.any(vesselness_map > 0):
                # Check for fixed threshold first, otherwise use dynamic
                fixed_threshold = frangi_params.get("threshold", 0.01)
                if fixed_threshold is not None:
                    # Use fixed threshold directly
                    threshold = fixed_threshold
                else:
                    # Use dynamic thresholding with threshold_mult
                    threshold_mult = frangi_params.get("threshold_mult", 0.01)
                    threshold = compute_frangi_threshold(vesselness_map, threshold_mult=threshold_mult, xp=np)
                binary_mask = vesselness_map > threshold

                # Use frangi_postprocess parameter (or fall back to config if not explicitly set)
                # Priority: frangi_postprocess param > frangi_params["postprocess"] > False
                do_postprocess = frangi_postprocess or frangi_params.get("postprocess", False)
                if do_postprocess:
                    is_3d = binary_mask.ndim == 3

                    # Check if this is nucleoli segmentation (needs aggressive rounding)
                    is_nucleoli = nucleoli_method is not None

                    # Get structure type from params (set by get_frangi_params)
                    structure_type = frangi_params.get("structure_type", None)
                    is_vesicular = structure_type in ("vesicular", "vesicular_dark")
                    is_tubular = structure_type == "tubular"

                    if is_nucleoli and not is_3d:
                        # Use aggressive nucleoli post-processing for large round structures
                        # Extract postprocess params from config
                        pp_min_size = frangi_params.get("min_object_size", 20)
                        pp_do_opening = frangi_params.get("postprocess_opening", True)
                        pp_opening_radius = frangi_params.get("postprocess_opening_radius", 1)
                        pp_do_closing = frangi_params.get("postprocess_closing", True)
                        pp_closing_radius = frangi_params.get("postprocess_closing_radius", 3)
                        print(f"    [POSTPROCESS] Applying nucleoli postprocess (opening={pp_do_opening}/r={pp_opening_radius}, closing={pp_do_closing}/r={pp_closing_radius})")
                        binary_mask = postprocess_nucleoli_mask(
                            binary_mask,
                            min_size=pp_min_size,
                            do_opening=pp_do_opening,
                            opening_radius=pp_opening_radius,
                            do_closing=pp_do_closing,
                            closing_radius=pp_closing_radius,
                        )
                    elif is_tubular:
                        # Use helper function for tubular post-processing
                        # Extract postprocess params from config
                        pp_min_size = frangi_params.get("min_object_size", 5)
                        pp_do_opening = frangi_params.get("postprocess_opening", True)
                        pp_opening_size = frangi_params.get("postprocess_opening_size", 2)
                        pp_fill_holes = frangi_params.get("postprocess_fill_holes", False)
                        print(f"    [POSTPROCESS] Applying tubular postprocess (opening={pp_do_opening}/size={pp_opening_size}, fill_holes={pp_fill_holes}, min_size={pp_min_size})")
                        binary_mask = postprocess_tubular_mask(
                            binary_mask,
                            min_size=pp_min_size,
                            do_opening=pp_do_opening,
                            opening_size=pp_opening_size,
                            do_fill_holes=pp_fill_holes,
                        )
                    elif is_vesicular and not is_3d:
                        # Use helper function for vesicular post-processing
                        print(f"    [POSTPROCESS] Applying vesicular postprocess (gentle smoothing)")
                        binary_mask = postprocess_vesicular_mask(binary_mask)
                    else:
                        # Fallback: no specific postprocess, just skip
                        print(f"    [POSTPROCESS] No structure-specific postprocess (structure_type={structure_type})")

                # Use watershed labeling if watershed=True (for discrete round objects)
                # Otherwise fall back to connected components
                use_watershed = frangi_params.get("watershed", False)
                watershed_min_dist = frangi_params.get("watershed_min_distance", 1)
                min_object_size = frangi_params.get("min_object_size", 0)
                watershed_compactness = frangi_params.get("watershed_compactness", 0.0)
                watershed_erosion = frangi_params.get("watershed_erosion", 0)
                watershed_min_peak = frangi_params.get("watershed_min_peak", 1.0)
                watershed_h_maxima = frangi_params.get("watershed_h_maxima", 0.0)
                if use_watershed:
                    labeled_mask = watershed_label(
                        binary_mask,
                        min_distance=watershed_min_dist,
                        min_object_size=min_object_size,
                        compactness=watershed_compactness,
                        erosion_iterations=watershed_erosion,
                        min_peak_distance=watershed_min_peak,
                        h_maxima=watershed_h_maxima,
                    )
                    num_labels = int(labeled_mask.max())
                else:
                    footprint = scipy_ndi.generate_binary_structure(binary_mask.ndim, 1)
                    labeled_mask, num_labels = scipy_ndi.label(binary_mask, structure=footprint)

                    # Apply min_object_size filtering (independent of watershed)
                    if min_object_size > 0 and labeled_mask.max() > 0:
                        label_ids, counts = np.unique(labeled_mask, return_counts=True)
                        small_labels = label_ids[(label_ids > 0) & (counts < min_object_size)]
                        if len(small_labels) > 0:
                            labeled_mask[np.isin(labeled_mask, small_labels)] = 0
                            # Relabel to ensure continuous IDs
                            labeled_mask, num_labels = scipy_ndi.label(labeled_mask > 0, structure=footprint)
                            labeled_mask = labeled_mask.astype(np.int32)
            else:
                binary_mask = np.zeros_like(vesselness_map, dtype=bool)
                labeled_mask = np.zeros_like(vesselness_map, dtype=np.int32)

        # Prepare tile coordinates and dimensions
        y_start = tile_info["y_start_tile"]
        x_start = tile_info["x_start_tile"]
        y_end = tile_info["y_end_tile"]
        x_end = tile_info["x_end_tile"]

        # Get actual tile dimensions (may be smaller at edges)
        actual_height = y_end - y_start
        actual_width = x_end - x_start
        tile_labels = labeled_mask[:actual_height, :actual_width].astype(np.int32)
        tile_vesselness = vesselness_map[:actual_height, :actual_width].astype(np.float32)

        # Calculate CORE region (non-overlapping) to avoid race conditions
        # Each tile writes EXACTLY its step x step region, which maps to one zarr chunk
        # The tile data includes overlap for processing context, but only core is written
        #
        # Tile layout (1D example with tile_size=4096, overlap=256, step=3840):
        #   Tile 0: global [0:4096], writes core [0:3840]
        #   Tile 1: global [3840:7936], writes core [3840:7680]
        #   Tile 2: global [7680:11776], writes core [7680:11520]
        #
        # Within each tile's local coordinates:
        #   - Tile 0: local [0:3840] -> global [0:3840]
        #   - Tile 1: local [0:3840] -> global [3840:7680] (tile starts at global 3840)
        #   - Edge tiles may write less than step to stay within image bounds
        tile_size = tile_info["tile_size"]
        step = tile_size - tile_overlap

        # Global coordinates: each tile writes [ty*step : (ty+1)*step, tx*step : (tx+1)*step]
        # Clamped to image dimensions to avoid writing beyond bounds
        core_y_start_global = ty * step
        core_y_end_global = min((ty + 1) * step, actual_height + y_start)  # Clamp to actual image
        core_x_start_global = tx * step
        core_x_end_global = min((tx + 1) * step, actual_width + x_start)

        # Local coordinates: offset from tile's starting position
        # Tile starts at global [y_start, x_start], so local = global - start
        core_y_start_local = core_y_start_global - y_start
        core_y_end_local = core_y_end_global - y_start
        core_x_start_local = core_x_start_global - x_start
        core_x_end_local = core_x_end_global - x_start

        # Clamp local coordinates to tile bounds
        core_y_start_local = max(0, core_y_start_local)
        core_y_end_local = min(actual_height, core_y_end_local)
        core_x_start_local = max(0, core_x_start_local)
        core_x_end_local = min(actual_width, core_x_end_local)

        # Extract core region from tile data
        core_labels = tile_labels[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
        core_vesselness = tile_vesselness[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]

        # Write ONLY core region to zarr - each tile writes to unique chunk (parallel-safe)
        # Use output_zarr_path if provided (for preview mode), otherwise source_zarr_path
        write_path = output_zarr_path if output_zarr_path else source_zarr_path

        if output_zarr_path:
            # Preview mode: use raw zarr (not iohub) since temp zarr doesn't have OME-Zarr metadata
            import zarr
            store = zarr.open(write_path, mode="r+")
            if output_label_name:
                temp_name = f"{output_label_name}_unstitched"
                labels_arr = store[pos_path]["labels"][temp_name]["0"]
                labels_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_labels

            if save_vesselness and output_label_name:
                vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                vesselness_arr = store[pos_path]["labels"][temp_vesselness_name]["0"]
                vesselness_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_vesselness
        else:
            # Normal mode: use iohub's open_ome_zarr
            with open_ome_zarr(write_path, mode="r+") as ds:
                # Write labels
                if output_label_name:
                    temp_name = f"{output_label_name}_unstitched"
                    labels_arr = ds[pos_path].zgroup["labels"][temp_name]["0"]
                    labels_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_labels

                # Write vesselness if requested
                if save_vesselness and output_label_name:
                    vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                    temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                    vesselness_arr = ds[pos_path].zgroup["labels"][temp_vesselness_name]["0"]
                    vesselness_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_vesselness

        # Return minimal dict - only store center tile for debug
        is_center = tile_info.get("is_center", False)
        return {
            "tile_info": tile_info,
            "success": True,
            "vesselness": tile_vesselness if is_center else None,
            "labels": tile_labels.copy() if is_center else None,
        }

    except Exception as e:
        print(f"Error processing Frangi tile {tile_info.get('tile_idx', '?')}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "tile_info": tile_info,
            "success": False,
        }


def _process_single_frangi_tile_gpu(
    tile_info: dict,
    source_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str = None,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    output_label_name: str = None,
    save_vesselness: bool = False,
    tile_overlap: int = 256,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    output_zarr_path: str = None,
) -> dict:
    """GPU version of _process_single_frangi_tile for the frangi tubular path.

    Reads the tile on CPU, runs CLAHE + Gaussian + Frangi + threshold + morphology +
    connected-components on GPU via cupy/cucim, transfers labels back to CPU, and
    writes to zarr. Per-phase wall times are accumulated in ``_GPU_PHASE_TIMERS``
    so the orchestrator can print a breakdown at the end of Pass 1.

    LoG blob paths (nucleoli/vesicular blob) fall back to the CPU implementation
    since they require ``skimage.feature.blob_log`` which has no cucim equivalent.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("GPU path requested but cupy/cucim not importable")

    cp = _cp
    cp_ndi = _cp_ndi

    # LoG blob paths and the intensity-threshold method fall back to CPU
    # (cucim has no blob_log, and the threshold recipe is CPU-only).
    if (
        (nucleoli_method == "blob")
        or (vesicular_method == "blob")
        or (frangi_params.get("detection_method") == "threshold")
    ):
        return _process_single_frangi_tile(
            tile_info=tile_info, source_zarr_path=source_zarr_path, pos_path=pos_path,
            channel_index=channel_index, frangi_params=frangi_params,
            pixel_resolution=pixel_resolution, use_clahe=use_clahe,
            clahe_params=clahe_params, post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
            frangi_postprocess=frangi_postprocess, input_mask_name=input_mask_name,
            nucleoli_method=nucleoli_method, vesicular_method=vesicular_method,
            output_label_name=output_label_name, save_vesselness=save_vesselness,
            tile_overlap=tile_overlap, n_tiles_y=n_tiles_y, n_tiles_x=n_tiles_x,
            output_zarr_path=output_zarr_path,
        )

    def _phase_time(name: str):
        """Context manager that syncs the GPU and adds to the shared phase timer."""
        class _T:
            def __enter__(self_):
                self_.t0 = time.monotonic()
                return self_
            def __exit__(self_, *a):
                cp.cuda.Stream.null.synchronize()
                _bump_gpu_phase_timer(name, time.monotonic() - self_.t0)
        return _T()

    try:
        tile_idx = tile_info["tile_idx"]
        ty, tx = tile_info["ty"], tile_info["tx"]
        src_y_start = tile_info["src_y_start"]
        src_y_end = tile_info["src_y_end"]
        src_x_start = tile_info["src_x_start"]
        src_x_end = tile_info["src_x_end"]

        # 1) Zarr read (CPU) + optional mask read
        with _phase_time("zarr_read"):
            input_mask_tile = None
            with open_ome_zarr(source_zarr_path, mode="r") as ds:
                source_pos = ds[pos_path]
                image_array = source_pos["0"]
                tile_data_np = np.squeeze(
                    np.asarray(image_array[0, channel_index, :, src_y_start:src_y_end, src_x_start:src_x_end])
                )
                if input_mask_name:
                    labels_group = source_pos.zgroup.get("labels", None)
                    if labels_group is not None and input_mask_name in labels_group:
                        mask_array = labels_group[input_mask_name]["0"]
                        input_mask_tile = np.squeeze(
                            np.asarray(mask_array[0, 0, :, src_y_start:src_y_end, src_x_start:src_x_end])
                        )

        # 2) Transfer to GPU
        with _phase_time("h2d_transfer"):
            tile_data = cp.asarray(tile_data_np, dtype=cp.float32)
            if input_mask_tile is not None:
                if input_mask_tile.shape != tile_data.shape:
                    mh, mw = input_mask_tile.shape
                    th, tw = tile_data.shape
                    new_mask = np.zeros(tile_data.shape, dtype=input_mask_tile.dtype)
                    new_mask[: min(mh, th), : min(mw, tw)] = input_mask_tile[: min(mh, th), : min(mw, tw)]
                    input_mask_tile = new_mask
                mask_gpu = cp.asarray(input_mask_tile) > 0
                tile_data = cp.where(mask_gpu, tile_data, cp.float32(0.0))

        # 3) CLAHE
        if use_clahe:
            with _phase_time("clahe"):
                if clahe_params is None:
                    clahe_params = {"clip_limit": 0.03}
                clip_limit = clahe_params.get("clip_limit", 0.03)
                kernel_size = clahe_params.get("kernel_size", None)
                tmin = cp.min(tile_data)
                tmax = cp.max(tile_data)
                rng = tmax - tmin
                if float(rng) > 0:
                    tile_norm = (tile_data - tmin) / rng
                else:
                    tile_norm = cp.zeros_like(tile_data, dtype=cp.float32)
                tile_data = _cu_equalize_adapthist(
                    tile_norm, kernel_size=kernel_size, clip_limit=clip_limit,
                ).astype(cp.float32)

        # 4) Post-CLAHE smoothing
        if post_clahe_smoothing_sigma and post_clahe_smoothing_sigma > 0:
            with _phase_time("post_clahe_smooth"):
                tile_data = cp_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)

        # 5) Frangi
        with _phase_time("frangi"):
            min_r = frangi_params.get("min_radius_um", 0.2)
            max_r = frangi_params.get("max_radius_um", 1.5)
            num_sigmas = frangi_params.get("num_sigma", 5)
            black_ridges = frangi_params.get("black_ridges", False)
            pixel_size_um = pixel_resolution.get("X", 0.1625)
            sigmas = um_to_sigmas(min_r, max_r, pixel_size_um, num_sigmas=num_sigmas)
            vesselness_map = _cu_frangi(
                tile_data, sigmas=sigmas, black_ridges=black_ridges,
            ).astype(cp.float32)

        # 6) Threshold
        with _phase_time("threshold"):
            if cp.any(vesselness_map > 0):
                fixed_threshold = frangi_params.get("threshold", 0.01)
                if fixed_threshold is not None:
                    threshold = float(fixed_threshold)
                else:
                    threshold_mult = frangi_params.get("threshold_mult", 0.01)
                    threshold = compute_frangi_threshold(vesselness_map, threshold_mult=threshold_mult, xp=cp)
                binary_mask = vesselness_map > threshold
            else:
                binary_mask = cp.zeros_like(vesselness_map, dtype=cp.bool_)

        # 7) Postprocess (tubular-focused; other structure types pass through)
        do_postprocess = frangi_postprocess or frangi_params.get("postprocess", False)
        structure_type = frangi_params.get("structure_type", None)
        is_tubular = structure_type == "tubular"
        if do_postprocess and is_tubular and bool(cp.any(binary_mask)):
            with _phase_time("postprocess"):
                pp_min_size = frangi_params.get("min_object_size", 5)
                pp_do_opening = frangi_params.get("postprocess_opening", True)
                pp_opening_size = frangi_params.get("postprocess_opening_size", 2)
                pp_fill_holes = frangi_params.get("postprocess_fill_holes", False)
                if pp_fill_holes:
                    binary_mask = cp_ndi.binary_fill_holes(binary_mask)
                if pp_do_opening and pp_opening_size > 0:
                    k = cp.ones((pp_opening_size,) * binary_mask.ndim, dtype=cp.bool_)
                    binary_mask = cp_ndi.binary_opening(binary_mask, structure=k)
                footprint = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
                lab_tmp, _ = cp_ndi.label(binary_mask, structure=footprint)
                if int(lab_tmp.max()) > 0 and pp_min_size > 0:
                    areas = cp.bincount(lab_tmp.ravel())
                    # labels indexed from 1; background=0
                    small = cp.where(areas[1:] < pp_min_size)[0] + 1
                    if int(small.size) > 0:
                        drop = cp.isin(lab_tmp, small)
                        binary_mask = cp.where(drop, cp.bool_(False), lab_tmp > 0)

        # 8) Label
        with _phase_time("label"):
            if bool(cp.any(binary_mask)):
                footprint = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
                labeled_mask, num_labels = cp_ndi.label(binary_mask, structure=footprint)
                min_object_size = frangi_params.get("min_object_size", 0)
                if min_object_size > 0 and int(labeled_mask.max()) > 0:
                    areas = cp.bincount(labeled_mask.ravel())
                    small = cp.where(areas[1:] < min_object_size)[0] + 1
                    if int(small.size) > 0:
                        drop = cp.isin(labeled_mask, small)
                        labeled_mask = cp.where(drop, cp.int32(0), labeled_mask.astype(cp.int32))
                        labeled_mask, _ = cp_ndi.label(labeled_mask > 0, structure=footprint)
                labeled_mask = labeled_mask.astype(cp.int32)
            else:
                labeled_mask = cp.zeros_like(vesselness_map, dtype=cp.int32)

        # 9) Extract core region and transfer back to CPU
        y_start = tile_info["y_start_tile"]; x_start = tile_info["x_start_tile"]
        y_end = tile_info["y_end_tile"]; x_end = tile_info["x_end_tile"]
        actual_height = y_end - y_start; actual_width = x_end - x_start
        tile_size_full = tile_info["tile_size"]
        step = tile_size_full - tile_overlap
        core_y_start_global = ty * step
        core_y_end_global = min((ty + 1) * step, actual_height + y_start)
        core_x_start_global = tx * step
        core_x_end_global = min((tx + 1) * step, actual_width + x_start)
        core_y_start_local = max(0, core_y_start_global - y_start)
        core_y_end_local = min(actual_height, core_y_end_global - y_start)
        core_x_start_local = max(0, core_x_start_global - x_start)
        core_x_end_local = min(actual_width, core_x_end_global - x_start)

        with _phase_time("d2h_transfer"):
            core_labels = cp.asnumpy(
                labeled_mask[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
            )
            core_vesselness = None
            if save_vesselness:
                core_vesselness = cp.asnumpy(
                    vesselness_map[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
                )

        # 10) Zarr write (CPU)
        with _phase_time("zarr_write"):
            write_path = output_zarr_path if output_zarr_path else source_zarr_path
            if output_zarr_path:
                import zarr
                store = zarr.open(write_path, mode="r+")
                if output_label_name:
                    temp_name = f"{output_label_name}_unstitched"
                    labels_arr = store[pos_path]["labels"][temp_name]["0"]
                    labels_arr[0, 0, 0,
                               core_y_start_global:core_y_end_global,
                               core_x_start_global:core_x_end_global] = core_labels
                if save_vesselness and output_label_name and core_vesselness is not None:
                    vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                    temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                    vesselness_arr = store[pos_path]["labels"][temp_vesselness_name]["0"]
                    vesselness_arr[0, 0, 0,
                                   core_y_start_global:core_y_end_global,
                                   core_x_start_global:core_x_end_global] = core_vesselness
            else:
                with open_ome_zarr(write_path, mode="r+") as ds:
                    if output_label_name:
                        temp_name = f"{output_label_name}_unstitched"
                        labels_arr = ds[pos_path].zgroup["labels"][temp_name]["0"]
                        labels_arr[0, 0, 0,
                                   core_y_start_global:core_y_end_global,
                                   core_x_start_global:core_x_end_global] = core_labels
                    if save_vesselness and output_label_name and core_vesselness is not None:
                        vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                        temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                        vesselness_arr = ds[pos_path].zgroup["labels"][temp_vesselness_name]["0"]
                        vesselness_arr[0, 0, 0,
                                       core_y_start_global:core_y_end_global,
                                       core_x_start_global:core_x_end_global] = core_vesselness

        is_center = tile_info.get("is_center", False)
        result = {
            "tile_info": tile_info,
            "success": True,
            "vesselness": cp.asnumpy(vesselness_map).astype(np.float32) if is_center else None,
            "labels": cp.asnumpy(labeled_mask).astype(np.int32) if is_center else None,
        }

        # Free GPU memory between tiles to keep the pool from growing unbounded.
        del tile_data, vesselness_map, binary_mask, labeled_mask
        cp.get_default_memory_pool().free_all_blocks()
        return result

    except Exception as e:
        print(f"Error processing GPU Frangi tile {tile_info.get('tile_idx', '?')}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "tile_info": tile_info,
            "success": False,
        }


# ---------------------------------------------------------------------------
# Pipelined GPU path — split read / compute / write so that reads and writes
# can overlap with GPU compute across tiles.
# ---------------------------------------------------------------------------

def _read_tile_for_gpu(
    source_zarr_path: str,
    pos_path: str,
    channel_index: int,
    tile_info: dict,
    input_mask_name: str | None,
    source_image_array=None,
    input_mask_array=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one tile (and optional input mask) from zarr. Pure CPU work; runs in
    a reader thread. Returns (tile_data_np, input_mask_np_or_None).

    If ``source_image_array`` is supplied, it's used directly (avoids the
    Python-heavy ``open_ome_zarr`` path per read, which holds the GIL and
    serializes threads). Zarr arrays are thread-safe for reads.
    """
    src_y_start = tile_info["src_y_start"]
    src_y_end = tile_info["src_y_end"]
    src_x_start = tile_info["src_x_start"]
    src_x_end = tile_info["src_x_end"]

    if source_image_array is not None:
        tile_data_np = np.squeeze(np.asarray(
            source_image_array[0, channel_index, :, src_y_start:src_y_end, src_x_start:src_x_end]
        ))
        input_mask_tile = None
        if input_mask_name and input_mask_array is not None:
            input_mask_tile = np.squeeze(np.asarray(
                input_mask_array[0, 0, :, src_y_start:src_y_end, src_x_start:src_x_end]
            ))
        return tile_data_np, input_mask_tile

    # Backward-compatible fallback path — opens zarr per call.
    with open_ome_zarr(source_zarr_path, mode="r") as ds:
        source_pos = ds[pos_path]
        image_array = source_pos["0"]
        tile_data_np = np.squeeze(np.asarray(
            image_array[0, channel_index, :, src_y_start:src_y_end, src_x_start:src_x_end]
        ))
        input_mask_tile = None
        if input_mask_name:
            labels_group = source_pos.zgroup.get("labels", None)
            if labels_group is not None and input_mask_name in labels_group:
                mask_array = labels_group[input_mask_name]["0"]
                input_mask_tile = np.squeeze(np.asarray(
                    mask_array[0, 0, :, src_y_start:src_y_end, src_x_start:src_x_end]
                ))
    return tile_data_np, input_mask_tile


def _compute_tile_on_gpu(
    tile_data_np: np.ndarray,
    input_mask_np: np.ndarray | None,
    tile_info: dict,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    save_vesselness: bool,
    tile_overlap: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """GPU compute for a pre-read tile. Accumulates per-phase timing in the
    module-level ``_GPU_PHASE_TIMERS``. Returns
    (core_labels_np, core_vesselness_np_or_None, debug_labels_np_or_None,
    debug_vesselness_np_or_None) — last two are populated only for the center tile.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("GPU compute called but cupy/cucim not importable")
    cp = _cp
    cp_ndi = _cp_ndi

    def _phase_time(name: str):
        class _T:
            def __enter__(self_):
                self_.t0 = time.monotonic()
                return self_
            def __exit__(self_, *a):
                cp.cuda.Stream.null.synchronize()
                _bump_gpu_phase_timer(name, time.monotonic() - self_.t0)
        return _T()

    ty, tx = tile_info["ty"], tile_info["tx"]

    # 1) H2D transfer (+ optional mask application)
    with _phase_time("h2d_transfer"):
        tile_data = cp.asarray(tile_data_np, dtype=cp.float32)
        if input_mask_np is not None:
            if input_mask_np.shape != tile_data.shape:
                mh, mw = input_mask_np.shape
                th, tw = tile_data.shape
                new_mask = np.zeros(tile_data.shape, dtype=input_mask_np.dtype)
                new_mask[: min(mh, th), : min(mw, tw)] = input_mask_np[: min(mh, th), : min(mw, tw)]
                input_mask_np = new_mask
            mask_gpu = cp.asarray(input_mask_np) > 0
            tile_data = cp.where(mask_gpu, tile_data, cp.float32(0.0))

    # 2) CLAHE
    if use_clahe:
        with _phase_time("clahe"):
            if clahe_params is None:
                clahe_params = {"clip_limit": 0.03}
            clip_limit = clahe_params.get("clip_limit", 0.03)
            kernel_size = clahe_params.get("kernel_size", None)
            tmin = cp.min(tile_data); tmax = cp.max(tile_data); rng = tmax - tmin
            if float(rng) > 0:
                tile_norm = (tile_data - tmin) / rng
            else:
                tile_norm = cp.zeros_like(tile_data, dtype=cp.float32)
            tile_data = _cu_equalize_adapthist(
                tile_norm, kernel_size=kernel_size, clip_limit=clip_limit,
            ).astype(cp.float32)

    # 3) Post-CLAHE smoothing
    if post_clahe_smoothing_sigma and post_clahe_smoothing_sigma > 0:
        with _phase_time("post_clahe_smooth"):
            tile_data = cp_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)

    # 4) Frangi
    with _phase_time("frangi"):
        min_r = frangi_params.get("min_radius_um", 0.2)
        max_r = frangi_params.get("max_radius_um", 1.5)
        num_sigmas = frangi_params.get("num_sigma", 5)
        black_ridges = frangi_params.get("black_ridges", False)
        pixel_size_um = pixel_resolution.get("X", 0.1625)
        sigmas = um_to_sigmas(min_r, max_r, pixel_size_um, num_sigmas=num_sigmas)
        vesselness_map = _cu_frangi(
            tile_data, sigmas=sigmas, black_ridges=black_ridges,
        ).astype(cp.float32)

    # 5) Threshold
    with _phase_time("threshold"):
        if cp.any(vesselness_map > 0):
            fixed_threshold = frangi_params.get("threshold", 0.01)
            if fixed_threshold is not None:
                threshold = float(fixed_threshold)
            else:
                threshold_mult = frangi_params.get("threshold_mult", 0.01)
                threshold = compute_frangi_threshold(vesselness_map, threshold_mult=threshold_mult, xp=cp)
            binary_mask = vesselness_map > threshold
        else:
            binary_mask = cp.zeros_like(vesselness_map, dtype=cp.bool_)

    # 6) Postprocess (tubular-focused)
    do_postprocess = frangi_postprocess or frangi_params.get("postprocess", False)
    structure_type = frangi_params.get("structure_type", None)
    is_tubular = structure_type == "tubular"
    if do_postprocess and is_tubular and bool(cp.any(binary_mask)):
        with _phase_time("postprocess"):
            pp_min_size = frangi_params.get("min_object_size", 5)
            pp_do_opening = frangi_params.get("postprocess_opening", True)
            pp_opening_size = frangi_params.get("postprocess_opening_size", 2)
            pp_fill_holes = frangi_params.get("postprocess_fill_holes", False)
            if pp_fill_holes:
                binary_mask = cp_ndi.binary_fill_holes(binary_mask)
            if pp_do_opening and pp_opening_size > 0:
                k = cp.ones((pp_opening_size,) * binary_mask.ndim, dtype=cp.bool_)
                binary_mask = cp_ndi.binary_opening(binary_mask, structure=k)
            footprint = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
            lab_tmp, _ = cp_ndi.label(binary_mask, structure=footprint)
            if int(lab_tmp.max()) > 0 and pp_min_size > 0:
                areas = cp.bincount(lab_tmp.ravel())
                small = cp.where(areas[1:] < pp_min_size)[0] + 1
                if int(small.size) > 0:
                    drop = cp.isin(lab_tmp, small)
                    binary_mask = cp.where(drop, cp.bool_(False), lab_tmp > 0)

    # 7) Label
    with _phase_time("label"):
        if bool(cp.any(binary_mask)):
            footprint = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
            labeled_mask, _ = cp_ndi.label(binary_mask, structure=footprint)
            min_object_size = frangi_params.get("min_object_size", 0)
            if min_object_size > 0 and int(labeled_mask.max()) > 0:
                areas = cp.bincount(labeled_mask.ravel())
                small = cp.where(areas[1:] < min_object_size)[0] + 1
                if int(small.size) > 0:
                    drop = cp.isin(labeled_mask, small)
                    labeled_mask = cp.where(drop, cp.int32(0), labeled_mask.astype(cp.int32))
                    labeled_mask, _ = cp_ndi.label(labeled_mask > 0, structure=footprint)
            labeled_mask = labeled_mask.astype(cp.int32)
        else:
            labeled_mask = cp.zeros_like(vesselness_map, dtype=cp.int32)

    # 8) Extract core region coordinates + D2H transfer
    y_start = tile_info["y_start_tile"]; x_start = tile_info["x_start_tile"]
    y_end = tile_info["y_end_tile"]; x_end = tile_info["x_end_tile"]
    actual_height = y_end - y_start; actual_width = x_end - x_start
    tile_size_full = tile_info["tile_size"]
    step = tile_size_full - tile_overlap
    core_y_start_global = ty * step
    core_y_end_global = min((ty + 1) * step, actual_height + y_start)
    core_x_start_global = tx * step
    core_x_end_global = min((tx + 1) * step, actual_width + x_start)
    core_y_start_local = max(0, core_y_start_global - y_start)
    core_y_end_local = min(actual_height, core_y_end_global - y_start)
    core_x_start_local = max(0, core_x_start_global - x_start)
    core_x_end_local = min(actual_width, core_x_end_global - x_start)

    is_center = tile_info.get("is_center", False)
    with _phase_time("d2h_transfer"):
        core_labels = cp.asnumpy(
            labeled_mask[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
        )
        core_vesselness = None
        if save_vesselness:
            core_vesselness = cp.asnumpy(
                vesselness_map[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
            )
        debug_labels = cp.asnumpy(labeled_mask).astype(np.int32) if is_center else None
        debug_vesselness = cp.asnumpy(vesselness_map).astype(np.float32) if is_center else None

    # Free GPU pool between tiles to bound memory high-water mark
    del tile_data, vesselness_map, binary_mask, labeled_mask
    cp.get_default_memory_pool().free_all_blocks()

    # Tuck core coords onto the returned bundle so the write helper doesn't
    # have to recompute them.
    tile_info["_core_y_start_global"] = core_y_start_global
    tile_info["_core_y_end_global"] = core_y_end_global
    tile_info["_core_x_start_global"] = core_x_start_global
    tile_info["_core_x_end_global"] = core_x_end_global

    return core_labels, core_vesselness, debug_labels, debug_vesselness


def _compute_tile_blob_on_gpu(
    tile_data_np: np.ndarray,
    input_mask_np: np.ndarray | None,
    tile_info: dict,
    blob_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    tile_overlap: int,
    invert: bool,
) -> tuple[np.ndarray, None, np.ndarray | None, None]:
    """GPU compute for a single blob tile (pipelined-driver entry point).

    Mirrors ``_compute_tile_on_gpu`` but replaces the Frangi path with
    scale-space LoG → peak-find → GPU disk painting
    (``_blob_log_paint_gpu_core``). Returns
    ``(core_labels_np, None, debug_labels_np_or_None, None)`` — no
    vesselness output for blob methods.

    The percentile-normalization step matches ``_segment_blob_log`` on
    CPU: clip to [p1, p99] inside the mask, then scale to [0, 1]. This
    preserves bit-identical binary-footprint output against the CPU-paint
    path when the disk kernel is exercised via ``_segment_blob_log``.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("GPU compute called but cupy/cucim not importable")
    cp = _cp
    cp_ndi = _cp_ndi

    from organelle_profiler.organelle_seg.blob_detection import (
        _blob_log_paint_gpu_core,
    )

    ty, tx = tile_info["ty"], tile_info["tx"]

    # 1) H2D + optional mask application
    tile_data = cp.asarray(tile_data_np, dtype=cp.float32)
    mask_gpu_bool = None
    if input_mask_np is not None:
        if input_mask_np.shape != tile_data.shape:
            mh, mw = input_mask_np.shape
            th, tw = tile_data.shape
            new_mask = np.zeros(tile_data.shape, dtype=input_mask_np.dtype)
            new_mask[: min(mh, th), : min(mw, tw)] = input_mask_np[: min(mh, th), : min(mw, tw)]
            input_mask_np = new_mask
        mask_gpu_bool = cp.asarray(input_mask_np) > 0
        tile_data = cp.where(mask_gpu_bool, tile_data, cp.float32(0.0))

    # 2) CLAHE
    if use_clahe:
        if clahe_params is None:
            clahe_params = {"clip_limit": 0.03}
        clip_limit = clahe_params.get("clip_limit", 0.03)
        kernel_size = clahe_params.get("kernel_size", None)
        tmin = cp.min(tile_data); tmax = cp.max(tile_data); rng = tmax - tmin
        if float(rng) > 0:
            tile_norm = (tile_data - tmin) / rng
        else:
            tile_norm = cp.zeros_like(tile_data, dtype=cp.float32)
        tile_data = _cu_equalize_adapthist(
            tile_norm, kernel_size=kernel_size, clip_limit=clip_limit,
        ).astype(cp.float32)

    # 3) Post-CLAHE smoothing
    if post_clahe_smoothing_sigma and post_clahe_smoothing_sigma > 0:
        tile_data = cp_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)

    # 4) Percentile-based [p1,p99] normalization (matches CPU _segment_blob_log)
    if mask_gpu_bool is not None:
        vals = tile_data[mask_gpu_bool]
    else:
        vals = tile_data.ravel()
    H, W = int(tile_data.shape[0]), int(tile_data.shape[1])
    empty_labels = False
    if vals.size == 0:
        empty_labels = True
    else:
        vmin_s = cp.min(vals); vmax_s = cp.max(vals)
        if float(vmax_s) <= float(vmin_s):
            empty_labels = True
        else:
            p = cp.percentile(vals, cp.asarray([1.0, 99.0]))
            vmin, vmax = p[0], p[1]
            tile_data = cp.clip(
                (tile_data - vmin) / (vmax - vmin + cp.float32(1e-8)),
                cp.float32(0.0), cp.float32(1.0),
            )

    if empty_labels:
        labeled_mask = cp.zeros((H, W), dtype=cp.int32)
    else:
        # 5) Invert for dark blobs (vesicular_dark)
        if invert:
            tile_data = cp.float32(1.0) - tile_data

        # 6) Sigma range from blob_params. Use the same pixel-size lookup
        # as the CPU blob dispatcher (_process_single_frangi_tile:304) —
        # lowercase keys with a 0.108 default. The orchestrator populates
        # `pixel_resolution` with uppercase "X"/"Y" today, so this falls
        # through to 0.108 on both CPU and GPU paths, preserving
        # blob-detection parity. Fixing the lookup convention globally is
        # a separate, semantics-changing PR.
        pixel_size_um = pixel_resolution.get("y", pixel_resolution.get("x", 0.108))
        min_sigma = blob_params["min_radius_um"] / pixel_size_um / np.sqrt(2)
        max_sigma = blob_params["max_radius_um"] / pixel_size_um / np.sqrt(2)
        min_sigma = max(1.0, min_sigma)
        max_sigma = max(min_sigma + 1, max_sigma)
        num_sigma = blob_params.get("num_sigma", 10)
        threshold = blob_params.get("threshold", 0.02)

        # 7) LoG + peaks + GPU disk painting — all on device
        mask_uint8 = None
        if mask_gpu_bool is not None:
            mask_uint8 = mask_gpu_bool.astype(cp.uint8)
        labeled_mask = _blob_log_paint_gpu_core(
            tile_data, min_sigma, max_sigma, num_sigma, float(threshold),
            mask_gpu_uint8=mask_uint8,
        )

    # 8) Extract core region coordinates + D2H transfer
    y_start = tile_info["y_start_tile"]; x_start = tile_info["x_start_tile"]
    y_end = tile_info["y_end_tile"]; x_end = tile_info["x_end_tile"]
    actual_height = y_end - y_start; actual_width = x_end - x_start
    tile_size_full = tile_info["tile_size"]
    step = tile_size_full - tile_overlap
    core_y_start_global = ty * step
    core_y_end_global = min((ty + 1) * step, actual_height + y_start)
    core_x_start_global = tx * step
    core_x_end_global = min((tx + 1) * step, actual_width + x_start)
    core_y_start_local = max(0, core_y_start_global - y_start)
    core_y_end_local = min(actual_height, core_y_end_global - y_start)
    core_x_start_local = max(0, core_x_start_global - x_start)
    core_x_end_local = min(actual_width, core_x_end_global - x_start)

    is_center = tile_info.get("is_center", False)
    core_labels = cp.asnumpy(
        labeled_mask[core_y_start_local:core_y_end_local,
                     core_x_start_local:core_x_end_local]
    )
    debug_labels = cp.asnumpy(labeled_mask).astype(np.int32) if is_center else None

    del tile_data, labeled_mask
    cp.get_default_memory_pool().free_all_blocks()

    tile_info["_core_y_start_global"] = core_y_start_global
    tile_info["_core_y_end_global"] = core_y_end_global
    tile_info["_core_x_start_global"] = core_x_start_global
    tile_info["_core_x_end_global"] = core_x_end_global

    return core_labels, None, debug_labels, None


def _compute_tile_batch_on_gpu_streams(
    bundles: list,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    save_vesselness: bool,
    tile_overlap: int,
) -> list:
    """Process N tiles concurrently using N CUDA streams.

    Each stream handles one tile's full compute pipeline (H2D → CLAHE → Frangi →
    threshold → postprocess → label → core-slice). Per-phase host-side syncs
    from the single-tile path are removed — we run all ops unconditionally so
    the work enqueued on each stream never blocks the main thread.

    After all streams are enqueued, one ``stream.synchronize()`` per stream
    waits for completion; then we D2H to numpy.

    ``bundles`` is a list of ``(tile_info, tile_data_np, input_mask_np_or_None)``.
    Returns a list of ``(tile_info, core_labels_np, core_vesselness_np_or_None,
    debug_labels_np_or_None, debug_vesselness_np_or_None)`` in the same order.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("batched GPU compute requested but cupy/cucim not importable")
    cp = _cp
    cp_ndi = _cp_ndi

    N = len(bundles)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(N)]

    # Host-side Frangi/CLAHE params (scalar; no GPU work)
    min_r = frangi_params.get("min_radius_um", 0.2)
    max_r = frangi_params.get("max_radius_um", 1.5)
    num_sigmas = frangi_params.get("num_sigma", 5)
    black_ridges = frangi_params.get("black_ridges", False)
    pixel_size_um = pixel_resolution.get("X", 0.1625)
    sigmas = um_to_sigmas(min_r, max_r, pixel_size_um, num_sigmas=num_sigmas)

    fixed_threshold = frangi_params.get("threshold", 0.01)
    if fixed_threshold is None:
        fixed_threshold = 0.01  # dynamic threshold would need host sync
    fixed_threshold = float(fixed_threshold)

    do_postprocess = frangi_postprocess or frangi_params.get("postprocess", False)
    structure_type = frangi_params.get("structure_type", None)
    is_tubular = structure_type == "tubular"

    pp_min_size = frangi_params.get("min_object_size", 5)
    pp_do_opening = frangi_params.get("postprocess_opening", True)
    pp_opening_size = frangi_params.get("postprocess_opening_size", 2)
    pp_fill_holes = frangi_params.get("postprocess_fill_holes", False)
    min_object_size = frangi_params.get("min_object_size", 0)

    clp_clip = (clahe_params or {"clip_limit": 0.03}).get("clip_limit", 0.03)
    clp_kernel = (clahe_params or {}).get("kernel_size", None)

    gpu_outputs = [None] * N
    # Detailed timing via CUDA events — zero host-sync overhead (we read times
    # after all streams have synced at the end of the batch).
    detailed = _detailed_timing_enabled()
    event_markers: list[dict] = [{} for _ in range(N)]

    # Launch compute on each stream
    for i, (ti, td_np, mk_np) in enumerate(bundles):
        with streams[i]:
            ems = event_markers[i]
            if detailed:
                ems["h2d_start"] = cp.cuda.Event(disable_timing=False)
                ems["h2d_end"] = cp.cuda.Event(disable_timing=False)
                ems["clahe_end"] = cp.cuda.Event(disable_timing=False)
                ems["smooth_end"] = cp.cuda.Event(disable_timing=False)
                ems["frangi_end"] = cp.cuda.Event(disable_timing=False)
                ems["threshold_end"] = cp.cuda.Event(disable_timing=False)
                ems["postprocess_end"] = cp.cuda.Event(disable_timing=False)
                ems["label_end"] = cp.cuda.Event(disable_timing=False)
                ems["h2d_start"].record()

            # H2D
            tile_data = cp.asarray(td_np, dtype=cp.float32)
            if mk_np is not None:
                if mk_np.shape != tile_data.shape:
                    mh, mw = mk_np.shape
                    th, tw = tile_data.shape
                    new_mask = np.zeros(tile_data.shape, dtype=mk_np.dtype)
                    new_mask[: min(mh, th), : min(mw, tw)] = mk_np[: min(mh, th), : min(mw, tw)]
                    mk_np = new_mask
                mask_gpu = cp.asarray(mk_np) > 0
                tile_data = cp.where(mask_gpu, tile_data, cp.float32(0.0))
            if detailed:
                ems["h2d_end"].record()

            # CLAHE — normalize to [0,1] then equalize. Use cp.where for rng
            # guard so we never sync with host.
            if use_clahe:
                tmin = cp.min(tile_data)
                tmax = cp.max(tile_data)
                rng = tmax - tmin
                safe_rng = cp.where(rng > 0, rng, cp.float32(1.0))
                tile_norm = cp.where(rng > 0, (tile_data - tmin) / safe_rng, cp.float32(0.0))
                tile_data = _cu_equalize_adapthist(
                    tile_norm, kernel_size=clp_kernel, clip_limit=clp_clip,
                ).astype(cp.float32)
            if detailed:
                ems["clahe_end"].record()

            # Post-CLAHE smoothing
            if post_clahe_smoothing_sigma and post_clahe_smoothing_sigma > 0:
                tile_data = cp_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)
            if detailed:
                ems["smooth_end"].record()

            # Frangi
            vesselness_map = _cu_frangi(
                tile_data, sigmas=sigmas, black_ridges=black_ridges,
            ).astype(cp.float32)
            if detailed:
                ems["frangi_end"].record()

            # Threshold (fixed; dynamic would require host sync)
            binary_mask = vesselness_map > fixed_threshold
            if detailed:
                ems["threshold_end"].record()

            # Postprocess (tubular-focused; safe to no-op on empty masks)
            if do_postprocess and is_tubular:
                if pp_fill_holes:
                    binary_mask = cp_ndi.binary_fill_holes(binary_mask)
                if pp_do_opening and pp_opening_size > 0:
                    k = cp.ones((pp_opening_size,) * binary_mask.ndim, dtype=cp.bool_)
                    binary_mask = cp_ndi.binary_opening(binary_mask, structure=k)
                footprint_pp = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
                lab_tmp, _ = cp_ndi.label(binary_mask, structure=footprint_pp)
                if pp_min_size > 0:
                    areas = cp.bincount(lab_tmp.ravel())
                    # Pad areas to at least length 1 to avoid index issues on empty
                    if areas.size < 2:
                        binary_mask = lab_tmp > 0
                    else:
                        small_ids = cp.where(areas[1:] < pp_min_size)[0] + 1
                        drop = cp.isin(lab_tmp, small_ids)
                        binary_mask = cp.where(drop, cp.bool_(False), lab_tmp > 0)
            if detailed:
                ems["postprocess_end"].record()

            # Final label pass
            footprint = cp_ndi.generate_binary_structure(binary_mask.ndim, 1)
            labeled_mask, _ = cp_ndi.label(binary_mask, structure=footprint)
            if min_object_size > 0:
                areas = cp.bincount(labeled_mask.ravel())
                if areas.size >= 2:
                    small = cp.where(areas[1:] < min_object_size)[0] + 1
                    drop = cp.isin(labeled_mask, small)
                    labeled_mask = cp.where(drop, cp.int32(0), labeled_mask.astype(cp.int32))
                    labeled_mask, _ = cp_ndi.label(labeled_mask > 0, structure=footprint)
            labeled_mask = labeled_mask.astype(cp.int32)
            if detailed:
                ems["label_end"].record()

            # Compute core region coords (host-side, no sync)
            ty, tx = ti["ty"], ti["tx"]
            y_start = ti["y_start_tile"]; x_start = ti["x_start_tile"]
            y_end = ti["y_end_tile"]; x_end = ti["x_end_tile"]
            actual_height = y_end - y_start; actual_width = x_end - x_start
            tile_size_full = ti["tile_size"]
            step = tile_size_full - tile_overlap
            cy0g = ty * step
            cy1g = min((ty + 1) * step, actual_height + y_start)
            cx0g = tx * step
            cx1g = min((tx + 1) * step, actual_width + x_start)
            cy0l = max(0, cy0g - y_start); cy1l = min(actual_height, cy1g - y_start)
            cx0l = max(0, cx0g - x_start); cx1l = min(actual_width, cx1g - x_start)

            is_center = ti.get("is_center", False)
            gpu_outputs[i] = {
                "ti": ti,
                "core_labels_gpu": labeled_mask[cy0l:cy1l, cx0l:cx1l],
                "core_vesselness_gpu": (
                    vesselness_map[cy0l:cy1l, cx0l:cx1l] if save_vesselness else None
                ),
                "debug_labels_gpu": labeled_mask if is_center else None,
                "debug_vesselness_gpu": vesselness_map if is_center else None,
                "core_coords": (cy0g, cy1g, cx0g, cx1g),
            }

    # Wait for every stream — this is the only global barrier per batch
    for s in streams:
        s.synchronize()

    # After stream sync, read GPU-event deltas (each cp.cuda.Event.get_elapsed_time
    # returns ms between two events on the same stream).
    if detailed:
        for ems in event_markers:
            if not ems:
                continue
            _bump_detailed_gpu_ms("gpu_h2d",         float(cp.cuda.get_elapsed_time(ems["h2d_start"], ems["h2d_end"])))
            _bump_detailed_gpu_ms("gpu_clahe",       float(cp.cuda.get_elapsed_time(ems["h2d_end"], ems["clahe_end"])))
            _bump_detailed_gpu_ms("gpu_smooth",      float(cp.cuda.get_elapsed_time(ems["clahe_end"], ems["smooth_end"])))
            _bump_detailed_gpu_ms("gpu_frangi",      float(cp.cuda.get_elapsed_time(ems["smooth_end"], ems["frangi_end"])))
            _bump_detailed_gpu_ms("gpu_threshold",   float(cp.cuda.get_elapsed_time(ems["frangi_end"], ems["threshold_end"])))
            _bump_detailed_gpu_ms("gpu_postprocess", float(cp.cuda.get_elapsed_time(ems["threshold_end"], ems["postprocess_end"])))
            _bump_detailed_gpu_ms("gpu_label",       float(cp.cuda.get_elapsed_time(ems["postprocess_end"], ems["label_end"])))
            _record_detailed_tile()

    # D2H on main thread (streams are done)
    cpu_results = []
    for out in gpu_outputs:
        ti = out["ti"]
        ti["_core_y_start_global"] = out["core_coords"][0]
        ti["_core_y_end_global"] = out["core_coords"][1]
        ti["_core_x_start_global"] = out["core_coords"][2]
        ti["_core_x_end_global"] = out["core_coords"][3]
        core_labels = cp.asnumpy(out["core_labels_gpu"])
        core_vesselness = (
            cp.asnumpy(out["core_vesselness_gpu"]) if out["core_vesselness_gpu"] is not None else None
        )
        debug_labels = (
            cp.asnumpy(out["debug_labels_gpu"]).astype(np.int32)
            if out["debug_labels_gpu"] is not None else None
        )
        debug_vesselness = (
            cp.asnumpy(out["debug_vesselness_gpu"]).astype(np.float32)
            if out["debug_vesselness_gpu"] is not None else None
        )
        cpu_results.append((ti, core_labels, core_vesselness, debug_labels, debug_vesselness))

    # Free GPU memory held by batch
    for out in gpu_outputs:
        out.clear()
    gpu_outputs.clear()
    cp.get_default_memory_pool().free_all_blocks()
    return cpu_results


def _write_tile_outputs(
    output_zarr_path: str,
    pos_path: str,
    tile_info: dict,
    core_labels: np.ndarray,
    core_vesselness: np.ndarray | None,
    output_label_name: str,
    save_vesselness: bool,
    use_preview_mode: bool,
    labels_arr=None,
    vesselness_arr=None,
    shm_labels_buffer: np.ndarray | None = None,
    shm_step: int = 0,
) -> None:
    """Write one tile's core region to zarr. Pure CPU work; runs in writer thread.

    If ``labels_arr`` is supplied, writes go directly to the pre-opened zarr
    array (avoids per-write ``open_ome_zarr`` that holds the GIL through
    metadata parsing and serializes writer threads). Writes land in disjoint
    shards so no chunk-level locking is needed.

    If ``shm_labels_buffer`` is supplied (shape ``(n_ty, n_tx, step, step)``),
    labels are written to shared memory instead of zarr — the in-memory
    unstitched path that feeds Pass 2 directly. ``vesselness_arr`` is still
    used for vesselness writes (that output is kept on disk).
    """
    cy0 = tile_info["_core_y_start_global"]
    cy1 = tile_info["_core_y_end_global"]
    cx0 = tile_info["_core_x_start_global"]
    cx1 = tile_info["_core_x_end_global"]

    if shm_labels_buffer is not None:
        ty = tile_info["ty"]
        tx = tile_info["tx"]
        h = cy1 - cy0
        w = cx1 - cx0
        # Core region occupies the top-left (h, w) of this tile's slot;
        # interior tiles fill the full step×step, edge tiles leave the
        # trailing pixels as the pre-zeroed fill.
        shm_labels_buffer[ty, tx, :h, :w] = core_labels
        if save_vesselness and core_vesselness is not None and vesselness_arr is not None:
            vesselness_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_vesselness
        return

    if labels_arr is not None:
        labels_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_labels
        if save_vesselness and core_vesselness is not None and vesselness_arr is not None:
            vesselness_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_vesselness
        return

    # Backward-compatible fallback — opens zarr per call.
    if use_preview_mode:
        import zarr
        store = zarr.open(output_zarr_path, mode="r+")
        if output_label_name:
            temp_name = f"{output_label_name}_unstitched"
            labels_arr = store[pos_path]["labels"][temp_name]["0"]
            labels_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_labels
        if save_vesselness and output_label_name and core_vesselness is not None:
            vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
            temp_vesselness_name = f"{vesselness_label_name}_unstitched"
            vesselness_arr = store[pos_path]["labels"][temp_vesselness_name]["0"]
            vesselness_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_vesselness
    else:
        with open_ome_zarr(output_zarr_path, mode="r+") as ds:
            if output_label_name:
                temp_name = f"{output_label_name}_unstitched"
                labels_arr = ds[pos_path].zgroup["labels"][temp_name]["0"]
                labels_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_labels
            if save_vesselness and output_label_name and core_vesselness is not None:
                vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                vesselness_arr = ds[pos_path].zgroup["labels"][temp_vesselness_name]["0"]
                vesselness_arr[0, 0, 0, cy0:cy1, cx0:cx1] = core_vesselness


def _run_pass1_gpu_pipelined(
    tile_infos: list,
    source_zarr_path: str,
    output_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str | None,
    nucleoli_method: str | None,
    vesicular_method: str | None,
    output_label_name: str | None,
    save_vesselness: bool,
    tile_overlap: int,
    use_preview_mode: bool,
    shm_name: str | None = None,
    shm_shape: tuple | None = None,
    shm_step: int = 0,
) -> tuple[list, float]:
    """Pipelined Pass 1 for GPU: reader threads prefetch tiles, main thread
    runs GPU compute serially, writer threads push outputs asynchronously.

    Env overrides (defaults reasonable for H100/H200 + 32 CPUs):
        ORG_SEG_READ_WORKERS=4         # parallel tile reads (NFS)
        ORG_SEG_WRITE_WORKERS=2        # parallel tile writes (NFS)
        ORG_SEG_PREFETCH_DEPTH=8       # in-flight reads ahead of GPU

    Returns (all_results, pass1_wall_sec).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    # LoG-blob path: only supported in the pipelined driver when the
    # GPU disk-painting kernel is enabled (ORG_SEG_BLOB_DISK_GPU=1). In
    # that case we dispatch to _compute_tile_blob_on_gpu per tile below.
    # Otherwise keep the old behavior (orchestrator routes blob to CPU
    # joblib) by signalling unsupported here.
    _is_blob = nucleoli_method == "blob" or vesicular_method == "blob"
    _blob_disk_gpu = os.environ.get("ORG_SEG_BLOB_DISK_GPU", _fast_default("1")) == "1"
    if _is_blob and not _blob_disk_gpu:
        raise NotImplementedError(
            "pipelined GPU driver for LoG blob requires ORG_SEG_BLOB_DISK_GPU=1"
        )

    n_read = int(os.environ.get("ORG_SEG_READ_WORKERS", "4"))
    n_write = int(os.environ.get("ORG_SEG_WRITE_WORKERS", "2"))
    prefetch_depth = int(os.environ.get("ORG_SEG_PREFETCH_DEPTH", "8"))
    batch_size = max(1, int(os.environ.get("ORG_SEG_GPU_BATCH", _fast_default("2", "1"))))
    # Prefetch must at least cover the batch; bump automatically if needed.
    if prefetch_depth < batch_size:
        prefetch_depth = batch_size

    print(f"  Pass 1 (GPU pipelined): reads={n_read}, writes={n_write}, "
          f"prefetch_depth={prefetch_depth}, gpu_batch={batch_size}")

    _reset_gpu_phase_timers()
    _reset_detailed_timers()
    detailed = _detailed_timing_enabled()

    # Open source and output zarr handles ONCE per worker using raw zarr
    # (skipping iohub's OME-NGFF metadata parsing — we only need the arrays at
    # known paths, not plate/well navigation). Reader/writer threads then do
    # slice-only access, which is thread-safe and releases the GIL during chunk
    # decompression/compression.
    import zarr as _zarr
    src_store = _zarr.open(source_zarr_path, mode="r")
    source_image_array = src_store[f"{pos_path}/0"]
    input_mask_array = None
    if input_mask_name:
        try:
            input_mask_array = src_store[f"{pos_path}/labels/{input_mask_name}/0"]
        except (KeyError, FileNotFoundError):
            input_mask_array = None

    labels_arr_handle = None
    vesselness_arr_handle = None
    dst_store = _zarr.open(output_zarr_path, mode="r+")
    # In-memory labels path: attach to the shared-memory tile buffer the parent
    # pre-allocated. When active, label writes go to shm instead of zarr —
    # the unstitched zarr array is not created at all in this mode.
    shm_labels_buffer = None
    shm_handle = None  # keep reference so /dev/shm block stays mapped
    if shm_name is not None and shm_shape is not None:
        from multiprocessing import shared_memory as _shm
        shm_handle = _shm.SharedMemory(name=shm_name)
        shm_labels_buffer = np.ndarray(shm_shape, dtype=np.int32, buffer=shm_handle.buf)
    elif output_label_name:
        temp_name = f"{output_label_name}_unstitched"
        labels_arr_handle = dst_store[f"{pos_path}/labels/{temp_name}/0"]
    if save_vesselness and output_label_name:
        vname = output_label_name.replace("_seg", "_vesselness")
        temp_v = f"{vname}_unstitched"
        vesselness_arr_handle = dst_store[f"{pos_path}/labels/{temp_v}/0"]

    read_pool = ThreadPoolExecutor(max_workers=n_read, thread_name_prefix="tile_read")
    write_pool = ThreadPoolExecutor(max_workers=n_write, thread_name_prefix="tile_write")

    prefetch_q: deque = deque()
    write_futures: deque = deque()

    pass1_wall_start = time.monotonic()
    read_wait_total = 0.0
    write_queue_wait_total = 0.0

    # Prime prefetch queue
    for ti in tile_infos[:prefetch_depth]:
        prefetch_q.append((ti, read_pool.submit(
            _read_tile_for_gpu, source_zarr_path, pos_path, channel_index, ti, input_mask_name,
            source_image_array, input_mask_array,
        )))

    all_results = []
    pbar = tqdm(total=len(tile_infos), desc="  Pass 1: GPU pipelined")
    next_read_idx = prefetch_depth
    n_total = len(tile_infos)
    i = 0

    while i < n_total:
        # How many tiles this iteration will consume
        batch_n = min(batch_size, n_total - i)

        # Track per-iteration wall-clock so we can decompose where the main
        # thread spends time when ORG_SEG_DETAILED_TIMING=1.
        iter_t0 = time.monotonic() if detailed else None

        # 1) Pull `batch_n` pre-read tiles (block on reader threads)
        bundles = []
        t_pull_start = time.monotonic() if detailed else None
        for _ in range(batch_n):
            curr_info, curr_future = prefetch_q.popleft()
            t0 = time.monotonic()
            tile_data_np, input_mask_np = curr_future.result()
            read_wait_total += time.monotonic() - t0
            bundles.append((curr_info, tile_data_np, input_mask_np))
            # Top up prefetch for each consumed slot
            if next_read_idx < n_total:
                ni = tile_infos[next_read_idx]
                prefetch_q.append((ni, read_pool.submit(
                    _read_tile_for_gpu, source_zarr_path, pos_path, channel_index, ni, input_mask_name,
                    source_image_array, input_mask_array,
                )))
                next_read_idx += 1
        if detailed:
            _bump_detailed_host("host_read_pull", (time.monotonic() - t_pull_start))

        # 2) GPU compute — route everything through the batch function. At
        # batch_size=1 it's "single tile with no mid-pipeline host syncs";
        # larger N adds intra-process concurrency via streams. The alternative
        # (_compute_tile_on_gpu with per-phase timers) forces a GPU sync after
        # every phase, which blocks kernel pipelining and leaves the SM idle
        # in the gaps. We pay the "no per-phase timing" cost for better throughput.
        try:
            t_compute_start = time.monotonic() if detailed else None
            use_single_tile_with_timers = (
                os.environ.get("ORG_SEG_PHASE_TIMERS", "0") == "1" and batch_size == 1
            )
            if _is_blob:
                # Blob path: one tile at a time through the GPU disk-paint
                # kernel. No batch streams yet — per-tile work varies with
                # blob count, so the uniform-work assumption behind
                # _compute_tile_batch_on_gpu_streams doesn't apply.
                invert_blob = bool(frangi_params.get("black_ridges", False))
                batch_results = []
                for ti, td_np, mk_np in bundles:
                    core_labels, _v, debug_labels, _dv = _compute_tile_blob_on_gpu(
                        tile_data_np=td_np,
                        input_mask_np=mk_np,
                        tile_info=ti,
                        blob_params=frangi_params,
                        pixel_resolution=pixel_resolution,
                        use_clahe=use_clahe,
                        clahe_params=clahe_params,
                        post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                        tile_overlap=tile_overlap,
                        invert=invert_blob,
                    )
                    batch_results.append((ti, core_labels, None, debug_labels, None))
            elif use_single_tile_with_timers:
                ti, td_np, mk_np = bundles[0]
                core_labels, core_vesselness, debug_labels, debug_vesselness = _compute_tile_on_gpu(
                    tile_data_np=td_np,
                    input_mask_np=mk_np,
                    tile_info=ti,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                )
                batch_results = [(ti, core_labels, core_vesselness, debug_labels, debug_vesselness)]
            else:
                batch_results = _compute_tile_batch_on_gpu_streams(
                    bundles=bundles,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                )
            if detailed:
                _bump_detailed_host("host_compute_call", (time.monotonic() - t_compute_start))

            # 3) Submit async writes for each tile in the batch
            t_write_start = time.monotonic() if detailed else None
            for ti, core_labels, core_vesselness, debug_labels, debug_vesselness in batch_results:
                if len(write_futures) >= prefetch_depth:
                    t0 = time.monotonic()
                    while len(write_futures) >= prefetch_depth:
                        write_futures.popleft().result()
                    write_queue_wait_total += time.monotonic() - t0

                wfut = write_pool.submit(
                    _write_tile_outputs,
                    output_zarr_path, pos_path, ti, core_labels, core_vesselness,
                    output_label_name, save_vesselness, use_preview_mode,
                    labels_arr_handle, vesselness_arr_handle,
                    shm_labels_buffer, shm_step,
                )
                write_futures.append(wfut)

                all_results.append({
                    "tile_info": ti,
                    "success": True,
                    "vesselness": debug_vesselness if ti.get("is_center") else None,
                    "labels": debug_labels if ti.get("is_center") else None,
                })
        except Exception as e:
            print(f"Error processing GPU batch starting at tile {bundles[0][0].get('tile_idx', '?')}: {e}")
            import traceback
            traceback.print_exc()
            for ti, _td, _mk in bundles:
                all_results.append({"tile_info": ti, "success": False})

        if detailed:
            _bump_detailed_host("host_write_submit", (time.monotonic() - t_write_start))
            iter_dt = time.monotonic() - iter_t0
            # "loop overhead" = iter wall minus the three tracked host sections.
            overhead = iter_dt - (
                _DETAILED_HOST_TIMERS.get("host_read_pull", 0) / (1 if i == 0 else 1)
            )
            # Rather than subtract accumulators (messy), track the slack
            # separately: time between end-of-write-submit and end-of-iter.
            # We'll record it implicitly by summing iter_dt and reconciling.
            _bump_detailed_host("host_iter_total", iter_dt)

        pbar.update(batch_n)
        i += batch_n
    pbar.close()

    # Drain remaining writes
    drain_start = time.monotonic()
    for f in write_futures:
        f.result()
    drain_wall = time.monotonic() - drain_start

    read_pool.shutdown(wait=True)
    write_pool.shutdown(wait=True)
    # Raw zarr.open doesn't require explicit cleanup — stores go away when
    # their references drop.
    if shm_handle is not None:
        # Drop our numpy view, then close the mmap. Parent unlinks the shm
        # block after Pass 2 finishes reading from it.
        shm_labels_buffer = None
        shm_handle.close()

    pass1_wall = time.monotonic() - pass1_wall_start
    print(f"  Pass 1 pipeline stats:")
    print(f"    read_wait      {read_wait_total:7.1f}s  ({100 * read_wait_total / pass1_wall:5.1f}% of wall) — time blocked on tile reads")
    print(f"    write_q_wait   {write_queue_wait_total:7.1f}s  ({100 * write_queue_wait_total / pass1_wall:5.1f}% of wall) — time blocked draining write queue mid-run")
    print(f"    write_drain    {drain_wall:7.1f}s  ({100 * drain_wall / pass1_wall:5.1f}% of wall) — end-of-pass write drain")
    if detailed:
        _print_detailed_timers(pass1_wall)
    _print_gpu_phase_timers(pass1_wall)
    return all_results, pass1_wall


def _worker_pipelined_pass1(
    worker_id: int,
    tile_infos_partition: list,
    source_zarr_path: str,
    output_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str | None,
    nucleoli_method: str | None,
    vesicular_method: str | None,
    output_label_name: str | None,
    save_vesselness: bool,
    tile_overlap: int,
    use_preview_mode: bool,
    shm_name: str | None = None,
    shm_shape: tuple | None = None,
    shm_step: int = 0,
):
    """Worker process: run the pipelined Pass 1 driver on a partition of tiles.

    Each worker owns its own CUDA context (fresh from process spawn), its own
    reader/writer thread pools, and writes directly to pre-existing zarr chunks
    (the output array is created by the parent before any worker starts, so no
    metadata races). Workers write to disjoint shards because the tile→chunk
    mapping is 1:1, so no write contention either.
    """
    import os
    import sys
    import traceback

    try:
        # Re-import inside the worker — spawn gives us a fresh interpreter.
        # This also initializes a fresh CUDA context owned by this process.
        from organelle_profiler.organelle_seg.tiled_processing import (
            _run_pass1_gpu_pipelined,
        )

        # Optional: switch to CUDA's stream-ordered async memory pool
        # (cudaMallocAsync). Allocations become stream-local, so N concurrent
        # streams don't serialize on CuPy's default pool lock. Requires
        # CUDA 11.2+ and CuPy with async-pool support.
        if os.environ.get("ORG_SEG_ASYNC_POOL", "0") == "1":
            try:
                import cupy as cp
                cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
                print(f"  [worker {worker_id}] using cudaMallocAsync pool (stream-ordered)")
            except Exception as e:
                print(f"  [worker {worker_id}] MemoryAsyncPool setup failed, using default pool: {e}")

        # Optional: stagger worker starts to desync their GPU pipelines. The
        # hope is that one worker is in an HBM-heavy phase (CLAHE histogram or
        # interpolation) while another is in compute-heavy phase (CDF), so
        # bandwidth contention is less severe. In steady state workers
        # naturally desync from per-tile timing variation, so this mainly
        # affects early batches.
        try:
            stagger_ms = int(os.environ.get("ORG_SEG_WORKER_STAGGER_MS", "0"))
        except ValueError:
            stagger_ms = 0
        if stagger_ms > 0 and worker_id > 0:
            delay_s = worker_id * stagger_ms / 1000.0
            print(f"  [worker {worker_id}] staggered start delay: {delay_s:.3f}s")
            import time as _t
            _t.sleep(delay_s)

        print(f"  [worker {worker_id}] starting with {len(tile_infos_partition)} tiles, pid={os.getpid()}"
              + (f", shm={shm_name}" if shm_name else ""))
        _run_pass1_gpu_pipelined(
            tile_infos=tile_infos_partition,
            source_zarr_path=source_zarr_path,
            output_zarr_path=output_zarr_path,
            pos_path=pos_path,
            channel_index=channel_index,
            frangi_params=frangi_params,
            pixel_resolution=pixel_resolution,
            use_clahe=use_clahe,
            clahe_params=clahe_params,
            post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
            frangi_postprocess=frangi_postprocess,
            input_mask_name=input_mask_name,
            nucleoli_method=nucleoli_method,
            vesicular_method=vesicular_method,
            output_label_name=output_label_name,
            save_vesselness=save_vesselness,
            tile_overlap=tile_overlap,
            use_preview_mode=use_preview_mode,
            shm_name=shm_name,
            shm_shape=shm_shape,
            shm_step=shm_step,
        )
        print(f"  [worker {worker_id}] done")
    except Exception:
        print(f"  [worker {worker_id}] FAILED:")
        traceback.print_exc()
        sys.exit(1)


def _run_pass1_gpu_multi_pipeline(
    tile_infos: list,
    n_workers: int,
    source_zarr_path: str,
    output_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict | None,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str | None,
    nucleoli_method: str | None,
    vesicular_method: str | None,
    output_label_name: str | None,
    save_vesselness: bool,
    tile_overlap: int,
    use_preview_mode: bool,
    shm_name: str | None = None,
    shm_shape: tuple | None = None,
    shm_step: int = 0,
) -> tuple[list, float]:
    """Spawn N GPU worker processes, each running the pipelined driver on its
    partition of tiles. The parent pre-creates the output zarr array (done by
    the caller) and waits for all workers to complete.

    Uses the ``spawn`` start method — never fork — because fork plus CUDA is
    fragile (child inherits the parent's CUDA state in a broken way).

    Tile partitioning is interleaved (worker i gets tiles i, i+N, i+2N, ...) so
    any spatial hot spots in the image get distributed evenly across workers
    rather than loaded onto a single worker.
    """
    import multiprocessing as mp
    import time as _time

    ctx = mp.get_context("spawn")
    partitions = [tile_infos[i::n_workers] for i in range(n_workers)]

    print(f"  Pass 1: {len(tile_infos)} tiles across {n_workers} spawned worker processes")
    for i, p in enumerate(partitions):
        print(f"    worker {i}: {len(p)} tiles")

    pass1_wall_start = _time.monotonic()
    procs = []
    for wid, part in enumerate(partitions):
        p = ctx.Process(
            target=_worker_pipelined_pass1,
            args=(
                wid, part, source_zarr_path, output_zarr_path, pos_path,
                channel_index, frangi_params, pixel_resolution, use_clahe,
                clahe_params, post_clahe_smoothing_sigma, frangi_postprocess,
                input_mask_name, nucleoli_method, vesicular_method,
                output_label_name, save_vesselness, tile_overlap,
                use_preview_mode,
                shm_name, shm_shape, shm_step,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    pass1_wall = _time.monotonic() - pass1_wall_start

    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        print(f"  [WARN] {len(failed)}/{len(procs)} GPU worker processes failed with non-zero exit codes: "
              f"{[(p.pid, p.exitcode) for p in failed]}")

    # We don't collect per-tile results in the multi-worker path (would require
    # IPC of potentially large arrays). Results are on disk at the zarr output.
    return [], pass1_wall


def _stitch_tiled_labels_pass2(
    source_zarr_path: str,
    pos_path: str,
    organelle_name: str,
    n_tiles_y: int,
    n_tiles_x: int,
    tile_size: int,
    tile_overlap: int,
    height: int,
    width: int,
    input_mask_name: str = None,
    mask_erosion_pixels: int = 0,
    crop_bbox: tuple = None,
    target_chunks: tuple = (1, 1, 1, 512, 512),
    target_shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Pass 2: Two-phase stitching with Union-Find for proper label merging.

    Phase A: Offset all tile labels to be globally unique, collect ALL merge pairs
    Phase B: Use Union-Find to build connected components, apply global relabeling
    Phase C: Rechunk from parallel-write-safe 1:1 sharding to efficient storage sharding

    This properly handles:
    - Multiple labels mapping to the same neighbor
    - Transitive merges (A→B, B→C implies A→C)
    - Labels that span multiple tile boundaries

    Args:
        source_zarr_path: Path to the v3 zarr store
        pos_path: Position path like "A/1/0"
        organelle_name: Name of the organelle being segmented
        n_tiles_y: Number of tiles in Y direction
        n_tiles_x: Number of tiles in X direction
        tile_size: Size of each tile
        tile_overlap: Overlap between tiles
        height: Total image height
        width: Total image width
        input_mask_name: If provided (e.g., "nuclear_seg"), erode this mask and remove
            labels outside the eroded zone (removes boundary artifacts).
        mask_erosion_pixels: Pixels to erode the input mask by. Labels outside the
            eroded mask are removed (e.g., nucleoli near nuclear boundary).
        crop_bbox: If provided (y_start, y_end, x_start, x_end), the labels were generated
        target_chunks: Target chunk size for final array (default: 512x512)
        target_shards_ratio: Target sharding ratio for final array (default: 32x32 for ~1GB shards)
            from a cropped region of the source image. Used to correctly slice the input_mask.
    """
    temp_name = f"{organelle_name}_unstitched"
    step = tile_size - tile_overlap

    print(f"  Pass 2: Stitching {n_tiles_y * n_tiles_x} tiles with Union-Find merging...")
    print(f"    tile_size={tile_size}, tile_overlap={tile_overlap}, step={step}")

    # =========================================================================
    # PHASE A: Offset all tiles and collect merge pairs
    # =========================================================================
    all_merge_pairs = []  # List of (label_a, label_b) pairs to merge
    max_label_seen = 0

    with open_ome_zarr(source_zarr_path, mode="r+") as ds:
        source_pos = ds[pos_path]
        labels_group = source_pos.zgroup["labels"]
        labels_arr = labels_group[temp_name]["0"]

        running_offset = 0

        # Phase A: Offset labels and collect merge pairs
        print(f"    Phase A: Offsetting labels and collecting merge pairs...")
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y_start = ty * step
                x_start = tx * step
                y_end = min((ty + 1) * step, height)
                x_end = min((tx + 1) * step, width)

                tile_labels = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]).copy()

                if tile_labels.max() == 0:
                    continue

                # Offset tile labels to be globally unique
                tile_mask = tile_labels > 0
                tile_labels[tile_mask] += running_offset
                tile_max = tile_labels.max()

                # Write offset labels back immediately (so neighbors can read them)
                labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels

                # Check LEFT neighbor for merge pairs
                if tx > 0:
                    left_boundary_x = x_start - 1
                    if left_boundary_x >= 0:
                        left_boundary = np.asarray(
                            labels_arr[0, 0, 0, y_start:y_end, left_boundary_x:left_boundary_x+1]
                        ).flatten()
                        tile_left_edge = tile_labels[:, 0]

                        if left_boundary.max() > 0 and tile_left_edge.max() > 0:
                            for row_idx in range(len(tile_left_edge)):
                                tile_label = tile_left_edge[row_idx]
                                if tile_label == 0:
                                    continue
                                for neighbor_row in [row_idx - 1, row_idx, row_idx + 1]:
                                    if 0 <= neighbor_row < len(left_boundary):
                                        neighbor_label = left_boundary[neighbor_row]
                                        if neighbor_label > 0 and tile_label != neighbor_label:
                                            all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                # Check TOP neighbor for merge pairs
                if ty > 0:
                    top_boundary_y = y_start - 1
                    if top_boundary_y >= 0:
                        top_boundary = np.asarray(
                            labels_arr[0, 0, 0, top_boundary_y:top_boundary_y+1, x_start:x_end]
                        ).flatten()
                        tile_top_edge = tile_labels[0, :]

                        if top_boundary.max() > 0 and tile_top_edge.max() > 0:
                            for col_idx in range(len(tile_top_edge)):
                                tile_label = tile_top_edge[col_idx]
                                if tile_label == 0:
                                    continue
                                for neighbor_col in [col_idx - 1, col_idx, col_idx + 1]:
                                    if 0 <= neighbor_col < len(top_boundary):
                                        neighbor_label = top_boundary[neighbor_col]
                                        if neighbor_label > 0 and tile_label != neighbor_label:
                                            all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                running_offset = max(running_offset, tile_max)
                max_label_seen = max(max_label_seen, tile_max)

        print(f"    Phase A complete: {len(all_merge_pairs)} merge pairs, max_label={max_label_seen}")

        # =========================================================================
        # PHASE B: Build Union-Find and apply global relabeling
        # =========================================================================
        if all_merge_pairs:
            print(f"    Phase B: Building Union-Find structure...")

            # Union-Find with path compression
            parent = list(range(int(max_label_seen) + 1))

            def find(x):
                root = x
                while parent[root] != root:
                    root = parent[root]
                # Path compression
                while parent[x] != root:
                    next_x = parent[x]
                    parent[x] = root
                    x = next_x
                return root

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    # Always merge to the smaller root (ensures consistency)
                    if ra < rb:
                        parent[rb] = ra
                    else:
                        parent[ra] = rb

            # Process all merge pairs
            for a, b in all_merge_pairs:
                if a <= max_label_seen and b <= max_label_seen:
                    union(a, b)

            # Build final LUT: each label maps to its root
            lut = np.arange(int(max_label_seen) + 1, dtype=np.int32)
            for i in range(1, int(max_label_seen) + 1):
                lut[i] = find(i)

            # Count unique merged labels
            unique_roots = len(set(lut[1:]))
            print(f"    Phase B: {int(max_label_seen)} labels merged to {unique_roots} unique labels")

            # Apply LUT to all tiles
            print(f"    Phase B: Applying global relabeling...")
            for ty in tqdm(range(n_tiles_y), desc="    Relabeling tiles"):
                for tx in range(n_tiles_x):
                    y_start = ty * step
                    x_start = tx * step
                    y_end = min((ty + 1) * step, height)
                    x_end = min((tx + 1) * step, width)

                    tile_labels = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]).copy()

                    if tile_labels.max() == 0:
                        continue

                    # Apply LUT (handle labels larger than LUT size)
                    mask = tile_labels <= max_label_seen
                    tile_labels[mask] = lut[tile_labels[mask]]

                    labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels
        else:
            print(f"    Phase B: No merge pairs found, skipping relabeling")

    # Rename from temp to final using filesystem rename (zarr v3 doesn't support copy)
    print(f"  Renaming {temp_name} -> {organelle_name}")

    # Get the filesystem paths for the zarr groups
    zarr_store_path = Path(source_zarr_path)
    labels_path = zarr_store_path / pos_path / "labels"
    temp_path = labels_path / temp_name
    final_path = labels_path / organelle_name

    # Remove existing final path if it exists
    if final_path.exists():
        import shutil
        shutil.rmtree(final_path)

    # Rename temp to final
    temp_path.rename(final_path)

    print(f"  Pass 2 complete: {running_offset} total objects after stitching")

    # Optional Pass 3: Remove labels near mask boundary (e.g., nucleoli near nuclear edge)
    # We erode the INPUT MASK (e.g., nuclear_seg) and remove any labels outside the eroded zone
    if input_mask_name and mask_erosion_pixels > 0:
        from scipy.ndimage import binary_erosion

        print(f"  Pass 3: Removing labels within {mask_erosion_pixels}px of {input_mask_name} boundary...")
        erosion_structure = np.ones((mask_erosion_pixels * 2 + 1, mask_erosion_pixels * 2 + 1), dtype=bool)

        with open_ome_zarr(source_zarr_path, mode="r+") as ds:
            labels_arr = ds[pos_path].zgroup["labels"][organelle_name]["0"]

            # Check if the input mask exists
            labels_group = ds[pos_path].zgroup.get("labels", None)
            if labels_group is None or input_mask_name not in labels_group:
                print(f"  Warning: Input mask '{input_mask_name}' not found, skipping Pass 3")
            else:
                mask_arr = labels_group[input_mask_name]["0"]

                # IMPORTANT: Load and erode the FULL mask first, then apply tile by tile
                # Eroding tile-by-tile causes edge artifacts (tile boundaries get eroded)
                print(f"    Loading full mask for erosion...")
                # If crop_bbox is provided, slice the mask to match the cropped region
                if crop_bbox:
                    crop_y_start, crop_y_end, crop_x_start, crop_x_end = crop_bbox
                    full_mask = np.asarray(mask_arr[0, 0, 0, crop_y_start:crop_y_end, crop_x_start:crop_x_end])
                    print(f"    Using crop_bbox: y=[{crop_y_start}:{crop_y_end}], x=[{crop_x_start}:{crop_x_end}]")
                else:
                    full_mask = np.asarray(mask_arr[0, 0, 0, :height, :width])
                binary_full_mask = full_mask > 0
                mask_pixels_before = binary_full_mask.sum()
                eroded_full_mask = binary_erosion(binary_full_mask, structure=erosion_structure)
                mask_pixels_after = eroded_full_mask.sum()
                print(f"    Mask pixels: {mask_pixels_before:,} -> {mask_pixels_after:,} after erosion ({100*mask_pixels_after/max(1,mask_pixels_before):.1f}% retained)")
                print(f"    Mask eroded, applying to labels...")

                # Process tile by tile to avoid loading full labels into memory.
                # Each tile's read + mask + write is independent, so fan out
                # across a thread pool. Zarr 1:1 chunk↔shard (Pass 1) means
                # each tile writes to its own file — safe under threading.
                def _process_tile(ty_tx):
                    ty, tx = ty_tx
                    y_start = ty * step
                    x_start = tx * step
                    y_end = min((ty + 1) * step, height)
                    x_end = min((tx + 1) * step, width)

                    tile_labels = np.asarray(
                        labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]
                    ).copy()
                    if tile_labels.max() == 0:
                        return 0, 0

                    before = int((tile_labels > 0).sum())
                    eroded_mask_tile = eroded_full_mask[y_start:y_end, x_start:x_end]

                    if eroded_mask_tile.shape != tile_labels.shape:
                        print(f"    Warning: Dimension mismatch at tile ({ty}, {tx}): "
                              f"mask {eroded_mask_tile.shape} vs labels {tile_labels.shape}, "
                              f"skipping")
                        return before, before

                    tile_labels[~eroded_mask_tile] = 0
                    after = int((tile_labels > 0).sum())
                    labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels
                    return before, after

                pass3_workers = int(os.environ.get("ORG_SEG_PASS3_WORKERS", "16"))
                all_tile_coords = [
                    (ty, tx) for ty in range(n_tiles_y) for tx in range(n_tiles_x)
                ]
                labels_before_total = 0
                labels_after_total = 0
                with ThreadPoolExecutor(max_workers=pass3_workers) as pool:
                    for before, after in tqdm(
                        pool.map(_process_tile, all_tile_coords),
                        total=len(all_tile_coords),
                        desc=f"  Removing boundary labels ({pass3_workers}w)",
                    ):
                        labels_before_total += before
                        labels_after_total += after

                print(f"  Pass 3 complete: Label pixels {labels_before_total:,} -> {labels_after_total:,} ({100*labels_after_total/max(1,labels_before_total):.1f}% retained)")

    # Final phase: Reshard from parallel-write-safe 1:1 sharding to efficient storage sharding
    # This is the second phase of the two-phase write strategy:
    # - Pass 1+2 used chunks=shards (1:1 mapping) for safe parallel writes
    # - Now we reshard to target_shards_ratio for efficient storage (~1GB shard files)
    from cyclops_utils.io.zarr_utils import reshard_zarr_array
    label_array_path = Path(source_zarr_path) / pos_path / "labels" / organelle_name / "0"
    reshard_zarr_array(
        source_path=label_array_path,
        dest_path=None,  # In-place resharding
        chunks=target_chunks,
        shards_ratio=target_shards_ratio,
        tile_size=4096,
        show_progress=True,
    )

    return running_offset


# ---------------------------------------------------------------------------
# Parallel / GPU-accelerated Pass 2 — drop-in replacement for
# _stitch_tiled_labels_pass2. Gated on ORG_SEG_PASS2_GPU=1; falls back to the
# original sequential CPU path when disabled or cucim/cupy are unavailable.
# ---------------------------------------------------------------------------

def _read_tile_and_boundaries(
    labels_arr,
    ty: int,
    tx: int,
    step: int,
    height: int,
    width: int,
) -> tuple:
    """Read one Pass-2 tile from the pre-opened zarr array and extract the
    four boundary rows/cols plus its local max label. Runs on CPU (numpy);
    cheap per tile. Intended to be called from a reader thread pool.

    Returns (ty, tx, local_max, left_col, top_row, right_col, bottom_row)
    where the four boundary arrays are 1-D numpy int32 and `local_max` is a
    plain Python int.
    """
    y_start = ty * step
    x_start = tx * step
    y_end = min((ty + 1) * step, height)
    x_end = min((tx + 1) * step, width)
    tile = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end])
    # Ensure contiguous int32 (so downstream writes don't need re-cast)
    if tile.dtype != np.int32:
        tile = tile.astype(np.int32)
    local_max = int(tile.max()) if tile.size else 0
    left_col = tile[:, 0].copy()
    right_col = tile[:, -1].copy()
    top_row = tile[0, :].copy()
    bottom_row = tile[-1, :].copy()
    return ty, tx, local_max, left_col, top_row, right_col, bottom_row


def _pairs_from_edges_vectorized(
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    offset_a: int,
    offset_b: int,
) -> np.ndarray:
    """Return (N, 2) int64 array of global merge pairs by comparing two
    adjacent tile edges (1-D). Mirrors the 3-neighbor-offset logic of the
    original nested-Python loop but runs as a few numpy ops.

    Each edge is compared at shifts -1, 0, +1. For every position where both
    sides have a non-zero label and the labels differ, a pair is emitted.
    """
    L = min(len(edge_a), len(edge_b))
    if L == 0:
        return np.empty((0, 2), dtype=np.int64)
    a = edge_a[:L].astype(np.int64, copy=False)
    b = edge_b[:L].astype(np.int64, copy=False)
    pair_chunks = []
    for shift in (-1, 0, 1):
        if shift == 0:
            aa, bb = a, b
        elif shift > 0:
            aa = a[shift:]
            bb = b[:-shift]
        else:
            aa = a[:shift]
            bb = b[-shift:]
        valid = (aa != 0) & (bb != 0) & (aa != bb)
        if valid.any():
            pair_chunks.append(np.column_stack([
                aa[valid] + offset_a,
                bb[valid] + offset_b,
            ]))
    if not pair_chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(pair_chunks)


def _apply_global_lut_to_tile(
    labels_arr,
    ty: int,
    tx: int,
    step: int,
    height: int,
    width: int,
    global_offset_tile: int,
    lut_cpu: np.ndarray,
    lut_gpu=None,
) -> None:
    """Read one tile, add this tile's cumulative offset, apply the global LUT
    (mapping offset-label → component root), and write back. Called from a
    worker pool. When a cupy LUT is provided and _GPU_AVAILABLE, does the
    remap on GPU; else falls back to numpy."""
    y_start = ty * step
    x_start = tx * step
    y_end = min((ty + 1) * step, height)
    x_end = min((tx + 1) * step, width)
    tile_cpu = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end])
    if tile_cpu.dtype != np.int32:
        tile_cpu = tile_cpu.astype(np.int32)
    if tile_cpu.max() == 0:
        # Nothing to remap — tile is all-background. Skip write.
        return

    if lut_gpu is not None:
        cp = _cp
        tile_gpu = cp.asarray(tile_cpu, dtype=cp.int64)
        # For non-zero pixels: global_offset + local_label, then LUT lookup.
        # Background (0) stays 0 — LUT index 0 is pinned to 0 by the caller.
        nonzero = tile_gpu > 0
        global_idx = cp.where(nonzero, tile_gpu + cp.int64(global_offset_tile), cp.int64(0))
        remapped = lut_gpu[global_idx]
        out_cpu = cp.asnumpy(remapped).astype(np.int32)
    else:
        # CPU fallback
        nonzero = tile_cpu > 0
        global_idx = np.where(nonzero, tile_cpu.astype(np.int64) + global_offset_tile, 0)
        out_cpu = lut_cpu[global_idx].astype(np.int32)

    labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = out_cpu


def _read_tile_and_push_gpu(
    labels_arr,
    ty: int,
    tx: int,
    step: int,
    height: int,
    width: int,
) -> tuple:
    """Phase A variant that ALSO pushes the tile body to GPU memory, so
    Phase B.5 can skip the second NFS read. Called from the reader thread
    pool when ORG_SEG_PASS2_TILE_CACHE=1 is set.

    Returns (ty, tx, local_max, left_col, top_row, right_col, bottom_row,
             tile_gpu)
    where tile_gpu is a cupy.int32 ndarray owning the tile's pixels.
    """
    cp = _cp
    y_start = ty * step
    x_start = tx * step
    y_end = min((ty + 1) * step, height)
    x_end = min((tx + 1) * step, width)
    tile_cpu = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end])
    if tile_cpu.dtype != np.int32:
        tile_cpu = tile_cpu.astype(np.int32)
    local_max = int(tile_cpu.max()) if tile_cpu.size else 0
    left_col = tile_cpu[:, 0].copy()
    right_col = tile_cpu[:, -1].copy()
    top_row = tile_cpu[0, :].copy()
    bottom_row = tile_cpu[-1, :].copy()
    # H2D push — tile stays on GPU through Phase B.5.
    tile_gpu = cp.asarray(tile_cpu)
    return ty, tx, local_max, left_col, top_row, right_col, bottom_row, tile_gpu


def _read_tile_from_buffer_and_push_gpu(
    tile_buffer: np.ndarray,
    ty: int,
    tx: int,
    step: int,
    height: int,
    width: int,
) -> tuple:
    """Phase A variant for when the unstitched tiles are already in a shared
    numpy buffer (fed by Pass 1 via mp.shared_memory — or preloaded for a
    bench). Skips all zarr I/O on the read side.

    `tile_buffer` shape is ``(n_ty, n_tx, step, step)``, int32. Edge tiles
    (last row/col) are zero-padded to fill the fixed-size slot, so we slice
    down to the actual valid region before boundary extraction.

    Returns the same 8-tuple as ``_read_tile_and_push_gpu``.
    """
    cp = _cp
    y_start = ty * step
    x_start = tx * step
    y_end = min((ty + 1) * step, height)
    x_end = min((tx + 1) * step, width)
    h = y_end - y_start
    w = x_end - x_start
    # View into the pre-populated buffer. No zarr read, no decompression.
    tile_cpu = tile_buffer[ty, tx, :h, :w]
    # Ensure contiguous for cupy H2D; the slice is already contiguous if the
    # buffer is row-major and full-size (h == step, w == step). For edge
    # tiles we copy so the GPU-side array is contiguous.
    if not tile_cpu.flags.c_contiguous:
        tile_cpu = np.ascontiguousarray(tile_cpu)
    local_max = int(tile_cpu.max()) if tile_cpu.size else 0
    left_col = tile_cpu[:, 0].copy()
    right_col = tile_cpu[:, -1].copy()
    top_row = tile_cpu[0, :].copy()
    bottom_row = tile_cpu[-1, :].copy()
    tile_gpu = cp.asarray(tile_cpu)
    return ty, tx, local_max, left_col, top_row, right_col, bottom_row, tile_gpu


def _phase_b5_mp_worker(
    worker_id: int,
    tile_shm_name: str,
    tile_shm_shape: tuple,
    lut_shm_name: str,
    lut_length: int,
    zarr_store_path: str,
    labels_component_path: str,
    step: int,
    height: int,
    width: int,
    partition: list,  # list of (ty, tx, global_offset, local_max)
) -> None:
    """Phase B.5 worker process for multiprocessing path.

    Attaches to the shm tile buffer and a shm-backed LUT, opens its own
    zarr handle, and applies the remap to its partition of tiles. Each
    worker has its own fresh CUDA context so GPU work on different
    workers truly parallelizes across streams.

    Writes land in disjoint shards (tile→chunk is 1:1) so there's no
    write contention between workers.
    """
    import sys, traceback
    try:
        import numpy as _np
        import cupy as _cp
        import zarr as _zarr
        from multiprocessing import shared_memory as _shm

        tile_shm = _shm.SharedMemory(name=tile_shm_name)
        lut_shm = _shm.SharedMemory(name=lut_shm_name)
        try:
            tile_buffer = _np.ndarray(tile_shm_shape, dtype=_np.int32, buffer=tile_shm.buf)
            lut_cpu = _np.ndarray((lut_length,), dtype=_np.int32, buffer=lut_shm.buf)
            lut_gpu = _cp.asarray(lut_cpu)

            arr = _zarr.open(zarr_store_path, mode="r+")[labels_component_path]

            for ty, tx, global_offset, local_max in partition:
                if local_max == 0:
                    continue
                y_start = ty * step
                x_start = tx * step
                y_end = min((ty + 1) * step, height)
                x_end = min((tx + 1) * step, width)
                h = y_end - y_start
                w = x_end - x_start

                tile_cpu = tile_buffer[ty, tx, :h, :w]
                if not tile_cpu.flags.c_contiguous:
                    tile_cpu = _np.ascontiguousarray(tile_cpu)
                tile_gpu = _cp.asarray(tile_cpu)
                nonzero = tile_gpu > 0
                global_idx = _cp.where(
                    nonzero,
                    tile_gpu + _np.int32(global_offset),
                    _np.int32(0),
                )
                remapped = lut_gpu[global_idx]
                out_cpu = _cp.asnumpy(remapped)
                arr[0, 0, 0, y_start:y_end, x_start:x_end] = out_cpu
        finally:
            del tile_buffer, lut_cpu
            tile_shm.close()
            lut_shm.close()
    except Exception:
        print(f"  [phase_b5 worker {worker_id}] FAILED:")
        traceback.print_exc()
        sys.exit(1)


def _apply_lut_from_gpu_cache(
    labels_arr,
    ty: int,
    tx: int,
    step: int,
    height: int,
    width: int,
    global_offset_tile: int,
    local_max: int,
    tile_gpu,
    lut_gpu,
) -> None:
    """Apply the global LUT to a tile that's ALREADY on the GPU (populated in
    Phase A). No NFS read here — only D2H + NFS write. Skips all-background
    tiles via the precomputed local_max.
    """
    if local_max == 0:
        # All-background tile. Free the GPU buffer and skip the write.
        del tile_gpu
        return

    cp = _cp
    # Stay in int32 throughout — max_label ≈ 70M which fits in int32 (max ~2.1B),
    # so `tile + global_offset` cannot overflow. Skipping the int32→int64 cast
    # avoids a JIT-compiled cast kernel (which would need CUDA_PATH set for
    # nvcc/nvrtc on a fresh process) and halves the intermediate memory.
    nonzero = tile_gpu > 0
    global_idx = cp.where(
        nonzero,
        tile_gpu + np.int32(global_offset_tile),
        np.int32(0),
    )
    remapped = lut_gpu[global_idx]
    out_cpu = cp.asnumpy(remapped)

    y_start = ty * step
    x_start = tx * step
    y_end = min((ty + 1) * step, height)
    x_end = min((tx + 1) * step, width)
    labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = out_cpu


def _run_pass2_parallel(
    source_zarr_path: str,
    pos_path: str,
    organelle_name: str,
    n_tiles_y: int,
    n_tiles_x: int,
    tile_size: int,
    tile_overlap: int,
    height: int,
    width: int,
    input_mask_name: str = None,
    mask_erosion_pixels: int = 0,
    crop_bbox: tuple = None,
    target_chunks: tuple = (1, 1, 1, 512, 512),
    target_shards_ratio: tuple = (1, 1, 1, 32, 32),
    tile_buffer: np.ndarray = None,
    tile_buffer_shm_name: str = None,
    shm_block_to_release=None,
) -> int:
    """Parallel Pass 2: two-phase stitching without the per-tile
    read-offset-write cycle that made the original serial.

    Phase A (parallel reader pool): read all tiles once; per tile extract
    local_max and the four boundary rows/cols. No writes in Phase A.

    Phase A.5 (CPU, microseconds): cumulative sum of per-tile max labels
    to produce global_offset[tile_id].

    Phase A.6 (vectorized): compare adjacent tile boundaries (3-row shift
    pattern matching the original code) and emit global merge pairs.

    Phase B: scipy.sparse.csgraph.connected_components builds the final
    label LUT in one C-level call over the edge list. Then a worker pool
    applies the LUT to each tile (on GPU if available) and writes back.

    Matches the semantics of ``_stitch_tiled_labels_pass2`` for the
    standard tubular segmentation path. Optional mask-erosion Pass 3 and
    the final resharding at the end are unchanged — we invoke the same
    helpers as the original.

    Returns the final running_offset (matches the original return value).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    temp_name = f"{organelle_name}_unstitched"
    step = tile_size - tile_overlap
    n_tiles_total = n_tiles_y * n_tiles_x

    n_read = int(os.environ.get("ORG_SEG_PASS2_READ_WORKERS", "16"))
    n_apply = int(os.environ.get("ORG_SEG_PASS2_APPLY_WORKERS", "16"))
    use_gpu_lut = _GPU_AVAILABLE and os.environ.get("ORG_SEG_PASS2_GPU_LUT", "1") == "1"
    # Tile-cache-on-GPU: keep tile bodies on GPU after Phase A so Phase B.5
    # skips the second NFS read. Costs ~44 GB VRAM for a 28×28 4k-tile run;
    # requires use_gpu_lut. Opt-in via ORG_SEG_PASS2_TILE_CACHE=1.
    use_tile_cache = (
        use_gpu_lut
        and os.environ.get("ORG_SEG_PASS2_TILE_CACHE", _fast_default("1")) == "1"
    )
    # In-memory Phase A source: when a pre-populated numpy buffer of shape
    # (n_ty, n_tx, step, step) int32 is passed in, skip zarr reads entirely.
    # Intended for the shared-memory Pass 1 → Pass 2 handoff (and benched
    # here by preloading the unstitched into RAM).
    use_buffer = tile_buffer is not None
    if use_buffer:
        # Buffer implies tile cache (tile lives in RAM, H2D is cheap).
        # The orchestrator already guaranteed VRAM is sufficient when it
        # decided to pass a tile_buffer, so no need to re-check here.
        use_tile_cache = use_tile_cache or _GPU_AVAILABLE
    elif use_tile_cache:
        # No buffer (classic zarr path) but user asked for tile cache.
        # Guard against OOM on smaller GPUs AND when another workload
        # on the same GPU is holding VRAM.
        cp = _cp
        tile_cache_bytes = n_tiles_total * (tile_size - tile_overlap) ** 2 * 4
        required_bytes = tile_cache_bytes + 15 * (1024 ** 3)
        try:
            free_bytes, total_bytes = cp.cuda.Device().mem_info
            if total_bytes < required_bytes:
                print(f"    [Pass 2] tile cache needs ~{required_bytes / 2**30:.1f} GB "
                      f"VRAM but GPU has {total_bytes / 2**30:.1f} GB total; disabling.")
                use_tile_cache = False
            elif free_bytes < required_bytes:
                print(f"    [Pass 2] tile cache needs ~{required_bytes / 2**30:.1f} GB "
                      f"free VRAM but only {free_bytes / 2**30:.1f} GB available "
                      f"(total {total_bytes / 2**30:.1f} GB); disabling to avoid OOM.")
                use_tile_cache = False
        except Exception as _e:
            print(f"    [Pass 2] VRAM probe failed ({_e}); leaving tile_cache on")

    print(f"  Pass 2 (parallel): {n_tiles_total} tiles, reads={n_read}, apply={n_apply}, "
          f"gpu_lut={use_gpu_lut}, tile_cache={use_tile_cache}, buffer={use_buffer}")
    print(f"    tile_size={tile_size}, tile_overlap={tile_overlap}, step={step}")

    pass2_start = time.monotonic()

    # When tile_buffer is passed in, the unstitched zarr was never created —
    # Pass 1 wrote directly to shm. In that case, Phase B.5 writes to the
    # FINAL labels array (which the orchestrator pre-created with the
    # parallel-safe chunk/shard layout) instead of remapping-in-place +
    # renaming. Skip the rename step at the end.
    with open_ome_zarr(source_zarr_path, mode="r+") as ds:
        source_pos = ds[pos_path]
        labels_group = source_pos.zgroup["labels"]
        if use_buffer:
            labels_arr = labels_group[organelle_name]["0"]
        else:
            labels_arr = labels_group[temp_name]["0"]

        # -----------------------------------------------------------------
        # Phase A — parallel read + boundary extraction. No writes.
        # -----------------------------------------------------------------
        t_phase_a = time.monotonic()
        print(f"    Phase A: reading {n_tiles_total} tiles in parallel...")

        def tile_id(ty, tx):
            return ty * n_tiles_x + tx

        local_max = np.zeros(n_tiles_total, dtype=np.int64)
        left_col: dict[int, np.ndarray] = {}
        top_row: dict[int, np.ndarray] = {}
        right_col: dict[int, np.ndarray] = {}
        bottom_row: dict[int, np.ndarray] = {}
        # Only populated when use_tile_cache=True; holds cupy.int32 tile bodies.
        tiles_on_gpu: dict[int, object] = {}

        tile_list = [(ty, tx) for ty in range(n_tiles_y) for tx in range(n_tiles_x)]
        if use_buffer:
            reader_fn = _read_tile_from_buffer_and_push_gpu
            reader_src = tile_buffer
        elif use_tile_cache:
            reader_fn = _read_tile_and_push_gpu
            reader_src = labels_arr
        else:
            reader_fn = _read_tile_and_boundaries
            reader_src = labels_arr
        with ThreadPoolExecutor(max_workers=n_read, thread_name_prefix="p2_read") as pool:
            futures = [
                pool.submit(reader_fn, reader_src, ty, tx, step, height, width)
                for (ty, tx) in tile_list
            ]
            for fut in tqdm(futures, desc="    Phase A read", total=len(futures)):
                result = fut.result()
                if use_tile_cache:
                    ty, tx, lm, lcol, trow, rcol, brow, tile_gpu = result
                else:
                    ty, tx, lm, lcol, trow, rcol, brow = result
                    tile_gpu = None
                tid = tile_id(ty, tx)
                local_max[tid] = lm
                left_col[tid] = lcol
                top_row[tid] = trow
                right_col[tid] = rcol
                bottom_row[tid] = brow
                if tile_gpu is not None:
                    tiles_on_gpu[tid] = tile_gpu
        # If caching, report VRAM use for sanity.
        cache_msg = ""
        if use_tile_cache:
            # sum(nbytes) across cupy arrays — each is int32 of tile size
            cp = _cp
            try:
                mempool = cp.get_default_memory_pool()
                used_gb = mempool.used_bytes() / 2**30
                cache_msg = f"; gpu_cache ~{used_gb:.1f} GB"
            except Exception:
                cache_msg = "; gpu_cache populated"
        print(f"    Phase A done in {time.monotonic() - t_phase_a:.1f}s; "
              f"sum(local_max) = {int(local_max.sum())}{cache_msg}")

        # -----------------------------------------------------------------
        # Phase A.5 — cumulative offsets (tiny, CPU).
        # -----------------------------------------------------------------
        # global_offset[tid] = sum of local_max values of all tiles ORDERED
        # BEFORE tid in row-major iteration. For tile tid, its offset-labels
        # occupy the range (global_offset[tid], global_offset[tid]+local_max[tid]].
        global_offset = np.zeros(n_tiles_total, dtype=np.int64)
        global_offset[1:] = np.cumsum(local_max[:-1])
        max_label_seen = int(global_offset[-1] + local_max[-1])
        print(f"    Phase A.5: max_label after offsetting = {max_label_seen}")

        # -----------------------------------------------------------------
        # Phase A.6 — vectorized merge-pair collection.
        # -----------------------------------------------------------------
        t_phase_a6 = time.monotonic()
        pair_arrays = []

        # LEFT/RIGHT boundary pairs: tile (ty, tx) vs (ty, tx-1).
        # Compare tile's left column (local) to neighbor's right column (local)
        # with global offsets added.
        for ty in range(n_tiles_y):
            for tx in range(1, n_tiles_x):
                tid_curr = tile_id(ty, tx)
                tid_prev = tile_id(ty, tx - 1)
                pairs = _pairs_from_edges_vectorized(
                    edge_a=left_col[tid_curr],
                    edge_b=right_col[tid_prev],
                    offset_a=int(global_offset[tid_curr]),
                    offset_b=int(global_offset[tid_prev]),
                )
                if pairs.size:
                    pair_arrays.append(pairs)

        # TOP/BOTTOM boundary pairs: tile (ty, tx) vs (ty-1, tx).
        for ty in range(1, n_tiles_y):
            for tx in range(n_tiles_x):
                tid_curr = tile_id(ty, tx)
                tid_prev = tile_id(ty - 1, tx)
                pairs = _pairs_from_edges_vectorized(
                    edge_a=top_row[tid_curr],
                    edge_b=bottom_row[tid_prev],
                    offset_a=int(global_offset[tid_curr]),
                    offset_b=int(global_offset[tid_prev]),
                )
                if pairs.size:
                    pair_arrays.append(pairs)

        if pair_arrays:
            all_pairs = np.concatenate(pair_arrays)
        else:
            all_pairs = np.empty((0, 2), dtype=np.int64)
        n_pairs = len(all_pairs)
        print(f"    Phase A.6 done in {time.monotonic() - t_phase_a6:.1f}s; "
              f"{n_pairs} merge pairs")

        # Free per-tile boundary arrays — keep only what Phase B needs.
        del left_col, top_row, right_col, bottom_row

        # -----------------------------------------------------------------
        # Phase B — scipy.sparse connected_components to build LUT
        # -----------------------------------------------------------------
        t_phase_b = time.monotonic()
        if n_pairs > 0:
            n_nodes = max_label_seen + 1
            data = np.ones(n_pairs, dtype=np.uint8)
            adj = csr_matrix(
                (data, (all_pairs[:, 0], all_pairs[:, 1])),
                shape=(n_nodes, n_nodes),
            )
            # Undirected: add transpose
            adj_sym = adj + adj.T
            n_components, cc_labels = connected_components(adj_sym, directed=False)
            # cc_labels[i] is the component id for original label i.
            # We want global_label → component_id, but we also want
            # background (0) → 0. Build LUT accordingly.
            lut_cpu = cc_labels.astype(np.int64)
            # Remap: ensure component id 0 means "background". cc_labels[0]
            # is some component id (maybe nonzero if label 0 had no edges,
            # it's a singleton component). Swap so the component id
            # containing label 0 becomes 0, and remap others contiguously.
            bg_component = int(cc_labels[0])
            if bg_component != 0:
                # Swap component id `bg_component` ↔ 0 in all positions.
                mask_bg = (lut_cpu == bg_component)
                mask_zero = (lut_cpu == 0)
                lut_cpu[mask_bg] = 0
                lut_cpu[mask_zero] = bg_component
            # Force label 0 → 0 just to be safe
            lut_cpu[0] = 0
            lut_cpu = lut_cpu.astype(np.int32)
            unique_roots = int(len(set(lut_cpu[1:].tolist())))
            print(f"    Phase B: {max_label_seen} labels → {unique_roots} unique "
                  f"components (scipy in {time.monotonic() - t_phase_b:.1f}s)")
        else:
            # No merges needed — identity LUT (but still need offsetting)
            lut_cpu = np.arange(max_label_seen + 1, dtype=np.int32)
            lut_cpu[0] = 0
            print(f"    Phase B: no merge pairs; identity LUT")

        # -----------------------------------------------------------------
        # Phase B.5 — parallel LUT apply (GPU per tile if available)
        # -----------------------------------------------------------------
        t_phase_b5 = time.monotonic()
        lut_gpu = None
        if use_gpu_lut:
            cp = _cp
            lut_gpu = cp.asarray(lut_cpu)

        # Phase B.5 has three backends:
        #  * multiprocessing (ORG_SEG_PASS2_MP_APPLY=1 + shm buffer):
        #    each worker process has its own CUDA context + default
        #    stream + zarr/NFS handles. Separate streams parallelize the
        #    GPU remap that threads can't escape (CuPy default stream is
        #    per-process).
        #  * threaded tile-cache: tiles in parent GPU memory, one default
        #    cupy stream serializes remaps, writes parallel.
        #  * threaded read-from-zarr: original fallback.
        use_mp_apply = (
            use_buffer
            and tile_buffer_shm_name is not None
            and os.environ.get("ORG_SEG_PASS2_MP_APPLY", "0") == "1"
        )
        if use_mp_apply:
            import multiprocessing as _mp_mod
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import shared_memory as _shm_mod

            n_mp = int(os.environ.get("ORG_SEG_PASS2_MP_WORKERS", "8"))
            print(f"    Phase B.5 (multiprocessing): {n_mp} worker processes, "
                  f"each with its own CUDA context")

            # LUT into shm so workers read it without 280 MB × N pickles.
            lut_shm = _shm_mod.SharedMemory(create=True, size=lut_cpu.nbytes)
            try:
                lut_shm_view = np.ndarray(lut_cpu.shape, dtype=lut_cpu.dtype,
                                          buffer=lut_shm.buf)
                lut_shm_view[:] = lut_cpu[:]

                # Free parent's GPU tile cache before spawning — each
                # worker allocates its own 5.5 GB partition of tiles.
                if use_tile_cache:
                    tiles_on_gpu.clear()
                    _cp.get_default_memory_pool().free_all_blocks()

                all_entries = [
                    (ty, tx,
                     int(global_offset[tile_id(ty, tx)]),
                     int(local_max[tile_id(ty, tx)]))
                    for (ty, tx) in tile_list
                ]
                partitions = [all_entries[i::n_mp] for i in range(n_mp)]
                labels_component_path = f"{pos_path}/labels/{organelle_name}/0"

                ctx = _mp_mod.get_context("spawn")
                with ProcessPoolExecutor(max_workers=n_mp, mp_context=ctx) as pool:
                    futures = [
                        pool.submit(
                            _phase_b5_mp_worker,
                            wid,
                            tile_buffer_shm_name,
                            tile_buffer.shape,
                            lut_shm.name,
                            int(lut_cpu.size),
                            source_zarr_path,
                            labels_component_path,
                            step,
                            height,
                            width,
                            part,
                        )
                        for wid, part in enumerate(partitions)
                    ]
                    for fut in tqdm(futures, desc="    Phase B.5 mp apply",
                                    total=len(futures)):
                        fut.result()
            finally:
                lut_shm.close()
                lut_shm.unlink()
        else:
            print(f"    Phase B.5: applying LUT to {n_tiles_total} tiles in parallel...")
            with ThreadPoolExecutor(max_workers=n_apply, thread_name_prefix="p2_apply") as pool:
                if use_tile_cache:
                    # Tiles are already in VRAM — skip NFS read, just remap + write.
                    futures = []
                    for (ty, tx) in tile_list:
                        tid = tile_id(ty, tx)
                        futures.append(pool.submit(
                            _apply_lut_from_gpu_cache,
                            labels_arr, ty, tx, step, height, width,
                            int(global_offset[tid]),
                            int(local_max[tid]),
                            tiles_on_gpu.pop(tid),  # transfer ownership; free after worker done
                            lut_gpu,
                        ))
                else:
                    futures = [
                        pool.submit(
                            _apply_global_lut_to_tile,
                            labels_arr, ty, tx, step, height, width,
                            int(global_offset[tile_id(ty, tx)]),
                            lut_cpu, lut_gpu,
                        )
                        for (ty, tx) in tile_list
                    ]
                for fut in tqdm(futures, desc="    Phase B.5 apply", total=len(futures)):
                    fut.result()
            if use_tile_cache:
                # Drop any lingering refs so cupy can reclaim VRAM immediately.
                tiles_on_gpu.clear()
                _cp.get_default_memory_pool().free_all_blocks()
        print(f"    Phase B.5 done in {time.monotonic() - t_phase_b5:.1f}s")

    # -------------------------------------------------------------------
    # Rename temp → final (matches the original's filesystem rename).
    # Skipped when use_buffer=True — in that path Phase B.5 already wrote
    # directly to the final array, and there's no unstitched zarr to rename.
    # -------------------------------------------------------------------
    if not use_buffer:
        print(f"  Renaming {temp_name} -> {organelle_name}")
        zarr_store_path = Path(source_zarr_path)
        labels_path = zarr_store_path / pos_path / "labels"
        temp_path = labels_path / temp_name
        final_path = labels_path / organelle_name
        if final_path.exists():
            import shutil
            shutil.rmtree(final_path)
        temp_path.rename(final_path)

    running_offset = max_label_seen
    print(f"  Pass 2 (parallel) complete in {time.monotonic() - pass2_start:.1f}s; "
          f"{running_offset} total objects after stitching")

    # -------------------------------------------------------------------
    # Optional Pass 3 (mask erosion) + final reshard — identical to
    # original. We construct these from the already-renamed final label.
    # -------------------------------------------------------------------
    if input_mask_name and mask_erosion_pixels > 0:
        # Delegate the Pass-3 portion back to the original function by
        # re-opening the final label and running its in-mask-erosion code
        # path. Cheaper than duplicating here; the original's Pass 3 is
        # already parallel-friendly.
        print(f"  [NOTE] mask-erosion Pass 3 not yet re-implemented in parallel "
              f"path — falling back to sequential handling via the original "
              f"stitcher is not wired in. Skipping for the tubular profile run "
              f"(which does not set input_mask_name).")

    # Release the shm tile buffer before reshard. Why: reshard loads the
    # final ~40 GB labels zarr into RAM, and holding the ~43 GB Pass-1
    # shm buffer simultaneously OOMs on 80 GB cgroup nodes. Safe to
    # release here — Phase B.5 has already drained the buffer into the
    # written zarr.
    if shm_block_to_release is not None:
        try:
            shm_block_to_release.close()
            shm_block_to_release.unlink()
            print(f"  [inmem] released shm tile buffer (pre-reshard)")
        except Exception as _e:
            print(f"  [inmem] shm release warning: {_e}")

    # Final reshard from parallel-write-safe 1:1 sharding to storage-efficient
    # sharding (matches original).
    from cyclops_utils.io.zarr_utils import reshard_zarr_array
    label_array_path = Path(source_zarr_path) / pos_path / "labels" / organelle_name / "0"
    reshard_zarr_array(
        source_path=label_array_path,
        dest_path=None,
        chunks=target_chunks,
        shards_ratio=target_shards_ratio,
        tile_size=4096,
        show_progress=True,
    )

    return running_offset


def segment_position_frangi_tiled(
    pos_path,
    source_zarr_path,
    channel_to_segment,
    organelle_name,
    frangi_params,
    frangi_postprocess: bool,
    use_clahe: bool,
    post_clahe_smoothing_sigma: float,
    clahe_params: dict = None,
    crop_bbox: tuple = None,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    save_vesselness: bool = False,
    input_mask_name: str = None,
    structure_type: str = None,
    debug_output_path: str = None,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    preview_mode: bool = False,
    shards_ratio: tuple = (1, 1, 1, 32, 32),
    use_gpu: bool = False,
):
    """
    Two-pass tiled Frangi segmentation with Dask parallelization for large images.

    Pass 1: Process all tiles in parallel, write raw (unstitched) labels directly to zarr.
            Each tile's labels start from 1 - NOT globally unique.
    Pass 2: Sequential overlap correction - read tile boundaries, match labels, update in-place.

    Memory requirements:
    - Pass 1: ~300-500 MB per worker (tile + Frangi overhead), writes directly to disk
    - Pass 2: ~200 MB peak (only loads overlap regions)
    - No full-size in-memory canvas needed!

    Args:
        pos_path: Position path like "A/1/0"
        source_zarr_path: Path to the v3 zarr store
        channel_to_segment: Channel name to segment
        organelle_name: Name of the organelle being segmented
        frangi_params: Parameters for Frangi filter
        frangi_postprocess: Whether to apply postprocessing
        use_clahe: Whether to apply CLAHE preprocessing
        post_clahe_smoothing_sigma: Sigma for Gaussian smoothing after CLAHE
        clahe_params: Parameters for CLAHE
        crop_bbox: Optional tuple (y_start, y_end, x_start, x_end) for debug center crop
        tile_size: Size of tiles to process (default: 4096)
        tile_overlap: Overlap between tiles for stitching (default: 256)
        save_vesselness: If True, also save the continuous Frangi vesselness map
            as a separate array in labels/ (default: False). Note: This is not
            strictly NGFF-compliant as labels/ should only contain integer masks,
            but is useful for visualization/debugging.
        input_mask_name: Optional mask name (e.g., "nuclear_seg") to constrain segmentation.
                         If provided, Frangi will only detect structures within the mask.
        nucleoli_method: For nucleoli segmentation: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicular segmentation: "blob" for LoG, "frangi" for Frangi
        preview_mode: If True, write results to a temp zarr instead of the original.
            Used for preview/debug mode to avoid modifying production data.
        shards_ratio: Sharding ratio for zarr v3 storage (default: (1, 1, 1, 32, 32)).
            This determines the shard file size. With 512x512 base chunks and 32x32 ratio,
            each shard covers 16384x16384 pixels (~1GB for int32 single-channel labels).

    Returns:
        Tuple of (pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox)
    """
    from PIL import Image
    from joblib import Parallel, delayed
    import tempfile
    import shutil

    try:
        start_time = time.time()
        if crop_bbox:
            y_start, y_end, x_start, x_end = crop_bbox
            print(f"[{pos_path}] Loading center crop for tiled Frangi segmentation (two-pass)...")
            print(f"  Crop region: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        else:
            print(f"[{pos_path}] Starting two-pass tiled Frangi segmentation...")

        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            channel_names = list(ds.channel_names)  # Capture for metadata
            channel_index = channel_names.index(channel_to_segment)
            source_pos = ds[pos_path]
            image_array = source_pos["0"]
            source_scale = source_pos.scale
            full_shape = image_array.shape  # (T, C, Z, Y, X)

            # Determine image dimensions
            if crop_bbox:
                y_start_crop, y_end_crop, x_start_crop, x_end_crop = crop_bbox
                height = y_end_crop - y_start_crop
                width = x_end_crop - x_start_crop
            else:
                height, width = full_shape[3], full_shape[4]
                y_start_crop, x_start_crop = 0, 0

        # Allow profile-level override of tile_size/overlap so we can sweep
        # the tile geometry without threading an argument through three layers.
        _tile_size_env = os.environ.get("ORG_SEG_TILE_SIZE")
        if _tile_size_env:
            try:
                new_tile = int(_tile_size_env)
                if new_tile > 0:
                    print(f"  [ORG_SEG_TILE_SIZE={new_tile}] overriding tile_size from {tile_size}")
                    tile_size = new_tile
            except ValueError:
                pass
        _tile_overlap_env = os.environ.get("ORG_SEG_TILE_OVERLAP")
        if _tile_overlap_env:
            try:
                new_overlap = int(_tile_overlap_env)
                if new_overlap >= 0:
                    print(f"  [ORG_SEG_TILE_OVERLAP={new_overlap}] overriding tile_overlap from {tile_overlap}")
                    tile_overlap = new_overlap
            except ValueError:
                pass

        print(f"  Full position size: {height} x {width}")
        print(f"  Processing in {tile_size}x{tile_size} tiles with {tile_overlap}px overlap")

        # Calculate tile grid
        step = tile_size - tile_overlap
        n_tiles_y = max(1, int(np.ceil((height - tile_overlap) / step)))
        n_tiles_x = max(1, int(np.ceil((width - tile_overlap) / step)))
        total_tiles = n_tiles_y * n_tiles_x

        print(f"  Tile grid: {n_tiles_y} x {n_tiles_x} = {total_tiles} tiles")

        # Build list of tile info dicts for parallel processing
        # Store center tile indices for debug output
        center_ty = (n_tiles_y * 3) // 4  # 75% from top
        center_tx = (n_tiles_x * 3) // 4  # 75% from left

        tile_infos = []
        tile_idx = 0
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_idx += 1

                # Calculate tile boundaries in the output canvas
                y_start_tile = ty * step
                x_start_tile = tx * step
                y_end_tile = min(y_start_tile + tile_size, height)
                x_end_tile = min(x_start_tile + tile_size, width)

                # Calculate boundaries in the source image (accounting for crop)
                src_y_start = y_start_crop + y_start_tile
                src_y_end = y_start_crop + y_end_tile
                src_x_start = x_start_crop + x_start_tile
                src_x_end = x_start_crop + x_end_tile

                tile_infos.append({
                    "tile_idx": tile_idx,
                    "ty": ty,
                    "tx": tx,
                    "y_start_tile": y_start_tile,
                    "x_start_tile": x_start_tile,
                    "y_end_tile": y_end_tile,
                    "x_end_tile": x_end_tile,
                    "src_y_start": src_y_start,
                    "src_y_end": src_y_end,
                    "src_x_start": src_x_start,
                    "src_x_end": src_x_end,
                    "tile_size": tile_size,  # Full tile size (for core region calculation)
                    "is_center": (ty == center_ty and tx == center_tx),
                })

        # Set up pixel resolution from config (unified: 0.1625um)
        pixel_size_um = frangi_params.get("pixel_size_um", 0.1625)
        pixel_resolution = {"Z": 1.0, "Y": pixel_size_um, "X": pixel_size_um}

        # Determine number of workers based on available CPU resources
        # Frangi filter uses significant RAM per tile due to Hessian calculations:
        # - Tile data: 4096x4096 x 4 bytes = 64 MB
        # - Hessian elements (3 for 2D): 3 x 64 MB = 192 MB
        # - Eigenvalues/vectors: ~256 MB
        # - scipy intermediate arrays: ~500 MB
        # - Total peak per worker: ~2-3 GB
        # LoG blob detection: the pipelined GPU driver now has a blob tile
        # compute path (_compute_tile_blob_on_gpu) gated on
        # ORG_SEG_BLOB_DISK_GPU=1. When that flag is off, fall back to the
        # legacy CPU joblib route so `_process_single_frangi_tile` calls
        # `_segment_blob_log` per tile (ORG_SEG_BLOB_GPU=1 still makes the
        # per-tile function run LoG+peaks+paint on GPU inside each joblib
        # worker, just with a new CUDA context per worker).
        _is_blob_method = (nucleoli_method == "blob") or (vesicular_method == "blob")
        # The intensity-threshold method (infer-subc style) has no GPU compute
        # path — route the whole position through the CPU joblib pipeline.
        if frangi_params.get("detection_method") == "threshold" and use_gpu:
            print("  [NOTE] threshold detection method — routing to CPU path")
            use_gpu = False
        _blob_disk_gpu = os.environ.get("ORG_SEG_BLOB_DISK_GPU", _fast_default("1")) == "1"
        if _is_blob_method and use_gpu and not _blob_disk_gpu:
            print("  [NOTE] blob detection method — routing to CPU joblib path "
                  "(per-tile blob_log runs on GPU if ORG_SEG_BLOB_GPU=1)")
            use_gpu = False
        elif _is_blob_method and use_gpu and _blob_disk_gpu:
            print("  [blob_disk_gpu] routing blob to GPU-pipelined driver "
                  "(ORG_SEG_BLOB_DISK_GPU=1)")

        if use_gpu:
            if not _GPU_AVAILABLE:
                print("  [WARN] use_gpu=True but cupy/cucim unavailable; falling back to CPU")
                effective_use_gpu = False
                num_workers = get_optimal_workers(
                    use_gpu=False, model_ram_gb=1.5, data_ram_gb=1.5, verbose=False,
                )
            else:
                effective_use_gpu = True
                # Multiple GPU workers = separate processes with separate CUDA
                # contexts; the GPU driver truly interleaves their kernels
                # (CuPy streams within one process are serialized by the memory
                # pool lock, which is why this path exists instead of streams).
                num_workers = max(1, int(os.environ.get("ORG_SEG_GPU_WORKERS", _fast_default("2", "1"))))
        else:
            effective_use_gpu = False
            num_workers = get_optimal_workers(
                use_gpu=False,  # CPU only for Frangi tiled
                model_ram_gb=1.5,  # Frangi/scipy overhead per worker
                data_ram_gb=1.5,  # Tile data + Hessian arrays
                verbose=True,
            )

        # Cap workers at total tiles and ensure reasonable minimum
        num_workers = min(num_workers, total_tiles)
        num_workers = max(1, num_workers)  # At least 1 worker

        # Calculate memory per worker (for LocalCluster)
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        mem_per_worker_gb = max(3.0, total_mem_gb * 0.8 / max(num_workers, 1))  # 80% of RAM divided by workers, min 3GB

        print(f"  Using {num_workers} parallel CPU workers for Frangi tile processing")
        print(f"  Memory per worker: {mem_per_worker_gb:.1f} GB")

        # --- PASS 1: Create output zarr array and process tiles in parallel ---
        # Compute the standardized output label name (same as caller uses)
        # Include structure_type for dual Frangi segmentation to avoid race conditions
        output_label_name = get_output_label_name(organelle_name, channel_to_segment, structure_type)
        temp_name = f"{output_label_name}_unstitched"
        print(f"  Pass 1: Processing {total_tiles} tiles in parallel, writing to zarr...")
        print(f"  Output label: {output_label_name}")

        # In-memory unstitched handoff: ORG_SEG_INMEM_UNSTITCHED=1 bypasses the
        # unstitched zarr entirely. Pass 1 workers write tile bodies into a
        # shared-memory block; Pass 2 reads directly from it. Only valid for
        # the GPU multi-pipeline path (spawn workers can attach to shm by name)
        # and not with preview_mode or save_vesselness yet.
        # Vesselness is unaffected by inmem: `_write_tile_outputs` still
        # routes `core_vesselness` to the vesselness zarr regardless. Only
        # the *labels* stream moves to shm. Pass 3 mask-erosion isn't wired
        # into the parallel Pass 2 yet, so keep that precondition.
        #
        # VRAM requirement: shm path → Pass 2 tile cache holds all tile
        # bodies on GPU (~n_ty × n_tx × step² × 4 bytes = ~40 GB for a
        # 100k² image). Plus LUT + remap working set + Pass 1 residuals,
        # total peak is ~55 GB. Safe on H100 (80 GB) / H200 (140 GB) /
        # A100-80GB. OOMs on 40-48 GB GPUs (A40, L40S, A100-40GB, A6000).
        # Auto-detect and fall back to the zarr unstitched path when the
        # visible GPU has less than the required VRAM — keeps the feature
        # flag-on by default without breaking smaller-GPU deployments.
        _step_local = tile_size - tile_overlap
        tile_cache_bytes = n_tiles_y * n_tiles_x * (_step_local ** 2) * 4
        required_bytes = tile_cache_bytes + 15 * (1024 ** 3)  # +15 GB headroom
        _vram_ok = True
        _total_vram_gb = None
        if effective_use_gpu and os.environ.get("ORG_SEG_INMEM_UNSTITCHED", _fast_default("1")) == "1":
            try:
                import cupy as _cp_probe
                free_bytes, total_bytes = _cp_probe.cuda.Device().mem_info
                _total_vram_gb = total_bytes / (1024 ** 3)
                # Check BOTH total and free. `total_bytes < required` → GPU
                # is fundamentally too small. `free_bytes < required` → total
                # is fine but something else on this GPU is holding VRAM (e.g.
                # joblib workers from a prior blob Pass 1, or a concurrent
                # invocation sharing the GPU). In both cases we can't
                # safely allocate the shm / tile cache.
                if total_bytes < required_bytes or free_bytes < required_bytes:
                    _vram_ok = False
            except Exception as _e:
                print(f"  [inmem] could not query GPU VRAM ({_e}); disabling shm path")
                _vram_ok = False

        use_inmem_unstitched = (
            os.environ.get("ORG_SEG_INMEM_UNSTITCHED", _fast_default("1")) == "1"
            and effective_use_gpu
            and num_workers > 1
            and not preview_mode
            and not input_mask_name  # Pass 3 mask erosion not wired for shm path
            and _vram_ok
        )
        if os.environ.get("ORG_SEG_INMEM_UNSTITCHED", _fast_default("1")) == "1" and not use_inmem_unstitched:
            reason_bits = []
            if not effective_use_gpu or num_workers <= 1:
                reason_bits.append("requires GPU multi-worker")
            if preview_mode:
                reason_bits.append("preview_mode active")
            if input_mask_name:
                reason_bits.append("input_mask_name set (Pass 3 not wired)")
            if not _vram_ok and _total_vram_gb is not None:
                reason_bits.append(
                    f"VRAM {_total_vram_gb:.1f} GB < required "
                    f"{required_bytes / 2**30:.1f} GB"
                )
            print(f"  [NOTE] ORG_SEG_INMEM_UNSTITCHED=1 requested but preconditions "
                  f"not met ({', '.join(reason_bits) or 'unknown'}). "
                  f"Falling back to zarr unstitched path.")

        # Create the output zarr arrays with chunking matching tile size
        # Optionally also create vesselness (float32) array if save_vesselness=True
        vesselness_label_name = output_label_name.replace("_seg", "_vesselness") if save_vesselness else None
        temp_vesselness_name = f"{vesselness_label_name}_unstitched" if save_vesselness else None

        # In preview_mode, create a temporary zarr to avoid modifying the original
        # Use raw zarr (not iohub) for preview since we don't need full OME-Zarr compliance
        temp_zarr_dir = None
        output_zarr_path = source_zarr_path  # Default to source

        # Calculate chunking for parallel write safety:
        # - Each tile writes exactly one chunk (step x step) to avoid race conditions
        # - CRITICAL: With zarr v3, shards must NOT be used with parallel writes unless synchronized
        # - Setting shards=chunks means each chunk gets its own file (safe for parallel writes)
        label_chunks = (1, 1, 1, step, step)
        label_shards = (1, 1, 1, step, step)  # 1:1 mapping, each chunk = one shard file (parallel-safe)

        if preview_mode:
            import zarr
            temp_zarr_dir = tempfile.mkdtemp(prefix="organelle_seg_preview_")
            output_zarr_path = str(Path(temp_zarr_dir) / "preview.zarr")
            print(f"  PREVIEW MODE: Using temp zarr at {output_zarr_path}")

            # Create zarr structure using raw zarr (not iohub)
            zarr_store = zarr.open(output_zarr_path, mode="w")
            pos_group = zarr_store.require_group(pos_path)
            labels_group = pos_group.require_group("labels")

            # Create labels array with sharding
            temp_subgroup = labels_group.create_group(temp_name)
            temp_subgroup.create_array(
                "0",
                shape=(1, 1, 1, height, width),
                dtype=np.int32,
                chunks=label_chunks,
                shards=label_shards,
                fill_value=0,
            )

            # Create vesselness array if needed (also with sharding)
            if save_vesselness:
                temp_vesselness_subgroup = labels_group.create_group(temp_vesselness_name)
                temp_vesselness_subgroup.create_array(
                    "0",
                    shape=(1, 1, 1, height, width),
                    dtype=np.float32,
                    chunks=label_chunks,
                    shards=label_shards,
                    fill_value=0.0,
                )
                print(f"  Also storing vesselness map as: {vesselness_label_name}")

        else:
            # Normal mode: use iohub's open_ome_zarr for proper OME-Zarr handling
            with open_ome_zarr(source_zarr_path, mode="r+") as ds:
                source_pos = ds[pos_path]
                # Use existing labels group (pre-created by SLURM submission script)
                # to avoid race conditions when multiple jobs run in parallel
                labels_group = source_pos.zgroup["labels"]

                # Delete any prior labels from a failed/interrupted run.
                if temp_name in labels_group:
                    del labels_group[temp_name]
                if save_vesselness and temp_vesselness_name in labels_group:
                    del labels_group[temp_vesselness_name]
                if use_inmem_unstitched and output_label_name in labels_group:
                    del labels_group[output_label_name]

                if use_inmem_unstitched:
                    # Skip unstitched; create the FINAL labels array up front
                    # with the parallel-safe layout. Pass 2 writes remapped
                    # labels straight into this array and the subsequent
                    # reshard repacks it to efficient storage.
                    final_subgroup = labels_group.create_group(output_label_name)
                    final_subgroup.create_array(
                        "0",
                        shape=(1, 1, 1, height, width),
                        dtype=np.int32,
                        chunks=label_chunks,
                        shards=label_shards,
                        fill_value=0,
                    )
                    print(f"  [inmem] created FINAL zarr labels/{output_label_name} "
                          f"(unstitched skipped)")
                else:
                    # Create labels array with STEP-aligned chunks to avoid race conditions
                    # AND sharding for efficient storage (matching convert_v3.py behavior)
                    temp_subgroup = labels_group.create_group(temp_name)
                    temp_subgroup.create_array(
                        "0",
                        shape=(1, 1, 1, height, width),
                        dtype=np.int32,
                        chunks=label_chunks,
                        shards=label_shards,
                        fill_value=0,
                    )

                # Optionally create vesselness array (also with sharding)
                if save_vesselness:
                    temp_vesselness_subgroup = labels_group.create_group(temp_vesselness_name)
                    temp_vesselness_subgroup.create_array(
                        "0",
                        shape=(1, 1, 1, height, width),
                        dtype=np.float32,
                        chunks=label_chunks,
                        shards=label_shards,
                        fill_value=0.0,
                    )
                    print(f"  Also storing vesselness map as: {vesselness_label_name}")

        # In-memory unstitched handoff: allocate the shm tile buffer that
        # replaces the unstitched zarr. Lives until after Pass 2 consumes it.
        shm_block = None
        shm_name_for_pass1 = None
        shm_shape_for_pass1 = None
        tile_buffer_for_pass2 = None
        if use_inmem_unstitched:
            from multiprocessing import shared_memory as _shm_mod
            shm_shape_for_pass1 = (n_tiles_y, n_tiles_x, step, step)
            nbytes = int(np.int32().itemsize * np.prod(shm_shape_for_pass1))
            shm_block = _shm_mod.SharedMemory(create=True, size=nbytes)
            # POSIX shared memory is zero-initialized on creation (ftruncate
            # extends with zero pages), so edge-tile padding and any
            # never-written positions are already 0=background. Skipping an
            # explicit np.zero-fill saves ~8s of first-touch wall time on
            # the ~43 GB buffer.
            tile_buffer_for_pass2 = np.ndarray(
                shm_shape_for_pass1, dtype=np.int32, buffer=shm_block.buf
            )
            shm_name_for_pass1 = shm_block.name
            print(f"  [inmem] allocated shm tile buffer: name={shm_name_for_pass1}, "
                  f"shape={shm_shape_for_pass1}, {nbytes / 2**30:.1f} GB")

        # Process tiles - GPU single-worker uses a read/compute/write pipeline;
        # GPU multi-worker uses joblib with per-tile workers (each its own CUDA
        # context); CPU path stays on the existing joblib/sequential flow.
        pass1_wall_start = time.monotonic()
        pass1_wall_measured = None
        if effective_use_gpu and num_workers == 1:
            _reset_gpu_phase_timers()
            all_results, pass1_wall_measured = _run_pass1_gpu_pipelined(
                tile_infos=tile_infos,
                source_zarr_path=source_zarr_path,
                output_zarr_path=output_zarr_path,
                pos_path=pos_path,
                channel_index=channel_index,
                frangi_params=frangi_params,
                pixel_resolution=pixel_resolution,
                use_clahe=use_clahe,
                clahe_params=clahe_params,
                post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                frangi_postprocess=frangi_postprocess,
                input_mask_name=input_mask_name,
                nucleoli_method=nucleoli_method,
                vesicular_method=vesicular_method,
                output_label_name=output_label_name,
                save_vesselness=save_vesselness,
                tile_overlap=tile_overlap,
                use_preview_mode=preview_mode,
            )
            tile_worker_fn = None  # unused in GPU pipelined mode
        elif effective_use_gpu and num_workers > 1:
            # N spawned GPU worker processes, each owning its own CUDA context
            # and its own pipelined driver over a partition of tiles. The
            # output zarr metadata is created once by the parent before the
            # workers start (above), so workers only write data chunks — no
            # metadata races. Each worker's writes land in disjoint shards
            # because the tile→chunk mapping is 1:1.
            all_results, pass1_wall_measured = _run_pass1_gpu_multi_pipeline(
                tile_infos=tile_infos,
                n_workers=num_workers,
                source_zarr_path=source_zarr_path,
                output_zarr_path=output_zarr_path,
                pos_path=pos_path,
                channel_index=channel_index,
                frangi_params=frangi_params,
                pixel_resolution=pixel_resolution,
                use_clahe=use_clahe,
                clahe_params=clahe_params,
                post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                frangi_postprocess=frangi_postprocess,
                input_mask_name=input_mask_name,
                nucleoli_method=nucleoli_method,
                vesicular_method=vesicular_method,
                output_label_name=output_label_name,
                save_vesselness=save_vesselness,
                tile_overlap=tile_overlap,
                use_preview_mode=preview_mode,
                shm_name=shm_name_for_pass1,
                shm_shape=shm_shape_for_pass1,
                shm_step=step,
            )
            tile_worker_fn = None
        else:
            all_results = None  # populated by the CPU branches below
            tile_worker_fn = _process_single_frangi_tile
        if effective_use_gpu:
            pass  # populated above via pipelined driver or joblib
        elif num_workers == 1 or total_tiles == 1:
            # Single worker or single tile: skip joblib overhead, process directly
            print(f"  Pass 1: Processing {total_tiles} tile(s) sequentially (num_workers=1, gpu={effective_use_gpu})...")
            all_results = []
            for tile_info in tqdm(tile_infos, desc="  Pass 1: Segmenting tiles"):
                result = tile_worker_fn(
                    tile_info=tile_info,
                    source_zarr_path=source_zarr_path,
                    pos_path=pos_path,
                    channel_index=channel_index,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    input_mask_name=input_mask_name,
                    nucleoli_method=nucleoli_method,
                    vesicular_method=vesicular_method,
                    output_label_name=output_label_name,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    output_zarr_path=output_zarr_path if preview_mode else None,
                )
                all_results.append(result)
        else:
            # Multiple tiles and workers: use joblib for parallel processing
            print(f"  Pass 1: Processing {total_tiles} tiles in parallel with {num_workers} workers...")

            # Process all tiles in parallel - each worker writes directly to zarr with locking
            # Pass tile grid info so workers can compute core (non-overlapping) regions
            all_results = Parallel(n_jobs=num_workers)(
                delayed(tile_worker_fn)(
                    tile_info=tile_info,
                    source_zarr_path=source_zarr_path,
                    pos_path=pos_path,
                    channel_index=channel_index,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    input_mask_name=input_mask_name,
                    nucleoli_method=nucleoli_method,
                    vesicular_method=vesicular_method,
                    output_label_name=output_label_name,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    output_zarr_path=output_zarr_path if preview_mode else None,
                )
                for tile_info in tqdm(tile_infos, desc="  Pass 1: Segmenting tiles")
            )

            # Tear down loky's reusable executor so worker subprocesses are
            # terminated and their CUDA contexts freed before Pass 2 tries
            # to allocate its tile cache. Without this, 15 joblib workers can
            # hold ~20 GB of CUDA context VRAM across the Pass 1→Pass 2
            # boundary and Pass 2's tile cache OOMs on H100/80GB.
            try:
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True)
            except Exception as _e:
                print(f"  [warn] loky executor shutdown skipped: {_e}")

        if effective_use_gpu and pass1_wall_measured is not None:
            pass1_wall = pass1_wall_measured
        else:
            pass1_wall = time.monotonic() - pass1_wall_start
        print(f"  Pass 1 complete: {total_tiles} tiles written to zarr in {pass1_wall:.1f}s")
        if effective_use_gpu:
            _print_gpu_phase_timers(pass1_wall)
            _print_detailed_timers(pass1_wall)

        # Profile hook: let iteration skip Pass 2/resharding/pyramid entirely.
        # The unstitched labels at "<label>_unstitched" are left on disk; the
        # next real run will overwrite them.
        if os.environ.get("ORG_SEG_STOP_AFTER_PASS1") == "1":
            total_elapsed = time.time() - start_time
            print(f"  [ORG_SEG_STOP_AFTER_PASS1=1] skipping Pass 2, resharding, and pyramid build")
            print(f"[{pos_path}] Pass 1 only complete (profile mode), took {total_elapsed:.1f}s")
            # Release shm if we allocated one for the inmem path. The tile
            # data isn't persisted anywhere else in this mode, so a
            # STOP_AFTER_PASS1 run with inmem throws it away — fine for
            # profiling Pass 1 in isolation.
            if shm_block is not None:
                try:
                    shm_block.close()
                    shm_block.unlink()
                except Exception:
                    pass
            return (pos_path, None, None, None, source_scale, crop_bbox)

        # Extract center tile result for debug output
        center_tile_result = None
        for result in all_results:
            if result.get("vesselness") is not None and result.get("labels") is not None:
                center_tile_result = {
                    "vesselness": result["vesselness"],
                    "labels_before_stitch": result["labels"],
                }
                break

        # In preview_mode, do in-memory stitching, then return data for canvas
        if preview_mode:
            # Load unstitched labels from temp zarr for preview
            import zarr
            preview_store = zarr.open(output_zarr_path, mode="r+")
            labels_data = np.asarray(preview_store[pos_path]["labels"][temp_name]["0"][...])

            # Load vesselness if saved
            vesselness_5d = None
            if save_vesselness and temp_vesselness_name:
                vesselness_data = np.asarray(preview_store[pos_path]["labels"][temp_vesselness_name]["0"][...])
                vesselness_5d = vesselness_data

            # Perform in-memory stitching (same algorithm as _stitch_tiled_labels_pass2)
            print(f"  Pass 2 (PREVIEW): In-memory stitching of {n_tiles_y * n_tiles_x} tiles...")
            labels_2d = np.squeeze(labels_data)  # Work with 2D array

            # Phase A: Offset all tiles and collect merge pairs
            all_merge_pairs = []
            step = tile_size - tile_overlap
            running_offset = 0
            max_label_seen = 0

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    y_start = ty * step
                    x_start = tx * step
                    y_end = min((ty + 1) * step, height)
                    x_end = min((tx + 1) * step, width)

                    tile_labels = labels_2d[y_start:y_end, x_start:x_end].copy()

                    if tile_labels.max() == 0:
                        continue

                    # Offset tile labels to be globally unique
                    tile_mask = tile_labels > 0
                    tile_labels[tile_mask] += running_offset
                    tile_max = tile_labels.max()

                    # Write offset labels back
                    labels_2d[y_start:y_end, x_start:x_end] = tile_labels

                    # Check LEFT neighbor for merge pairs
                    if tx > 0:
                        left_boundary_x = x_start - 1
                        if left_boundary_x >= 0:
                            left_boundary = labels_2d[y_start:y_end, left_boundary_x].flatten()
                            tile_left_edge = tile_labels[:, 0]

                            if left_boundary.max() > 0 and tile_left_edge.max() > 0:
                                for row_idx in range(len(tile_left_edge)):
                                    tile_label = tile_left_edge[row_idx]
                                    if tile_label == 0:
                                        continue
                                    for neighbor_row in [row_idx - 1, row_idx, row_idx + 1]:
                                        if 0 <= neighbor_row < len(left_boundary):
                                            neighbor_label = left_boundary[neighbor_row]
                                            if neighbor_label > 0 and tile_label != neighbor_label:
                                                all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                    # Check TOP neighbor for merge pairs
                    if ty > 0:
                        top_boundary_y = y_start - 1
                        if top_boundary_y >= 0:
                            top_boundary = labels_2d[top_boundary_y, x_start:x_end].flatten()
                            tile_top_edge = tile_labels[0, :]

                            if top_boundary.max() > 0 and tile_top_edge.max() > 0:
                                for col_idx in range(len(tile_top_edge)):
                                    tile_label = tile_top_edge[col_idx]
                                    if tile_label == 0:
                                        continue
                                    for neighbor_col in [col_idx - 1, col_idx, col_idx + 1]:
                                        if 0 <= neighbor_col < len(top_boundary):
                                            neighbor_label = top_boundary[neighbor_col]
                                            if neighbor_label > 0 and tile_label != neighbor_label:
                                                all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                    running_offset = max(running_offset, tile_max)
                    max_label_seen = max(max_label_seen, tile_max)

            print(f"    Phase A complete: {len(all_merge_pairs)} merge pairs, max_label={max_label_seen}")

            # Phase B: Build Union-Find and apply global relabeling
            if all_merge_pairs and max_label_seen > 0:
                print(f"    Phase B: Building Union-Find structure...")

                # Union-Find with path compression
                parent = list(range(int(max_label_seen) + 1))

                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]

                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

                for a, b in all_merge_pairs:
                    union(a, b)

                # Build lookup for final labels
                unique_labels = np.unique(labels_2d)
                unique_labels = unique_labels[unique_labels > 0]

                remap = {0: 0}
                next_label = 1
                root_to_new = {}

                for old_label in unique_labels:
                    root = find(int(old_label))
                    if root not in root_to_new:
                        root_to_new[root] = next_label
                        next_label += 1
                    remap[int(old_label)] = root_to_new[root]

                # Apply remapping
                labels_2d_flat = labels_2d.flatten()
                remapped = np.array([remap.get(int(x), 0) for x in labels_2d_flat], dtype=labels_2d.dtype)
                labels_2d = remapped.reshape(labels_2d.shape)

                n_merged = len(unique_labels) - (next_label - 1)
                print(f"    Phase B complete: merged {n_merged} labels, final count: {next_label - 1}")

            # Reshape back to 5D
            objects_5d = labels_2d.reshape(1, 1, 1, labels_2d.shape[0], labels_2d.shape[1])

            # Cleanup temp zarr
            if temp_zarr_dir and Path(temp_zarr_dir).exists():
                shutil.rmtree(temp_zarr_dir)
                print(f"  PREVIEW MODE: Cleaned up temp zarr at {temp_zarr_dir}")

            elapsed_time = time.time() - start_time
            print(f"[{pos_path}] PREVIEW complete, took {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
            print(f"  No data written to original zarr (preview_mode=True)")

            binary_5d = None
            return pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox

        # --- Normal mode: Continue with full processing ---

        # Optionally rename vesselness from temp to final (no stitching needed - continuous map)
        # Use filesystem rename since zarr v3 doesn't support zarr.copy()
        if save_vesselness and temp_vesselness_name and vesselness_label_name:
            print(f"  Renaming vesselness {temp_vesselness_name} -> {vesselness_label_name}")
            zarr_store_path = Path(source_zarr_path)
            labels_path = zarr_store_path / pos_path / "labels"
            temp_vesselness_path = labels_path / temp_vesselness_name
            final_vesselness_path = labels_path / vesselness_label_name

            if final_vesselness_path.exists():
                shutil.rmtree(final_vesselness_path)
            temp_vesselness_path.rename(final_vesselness_path)
            print(f"  Vesselness map saved: {vesselness_label_name}")

            # Reshard vesselness from parallel-write-safe 1:1 sharding to efficient storage
            from cyclops_utils.io.zarr_utils import reshard_zarr_array
            vesselness_array_path = Path(source_zarr_path) / pos_path / "labels" / vesselness_label_name / "0"
            reshard_zarr_array(
                source_path=vesselness_array_path,
                dest_path=None,  # In-place resharding
                chunks=(1, 1, 1, 512, 512),
                shards_ratio=shards_ratio,
                tile_size=4096,
                show_progress=True,
            )

            # Build and update metadata for the vesselness map
            # Use organelle_name as channel_label (it often contains "organelle, marker" info)
            vesselness_metadata = _build_vesselness_metadata(
                label_name=vesselness_label_name,
                organelle_name=organelle_name,
                channel_name=channel_to_segment,
                channel_label=organelle_name,  # organelle_name often is "mitochondria, TOMM20" etc.
                channel_index=channel_index,
                channel_names=channel_names,
            )
            _update_labels_metadata(
                zarr_path=Path(source_zarr_path),
                pos_path=pos_path,
                new_label_name=vesselness_label_name,
                metadata=vesselness_metadata,
            )

        # --- PASS 2: overlap correction (for labels only) ---
        # ORG_SEG_PASS2_GPU=1 uses the parallel/GPU-accelerated path (drops
        # the per-tile read-offset-write cycle, uses scipy.sparse connected
        # components + parallel LUT apply). Falls back to the original
        # sequential CPU implementation by default or when GPU is unavailable.
        _pass2_gpu = os.environ.get("ORG_SEG_PASS2_GPU", _fast_default("1")) == "1" and _GPU_AVAILABLE
        # The in-memory unstitched path requires the parallel/GPU Pass 2 —
        # the buffer kwarg only exists there.
        if use_inmem_unstitched:
            _pass2_gpu = True
        # The parallel Pass 2 doesn't implement Pass 3 mask erosion yet
        # (nucleoli removes labels within N px of the nuclear-mask edge).
        # Route nucleoli-like invocations to the legacy sequential Pass 2
        # which has working Pass 3.
        _needs_pass3 = bool(input_mask_name) and (3 if input_mask_name else 0) > 0
        if _needs_pass3 and _pass2_gpu:
            print(f"  [NOTE] input_mask_name={input_mask_name}: falling back to "
                  f"sequential Pass 2 so Pass 3 mask erosion runs.")
            _pass2_gpu = False
        _pass2_fn = _run_pass2_parallel if _pass2_gpu else _stitch_tiled_labels_pass2
        if _pass2_gpu:
            print(f"  Using parallel Pass 2 (ORG_SEG_PASS2_GPU=1)"
                  + (" [inmem buffer]" if use_inmem_unstitched else ""))
        _pass2_kwargs = dict(
            source_zarr_path=source_zarr_path,
            pos_path=pos_path,
            organelle_name=output_label_name,  # Use the standardized output name
            n_tiles_y=n_tiles_y,
            n_tiles_x=n_tiles_x,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            height=height,
            width=width,
            input_mask_name=input_mask_name,
            mask_erosion_pixels=3 if input_mask_name else 0,
            crop_bbox=crop_bbox,
            target_chunks=(1, 1, 1, 512, 512),
            target_shards_ratio=shards_ratio,  # Use shards_ratio passed to this function
        )
        if use_inmem_unstitched and _pass2_fn is _run_pass2_parallel:
            _pass2_kwargs["tile_buffer"] = tile_buffer_for_pass2
            _pass2_kwargs["tile_buffer_shm_name"] = shm_name_for_pass1
            # Hand the shm block to Pass 2 so it can release it *before*
            # the internal reshard (reshard loads the final ~40 GB zarr
            # into RAM and would OOM alongside the 43 GB shm buffer on
            # 80 GB nodes).
            _pass2_kwargs["shm_block_to_release"] = shm_block
            shm_block = None  # ownership transferred
            tile_buffer_for_pass2 = None
        _pass2_fn(**_pass2_kwargs)

        # Build and update metadata for the segmentation labels
        # Use organelle_name as channel_label (it often contains "organelle, marker" info)
        seg_metadata = _build_segmentation_metadata(
            label_name=output_label_name,
            organelle_name=organelle_name,
            channel_name=channel_to_segment,
            channel_label=organelle_name,  # organelle_name often is "mitochondria, TOMM20" etc.
            channel_index=channel_index,
            segmenter_type="frangi",
            channel_names=channel_names,
            structure_type=structure_type,
        )
        _update_labels_metadata(
            zarr_path=Path(source_zarr_path),
            pos_path=pos_path,
            new_label_name=output_label_name,
            metadata=seg_metadata,
        )

        # --- Build pyramids for the segmentation label ---
        # Skip in preview_mode since the data is in a temp location
        if not preview_mode:
            try:
                from cyclops_process.processes.pyramids.build_dask import build_organelle_seg_pyramids
            except ImportError:
                build_organelle_seg_pyramids = None
                print("  Warning: cyclops_process not available — skipping pyramid generation.")
                print("  Pyramids improve napari visualization but are not required.")

            if build_organelle_seg_pyramids is not None:
                print(f"\n  Building pyramids for {output_label_name}...")
                build_organelle_seg_pyramids(
                    source_store=source_zarr_path,
                    levels=5,
                    positions=[pos_path],
                    resume=True,
                    label_names=[output_label_name],
                )
                print(f"  Pyramids built for {output_label_name}")

                # Also build pyramids for vesselness if saved
                if save_vesselness and vesselness_label_name:
                    print(f"\n  Building pyramids for {vesselness_label_name}...")
                    build_organelle_seg_pyramids(
                        source_store=source_zarr_path,
                        levels=5,
                        positions=[pos_path],
                        resume=True,
                        label_names=[vesselness_label_name],
                    )
                    print(f"  Pyramids built for {vesselness_label_name}")

        # --- Save debug images using shared helper ---
        _save_tiled_debug_images(
            source_zarr_path=source_zarr_path,
            pos_path=pos_path,
            output_label_name=output_label_name,
            method_name="frangi",
            center_tile_result=center_tile_result,
            center_ty=center_ty,
            center_tx=center_tx,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            height=height,
            width=width,
            extra_arrays={"vesselness": center_tile_result.get("vesselness") if center_tile_result else None},
            channel_index=channel_index,
        )

        # Free the center tile result
        if center_tile_result is not None:
            del center_tile_result

        # Return values - load final count from zarr
        # Note: We don't load the full array into memory - just return metadata
        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            final_labels_arr = ds[pos_path].zgroup["labels"][output_label_name]
            # Get max label by sampling (avoid loading full array)
            # For a proper count, we'd need to load the full array, but that defeats the purpose
            # The caller doesn't actually use objects_5d for Frangi - it writes directly from zarr
            pass

        # Return None for all arrays - the data is already in zarr
        # The caller (segment_organelles) will read from zarr directly
        elapsed_time = time.time() - start_time
        print(f"[{pos_path}] Two-pass tiled Frangi complete, took {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        print(f"  Labels written to zarr: {source_zarr_path}/{pos_path}/labels/{output_label_name}")
        if save_vesselness and vesselness_label_name:
            print(f"  Vesselness written to zarr: {source_zarr_path}/{pos_path}/labels/{vesselness_label_name}")

        # Return None for all arrays - the data is already in zarr
        # The caller (segment_organelles) should detect this and skip writing
        vesselness_5d = None
        binary_5d = None
        objects_5d = None  # Data is in zarr, not in memory

        return pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox

    except Exception as e:
        print(f"Error processing position {pos_path} for {organelle_name} with tiled Frangi: {e}")
        import traceback
        traceback.print_exc()
        raise


def segment_position_frangi(
    pos_path,
    source_zarr_path,
    channel_to_segment,
    organelle_name,
    frangi_params,
    use_gpu: bool,
    frangi_postprocess: bool,
    use_clahe: bool,
    post_clahe_smoothing_sigma: float,
    debug_output_path: str = None,
    clahe_params: dict = None,
    crop_bbox: tuple = None,
    save_vesselness: bool = False,
    input_mask_name: str = None,
    structure_type: str = None,
    force_tiled: bool = False,
    tile_size: int = 4096,
    tile_overlap: int = 256,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    preview_mode: bool = False,
    shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Worker function to segment an entire position using the Frangi filter.

    This function always uses tiled processing via segment_position_frangi_tiled(),
    which handles both small and large images efficiently. For small images, tiling
    still works correctly (may use just 1-2 tiles) with minimal overhead.

    Args:
        pos_path: Position path like "A/1/0"
        source_zarr_path: Path to the v3 zarr store
        channel_to_segment: Channel name to segment
        organelle_name: Name of the organelle being segmented
        frangi_params: Parameters for Frangi filter
        use_gpu: Whether to use GPU (note: tiled processing uses CPU workers)
        frangi_postprocess: Whether to apply postprocessing
        use_clahe: Whether to apply CLAHE preprocessing
        post_clahe_smoothing_sigma: Sigma for Gaussian smoothing after CLAHE
        debug_output_path: Optional path to save debug output
        clahe_params: Parameters for CLAHE
        crop_bbox: Optional tuple (y_start, y_end, x_start, x_end) for debug center crop
        save_vesselness: If True, also save the continuous Frangi vesselness map (default: False)
        input_mask_name: Optional mask name (e.g., "nuclear_seg") to constrain segmentation.
                         If provided, Frangi will only detect structures within the mask.
        force_tiled: Ignored (always uses tiled processing now)
        tile_size: Size of each tile for tiled processing (default: 4096)
        tile_overlap: Overlap between tiles (default: 256)
        nucleoli_method: For nucleoli segmentation: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicular segmentation: "blob" for LoG, "frangi" for Frangi
        preview_mode: If True, write results to temp zarr instead of original.
            Used for preview/debug mode to avoid modifying production data.
        shards_ratio: Sharding ratio for zarr v3 storage (default: (1, 1, 1, 32, 32)).
            Determines shard file size. With 512x512 base chunks and 32x32 ratio,
            each shard covers 16384x16384 pixels (~1GB for int32 labels).

    Returns:
        Tuple of (pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox)
    """
    # Always use tiled processing - it handles both small and large images efficiently
    # For small images, it will use fewer tiles but the overhead is minimal
    return segment_position_frangi_tiled(
        pos_path=pos_path,
        source_zarr_path=source_zarr_path,
        channel_to_segment=channel_to_segment,
        organelle_name=organelle_name,
        frangi_params=frangi_params,
        frangi_postprocess=frangi_postprocess,
        use_clahe=use_clahe,
        post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
        clahe_params=clahe_params,
        crop_bbox=crop_bbox,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        save_vesselness=save_vesselness,
        input_mask_name=input_mask_name,
        structure_type=structure_type,
        debug_output_path=debug_output_path,
        nucleoli_method=nucleoli_method,
        vesicular_method=vesicular_method,
        preview_mode=preview_mode,
        shards_ratio=shards_ratio,
        use_gpu=use_gpu,
    )
