"""
Result Handler Module
=====================

Provides result processing and zarr writing helpers for organelle segmentation.

This module consolidates result handling logic that was previously duplicated
or scattered across the main segmentation functions.

Main functions:
- create_error_result(): Create standardized error result dictionary
- write_segmentation_to_zarr(): Write segmentation results to zarr
- process_segmentation_result(): Process and format segmentation results
- load_debug_arrays(): Load arrays for debug output
"""

from pathlib import Path
import numpy as np
import zarr
from iohub import open_ome_zarr


def create_error_result(
    error_message: str,
    position: str = None,
    channel: str = None,
    **extra_fields,
) -> dict:
    """
    Create standardized error result dictionary.

    Args:
        error_message: Error description
        position: Position identifier (optional)
        channel: Channel identifier (optional)
        **extra_fields: Additional fields to include in result

    Returns:
        Dict with success=False, error message, and provided identifiers

    Example:
        >>> create_error_result("Channel not found", position="A/1/0", channel="GFP")
        {'success': False, 'error': 'Channel not found', 'position': 'A/1/0', 'channel': 'GFP'}
    """
    result = {
        "success": False,
        "error": error_message,
    }
    if position is not None:
        result["position"] = position
    if channel is not None:
        result["channel"] = channel
    result.update(extra_fields)
    return result


def write_segmentation_to_zarr(
    store_path: Path,
    position: str,
    objects_name: str,
    objects: np.ndarray,
    crop_bbox: tuple = None,
) -> None:
    """
    Write segmentation result to zarr store.

    Handles both full position and cropped region writes.

    Args:
        store_path: Path to zarr store
        position: Position identifier (e.g., "A/1/0")
        objects_name: Label name (e.g., "gfp_tubular_seg")
        objects: Segmentation array (5D: T, C, Z, Y, X)
        crop_bbox: Optional (y_start, y_end, x_start, x_end) for cropped writes

    Example:
        >>> write_segmentation_to_zarr(
        ...     store_path, "A/1/0", "gfp_seg", objects_array
        ... )
        Wrote gfp_seg segmentation (full position)

        >>> write_segmentation_to_zarr(
        ...     store_path, "A/1/0", "gfp_seg", objects_array,
        ...     crop_bbox=(1000, 2000, 1000, 2000)
        ... )
        Wrote gfp_seg segmentation (crop region)
    """
    store = zarr.open(str(store_path), mode="r+")
    objects_array = store[position]["labels"][objects_name]["0"]

    if crop_bbox is not None:
        # Cropped region write
        y_start, y_end, x_start, x_end = crop_bbox
        objects_array[:, :, :, y_start:y_end, x_start:x_end] = objects
        print(f"Wrote {objects_name} segmentation (crop region)")
    else:
        # Full position write
        objects_array[:] = objects
        print(f"Wrote {objects_name} segmentation (full position)")


def load_debug_arrays(
    source_path: Path,
    position: str,
    source_channel: str,
    channel_names: list[str],
    objects_name: str,
    result_crop_bbox: tuple = None,
    load_vesselness: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load arrays for debug output (raw image, labels, vesselness).

    Args:
        source_path: Path to zarr store
        position: Position identifier
        source_channel: Channel name to load
        channel_names: List of all channel names
        objects_name: Label name for vesselness lookup
        result_crop_bbox: Optional crop coordinates
        load_vesselness: If True, attempt to load vesselness from zarr

    Returns:
        tuple: (raw_data, labels_2d, vesselness_2d)
            - raw_data: 2D raw image array
            - labels_2d: 2D segmentation labels
            - vesselness_2d: 2D vesselness (or None if not available)
    """
    with open_ome_zarr(source_path, mode="r") as ds:
        channel_index = list(ds.channel_names).index(source_channel)

        if result_crop_bbox:
            y_start, y_end, x_start, x_end = result_crop_bbox
            raw_data = np.squeeze(np.asarray(
                ds[position]["0"][0, channel_index, :, y_start:y_end, x_start:x_end]
            ))
        else:
            raw_data = np.squeeze(np.asarray(
                ds[position]["0"][0, channel_index, :, :, :]
            ))

        # Try to load vesselness from zarr if available
        vesselness_2d = None
        if load_vesselness:
            vesselness_name = objects_name.replace("_seg", "_vesselness")
            labels_group = ds[position].zgroup.get("labels", None)
            if labels_group is not None and vesselness_name in labels_group:
                vesselness_arr = labels_group[vesselness_name]["0"]
                vesselness_2d = np.squeeze(np.asarray(vesselness_arr[...]))

    return raw_data, vesselness_2d


def process_segmentation_result(
    result: tuple,
    source_path: Path,
    position: str,
    objects_name: str,
    source_channel: str,
    channel_names: list[str],
    channel_key: str,
    debug_only: bool = False,
    force_tiled: bool = False,
    start_time: float = None,
    local_frangi_params: dict = None,
    local_clahe_params: dict = None,
    structure_type: str = None,
) -> dict:
    """
    Process a single segmentation result tuple.

    Handles:
    - Result unpacking
    - Tiled result loading from zarr (if objects=None)
    - Debug output construction (arrays for canvas)
    - Normal zarr writing
    - Success/error result dict creation

    Args:
        result: Result tuple from segment_position_frangi
        source_path: Path to zarr store
        position: Position identifier
        objects_name: Output label name
        source_channel: Source channel name
        channel_names: List of all channel names
        channel_key: Channel key for display
        debug_only: If True, return debug arrays instead of writing
        force_tiled: If True, tiled processing was used
        start_time: Start timestamp for timing
        local_frangi_params: Frangi parameters used
        local_clahe_params: CLAHE parameters used
        structure_type: Structure type used

    Returns:
        dict: Result dictionary with keys:
            - success: bool
            - position: str
            - channel: str
            - output_label: str
            - num_objects: int
            - elapsed_time: float
            - error: str (if failure)
            - debug_only: bool (if debug mode)
            - tiled: bool (if tiled)
            - labels, raw, vesselness: arrays (if debug mode)
            - frangi_params, clahe_params, structure_type: configs

    Example:
        >>> result_dict = process_segmentation_result(
        ...     result_tuple, source_path, "A/1/0", "gfp_seg", "GFP",
        ...     channel_names, "GFP", debug_only=True, start_time=time.time()
        ... )
        >>> result_dict["success"]  # True
        >>> result_dict["labels"].shape  # (2048, 2048)
    """
    import time

    if result is None:
        return {
            "success": False,
            "position": position,
            "channel": channel_key,
            "error": "Segmentation returned None result",
        }

    # Unpack result tuple
    pos_path_result, vesselness, binary, objects, scale, result_crop_bbox = result

    # Handle tiled processing where objects may be None (data written directly to zarr)
    if objects is None and debug_only and force_tiled:
        # Load from zarr for debug output
        with open_ome_zarr(source_path, mode="r") as ds:
            labels_group = ds[position].zgroup.get("labels", None)
            if labels_group is not None and objects_name in labels_group:
                labels_arr = labels_group[objects_name]["0"]
                objects = np.asarray(labels_arr[...])
                print(f"  Loaded tiled results from zarr for debug output: shape {objects.shape}")

    if objects is None and not debug_only:
        # Tiled non-debug path writes directly to zarr and returns None rather than
        # shipping a 40+ GB in-memory array. Verify the label exists on disk; if so,
        # that's a successful run. We skip the expensive array read — num_objects=-1
        # signals "not counted" without pretending to be zero.
        import time
        try:
            store = zarr.open(str(source_path), mode="r")
            if objects_name in store[position]["labels"]:
                elapsed_time = time.time() - start_time if start_time else 0.0
                print(f"  Tiled write verified on disk: labels/{objects_name}")
                return {
                    "success": True,
                    "position": position,
                    "channel": channel_key,
                    "output_label": objects_name,
                    "num_objects": -1,  # not counted — reading 40GB to count would defeat the point
                    "elapsed_time": elapsed_time,
                    "tiled": True,
                }
        except Exception as e:
            print(f"  [warn] failed to verify tiled write on disk: {e}")

    if objects is None:
        return {
            "success": False,
            "position": position,
            "channel": channel_key,
            "error": "Segmentation returned None objects",
        }

    # Calculate timing
    elapsed_time = time.time() - start_time if start_time else 0.0
    num_objects = int(np.squeeze(objects).max())

    if debug_only:
        # Debug mode: return arrays for canvas
        labels_2d = np.squeeze(objects)

        # Load raw image and vesselness
        raw_data, vesselness_2d = load_debug_arrays(
            source_path, position, source_channel, channel_names,
            objects_name, result_crop_bbox, load_vesselness=force_tiled
        )

        # Get vesselness from result if available
        if vesselness is not None:
            vesselness_2d = np.squeeze(vesselness)

        print(f"  Channel complete: {num_objects} objects, {elapsed_time:.1f}s")

        return {
            "success": True,
            "position": position,
            "channel": channel_key,
            "output_label": objects_name,
            "num_objects": num_objects,
            "elapsed_time": elapsed_time,
            "debug_only": True,
            "tiled": force_tiled,
            # Arrays for combined canvas
            "labels": labels_2d,
            "raw": raw_data,
            "vesselness": vesselness_2d,
            # Params used for this run
            "frangi_params": local_frangi_params,
            "clahe_params": local_clahe_params,
            "structure_type": structure_type,
        }
    else:
        # Normal mode: write to zarr
        write_segmentation_to_zarr(
            source_path, position, objects_name, objects, result_crop_bbox
        )

        print(f"\n{'='*60}")
        print(f"Segmentation complete!")
        print(f"  Position: {position}")
        print(f"  Channel: {channel_key}")
        print(f"  Output label: {objects_name}")
        print(f"  Objects found: {num_objects}")
        print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "position": position,
            "channel": channel_key,
            "output_label": objects_name,
            "num_objects": num_objects,
            "elapsed_time": elapsed_time,
            # Params used for this run
            "frangi_params": local_frangi_params,
            "clahe_params": local_clahe_params,
            "structure_type": structure_type,
        }
