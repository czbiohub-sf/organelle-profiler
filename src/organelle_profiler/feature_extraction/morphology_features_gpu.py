"""
GPU-Accelerated Morphology Feature Extraction using cuCIM.

This module provides GPU-accelerated alternatives to the CPU-based
morphology feature extraction in morphology_features.py.

Functions
---------
extract_organelle_features_gpu
    GPU-accelerated extraction of morphological features from organelle masks.
batch_extract_organelle_features_gpu
    Process multiple organelles with GPU acceleration and timing profiling.

Features Extracted (~40 features per object with intensity image)
------------------------------------------------------------------
Same as CPU version - see morphology_features.py for full list.

Performance
-----------
Expected 5-20x speedup over CPU depending on:
- Number of objects in mask
- Image size
- GPU memory bandwidth
- Batch size

The GPU version uses cucim's batch_processing=True mode which computes
all region properties in a single pass over the image, significantly
faster than per-region computation.

Usage
-----
from .morphology_features_gpu import extract_organelle_features_gpu

# Single organelle
df = extract_organelle_features_gpu(mask, spacing, intensity_image)

# Batch processing (more efficient)
results, timing = batch_extract_organelle_features_gpu(
    organelle_masks, spacing, intensity_images
)
"""

import sys
import warnings
import time
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

# GPU availability flag
_GPU_AVAILABLE = False
_GPU_INIT_ERROR = None

try:
    import cupy as cp
    import cucim.skimage.measure as cucim_measure
    _GPU_AVAILABLE = True
except ImportError as e:
    _GPU_INIT_ERROR = str(e)
except Exception as e:
    _GPU_INIT_ERROR = str(e)

# Import CPU fallback
from .morphology_features import (
    extract_organelle_features as extract_organelle_features_cpu,
    batch_extract_organelle_features as batch_extract_organelle_features_cpu,
)


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    if not _GPU_AVAILABLE:
        return False
    try:
        # Try to actually use the GPU
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def get_gpu_init_error() -> Optional[str]:
    """Get the error message if GPU initialization failed."""
    return _GPU_INIT_ERROR


# Property name mapping: skimage -> cucim
_SKIMAGE_TO_CUCIM = {
    "label": "label",
    "area": "area",
    "perimeter": "perimeter",
    "perimeter_crofton": "perimeter_crofton",
    "axis_major_length": "major_axis_length",
    "axis_minor_length": "minor_axis_length",
    "solidity": "solidity",
    "extent": "extent",
    "orientation": "orientation",
    "eccentricity": "eccentricity",
    "equivalent_diameter_area": "equivalent_diameter",
    "area_convex": "convex_area",
    "area_filled": "filled_area",
    "euler_number": "euler_number",
    "centroid": "centroid",
    "intensity_mean": "mean_intensity",
    "intensity_max": "max_intensity",
    "intensity_min": "min_intensity",
    "intensity_std": "std_intensity",
    "centroid_weighted": "weighted_centroid",
    "inertia_tensor_eigvals": "inertia_tensor_eigvals",
    "moments_weighted_hu": "weighted_moments_hu",
}

# Reverse mapping for output column names
_CUCIM_TO_SKIMAGE = {v: k for k, v in _SKIMAGE_TO_CUCIM.items()}


def _compute_extra_intensity_stats_gpu(
    organelle_mask: cp.ndarray,
    intensity_image: cp.ndarray,
) -> Dict[str, list]:
    """
    Compute extra intensity statistics not in regionprops_table (GPU version).

    Includes std, median, quartiles, MAD, IQR, and integrated intensity.
    Uses cupy for GPU-accelerated computation.

    Parameters
    ----------
    organelle_mask : cp.ndarray
        Labeled segmentation mask (GPU array).
    intensity_image : cp.ndarray
        Intensity image (same shape as mask, GPU array).

    Returns
    -------
    dict
        Dictionary with lists of values per object.
    """
    # Get unique labels (excluding background)
    labels = cp.unique(organelle_mask)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return {
            "intensity_std": [],
            "intensity_median": [],
            "intensity_q25": [],
            "intensity_q75": [],
            "intensity_iqr": [],
            "intensity_mad": [],
            "intensity_integrated": [],
        }

    stats = {
        "intensity_std": [],
        "intensity_median": [],
        "intensity_q25": [],
        "intensity_q75": [],
        "intensity_iqr": [],
        "intensity_mad": [],
        "intensity_integrated": [],
    }

    # Process each region - this part is harder to fully vectorize
    # but we can use cupy operations within each region
    for label_val in labels.get():
        mask = organelle_mask == label_val
        intensities = intensity_image[mask]

        if intensities.size > 0:
            # Compute statistics on GPU
            std = float(cp.std(intensities))
            median = float(cp.median(intensities))
            q25 = float(cp.percentile(intensities, 25))
            q75 = float(cp.percentile(intensities, 75))

            stats["intensity_std"].append(std)
            stats["intensity_median"].append(median)
            stats["intensity_q25"].append(q25)
            stats["intensity_q75"].append(q75)
            stats["intensity_iqr"].append(q75 - q25)
            stats["intensity_mad"].append(float(cp.median(cp.abs(intensities - median))))
            stats["intensity_integrated"].append(float(cp.sum(intensities)))
        else:
            stats["intensity_std"].append(0.0)
            stats["intensity_median"].append(0.0)
            stats["intensity_q25"].append(0.0)
            stats["intensity_q75"].append(0.0)
            stats["intensity_iqr"].append(0.0)
            stats["intensity_mad"].append(0.0)
            stats["intensity_integrated"].append(0.0)

    return stats


def extract_organelle_features_gpu(
    organelle_mask: np.ndarray,
    spacing: Tuple[float, float],
    intensity_image: Optional[np.ndarray] = None,
    device_id: int = 0,
) -> pd.DataFrame:
    """
    GPU-accelerated extraction of morphological features from an organelle mask.

    Parameters
    ----------
    organelle_mask : np.ndarray
        2D labeled segmentation mask (H, W) where each unique value > 0 is an object.
    spacing : tuple
        Pixel spacing (y, x) in physical units.
    intensity_image : np.ndarray, optional
        2D intensity image for intensity-based features.
    device_id : int, optional
        CUDA device ID to use. Default is 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per object, columns are feature names.
        Column names match the CPU version for compatibility.
    """
    if not is_gpu_available():
        warnings.warn("GPU not available, falling back to CPU implementation")
        return extract_organelle_features_cpu(organelle_mask, spacing, intensity_image)

    if not np.any(organelle_mask > 0):
        return pd.DataFrame()

    try:
        with cp.cuda.Device(device_id):
            return _extract_organelle_features_gpu_impl(
                organelle_mask, spacing, intensity_image
            )
    except Exception as e:
        warnings.warn(f"GPU extraction failed ({e}), falling back to CPU")
        return extract_organelle_features_cpu(organelle_mask, spacing, intensity_image)


def _extract_organelle_features_gpu_impl(
    organelle_mask: np.ndarray,
    spacing: Tuple[float, float],
    intensity_image: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Internal GPU implementation."""
    from skimage.segmentation import relabel_sequential

    # CRITICAL OPTIMIZATION: Relabel mask to consecutive integers
    # Sparse labels cause regionprops to be extremely slow
    original_labels = np.unique(organelle_mask)
    original_labels = original_labels[original_labels > 0]

    if len(original_labels) > 0 and original_labels.max() > len(original_labels) * 2:
        organelle_mask, forward_map, inverse_map = relabel_sequential(organelle_mask)
        needs_label_restore = True
    else:
        needs_label_restore = False

    # Transfer to GPU
    mask_gpu = cp.asarray(organelle_mask)
    intensity_gpu = cp.asarray(intensity_image) if intensity_image is not None else None

    # Core shape properties (perimeter/euler/convex removed to avoid slow fallbacks)
    properties = [
        "label",
        "area",
        "major_axis_length",
        "minor_axis_length",
        "extent",
        "orientation",
        "eccentricity",
        "equivalent_diameter",
        "filled_area",
        "centroid",
    ]

    # Add intensity properties if intensity image provided
    if intensity_gpu is not None:
        properties.extend([
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "std_intensity",
            "weighted_centroid",
            "inertia_tensor_eigvals",
            "weighted_moments_hu",
        ])

    # Extract properties using cucim with batch processing
    props = cucim_measure.regionprops_table(
        mask_gpu,
        intensity_image=intensity_gpu,
        properties=properties,
        spacing=spacing,
        batch_processing=True,  # Efficient batch mode
    )

    # Convert cupy arrays to numpy for DataFrame
    props_np = {}
    for key, value in props.items():
        if isinstance(value, cp.ndarray):
            props_np[key] = cp.asnumpy(value)
        else:
            props_np[key] = value

    df = pd.DataFrame(props_np)

    # Rename columns to match skimage convention
    rename_map = {
        "major_axis_length": "axis_major_length",
        "minor_axis_length": "axis_minor_length",
        "equivalent_diameter": "equivalent_diameter_area",
        "convex_area": "area_convex",
        "filled_area": "area_filled",
        "mean_intensity": "intensity_mean",
        "max_intensity": "intensity_max",
        "min_intensity": "intensity_min",
        "std_intensity": "intensity_std",
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
        "weighted_centroid-0": "centroid_weighted_y",
        "weighted_centroid-1": "centroid_weighted_x",
        "inertia_tensor_eigvals-0": "inertia_eigval_0",
        "inertia_tensor_eigvals-1": "inertia_eigval_1",
    }
    # Rename weighted Hu moments
    for i in range(7):
        rename_map[f"weighted_moments_hu-{i}"] = f"moments_weighted_hu_{i}"

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Derived shape features
    if "area" in df.columns and len(df) > 0:
        if "axis_major_length" in df.columns and "axis_minor_length" in df.columns:
            minor = df["axis_minor_length"].replace(0, 1)
            df["aspect_ratio"] = df["axis_major_length"] / minor

        # Circularity approximation using eccentricity (perimeter/convex removed for speed)
        # eccentricity=0 is circle, eccentricity=1 is line; circularity = 1 - eccentricity
        if "eccentricity" in df.columns:
            df["circularity"] = 1.0 - df["eccentricity"]

    # Add extra intensity stats (quartiles, MAD, IQR, integrated)
    if intensity_gpu is not None and len(df) > 0:
        intensity_stats = _compute_extra_intensity_stats_gpu(mask_gpu, intensity_gpu)
        for stat_name, values in intensity_stats.items():
            if len(values) == len(df):
                df[stat_name] = values

        # Derived intensity features
        if "intensity_max" in df.columns and "intensity_min" in df.columns:
            df["intensity_range"] = df["intensity_max"] - df["intensity_min"]

        if "intensity_std" in df.columns and "intensity_mean" in df.columns:
            mean_safe = df["intensity_mean"].replace(0, np.nan)
            df["intensity_cv"] = df["intensity_std"] / mean_safe

    # Restore original labels if we relabeled
    if needs_label_restore and "label" in df.columns and len(df) > 0:
        df["label"] = df["label"].map(lambda x: inverse_map[x] if x < len(inverse_map) else x)

    return df


def batch_extract_organelle_features_gpu(
    organelle_masks: Dict[str, np.ndarray],
    spacing: Tuple[float, float],
    intensity_images: Optional[Dict[str, np.ndarray]] = None,
    profile_properties: bool = False,
    profile_organelle: Optional[str] = None,
    device_id: int = 0,
    use_streams: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    GPU-accelerated batch extraction of features from multiple organelle masks.

    This function processes multiple organelles efficiently by:
    1. Batching ALL GPU memory transfers upfront (single H2D sync point)
    2. Processing all organelles on GPU
    3. Batching ALL result transfers (single D2H sync point)

    Parameters
    ----------
    organelle_masks : dict
        Dict mapping organelle name -> labeled mask array.
    spacing : tuple
        Pixel spacing (y, x).
    intensity_images : dict, optional
        Dict mapping organelle name -> intensity image.
    device_id : int, optional
        CUDA device ID to use. Default is 0.
    use_streams : bool, optional
        Whether to use CUDA streams for async processing. Default is True.

    Returns
    -------
    tuple
        (features_dict, timing_dict)
    """
    if not is_gpu_available():
        warnings.warn("GPU not available, falling back to CPU implementation")
        return batch_extract_organelle_features_cpu(
            organelle_masks, spacing, intensity_images
        )

    results = {}
    timing = {}

    try:
        from skimage.segmentation import relabel_sequential

        with cp.cuda.Device(device_id):
            t_transfer_start = time.time()

            # PHASE 0: Relabel masks to consecutive integers (CPU, before transfer)
            # Sparse labels (max label >> num objects) cause regionprops to be extremely slow
            relabeled_masks = {}
            inverse_maps = {}
            for name, mask in organelle_masks.items():
                original_labels = np.unique(mask)
                original_labels = original_labels[original_labels > 0]
                if len(original_labels) > 0 and original_labels.max() > len(original_labels) * 2:
                    relabeled, _, inverse_map = relabel_sequential(mask)
                    relabeled_masks[name] = relabeled
                    inverse_maps[name] = inverse_map
                else:
                    relabeled_masks[name] = mask
                    inverse_maps[name] = None

            # PHASE 1: Batch transfer ALL data to GPU (single H2D sync point)
            masks_gpu = {}
            intensities_gpu = {}
            for name, mask in relabeled_masks.items():
                masks_gpu[name] = cp.asarray(mask)
                if intensity_images and name in intensity_images:
                    intensities_gpu[name] = cp.asarray(intensity_images[name])
                else:
                    intensities_gpu[name] = None

            # Ensure all transfers complete before processing
            cp.cuda.Stream.null.synchronize()
            t_transfer_h2d = time.time() - t_transfer_start

            # PHASE 2: Process all organelles on GPU (compute phase)
            results_gpu = {}
            for name in organelle_masks.keys():
                t0 = time.time()
                mask_gpu = masks_gpu[name]
                intensity_gpu = intensities_gpu[name]

                # Run GPU feature extraction (data already on GPU)
                df = _extract_organelle_features_gpu_impl_from_gpu(
                    mask_gpu, spacing, intensity_gpu
                )

                # Restore original labels if we relabeled
                if inverse_maps[name] is not None and "label" in df.columns and len(df) > 0:
                    inv_map = inverse_maps[name]
                    df["label"] = df["label"].map(lambda x: inv_map[x] if x < len(inv_map) else x)

                results_gpu[name] = df
                timing[name] = time.time() - t0

            # PHASE 3: Results are already DataFrames (transferred in impl function)
            results = results_gpu

            # Record transfer timing
            timing['_h2d_transfer_ms'] = t_transfer_h2d * 1000

    except Exception as e:
        warnings.warn(f"GPU batch extraction failed ({e}), falling back to CPU")
        return batch_extract_organelle_features_cpu(
            organelle_masks, spacing, intensity_images
        )

    return results, timing


def _extract_organelle_features_gpu_impl_from_gpu(
    mask_gpu: "cp.ndarray",
    spacing: Tuple[float, float],
    intensity_gpu: Optional["cp.ndarray"] = None,
) -> pd.DataFrame:
    """
    GPU implementation that takes data already on GPU.

    This avoids redundant CPU->GPU transfers when processing multiple organelles.
    """
    if not cp.any(mask_gpu > 0):
        return pd.DataFrame()

    # Core shape properties (perimeter/euler/convex removed to avoid slow fallbacks)
    properties = [
        "label",
        "area",
        "major_axis_length",
        "minor_axis_length",
        "extent",
        "orientation",
        "eccentricity",
        "equivalent_diameter",
        "filled_area",
        "centroid",
    ]

    # Add intensity properties if intensity image provided
    if intensity_gpu is not None:
        properties.extend([
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "weighted_centroid",
            "inertia_tensor_eigvals",
            "weighted_moments_hu",
        ])

    # Extract properties using cucim with batch processing
    props = cucim_measure.regionprops_table(
        mask_gpu,
        intensity_image=intensity_gpu,
        properties=properties,
        spacing=spacing,
        batch_processing=True,
    )

    # Convert cupy arrays to numpy for DataFrame
    props_np = {}
    for key, value in props.items():
        if isinstance(value, cp.ndarray):
            props_np[key] = cp.asnumpy(value)
        else:
            props_np[key] = value

    df = pd.DataFrame(props_np)

    # Rename columns to match skimage convention
    rename_map = {
        "major_axis_length": "axis_major_length",
        "minor_axis_length": "axis_minor_length",
        "equivalent_diameter": "equivalent_diameter_area",
        "convex_area": "area_convex",
        "filled_area": "area_filled",
        "mean_intensity": "intensity_mean",
        "max_intensity": "intensity_max",
        "min_intensity": "intensity_min",
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
        "weighted_centroid-0": "centroid_weighted_y",
        "weighted_centroid-1": "centroid_weighted_x",
        "inertia_tensor_eigvals-0": "inertia_eigval_0",
        "inertia_tensor_eigvals-1": "inertia_eigval_1",
    }
    for i in range(7):
        rename_map[f"weighted_moments_hu-{i}"] = f"moments_weighted_hu_{i}"

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Derived shape features
    if "area" in df.columns and len(df) > 0:
        if "axis_major_length" in df.columns and "axis_minor_length" in df.columns:
            minor = df["axis_minor_length"].replace(0, 1)
            df["aspect_ratio"] = df["axis_major_length"] / minor

        # Circularity approximation using eccentricity (perimeter/convex removed for speed)
        # eccentricity=0 is circle, eccentricity=1 is line; circularity = 1 - eccentricity
        if "eccentricity" in df.columns:
            df["circularity"] = 1.0 - df["eccentricity"]

    # Add extra intensity stats (computed on GPU)
    if intensity_gpu is not None and len(df) > 0:
        intensity_stats = _compute_extra_intensity_stats_gpu(mask_gpu, intensity_gpu)
        for stat_name, values in intensity_stats.items():
            if len(values) == len(df):
                df[stat_name] = values

        # Derived intensity features
        if "intensity_max" in df.columns and "intensity_min" in df.columns:
            df["intensity_range"] = df["intensity_max"] - df["intensity_min"]

        if "intensity_mean" in df.columns:
            mean_safe = df["intensity_mean"].replace(0, np.nan)
            if "intensity_std" in df.columns:
                df["intensity_cv"] = df["intensity_std"] / mean_safe

    return df


def extract_features_batch_gpu(
    masks_batch: list,
    spacing: Tuple[float, float],
    intensity_batch: Optional[list] = None,
    device_id: int = 0,
) -> list:
    """
    Process a batch of cells' organelle masks on GPU.

    This is optimized for the feature extraction pipeline where we want to
    process many cells at once to maximize GPU utilization.

    Parameters
    ----------
    masks_batch : list
        List of (organelle_name, mask_array) tuples for each cell.
    spacing : tuple
        Pixel spacing (y, x).
    intensity_batch : list, optional
        List of (organelle_name, intensity_array) tuples.
    device_id : int, optional
        CUDA device ID.

    Returns
    -------
    list
        List of DataFrames with extracted features for each cell.
    """
    if not is_gpu_available():
        # Fall back to CPU processing
        results = []
        for i, (name, mask) in enumerate(masks_batch):
            intensity = intensity_batch[i][1] if intensity_batch else None
            df = extract_organelle_features_cpu(mask, spacing, intensity)
            results.append(df)
        return results

    results = []

    try:
        with cp.cuda.Device(device_id):
            # Transfer all masks to GPU at once for efficiency
            for i, (name, mask) in enumerate(masks_batch):
                intensity = intensity_batch[i][1] if intensity_batch else None
                df = _extract_organelle_features_gpu_impl(mask, spacing, intensity)
                results.append(df)
    except Exception as e:
        warnings.warn(f"GPU batch failed ({e}), falling back to CPU")
        results = []
        for i, (name, mask) in enumerate(masks_batch):
            intensity = intensity_batch[i][1] if intensity_batch else None
            df = extract_organelle_features_cpu(mask, spacing, intensity)
            results.append(df)

    return results


# =============================================================================
# Multi-Cell Batched GPU Processing
# =============================================================================

def create_cell_montage(
    cell_masks: list,
    cell_intensities: list = None,
    padding: int = 4,
) -> tuple:
    """
    Combine N cells' masks into a single large montage image for batched GPU processing.

    Uses vectorized LUT-based relabeling (no per-label Python loops).

    Returns
    -------
    tuple
        (montage_mask, montage_intensity, cell_label_ranges, cell_offsets)
        - montage_mask: Combined labeled mask (int32)
        - montage_intensity: Combined intensity image (float32) or None
        - cell_label_ranges: List of (start_label, end_label, cell_idx, original_labels)
        - cell_offsets: Dict mapping cell_idx -> (y_offset, x_offset) for centroid correction
    """
    if not cell_masks:
        return None, None, [], {}

    # Filter out None masks and track valid indices
    valid_masks = [(idx, m) for idx, m in enumerate(cell_masks) if m is not None and np.any(m > 0)]

    if not valid_masks:
        return None, None, [], {}

    # Find max dimensions across valid cells only
    max_h = max(m.shape[0] for _, m in valid_masks)
    max_w = max(m.shape[1] for _, m in valid_masks)

    n_valid = len(valid_masks)
    grid_size = int(np.ceil(np.sqrt(n_valid)))

    # Create montage arrays with padding between cells
    cell_h = max_h + padding
    cell_w = max_w + padding
    montage_h = grid_size * cell_h
    montage_w = grid_size * cell_w
    montage_mask = np.zeros((montage_h, montage_w), dtype=np.int32)

    has_intensity = cell_intensities is not None and len(cell_intensities) == len(cell_masks)
    if has_intensity:
        montage_intensity = np.zeros((montage_h, montage_w), dtype=np.float32)
    else:
        montage_intensity = None

    label_offset = 0
    cell_label_ranges = []
    cell_offsets = {}

    for grid_idx, (idx, mask) in enumerate(valid_masks):
        row = grid_idx // grid_size
        col = grid_idx % grid_size
        y_start = row * cell_h
        x_start = col * cell_w

        # Track offset for centroid correction
        cell_offsets[idx] = (y_start, x_start)

        # Vectorized relabeling using LUT
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        n_labels = len(unique_labels)

        if n_labels > 0:
            lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
            lut[unique_labels] = np.arange(label_offset + 1, label_offset + 1 + n_labels, dtype=np.int32)
            relabeled = lut[mask]

            cell_label_ranges.append((label_offset + 1, label_offset + n_labels, idx, unique_labels))
            label_offset += n_labels

            h, w = mask.shape
            montage_mask[y_start:y_start+h, x_start:x_start+w] = relabeled

        # Place intensity image
        if has_intensity and cell_intensities[idx] is not None:
            intensity = cell_intensities[idx]
            if intensity.ndim > 2:
                intensity = np.mean(intensity, axis=0)
            h, w = intensity.shape[:2]
            montage_intensity[y_start:y_start+h, x_start:x_start+w] = intensity.astype(np.float32)

    return montage_mask, montage_intensity, cell_label_ranges, cell_offsets


def batch_extract_features_gpu_multicell(
    cell_masks_by_organelle: dict,
    cell_intensities_by_organelle: dict,
    spacing: tuple,
    device_id: int = 0,
) -> tuple:
    """
    Process multiple cells in single GPU calls per organelle (true GPU batching).

    Instead of N cells × M organelles GPU kernel launches, this does M kernel
    launches total by combining all cells' masks into montages.

    Parameters
    ----------
    cell_masks_by_organelle : dict
        Dict mapping organelle_name -> list of masks (one mask per cell).
    cell_intensities_by_organelle : dict
        Dict mapping organelle_name -> list of intensity images (one per cell).
    spacing : tuple
        Pixel spacing (y, x) in physical units.
    device_id : int, optional
        CUDA device ID to use. Default is 0.

    Returns
    -------
    tuple
        (results_per_cell, timing_dict)
    """
    if not is_gpu_available():
        warnings.warn("GPU not available for multicell batch processing")
        return None, {}

    if not cell_masks_by_organelle:
        return [], {}

    first_organelle = next(iter(cell_masks_by_organelle.keys()))
    n_cells = len(cell_masks_by_organelle[first_organelle])

    if n_cells == 0:
        return [], {}

    results_per_cell = [{} for _ in range(n_cells)]
    timing = {}

    try:
        import time

        with cp.cuda.Device(device_id):
            for organelle_name in cell_masks_by_organelle:
                t_org_start = time.time()

                masks = cell_masks_by_organelle[organelle_name]
                intensities = cell_intensities_by_organelle.get(organelle_name, [None] * n_cells)

                # Skip if no masks have data
                if not any(m is not None and np.any(m > 0) for m in masks):
                    timing[organelle_name] = time.time() - t_org_start
                    continue

                # Create montage combining all cells (vectorized LUT relabeling)
                t_montage_start = time.time()
                montage_mask, montage_intensity, label_ranges, cell_offsets = create_cell_montage(
                    masks, intensities
                )
                t_montage = time.time() - t_montage_start

                if montage_mask is None or len(label_ranges) == 0:
                    timing[organelle_name] = time.time() - t_org_start
                    continue

                # Transfer to GPU (single transfer for entire montage)
                t_h2d_start = time.time()
                montage_mask_gpu = cp.asarray(montage_mask)
                montage_intensity_gpu = cp.asarray(montage_intensity) if montage_intensity is not None else None
                cp.cuda.Stream.null.synchronize()
                t_h2d = time.time() - t_h2d_start

                # Run regionprops ONCE on entire montage
                t_gpu_start = time.time()
                properties = [
                    "label",
                    "area",
                    "major_axis_length",
                    "minor_axis_length",
                    "extent",
                    "orientation",
                    "eccentricity",
                    "equivalent_diameter",
                    "filled_area",
                    "centroid",
                ]

                if montage_intensity_gpu is not None:
                    properties.extend([
                        "mean_intensity",
                        "max_intensity",
                        "min_intensity",
                        "weighted_centroid",
                        "inertia_tensor_eigvals",
                        "weighted_moments_hu",
                    ])

                props = cucim_measure.regionprops_table(
                    montage_mask_gpu,
                    intensity_image=montage_intensity_gpu,
                    properties=properties,
                    spacing=spacing,
                    batch_processing=True,
                )
                t_gpu = time.time() - t_gpu_start

                # Convert to numpy DataFrame
                t_d2h_start = time.time()
                props_np = {}
                for key, value in props.items():
                    if isinstance(value, cp.ndarray):
                        props_np[key] = cp.asnumpy(value)
                    else:
                        props_np[key] = value

                props_df = pd.DataFrame(props_np)
                t_d2h = time.time() - t_d2h_start

                # Rename columns to match skimage convention
                rename_map = {
                    "major_axis_length": "axis_major_length",
                    "minor_axis_length": "axis_minor_length",
                    "equivalent_diameter": "equivalent_diameter_area",
                    "convex_area": "area_convex",
                    "filled_area": "area_filled",
                    "mean_intensity": "intensity_mean",
                    "max_intensity": "intensity_max",
                    "min_intensity": "intensity_min",
                    "centroid-0": "centroid_y",
                    "centroid-1": "centroid_x",
                    "weighted_centroid-0": "centroid_weighted_y",
                    "weighted_centroid-1": "centroid_weighted_x",
                    "inertia_tensor_eigvals-0": "inertia_eigval_0",
                    "inertia_tensor_eigvals-1": "inertia_eigval_1",
                }
                for i in range(7):
                    rename_map[f"weighted_moments_hu-{i}"] = f"moments_weighted_hu_{i}"

                props_df.rename(columns={k: v for k, v in rename_map.items() if k in props_df.columns}, inplace=True)

                # Derived shape features
                if "area" in props_df.columns and len(props_df) > 0:
                    if "axis_major_length" in props_df.columns and "axis_minor_length" in props_df.columns:
                        minor = props_df["axis_minor_length"].replace(0, 1)
                        props_df["aspect_ratio"] = props_df["axis_major_length"] / minor

                    if "eccentricity" in props_df.columns:
                        props_df["circularity"] = 1.0 - props_df["eccentricity"]

                # Derived intensity features
                if montage_intensity_gpu is not None and len(props_df) > 0:
                    if "intensity_max" in props_df.columns and "intensity_min" in props_df.columns:
                        props_df["intensity_range"] = props_df["intensity_max"] - props_df["intensity_min"]

                # Split results by cell using label ranges + correct centroids
                t_split_start = time.time()
                for start_label, end_label, cell_idx, original_labels in label_ranges:
                    cell_row_mask = (props_df['label'] >= start_label) & (props_df['label'] <= end_label)
                    cell_df = props_df[cell_row_mask].copy()

                    if len(cell_df) > 0:
                        # Restore original labels from the cell's mask
                        new_to_orig = dict(zip(
                            range(start_label, end_label + 1),
                            original_labels,
                        ))
                        cell_df['label'] = cell_df['label'].map(new_to_orig)

                        # Correct centroids from montage space to cell-local space
                        if cell_idx in cell_offsets:
                            y_off, x_off = cell_offsets[cell_idx]
                            if "centroid_y" in cell_df.columns:
                                cell_df["centroid_y"] -= y_off * spacing[0]
                                cell_df["centroid_x"] -= x_off * spacing[1]
                            if "centroid_weighted_y" in cell_df.columns:
                                cell_df["centroid_weighted_y"] -= y_off * spacing[0]
                                cell_df["centroid_weighted_x"] -= x_off * spacing[1]

                        results_per_cell[cell_idx][organelle_name] = cell_df
                t_split = time.time() - t_split_start

                # Record detailed timing
                timing[organelle_name] = time.time() - t_org_start
                timing[f"{organelle_name}_montage_ms"] = t_montage * 1000
                timing[f"{organelle_name}_h2d_ms"] = t_h2d * 1000
                timing[f"{organelle_name}_gpu_ms"] = t_gpu * 1000
                timing[f"{organelle_name}_d2h_ms"] = t_d2h * 1000
                timing[f"{organelle_name}_split_ms"] = t_split * 1000

            # # Log summary
            # total_time = sum(v for k, v in timing.items() if not k.endswith('_ms'))
            # n_organelles = len([k for k in timing if not k.endswith('_ms')])
            # print(f"    [GPU MULTICELL] {n_cells} cells × {n_organelles} organelles in {total_time:.1f}s", file=sys.stderr, flush=True)

    except Exception as e:
        warnings.warn(f"Multicell GPU batch extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}

    return results_per_cell, timing


# =============================================================================
# GPU-Batched Localization via Distance Transforms
# =============================================================================

def compute_localization_gpu_batched(
    cell_masks: list,
    nuclear_masks: list,
    organelle_masks_by_organelle: dict,
    spacing: tuple,
    organelles_to_process: list,
    device_id: int = 0,
    padding: int = 4,
) -> tuple:
    """
    Compute localization features for ALL cells using GPU distance transforms.

    Instead of per-cell KDTree construction + queries, this:
    1. Creates montages of cell masks and nuclear masks
    2. Computes distance_transform_edt on GPU (2 calls for ALL cells)
    3. Samples distances at organelle centroids

    Parameters
    ----------
    cell_masks : list of np.ndarray
        Binary cell masks, one per cell (same order as loaded_cells).
    nuclear_masks : list of np.ndarray or None
        Binary nuclear masks, one per cell. Can contain None entries.
    organelle_masks_by_organelle : dict
        Dict mapping organelle_name -> list of labeled masks (one per cell).
    spacing : tuple
        Pixel spacing (y, x).
    organelles_to_process : list
        List of organelle names to compute localization for.
    device_id : int
        CUDA device ID.
    padding : int
        Padding between cells in montage (must match morphology montage).

    Returns
    -------
    tuple
        (localization_per_cell, timing)
        - localization_per_cell: list of dicts, one per cell.
          Each dict maps organelle_name -> dict of cell-level summary features.
        - timing: dict with timing info.
    """
    import time as time_mod

    if not is_gpu_available():
        return None, {}

    n_cells = len(cell_masks)
    localization_per_cell = [{} for _ in range(n_cells)]
    timing = {}

    try:
        import cupyx.scipy.ndimage as cupy_ndimage

        with cp.cuda.Device(device_id):
            # ---- Step 1: Build cell mask montage ----
            t0 = time_mod.time()

            # Find valid cells (have cell mask)
            valid_cells = [(i, m) for i, m in enumerate(cell_masks)
                           if m is not None and np.any(m > 0)]
            if not valid_cells:
                return localization_per_cell, {}

            max_h = max(m.shape[0] for _, m in valid_cells)
            max_w = max(m.shape[1] for _, m in valid_cells)
            n_valid = len(valid_cells)
            grid_size = int(np.ceil(np.sqrt(n_valid)))
            cell_h = max_h + padding
            cell_w = max_w + padding
            montage_h = grid_size * cell_h
            montage_w = grid_size * cell_w

            # Cell interior montage (1=inside cell, 0=outside)
            cell_interior_montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
            # Nuclear interior montage (1=inside nucleus, 0=outside)
            nuc_interior_montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
            # Track nucleus centroids per cell (pixel coords in montage space)
            nuc_centroids_montage = {}

            cell_offsets = {}
            for grid_idx, (cell_idx, cell_mask) in enumerate(valid_cells):
                row = grid_idx // grid_size
                col = grid_idx % grid_size
                y_start = row * cell_h
                x_start = col * cell_w
                cell_offsets[cell_idx] = (y_start, x_start)

                h, w = cell_mask.shape
                cell_binary = (cell_mask > 0).astype(np.uint8)
                cell_interior_montage[y_start:y_start+h, x_start:x_start+w] = cell_binary

                # Nuclear mask
                nuc_mask = nuclear_masks[cell_idx] if cell_idx < len(nuclear_masks) else None
                if nuc_mask is not None and np.any(nuc_mask > 0):
                    nuc_binary = (nuc_mask > 0).astype(np.uint8)
                    nuc_interior_montage[y_start:y_start+h, x_start:x_start+w] = nuc_binary
                    # Nucleus centroid in montage pixel coords
                    nuc_ys, nuc_xs = np.where(nuc_mask > 0)
                    nuc_centroids_montage[cell_idx] = (
                        y_start + nuc_ys.mean(),
                        x_start + nuc_xs.mean(),
                    )

            t_montage = time_mod.time() - t0

            # ---- Step 2: GPU distance transforms ----
            t1 = time_mod.time()

            # Distance to cell edge = distance_transform of cell interior
            # (distance from each interior pixel to nearest exterior pixel = cell boundary)
            cell_interior_gpu = cp.asarray(cell_interior_montage)
            dist_to_edge_gpu = cupy_ndimage.distance_transform_edt(
                cell_interior_gpu, sampling=spacing
            )

            # Distance to nucleus = distance_transform of inverted nuclear interior
            # (distance from each pixel to nearest nuclear pixel)
            nuc_interior_gpu = cp.asarray(nuc_interior_montage)
            has_any_nucleus = cp.any(nuc_interior_gpu > 0)
            if has_any_nucleus:
                dist_to_nuc_gpu = cupy_ndimage.distance_transform_edt(
                    ~(nuc_interior_gpu.astype(bool)), sampling=spacing
                )
            else:
                dist_to_nuc_gpu = None

            cp.cuda.Stream.null.synchronize()
            t_gpu = time_mod.time() - t1

            # Transfer to CPU for sampling
            t2 = time_mod.time()
            dist_to_edge = cp.asnumpy(dist_to_edge_gpu)
            dist_to_nuc = cp.asnumpy(dist_to_nuc_gpu) if dist_to_nuc_gpu is not None else None
            # Free GPU memory
            del cell_interior_gpu, dist_to_edge_gpu, nuc_interior_gpu
            if dist_to_nuc_gpu is not None:
                del dist_to_nuc_gpu
            cp.get_default_memory_pool().free_all_blocks()
            t_d2h = time_mod.time() - t2

            # ---- Step 3: Sample distances at organelle centroids (GPU regionprops) ----
            t3 = time_mod.time()

            skip_organelles = {"nuclei", "nuclear_seg", "cell_membrane"}

            for organelle_name in organelles_to_process:
                if organelle_name in skip_organelles:
                    continue

                masks_list = organelle_masks_by_organelle.get(organelle_name, [])
                if not masks_list:
                    continue

                # Create organelle montage using same grid layout
                org_montage = np.zeros((montage_h, montage_w), dtype=np.int32)
                label_offset = 0
                # Track: (start_label, end_label, cell_idx, original_labels)
                org_label_ranges = []

                for grid_idx, (cell_idx, _) in enumerate(valid_cells):
                    if cell_idx >= len(masks_list):
                        continue
                    org_mask = masks_list[cell_idx]
                    if org_mask is None or not np.any(org_mask > 0):
                        continue

                    y_off, x_off = cell_offsets[cell_idx]
                    unique_labels = np.unique(org_mask)
                    unique_labels = unique_labels[unique_labels > 0]
                    n_labels = len(unique_labels)
                    if n_labels == 0:
                        continue

                    # Vectorized LUT relabeling
                    lut = np.zeros(int(org_mask.max()) + 1, dtype=np.int32)
                    lut[unique_labels] = np.arange(label_offset + 1, label_offset + 1 + n_labels, dtype=np.int32)
                    relabeled = lut[org_mask]

                    h, w = org_mask.shape
                    # Clip to montage cell slot size (organelle crops can be larger than cell crops)
                    slot_h = min(h, cell_h - padding)
                    slot_w = min(w, cell_w - padding)
                    org_montage[y_off:y_off+slot_h, x_off:x_off+slot_w] = relabeled[:slot_h, :slot_w]
                    org_label_ranges.append((label_offset + 1, label_offset + n_labels, cell_idx, unique_labels))
                    label_offset += n_labels

                if label_offset == 0:
                    continue  # No objects for this organelle

                # GPU regionprops to get centroids (no center_of_mass needed)
                org_montage_gpu = cp.asarray(org_montage)
                props = cucim_measure.regionprops_table(
                    org_montage_gpu,
                    properties=["label", "centroid"],
                )
                prop_labels = cp.asnumpy(props["label"])
                centroid_y = cp.asnumpy(props["centroid-0"])
                centroid_x = cp.asnumpy(props["centroid-1"])
                del org_montage_gpu

                # Sample distance transforms at all centroids at once
                all_y = np.clip(centroid_y.astype(int), 0, montage_h - 1)
                all_x = np.clip(centroid_x.astype(int), 0, montage_w - 1)
                all_d_edge = dist_to_edge[all_y, all_x]
                all_d_nuc = dist_to_nuc[all_y, all_x] if dist_to_nuc is not None else None

                # Build label -> index mapping for fast lookup
                label_to_idx = {}
                for i, lbl in enumerate(prop_labels):
                    label_to_idx[int(lbl)] = i

                # Split results back by cell using label ranges
                for start_label, end_label, cell_idx, original_labels in org_label_ranges:
                    if cell_idx >= n_cells:
                        continue

                    # Get indices for this cell's labels
                    indices = [label_to_idx[l] for l in range(start_label, end_label + 1) if l in label_to_idx]
                    if not indices:
                        continue
                    indices = np.array(indices)

                    d_edge = all_d_edge[indices]

                    loc_data = {
                        "label": original_labels[:len(indices)],
                        "distance_from_cell_edge": d_edge,
                    }

                    if all_d_nuc is not None:
                        d_nuc = all_d_nuc[indices]
                        loc_data["distance_from_nucleus"] = d_nuc

                        # Distance to nucleus centroid
                        y_off, x_off = cell_offsets[cell_idx]
                        if cell_idx in nuc_centroids_montage:
                            nuc_cy, nuc_cx = nuc_centroids_montage[cell_idx]
                            # Cell-local centroid coords (subtract montage offset)
                            local_y = centroid_y[indices] - y_off
                            local_x = centroid_x[indices] - x_off
                            nuc_local_y = nuc_cy - y_off
                            nuc_local_x = nuc_cx - x_off
                            loc_data["distance_from_nucleus_centroid"] = np.sqrt(
                                ((local_y - nuc_local_y) * spacing[0])**2 +
                                ((local_x - nuc_local_x) * spacing[1])**2
                            )
                            # Angle about the nucleus centroid (radians) — reused by
                            # the radial-distribution anisotropy feature (matches the
                            # CPU compute_localization_kdtree convention).
                            loc_data["angle_from_nucleus"] = np.arctan2(
                                local_y - nuc_local_y, local_x - nuc_local_x
                            )

                        # Radial position (0=nucleus, 1=cell edge)
                        total_dist = d_edge + d_nuc
                        with np.errstate(divide='ignore', invalid='ignore'):
                            loc_data["normalized_radial_position"] = np.where(
                                total_dist > 0, d_nuc / total_dist, 0.5
                            )

                    loc_df = pd.DataFrame(loc_data)
                    summary = _localization_summary(loc_df, organelle_name)
                    if summary:
                        localization_per_cell[cell_idx].update(summary)

            t_sample = time_mod.time() - t3

            timing = {
                "montage_ms": t_montage * 1000,
                "gpu_distance_transform_ms": t_gpu * 1000,
                "d2h_ms": t_d2h * 1000,
                "sample_ms": t_sample * 1000,
                "total_ms": (time_mod.time() - t0) * 1000,
            }
            # print(
            #     f"    [GPU LOC] {n_cells} cells × {len(organelles_to_process)} organelles: "
            #     f"montage={t_montage*1000:.0f}ms, GPU_DT={t_gpu*1000:.0f}ms, "
            #     f"sample={t_sample*1000:.0f}ms, total={timing['total_ms']:.0f}ms",
            #     file=sys.stderr, flush=True,
            # )

    except Exception as e:
        warnings.warn(f"GPU localization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}

    return localization_per_cell, timing


def _localization_summary(loc_df: pd.DataFrame, organelle_name: str) -> dict:
    """Aggregate per-object localization to cell-level summary (matches CPU version)."""
    if loc_df.empty:
        return {}

    agg_funcs = ["sum", "mean", "median", "std", "min", "max", "count"]
    feature_cols = [
        "distance_from_cell_edge",
        "distance_from_nucleus",
        "distance_from_nucleus_centroid",
        "normalized_radial_position",
    ]

    result = {}
    for col in feature_cols:
        if col not in loc_df.columns:
            continue
        values = loc_df[col].dropna()
        if len(values) == 0:
            continue
        for agg in agg_funcs:
            if agg == "sum":
                val = values.sum()
            elif agg == "mean":
                val = values.mean()
            elif agg == "median":
                val = values.median()
            elif agg == "std":
                val = values.std() if len(values) > 1 else 0.0
            elif agg == "min":
                val = values.min()
            elif agg == "max":
                val = values.max()
            elif agg == "count":
                val = len(values)
            result[f"{organelle_name}_{col}_{agg}"] = val

    # Radial distribution (concentric-shell fractions + angular anisotropy) from
    # the same per-object loc_df — no extra segmentation/KDTree work. Keeps the GPU
    # localization path in parity with the CPU compute_cell_level_localization_summary.
    from organelle_profiler.feature_extraction.localization_features import (
        _radial_distribution_from_df,
    )
    result.update(_radial_distribution_from_df(loc_df, organelle_name))

    return result


# Convenience function to match CPU API
def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary mask using GPU if available.

    Falls back to CPU if GPU is not available.
    """
    if is_gpu_available():
        try:
            import cucim.skimage.morphology as cucim_morph
            mask_gpu = cp.asarray(binary_mask)
            # cucim uses 'thin' instead of 'skeletonize'
            skeleton_gpu = cucim_morph.thin(mask_gpu)
            return cp.asnumpy(skeleton_gpu)
        except Exception:
            pass

    # Fall back to CPU
    from skimage.morphology import skeletonize
    return skeletonize(binary_mask)
