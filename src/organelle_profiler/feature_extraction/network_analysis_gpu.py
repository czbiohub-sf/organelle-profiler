"""
GPU-Accelerated Network/Skeleton Analysis for Organelle Feature Extraction.

This module provides GPU-accelerated preprocessing for network analysis,
including skeletonization and distance transforms. The actual network
topology analysis (using skan) remains CPU-based as it requires numba.

GPU Acceleration Targets
------------------------
1. Skeletonization: cucim.skimage.morphology.thin
2. Distance Transform: cupyx.scipy.ndimage.distance_transform_edt
3. Labeling: cucim.skimage.measure.label
4. Region properties: cucim.skimage.measure.regionprops

Performance
-----------
Expected 3-10x speedup for preprocessing steps.
The skan analysis portion remains CPU-bound.
"""

import warnings
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# GPU availability
_GPU_AVAILABLE = False
_GPU_INIT_ERROR = None

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    import cucim.skimage.morphology as cucim_morph
    import cucim.skimage.measure as cucim_measure
    from cucim.skimage.filters import threshold_otsu as cucim_threshold_otsu
    _GPU_AVAILABLE = True
except ImportError as e:
    _GPU_INIT_ERROR = str(e)
except Exception as e:
    _GPU_INIT_ERROR = str(e)

# Import CPU versions for fallback and skan analysis
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize
from skimage.measure import label as skimage_label, regionprops, euler_number
from skimage.filters import threshold_otsu, threshold_triangle

try:
    from skan import Skeleton, summarize
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    if not _GPU_AVAILABLE:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _skeletonize_gpu(mask: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated skeletonization using cucim's thin operation.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask to skeletonize.

    Returns
    -------
    np.ndarray
        Skeletonized binary mask.
    """
    if not is_gpu_available():
        return skimage_skeletonize(mask)

    try:
        mask_gpu = cp.asarray(mask.astype(np.uint8))
        # cucim uses 'thin' which is morphological thinning
        skeleton_gpu = cucim_morph.thin(mask_gpu)
        return cp.asnumpy(skeleton_gpu).astype(bool)
    except Exception as e:
        warnings.warn(f"GPU skeletonization failed ({e}), using CPU")
        return skimage_skeletonize(mask)


def _distance_transform_gpu(mask: np.ndarray, spacing: tuple) -> np.ndarray:
    """
    GPU-accelerated distance transform.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    spacing : tuple
        Pixel spacing.

    Returns
    -------
    np.ndarray
        Distance transform array.
    """
    if not is_gpu_available():
        return scipy_distance_transform_edt(mask, sampling=spacing)

    try:
        mask_gpu = cp.asarray(mask.astype(bool))
        # cupy distance transform
        dist_gpu = cupy_ndimage.distance_transform_edt(mask_gpu, sampling=spacing)
        return cp.asnumpy(dist_gpu)
    except Exception as e:
        warnings.warn(f"GPU distance transform failed ({e}), using CPU")
        return scipy_distance_transform_edt(mask, sampling=spacing)


def _label_gpu(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    GPU-accelerated connected component labeling.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.

    Returns
    -------
    tuple
        (labeled_array, num_labels)
    """
    if not is_gpu_available():
        return skimage_label(mask, return_num=True)

    try:
        mask_gpu = cp.asarray(mask.astype(bool))
        labeled_gpu = cucim_measure.label(mask_gpu)
        num_labels = int(labeled_gpu.max())
        return cp.asnumpy(labeled_gpu), num_labels
    except Exception as e:
        warnings.warn(f"GPU labeling failed ({e}), using CPU")
        return skimage_label(mask, return_num=True)


def _fractal_dimension(mask: np.ndarray) -> float:
    """
    Calculates the fractal dimension of a 2D binary mask using the box-counting method.
    This is CPU-only as it's not easily GPU-parallelizable.
    """
    if not np.any(mask) or mask.ndim != 2:
        return np.nan

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.nan

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    trimmed_mask = mask[rmin : rmax + 1, cmin : cmax + 1]

    min_dim = min(trimmed_mask.shape)
    n = int(np.floor(np.log2(min_dim)))
    if n < 4:
        return np.nan

    scales = np.logspace(0, n, num=n + 1, base=2, dtype=int)
    counts = []

    for scale in scales:
        if scale == 0:
            continue
        count = 0
        for y in range(0, trimmed_mask.shape[0], scale):
            for x in range(0, trimmed_mask.shape[1], scale):
                box = trimmed_mask[y : y + scale, x : x + scale]
                if np.any(box):
                    count += 1
        counts.append(count)

    scales_log = np.log(scales[scales > 0])
    counts_log = np.log(np.array(counts)[np.array(counts) > 0])

    if len(scales_log) < 2 or len(counts_log) < 2:
        return np.nan

    coeffs = np.polyfit(scales_log, counts_log, 1)
    return -coeffs[0]


def _get_intensity_threshold(intensity_values: np.ndarray) -> float:
    """
    Calculates intensity threshold using Otsu and Triangle methods.
    """
    if intensity_values.size == 0:
        return 0.0

    positive_intensities = intensity_values[intensity_values > 0]
    if positive_intensities.size == 0:
        return 0.0

    log_intensities = np.log10(positive_intensities)

    try:
        thresh_otsu = threshold_otsu(log_intensities)
    except ValueError:
        thresh_otsu = np.inf
    try:
        thresh_tri = threshold_triangle(log_intensities)
    except (ValueError, IndexError):
        thresh_tri = np.inf

    final_thresh_otsu = 10**thresh_otsu
    final_thresh_tri = 10**thresh_tri

    return min(final_thresh_otsu, final_thresh_tri)


def _get_cleaned_skeleton_gpu(
    mask: np.ndarray, intensity_image: np.ndarray = None, min_branch_size: int = 2
) -> np.ndarray:
    """
    GPU-accelerated skeleton generation and cleaning.

    Uses GPU for:
    - Skeletonization (cucim.thin)
    - Labeling (cucim.label)

    Parameters
    ----------
    mask : np.ndarray
        The input binary mask.
    intensity_image : np.ndarray, optional
        A grayscale image for intensity-based pruning.
    min_branch_size : int, optional
        Minimum size in pixels for skeleton branches.

    Returns
    -------
    np.ndarray
        Cleaned binary skeleton.
    """
    # GPU-accelerated skeletonization
    skeleton = _skeletonize_gpu(mask)

    # Intensity-based cleaning (CPU - involves indexing)
    if intensity_image is not None and np.any(skeleton):
        skeleton_intensities = intensity_image[skeleton]
        threshold = _get_intensity_threshold(skeleton_intensities)
        pruned_skeleton = skeleton & (intensity_image > threshold)
        skeleton = pruned_skeleton if np.any(pruned_skeleton) else skeleton

    # Small fragment removal using GPU labeling
    if np.any(skeleton):
        labeled_skeleton, num_labels = _label_gpu(skeleton)
        if num_labels > 0:
            component_sizes = np.bincount(labeled_skeleton.ravel())
            too_small = component_sizes < min_branch_size
            removal_mask = too_small[labeled_skeleton]
            skeleton[removal_mask] = False

    return skeleton


def calculate_network_features_gpu(
    organelle_mask: np.ndarray,
    spacing: tuple,
    intensity_image: np.ndarray = None,
    full_features: bool = False,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    GPU-accelerated network feature calculation.

    Uses GPU acceleration for:
    - Skeletonization
    - Distance transform
    - Connected component labeling

    The skan topology analysis remains CPU-based.

    Parameters
    ----------
    organelle_mask : np.ndarray
        Binary mask of the organelle.
    spacing : tuple
        Pixel spacing (y, x) or (z, y, x).
    intensity_image : np.ndarray, optional
        Intensity image for skeleton pruning.
    full_features : bool, optional
        Calculate expensive features like fractal dimension.

    Returns
    -------
    tuple
        (branch_data, network_summary, per_object_features)
    """
    if not _SKAN_AVAILABLE:
        raise ImportError("skan library required for network analysis")

    if not np.any(organelle_mask):
        return pd.DataFrame(), {}, pd.DataFrame()

    organelle_mask = organelle_mask > 0
    network_summary = {}

    # GPU-accelerated labeling for initial analysis
    labeled_mask, _ = _label_gpu(organelle_mask)
    props = regionprops(labeled_mask)
    if not props:
        return pd.DataFrame(), {}, pd.DataFrame()

    main_prop = props[0]
    convex_area = getattr(main_prop, 'area_convex', None) or main_prop.convex_area
    convex_area_safe = convex_area if convex_area > 0 else 1.0

    network_summary["euler_number"] = euler_number(
        organelle_mask, connectivity=organelle_mask.ndim
    )

    # GPU-accelerated skeleton cleaning
    skeleton_to_process = _get_cleaned_skeleton_gpu(organelle_mask, intensity_image)

    if not np.any(skeleton_to_process):
        return pd.DataFrame(), network_summary, pd.DataFrame()

    network_summary["skeleton_pixel_count"] = int(np.sum(skeleton_to_process))

    # GPU-accelerated labeling of skeleton
    labeled_skeleton, num_skeleton_components = _label_gpu(skeleton_to_process)
    network_summary["num_skeleton_components"] = num_skeleton_components

    # skan analysis (CPU-based, uses numba internally)
    skan_obj = Skeleton(skeleton_to_process, spacing=spacing)
    branch_data = summarize(skan_obj, separator="-")

    if branch_data.empty:
        return pd.DataFrame(), network_summary, pd.DataFrame()

    # Network-wide features
    network_summary["num_branches"] = len(branch_data)

    endpoint_branches = branch_data[branch_data["branch-type"] < 2]
    junction_branches = branch_data[branch_data["branch-type"] > 0]

    endpoint_node_ids = set(endpoint_branches["node-id-src"]) | set(
        endpoint_branches["node-id-dst"]
    )
    junction_node_ids = set(junction_branches["node-id-src"]) | set(
        junction_branches["node-id-dst"]
    )

    final_endpoint_ids = endpoint_node_ids - junction_node_ids
    network_summary["num_endpoints"] = len(final_endpoint_ids)
    network_summary["num_nodes"] = len(junction_node_ids)

    # Calculate degrees
    node_ids_series = pd.concat(
        [branch_data["node-id-src"], branch_data["node-id-dst"]]
    )
    node_degrees = node_ids_series.value_counts().reset_index()
    node_degrees.columns = ["node_id", "degree"]

    if not junction_node_ids:
        network_summary["average_degree"] = 0
    else:
        junction_degrees = node_degrees[node_degrees["node_id"].isin(junction_node_ids)]
        if not junction_degrees.empty:
            network_summary["average_degree"] = junction_degrees["degree"].mean()
        else:
            network_summary["average_degree"] = 0

    # GPU-accelerated distance transform for branch thickness
    dist_transform = _distance_transform_gpu(organelle_mask, spacing)

    thicknesses = []
    for i in branch_data.index:
        path_coords = skan_obj.path_coordinates(i).astype(int)
        branch_radii = dist_transform[tuple(path_coords.T)]
        median_thickness = 2 * np.median(branch_radii)
        thicknesses.append(median_thickness)

    branch_data["branch_thickness"] = thicknesses

    # Tortuosity calculation
    ndim = organelle_mask.ndim
    src_cols = [f"coord-src-{i}" for i in range(ndim)]
    dst_cols = [f"coord-dst-{i}" for i in range(ndim)]

    src_coords = branch_data[src_cols].values
    dst_coords = branch_data[dst_cols].values
    end_to_end_dist = np.linalg.norm(dst_coords - src_coords, axis=1)

    branch_lengths = branch_data["branch-distance"].values
    tortuosity = np.divide(
        branch_lengths,
        end_to_end_dist,
        out=np.ones_like(branch_lengths),
        where=(end_to_end_dist != 0),
    )
    branch_data["tortuosity"] = tortuosity

    # More network-wide features
    total_branch_length = branch_data["branch-distance"].sum()
    network_summary["total_branch_length"] = total_branch_length
    network_summary["network_length_density"] = total_branch_length / convex_area_safe
    network_summary["branching_density"] = (
        network_summary["num_nodes"] / convex_area_safe
    )

    if (
        "skeleton-id" in branch_data.columns
        and branch_data["skeleton-id"].nunique() > 1
    ):
        lcc_length = branch_data.groupby("skeleton-id")["branch-distance"].sum().max()
        network_summary["largest_connected_component_size"] = lcc_length
    else:
        network_summary["largest_connected_component_size"] = total_branch_length

    # Expensive features
    if full_features:
        if organelle_mask.ndim == 2:
            network_summary["fractal_dimension"] = _fractal_dimension(organelle_mask)
        else:
            network_summary["fractal_dimension"] = np.nan

        cycle_branches = branch_data[branch_data.get("branch_type", branch_data.get("branch-type", pd.Series())) == 3]
        mesh_areas = []
        if not cycle_branches.empty and organelle_mask.ndim == 2:
            for branch_id in cycle_branches.get("branch-id", cycle_branches.index):
                path = skan_obj.path_coordinates(branch_id)
                x, y = path[:, 1], path[:, 0]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                mesh_areas.append(area)

        if mesh_areas:
            network_summary["mesh_area_mean"] = np.mean(mesh_areas)
            network_summary["mesh_area_std"] = np.std(mesh_areas)
            network_summary["mesh_area_sum"] = np.sum(mesh_areas)
        else:
            network_summary["mesh_area_mean"] = 0
            network_summary["mesh_area_std"] = 0
            network_summary["mesh_area_sum"] = 0

    # Clean up column names
    branch_data.rename(columns={"branch-distance": "branch_length"}, inplace=True)

    feature_cols = [
        "branch_length",
        "branch_thickness",
        "tortuosity",
        "branch_type",
    ]
    final_cols = [col for col in feature_cols if col in branch_data.columns]

    # Per-object features
    per_object_features = pd.DataFrame()

    if "skeleton-id" in branch_data.columns:
        per_object_data = []

        for skel_id in branch_data["skeleton-id"].unique():
            skel_branches = branch_data[branch_data["skeleton-id"] == skel_id]
            num_branches = len(skel_branches)

            all_node_ids = pd.concat([skel_branches["node-id-src"], skel_branches["node-id-dst"]])
            node_degrees_local = all_node_ids.value_counts()

            junction_branches_local = skel_branches[skel_branches["branch-type"] > 0]
            endpoint_branches_local = skel_branches[skel_branches["branch-type"] < 2]

            junction_node_ids_local = set()
            if not junction_branches_local.empty:
                junction_node_ids_local = set(junction_branches_local["node-id-src"]) | set(junction_branches_local["node-id-dst"])

            endpoint_node_ids_local = set()
            if not endpoint_branches_local.empty:
                endpoint_candidates = set(endpoint_branches_local["node-id-src"]) | set(endpoint_branches_local["node-id-dst"])
                endpoint_node_ids_local = endpoint_candidates - junction_node_ids_local

            num_nodes = len(junction_node_ids_local)
            num_endpoints = len(endpoint_node_ids_local)

            if junction_node_ids_local:
                junction_degrees = node_degrees_local[node_degrees_local.index.isin(junction_node_ids_local)]
                average_degree = junction_degrees.mean() if len(junction_degrees) > 0 else 0
            else:
                average_degree = 0

            mean_branch_length = skel_branches["branch_length"].mean() if "branch_length" in skel_branches.columns else 0
            mean_branch_thickness = skel_branches["branch_thickness"].mean() if "branch_thickness" in skel_branches.columns else 0
            mean_tortuosity = skel_branches["tortuosity"].mean() if "tortuosity" in skel_branches.columns else 1.0

            skel_mask = (skan_obj.skeleton_image == skel_id)
            label_id = None
            if np.any(skel_mask):
                overlapping_labels = labeled_mask[skel_mask]
                overlapping_labels = overlapping_labels[overlapping_labels > 0]
                if len(overlapping_labels) > 0:
                    label_id = int(np.bincount(overlapping_labels).argmax())

            if label_id is not None:
                per_object_data.append({
                    "label": label_id,
                    "num_branches": num_branches,
                    "num_nodes": num_nodes,
                    "num_endpoints": num_endpoints,
                    "average_degree": average_degree,
                    "branch_length": mean_branch_length,
                    "branch_thickness": mean_branch_thickness,
                    "tortuosity": mean_tortuosity,
                })

        if per_object_data:
            per_object_features = pd.DataFrame(per_object_data)

    return branch_data[final_cols], network_summary, per_object_features


# Convenience alias matching original module API
calculate_network_features = calculate_network_features_gpu
