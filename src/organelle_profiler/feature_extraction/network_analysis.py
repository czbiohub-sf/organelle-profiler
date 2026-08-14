import time as time_module
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, median as ndimage_median
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, euler_number
from skimage.filters import threshold_otsu, threshold_triangle


# need to install skan
# pip install skan --no-deps
# need to integrate with feature_extraction.py
try:
    from skan import Skeleton, summarize
except ImportError:
    raise ImportError(
        "The 'skan' library is required for network analysis. Please install it with 'pip install skan'"
    )


def _fractal_dimension(mask: np.ndarray) -> float:
    """
    Calculates the fractal dimension of a 2D binary mask using the box-counting method.
    This should only be used for complex, filamentous structures.
    """
    if not np.any(mask) or mask.ndim != 2:
        return np.nan

    # Find bounding box to avoid counting empty parts of the image
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # If mask is empty or 1D, fractal dimension is not well-defined for this method
    if not np.any(rows) or not np.any(cols):
        return np.nan

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    trimmed_mask = mask[rmin : rmax + 1, cmin : cmax + 1]

    # Use powers of 2 for scales, ensuring we have enough scales for a fit
    min_dim = min(trimmed_mask.shape)
    n = int(np.floor(np.log2(min_dim)))
    if n < 4:  # Need at least a few scales for a meaningful fit
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

    # linear regression on log-log scale
    # Filter out scales where count is 0, which can happen for very sparse masks
    scales_log = np.log(scales[scales > 0])
    counts_log = np.log(np.array(counts)[np.array(counts) > 0])

    if len(scales_log) < 2 or len(counts_log) < 2:
        return np.nan

    coeffs = np.polyfit(scales_log, counts_log, 1)
    return -coeffs[0]


def _get_intensity_threshold(intensity_values: np.ndarray) -> float:
    """
    Calculates an intensity threshold using a combination of Otsu and Triangle methods,
    mimicking the approach in nellie_networking.py.

    Parameters
    ----------
    intensity_values : np.ndarray
        An array of intensity values from which to calculate the threshold.

    Returns
    -------
    float
        The calculated intensity threshold.
    """
    if intensity_values.size == 0:
        return 0.0

    # Filter out non-positive values for log transform
    positive_intensities = intensity_values[intensity_values > 0]
    if positive_intensities.size == 0:
        return 0.0

    log_intensities = np.log10(positive_intensities)

    try:
        thresh_otsu = threshold_otsu(log_intensities)
    except ValueError:  # Can happen if all values are the same
        thresh_otsu = np.inf
    try:
        thresh_tri = threshold_triangle(log_intensities)
    except (ValueError, IndexError):  # Can happen with certain distributions
        thresh_tri = np.inf

    # Convert back from log scale
    final_thresh_otsu = 10**thresh_otsu
    final_thresh_tri = 10**thresh_tri

    return min(final_thresh_otsu, final_thresh_tri)


def _get_cleaned_skeleton(
    mask: np.ndarray, intensity_image: np.ndarray = None, min_branch_size: int = 2
) -> np.ndarray:
    """
    Generates a cleaned skeleton from a binary mask, optionally using an
    intensity image for pruning.

    Parameters
    ----------
    mask : np.ndarray
        The input binary mask.
    intensity_image : np.ndarray, optional
        A grayscale image used for intensity-based pruning of the skeleton.
    min_branch_size : int, optional
        The minimum size in pixels for a skeleton branch to be kept.

    Returns
    -------
    np.ndarray
        A cleaned, binary skeleton image.
    """
    skeleton = skeletonize(mask)

    # 1. Intensity-based cleaning
    if intensity_image is not None and np.any(skeleton):
        skeleton_intensities = intensity_image[skeleton]
        threshold = _get_intensity_threshold(skeleton_intensities)

        # Prune skeleton based on the intensity threshold
        pruned_skeleton = skeleton & (intensity_image > threshold)

        # If pruning removed everything, fall back to the original skeleton
        skeleton = pruned_skeleton if np.any(pruned_skeleton) else skeleton

    # 2. Small fragment removal
    if np.any(skeleton):
        labeled_skeleton, num_labels = label(skeleton, return_num=True, connectivity=1)
        if num_labels > 0:
            component_sizes = np.bincount(labeled_skeleton.ravel())
            too_small = component_sizes < min_branch_size
            # Create a removal mask and apply it
            removal_mask = too_small[labeled_skeleton]
            skeleton[removal_mask] = False

    return skeleton


def calculate_network_features(
    organelle_mask: np.ndarray,
    spacing: tuple,
    intensity_image: np.ndarray = None,
    full_features: bool = False,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Calculates skeleton-based features for an organelle mask, using an
    intensity-based cleaning approach inspired by nellie_networking.py.

    This function uses 'skan' to analyze the skeleton of the input mask
    and computes network/filament-specific features.

    Parameters
    ----------
    organelle_mask : np.ndarray
        A 2D or 3D numpy array representing the binary mask of an organelle.
    spacing : tuple
        The pixel spacing/resolution for each dimension (e.g., (y, x) or (z, y, x)).
    intensity_image : np.ndarray, optional
        A corresponding intensity image (e.g., Frangi-filtered) used to
        prune the skeleton and remove low-confidence branches before analysis.
    full_features : bool, optional
        If True, calculate computationally expensive features like fractal dimension.

    Returns
    -------
    tuple[pd.DataFrame, dict, pd.DataFrame]
        A tuple containing:
        - A DataFrame with per-branch features.
        - A dictionary with network-wide summary features.
        - A DataFrame with per-object (connected component) network features
          including 'label' and 'num_branches' for visualization.
    """
    timings = {}

    if not np.any(organelle_mask):
        return pd.DataFrame(), {}, pd.DataFrame(), timings

    organelle_mask = organelle_mask > 0
    network_summary = {}

    # --- Part 0: Calculate some whole-mask features ---
    t0 = time_module.time()
    labeled_mask = label(organelle_mask)
    props = regionprops(labeled_mask)
    if not props:
        return pd.DataFrame(), {}, pd.DataFrame(), timings
    main_prop = props[0]

    # Use area_convex (modern API) with fallback to convex_area (deprecated)
    convex_area = getattr(main_prop, 'area_convex', None) or main_prop.convex_area
    # Avoid division by zero for density calculations
    convex_area_safe = convex_area if convex_area > 0 else 1.0

    network_summary["euler_number"] = euler_number(
        organelle_mask, connectivity=organelle_mask.ndim
    )
    timings["label_regionprops_euler"] = time_module.time() - t0

    # 1. Get a cleaned skeleton
    t0 = time_module.time()
    skeleton_to_process = _get_cleaned_skeleton(organelle_mask, intensity_image)
    timings["skeletonize_clean"] = time_module.time() - t0

    if not np.any(skeleton_to_process):
        return pd.DataFrame(), network_summary, pd.DataFrame(), timings

    # Track skeleton pixel count for network complexity metric
    network_summary["skeleton_pixel_count"] = int(np.sum(skeleton_to_process))

    # Count separate connected components in skeleton (disconnected network fragments)
    t0 = time_module.time()
    labeled_skeleton, num_skeleton_components = label(skeleton_to_process, return_num=True)
    timings["skeleton_label"] = time_module.time() - t0
    network_summary["num_skeleton_components"] = num_skeleton_components

    # 2. Use skan to analyze the skeleton and get branch properties
    t0 = time_module.time()
    skan_obj = Skeleton(skeleton_to_process, spacing=spacing)
    branch_data = summarize(skan_obj, separator="-")
    timings["skan_analysis"] = time_module.time() - t0
    # print("SKAN COLUMNS AVAILABLE:", branch_data.columns) # <-- ADD THIS DEBUG LINE

    # If no branches are found, return the empty dataframe and the current summary
    if branch_data.empty:
        return pd.DataFrame(), network_summary, pd.DataFrame(), timings

    # --- Part A: Network-wide features (from the branch_data DataFrame) ---
    t0 = time_module.time()
    network_summary["num_branches"] = len(branch_data)

    # Count endpoints and junctions using the 'branch-type' column.
    # skan branch types: 0=endpoint-endpoint, 1=endpoint-junction, 2=junction-junction.
    endpoint_branches = branch_data[branch_data["branch-type"] < 2]
    junction_branches = branch_data[branch_data["branch-type"] > 0]

    # Find the unique node IDs associated with each type of branch.
    endpoint_node_ids = set(endpoint_branches["node-id-src"]) | set(
        endpoint_branches["node-id-dst"]
    )
    junction_node_ids = set(junction_branches["node-id-src"]) | set(
        junction_branches["node-id-dst"]
    )

    # A true endpoint node appears in endpoint branches but NOT in junction branches.
    final_endpoint_ids = endpoint_node_ids - junction_node_ids
    network_summary["num_endpoints"] = len(final_endpoint_ids)
    network_summary["num_nodes"] = len(junction_node_ids)

    # --- Calculate degrees manually since the columns are not provided ---
    # The degree is the number of times a node ID appears in the table.
    node_ids_series = pd.concat(
        [branch_data["node-id-src"], branch_data["node-id-dst"]]
    )
    node_degrees = node_ids_series.value_counts().reset_index()
    node_degrees.columns = ["node_id", "degree"]

    # Calculate the average degree for only the junction nodes.
    if not junction_node_ids:
        network_summary["average_degree"] = 0
    else:
        junction_degrees = node_degrees[node_degrees["node_id"].isin(junction_node_ids)]
        if not junction_degrees.empty:
            network_summary["average_degree"] = junction_degrees["degree"].mean()
        else:
            network_summary["average_degree"] = 0
    timings["network_wide_features"] = time_module.time() - t0

    # 3. Calculate branch thickness (vectorized via path_label_image)
    t0 = time_module.time()
    dist_transform = distance_transform_edt(organelle_mask, sampling=spacing)
    timings["distance_transform"] = time_module.time() - t0

    t0 = time_module.time()
    # Get branch-labeled skeleton: each pixel = branch_id + 1
    branch_label_img = skan_obj.path_label_image()
    # branch_data.index is 0-based, path_label_image uses index+1
    branch_indices = np.array(branch_data.index) + 1
    # Single vectorized call: median of dist_transform per branch region
    median_radii = ndimage_median(dist_transform, labels=branch_label_img, index=branch_indices)
    branch_data["branch_thickness"] = 2 * np.asarray(median_radii)
    timings["branch_thickness"] = time_module.time() - t0

    # 4. Calculate tortuosity (arc-chord ratio)
    t0 = time_module.time()
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
    timings["tortuosity"] = time_module.time() - t0

    # --- Part B: More network-wide features (from branch_data) ---
    total_branch_length = branch_data["branch-distance"].sum()
    network_summary["total_branch_length"] = total_branch_length  # Total skeleton length in physical units
    network_summary["network_length_density"] = total_branch_length / convex_area_safe
    network_summary["branching_density"] = (
        network_summary["num_nodes"] / convex_area_safe
    )

    # Largest connected component is the sum of branch lengths in the largest subgraph
    if (
        "skeleton-id" in branch_data.columns
        and branch_data["skeleton-id"].nunique() > 1
    ):
        lcc_length = branch_data.groupby("skeleton-id")["branch-distance"].sum().max()
        network_summary["largest_connected_component_size"] = lcc_length
    else:
        network_summary["largest_connected_component_size"] = total_branch_length

    # --- Part C: Computationally expensive features ---
    if full_features:
        if organelle_mask.ndim == 2:
            network_summary["fractal_dimension"] = _fractal_dimension(organelle_mask)
        else:
            network_summary["fractal_dimension"] = np.nan  # Not implemented for 3D

        # Mesh Area (average area of network loops)
        cycle_branches = branch_data[branch_data["branch_type"] == 3]
        mesh_areas = []
        if not cycle_branches.empty and organelle_mask.ndim == 2:
            for branch_id in cycle_branches["branch-id"]:
                path = skan_obj.path_coordinates(branch_id)
                # Shoelace formula for polygon area from coordinates
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

    # 5. Clean up and rename columns
    branch_data.rename(columns={"branch-distance": "branch_length"}, inplace=True)

    feature_cols = [
        "branch_length",
        "branch_thickness",
        "tortuosity",
        "branch_type",
    ]

    final_cols = [col for col in feature_cols if col in branch_data.columns]

    # --- Part D: Per-object (connected component) network features ---
    # Compute per-object: num_branches, num_nodes, num_endpoints, average_degree,
    # and aggregated branch metrics (mean branch_length, branch_thickness, tortuosity)
    t0 = time_module.time()
    per_object_features = pd.DataFrame()

    if "skeleton-id" in branch_data.columns:
        # --- Step 1: Pre-compute skeleton-id → organelle label mapping (one pass) ---
        # Instead of scanning entire image per skeleton-id, extract all at once
        skel_img = skan_obj.skeleton_image
        skel_flat = skel_img.ravel()
        org_flat = labeled_mask.ravel()
        skel_pixel_mask = skel_flat > 0
        skel_ids_at_pixels = skel_flat[skel_pixel_mask]
        org_labels_at_pixels = org_flat[skel_pixel_mask]

        # Sort by skeleton-id for efficient grouping
        sort_idx = np.argsort(skel_ids_at_pixels, kind='mergesort')
        skel_sorted = skel_ids_at_pixels[sort_idx]
        org_sorted = org_labels_at_pixels[sort_idx]
        unique_skel_pixel, group_starts = np.unique(skel_sorted, return_index=True)
        group_ends = np.append(group_starts[1:], len(skel_sorted))

        skel_to_label = {}
        for i, sid in enumerate(unique_skel_pixel):
            chunk = org_sorted[group_starts[i]:group_ends[i]]
            chunk = chunk[chunk > 0]
            if len(chunk) > 0:
                skel_to_label[int(sid)] = int(np.bincount(chunk).argmax())

        # --- Step 2: Vectorized branch metric aggregation ---
        agg_df = branch_data.groupby("skeleton-id", sort=False).agg(
            num_branches=("branch_length", "size"),
            branch_length=("branch_length", "mean"),
            branch_thickness=("branch_thickness", "mean"),
            tortuosity=("tortuosity", "mean"),
        )

        # --- Step 3: Vectorized node/endpoint counting per skeleton ---
        # Extract numpy arrays for speed
        skel_ids_arr = branch_data["skeleton-id"].values
        btypes = branch_data["branch-type"].values
        src_nodes = branch_data["node-id-src"].values
        dst_nodes = branch_data["node-id-dst"].values

        # Build per-skeleton node counts using numpy
        node_stats = {}  # skel_id -> (num_nodes, num_endpoints, avg_degree)
        for sid in agg_df.index:
            mask = skel_ids_arr == sid
            bt = btypes[mask]
            src = src_nodes[mask]
            dst = dst_nodes[mask]

            # Junction branches (type > 0) and endpoint branches (type < 2)
            junc_mask = bt > 0
            ep_mask = bt < 2

            junction_node_set = set()
            if np.any(junc_mask):
                junction_node_set = set(src[junc_mask]) | set(dst[junc_mask])

            endpoint_node_set = set()
            if np.any(ep_mask):
                ep_candidates = set(src[ep_mask]) | set(dst[ep_mask])
                endpoint_node_set = ep_candidates - junction_node_set

            # Average degree for junction nodes
            avg_deg = 0
            if junction_node_set:
                all_nodes = np.concatenate([src, dst])
                node_counts = np.bincount(all_nodes.astype(np.intp))
                deg_sum = sum(node_counts[n] for n in junction_node_set if n < len(node_counts))
                avg_deg = deg_sum / len(junction_node_set)

            node_stats[sid] = (len(junction_node_set), len(endpoint_node_set), avg_deg)

        # --- Step 4: Assemble results ---
        per_object_data = []
        for sid in agg_df.index:
            label_id = skel_to_label.get(int(sid))
            if label_id is not None:
                row = agg_df.loc[sid]
                num_nodes, num_endpoints, avg_degree = node_stats.get(sid, (0, 0, 0))
                per_object_data.append({
                    "label": label_id,
                    "num_branches": int(row["num_branches"]),
                    "num_nodes": num_nodes,
                    "num_endpoints": num_endpoints,
                    "average_degree": avg_degree,
                    "branch_length": row["branch_length"],
                    "branch_thickness": row["branch_thickness"],
                    "tortuosity": row["tortuosity"],
                })

        if per_object_data:
            per_object_features = pd.DataFrame(per_object_data)
    timings["per_object_features"] = time_module.time() - t0

    # Record num_branches for correlation analysis
    timings["num_branches"] = len(branch_data)
    timings["mask_pixels"] = int(np.sum(organelle_mask))

    return branch_data[final_cols], network_summary, per_object_features, timings
