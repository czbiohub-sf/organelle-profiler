"""
Morphological Postprocessing for Organelle Segmentation
=======================================================

This module provides morphological postprocessing utilities for
organelle segmentation, including watershed labeling and structure-specific
postprocessing for tubular, vesicular, and nucleoli structures.

Functions:
- watershed_label: Label objects using watershed on distance transform
- postprocess_vesicular_mask: Postprocess vesicular structures
- postprocess_nucleoli_mask: Postprocess nucleoli structures
- postprocess_tubular_mask: Postprocess tubular structures
"""

import numpy as np


def watershed_label(
    binary_mask: np.ndarray,
    min_distance: int = 3,
    min_object_size: int = 0,
    compactness: float = 0.0,
    erosion_iterations: int = 0,
    min_peak_distance: float = 1.0,
    h_maxima: float = 0.0,
) -> np.ndarray:
    """
    Label objects using watershed on distance transform.

    Separates touching round objects that would merge with simple connected
    components labeling. This is the preferred labeling method for vesicular
    and nucleoli structures that should be discrete spherical objects.

    Args:
        binary_mask: Binary mask of detected objects (2D array)
        min_distance: Minimum pixels between object centers (lower = more aggressive separation)
            - Vesicles (~0.5μm, ~3-5px): use min_distance=1-2
            - Nucleoli (~1-3μm, ~6-20px): use min_distance=3-4
        min_object_size: Minimum object size in pixels. Objects smaller than this are removed.
            - Vesicles: use 4 (removes single-pixel noise)
            - Nucleoli: use 50 (removes small false positives, keeps real nucleoli ~50-500px)
        compactness: Controls watershed compactness (0.0 = standard, higher = more compact/round).
            Higher values bias towards more spherical regions by penalizing distance from markers.
            - 0.0: Standard watershed (no compactness bias)
            - 0.01-0.1: Mild compactness bias (good starting point)
            - 1.0+: Strong compactness bias (very round regions)
        erosion_iterations: Number of binary erosion iterations before watershed.
            This physically separates touching objects by shrinking them.
            - 0: No erosion (default)
            - 1-2: Mild separation (good for larger structures)
            - 2-3: Stronger separation (good for nucleoli)
            NOTE: Don't use for small vesicles - will remove them entirely
        min_peak_distance: Minimum distance transform value for a peak to be valid.
            Filters out shallow peaks in thin/elongated regions.
            - 1.0: Accept all peaks (default)
            - 2.0: Require object to be at least 4px diameter
            - 3.0: Require object to be at least 6px diameter
        h_maxima: H-maxima transform height. Suppresses peaks that are less than h
            below the highest point in their neighborhood. Good for small vesicles.
            - 0.0: No suppression (default)
            - 0.3-0.5: Mild suppression (merges very shallow ridges)
            - 1.0+: Strong suppression (only keeps prominent peaks)

    Returns:
        Labeled mask (int32) with separated objects

    Example:
        >>> binary_mask = vesselness_map > threshold
        >>> labels = watershed_label(binary_mask, min_distance=1, h_maxima=0.5)  # For vesicles
        >>> labels = watershed_label(binary_mask, min_distance=3, erosion_iterations=2)  # For nucleoli
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as scipy_ndi
    from skimage.morphology import h_maxima as skimage_h_maxima

    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    # Apply erosion to physically separate touching objects (for larger structures)
    if erosion_iterations > 0:
        eroded_mask = binary_erosion(binary_mask, iterations=erosion_iterations)
        # Use eroded mask for finding peaks, but original mask for final labels
        peak_mask = eroded_mask
    else:
        peak_mask = binary_mask

    # Distance transform on peak mask
    distance = distance_transform_edt(peak_mask)

    # Apply h-maxima transform to suppress shallow peaks (good for small vesicles)
    if h_maxima > 0:
        # h_maxima suppresses all maxima that are less than h below the regional maximum
        # This merges shallow ridges while keeping distinct round peaks
        distance_filtered = skimage_h_maxima(distance, h_maxima)
    else:
        distance_filtered = distance

    # Find local maxima as markers, filtering by minimum peak height
    coords = peak_local_max(
        distance_filtered,
        min_distance=min_distance,
        labels=peak_mask,
        exclude_border=False,
        threshold_abs=min_peak_distance,  # Filter shallow peaks
    )

    if len(coords) == 0:
        # No peaks found, fall back to simple connected components on original mask
        footprint = scipy_ndi.generate_binary_structure(binary_mask.ndim, 1)
        labels, _ = scipy_ndi.label(binary_mask, structure=footprint)
    else:
        # Create marker array
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

        # Watershed on original mask's distance transform (not eroded/filtered)
        # This gives better boundary placement
        original_distance = distance_transform_edt(binary_mask)
        labels = watershed(-original_distance, markers, mask=binary_mask, compactness=compactness)

    labels = labels.astype(np.int32)

    # Remove small objects if min_object_size > 0
    if min_object_size > 0 and labels.max() > 0:
        # Count pixels per label
        label_ids, counts = np.unique(labels, return_counts=True)
        # Find labels that are too small (excluding background label 0)
        small_labels = label_ids[(label_ids > 0) & (counts < min_object_size)]
        if len(small_labels) > 0:
            # Remove small objects
            labels[np.isin(labels, small_labels)] = 0
            # Relabel to ensure continuous IDs
            labels, _ = scipy_ndi.label(labels > 0)
            labels = labels.astype(np.int32)

    return labels


def postprocess_vesicular_mask(
    binary_mask: np.ndarray,
    min_size: int = 3,
) -> np.ndarray:
    """
    Passthrough for vesicular structures - no morphological operations.

    Watershed labeling (called after this) handles all separation.
    This function exists for API compatibility but does nothing.

    Args:
        binary_mask: Binary segmentation mask (2D array)
        min_size: Unused (kept for API compatibility)

    Returns:
        Unchanged binary mask
    """
    # No-op: watershed handles separation, small object removal is in watershed_label
    return binary_mask


def postprocess_nucleoli_mask(
    binary_mask: np.ndarray,
    min_size: int = 20,
    do_opening: bool = True,
    opening_radius: int = 1,
    do_closing: bool = True,
    closing_radius: int = 3,
) -> np.ndarray:
    """
    Apply morphological post-processing for nucleoli to favor round, filled structures.

    Strategy:
    1. Opening to clean up edges (gentle noise removal) - optional
    2. Morphological closing + fill_holes to fill internal gaps - optional
    3. Distance transform + watershed to split merged nucleoli
    4. Remove small objects

    Note: Opening = erosion followed by dilation (removes small protrusions)
          Closing = dilation followed by erosion (fills small gaps/holes)

    Args:
        binary_mask: Binary segmentation mask (2D array)
        min_size: Minimum object size in pixels to keep (default: 20)
        do_opening: Enable opening operation (default: True)
        opening_radius: Disk radius for opening (default: 1 = 3x3 disk)
        do_closing: Enable closing + fill_holes operation (default: True).
            Closing fills boundary gaps, fill_holes fills enclosed interior regions.
        closing_radius: Disk radius for morphological closing (default: 3).
            Controls how large of gaps/holes can be filled. Larger = fills bigger holes
            but may also connect nearby objects if too large.

    Returns:
        Post-processed binary mask with rounder, filled nucleoli
    """
    from skimage.morphology import disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as scipy_ndi

    if binary_mask.ndim != 2:
        raise ValueError(f"postprocess_nucleoli_mask expects 2D mask, got shape {binary_mask.shape}")

    # Count initial pixels for debugging
    initial_pixels = binary_mask.sum()
    # print(f"      [nucleoli postprocess] Input: {initial_pixels} pixels")

    result = binary_mask.copy()

    # Step 1: Opening to clean up edges (removes noise/protrusions)
    if do_opening and opening_radius > 0:
        disk_open = disk(opening_radius)
        result = scipy_ndi.binary_opening(result, structure=disk_open)
         #print(f"      [nucleoli postprocess] After opening (r={opening_radius}): {result.sum()} pixels")

    # Step 2: Morphological closing + fill_holes to fill internal gaps
    # Closing fills boundary gaps, fill_holes fills enclosed interior regions
    if do_closing and closing_radius > 0:
        disk_close = disk(closing_radius)
        result = scipy_ndi.binary_closing(result, structure=disk_close)
        result = scipy_ndi.binary_fill_holes(result)
         #print(f"      [nucleoli postprocess] After closing+fill_holes (r={closing_radius}): {result.sum()} pixels")

    # Step 4: Distance transform for watershed seeds
    # The distance transform gives distance to nearest background pixel
    distance = scipy_ndi.distance_transform_edt(result)

    # Step 5: Find local maxima as watershed seeds
    # min_distance controls minimum separation between nucleoli centers
    # Larger value = fewer, more separated objects
    coords = peak_local_max(
        distance,
        min_distance=5,  # Minimum 5 pixels between nucleoli centers
        threshold_abs=2,  # Minimum distance value (excludes tiny objects)
        labels=result.astype(int),
    )

    # Create markers from peak coordinates
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    # Step 6: Watershed to split merged nucleoli
    # Use negative distance so watershed flows from peaks outward
    if markers.max() > 0:
        labels = watershed(-distance, markers, mask=result)
        result = labels > 0
    else:
        pass  # Keep result as is

    # Step 7: Remove small objects
    labeled_temp, _ = scipy_ndi.label(result)
    if labeled_temp.max() > 0:
        areas = np.bincount(labeled_temp.ravel())[1:]
        small_objects = np.where(areas < min_size)[0] + 1
        result[np.isin(labeled_temp, small_objects)] = False

    n_objects = scipy_ndi.label(result)[1]
    # print(f"      [nucleoli postprocess] Output: {result.sum()} pixels, {n_objects} objects ({100*result.sum()/max(1,initial_pixels):.1f}% retained)")

    return result


def postprocess_tubular_mask(
    binary_mask: np.ndarray,
    min_size: int = 5,
    do_opening: bool = True,
    opening_size: int = 2,
    do_fill_holes: bool = False,
) -> np.ndarray:
    """
    Apply morphological post-processing for tubular structures.

    Uses square structuring elements to preserve elongated shapes:
    1. Fill holes (optional, default only for 3D)
    2. Binary opening with small square kernel (optional)
    3. Remove small objects

    Args:
        binary_mask: Binary segmentation mask (2D or 3D array)
        min_size: Minimum object size in pixels to keep (default: 5)
        do_opening: Enable opening operation (default: True)
        opening_size: Square kernel size for opening (default: 2 = 2x2)
        do_fill_holes: Fill holes in mask (default: False, but auto-enabled for 3D)

    Returns:
        Post-processed binary mask
    """
    from scipy import ndimage as scipy_ndi

    is_3d = binary_mask.ndim == 3

    result = binary_mask.copy()

    # Fill holes (auto-enabled for 3D, or if explicitly requested)
    if do_fill_holes or is_3d:
        result = scipy_ndi.binary_fill_holes(result)

    # Binary opening with small square kernel
    if do_opening and opening_size > 0:
        structure = np.ones((opening_size,) * result.ndim)
        result = scipy_ndi.binary_opening(result, structure=structure)

    # Remove small objects
    footprint = scipy_ndi.generate_binary_structure(result.ndim, 1)
    labeled_temp, _ = scipy_ndi.label(result, structure=footprint)
    if labeled_temp.max() > 0:
        areas = np.bincount(labeled_temp.ravel())[1:]
        small_objects = np.where(areas < min_size)[0] + 1
        result[np.isin(labeled_temp, small_objects)] = False

    return result


def topology_preserving_thinning(
    binary_mask: np.ndarray,
    min_thickness: float = 1.6,
    thin_dist: int = 1,
) -> np.ndarray:
    """Thin a mask while preserving connectivity (infer-subc Golgi recipe).

    Erodes the mask by ``thin_dist`` pixels but only where it is thicker than
    ``min_thickness`` (distance-transform based), and always unions back the
    morphological skeleton so thin bridges between regions are never broken.
    This collapses thick sheet/ribbon structures (Golgi) toward their medial
    axis without fragmenting them.

    Mirrors aicssegmentation's ``topology_preserving_thinning``: thick cores get
    thinned, sub-``min_thickness`` features are preserved as-is, and the skeleton
    guarantees topology preservation.

    Args:
        binary_mask: Binary segmentation mask (2D array)
        min_thickness: Minimum half-width (in pixels) a region must exceed before
            it is eroded. Features thinner than this are kept unchanged.
        thin_dist: Number of erosion iterations applied in the thick regions.

    Returns:
        Thinned boolean mask with topology preserved.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    from skimage.morphology import skeletonize

    if binary_mask.ndim != 2:
        raise ValueError(
            f"topology_preserving_thinning expects 2D mask, got shape {binary_mask.shape}"
        )

    bw = binary_mask > 0
    if not bw.any():
        return bw

    # Distance from the medial axis (skeleton). Pixels far from the centerline
    # are "safe" to erode; a core band of half-width ~(min_thickness + thin_dist)
    # around the centerline is always preserved so topology can't break.
    skeleton = skeletonize(bw)
    dist_from_center = distance_transform_edt(~skeleton)
    safe_zone = dist_from_center > (min_thickness + thin_dist)

    eroded = binary_erosion(bw, iterations=max(1, int(thin_dist)))
    result = eroded | (bw & ~safe_zone)
    return result


def filter_objects_by_physical_size(
    labels: np.ndarray,
    pixel_size_um: float,
    min_size_um2: float = None,
    max_size_um2: float = None,
    z_size_um: float = None,
) -> np.ndarray:
    """Remove labeled objects outside a physical size range.

    Filters in physical units (µm² for 2D, µm³ for 3D) rather than raw pixels so
    that thresholds transfer across acquisitions with different pixel sizes.
    Converts the µm size bounds to a voxel-count bound via ``pixel_size_um``
    (and ``z_size_um`` for 3D), then drops objects below ``min_size_um2`` or
    above ``max_size_um2`` and relabels to keep IDs contiguous.

    Args:
        labels: Labeled integer mask (2D or 3D)
        pixel_size_um: In-plane pixel size (µm/px)
        min_size_um2: Minimum object size (µm² for 2D, µm³ for 3D); None disables
        max_size_um2: Maximum object size (µm² for 2D, µm³ for 3D); None disables
        z_size_um: Z step (µm); required for 3D, defaults to pixel_size_um

    Returns:
        Relabeled int32 mask with out-of-range objects removed.
    """
    from scipy import ndimage as scipy_ndi

    if min_size_um2 is None and max_size_um2 is None:
        return labels
    if labels.max() == 0:
        return labels

    if labels.ndim == 3:
        voxel_um = (pixel_size_um ** 2) * (z_size_um if z_size_um else pixel_size_um)
    else:
        voxel_um = pixel_size_um ** 2

    label_ids, counts = np.unique(labels, return_counts=True)
    drop = np.zeros(labels.shape, dtype=bool)
    for lid, count in zip(label_ids, counts):
        if lid == 0:
            continue
        size_um = count * voxel_um
        if (min_size_um2 is not None and size_um < min_size_um2) or (
            max_size_um2 is not None and size_um > max_size_um2
        ):
            drop |= labels == lid

    if drop.any():
        result = labels.copy()
        result[drop] = 0
        footprint = scipy_ndi.generate_binary_structure(result.ndim, 1)
        result, _ = scipy_ndi.label(result > 0, structure=footprint)
        return result.astype(np.int32)
    return labels
