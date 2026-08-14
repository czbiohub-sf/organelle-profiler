

# --- nellie imports ---
import numpy as np

# --- end nellie imports ---


def otsu_effectiveness(image, inter_variance, xp):
    # flatten image and create histogram
    flattened_image = image.flatten()
    sigma_total_squared = xp.var(flattened_image)
    normalized_sigma_B_squared = inter_variance / sigma_total_squared
    return normalized_sigma_B_squared


def otsu_threshold(matrix, nbins=256, xp=np):
    # gpu version of skimage.filters.threshold_otsu
    counts, bin_edges = xp.histogram(
        matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max())
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Ensure counts is not empty and sum is not zero
    if xp.sum(counts) == 0:
        return bin_centers[0], 0  # Or handle as an error

    counts = counts / xp.sum(counts)

    weight1 = xp.cumsum(counts)
    mean1 = xp.cumsum(counts * bin_centers) / xp.where(
        weight1 > 0, weight1, 1
    )  # Avoid division by zero

    weight2 = xp.cumsum(counts[::-1])[::-1]
    mean2 = (
        xp.cumsum((counts * bin_centers)[::-1])
        / xp.where(weight2 > 0, weight2, 1)[::-1]
    )[::-1]

    # Prevent issues with empty slices
    if len(weight1) < 2 or len(weight2) < 2:
        return bin_centers[0], 0

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Handle case where variance is all zero
    if xp.max(variance12) == 0:
        return bin_centers[0], 0

    idx = xp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold, variance12[idx]


def triangle_threshold(matrix, nbins=256, xp=np):
    # gpu version of skimage.filters.threshold_triangle
    hist, bin_edges = xp.histogram(
        matrix.reshape(-1), bins=nbins, range=(xp.min(matrix), xp.max(matrix))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if xp.sum(hist) == 0:
        return bin_centers[0]

    hist = hist / xp.sum(hist)

    arg_peak_height = xp.argmax(hist)
    peak_height = hist[arg_peak_height]

    try:
        non_zero_indices = xp.flatnonzero(hist)
        if len(non_zero_indices) < 2:
            return bin_centers[0]
        arg_low_level, arg_high_level = non_zero_indices[[0, -1]]
    except IndexError:
        return bin_centers[0]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = xp.flip(hist, axis=0)
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del arg_high_level

    width = arg_peak_height - arg_low_level
    if width <= 0:
        return bin_centers[arg_low_level]

    x1 = xp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = xp.sqrt(peak_height**2 + width**2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1
    arg_level = xp.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


# =============================================================================
# Intensity-image thresholding (infer-subc style)
# =============================================================================
# These operate on the (CLAHE'd, smoothed) raw intensity image rather than the
# Frangi vesselness map, providing the "threshold" detection method used for
# filled / blobby organelles (nuclei, lipid droplets, Golgi, ER). They mirror
# the recipes in infer-subc's core/img.py (log-Li threshold + Masked Object
# thresholding). CPU/numpy only — the "threshold" method runs on the CPU tile
# worker, so GPU variants are unnecessary.


def li_threshold_log(image):
    """Li minimum-cross-entropy threshold computed in log space.

    Mirrors infer-subc's ``apply_log_li_threshold``: take ``log`` of the
    positive intensities, run Li's iterative min-cross-entropy threshold, then
    map back to intensity units via ``exp``. Robust for low-contrast organelles
    (e.g. nuclei) where a linear Otsu/Li over-segments background.

    Returns a scalar threshold in the image's intensity units. If no positive
    pixels exist, returns a value above the image max so nothing passes.
    """
    from skimage.filters import threshold_li

    pos = image[image > 0]
    if pos.size < 2:
        return float(image.max()) + 1.0
    log_pos = np.log(pos.astype(np.float64))
    t_log = threshold_li(log_pos)
    return float(np.exp(t_log))


def _scalar_intensity_threshold(image, method, multiotsu_classes=3, multiotsu_level=0):
    """Compute a single global threshold value for an intensity image.

    Supported methods: "otsu", "triangle", "li", "li_log", "median", "ave"
    (mean of triangle+otsu, matching infer-subc MO's "ave_tri_med"-style
    blends), "multiotsu".

    For "multiotsu", ``multiotsu_classes`` sets the number of intensity classes
    (so ``classes-1`` boundaries) and ``multiotsu_level`` picks which boundary is
    the foreground cutoff: 0 = lowest (most inclusive, foreground = all but the
    darkest class), higher = more stringent (only the brightest classes).
    """
    from skimage.filters import (
        threshold_otsu,
        threshold_triangle,
        threshold_li,
        threshold_multiotsu,
    )

    method = (method or "otsu").lower()
    img = np.asarray(image)
    if method == "otsu":
        return float(threshold_otsu(img))
    if method == "triangle":
        return float(threshold_triangle(img))
    if method == "li":
        return float(threshold_li(img))
    if method == "li_log":
        return li_threshold_log(img)
    if method == "median":
        return float(np.median(img[img > 0])) if np.any(img > 0) else float(img.max())
    if method == "ave":
        return float((threshold_triangle(img) + threshold_otsu(img)) / 2.0)
    if method == "multiotsu":
        try:
            thresholds = threshold_multiotsu(img, classes=int(multiotsu_classes))
            idx = min(max(0, int(multiotsu_level)), len(thresholds) - 1)
            return float(thresholds[idx])
        except Exception:
            return float(threshold_otsu(img))
    raise ValueError(f"Unknown intensity threshold method '{method}'")


def masked_object_threshold(
    image,
    global_method="triangle",
    local_adjust=0.98,
    object_min_area_px=100,
):
    """Masked Object (MO) thresholding (infer-subc / aicssegmentation style).

    Two-pass thresholding that adapts to per-object intensity:
      1. Global threshold (``global_method``) to find candidate object regions.
      2. Drop regions smaller than ``object_min_area_px``.
      3. Re-threshold *within* each surviving object using a local Otsu scaled
         by ``local_adjust`` (<1 grows objects, >1 shrinks them).

    Robust for organelles with heterogeneous intensity across the field
    (ER, Golgi, cell mask) where a single global threshold clips dim objects.

    Returns a boolean mask.
    """
    from skimage.morphology import remove_small_objects
    from skimage.filters import threshold_otsu
    from scipy.ndimage import label as ndi_label, find_objects

    img = np.asarray(image)
    global_t = _scalar_intensity_threshold(img, global_method)
    bw_global = img > global_t
    if object_min_area_px and object_min_area_px > 0:
        bw_global = remove_small_objects(bw_global, min_size=int(object_min_area_px))

    if not bw_global.any():
        return np.zeros(img.shape, dtype=bool)

    labels, _ = ndi_label(bw_global)
    out = np.zeros(img.shape, dtype=bool)
    for i, sl in enumerate(find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == i
        local_img = img[sl]
        vals = local_img[region]
        if vals.size < 2:
            out[sl] |= region
            continue
        try:
            local_t = threshold_otsu(vals) * local_adjust
        except Exception:
            local_t = float(vals.min())
        out[sl] |= region & (local_img > local_t)
    return out


def apply_intensity_threshold(
    image,
    method="otsu",
    threshold_factor=1.0,
    mo_global_method="triangle",
    mo_local_adjust=0.98,
    mo_object_min_area_px=100,
    multiotsu_classes=3,
    multiotsu_level=0,
):
    """Threshold a raw intensity image to a boolean mask.

    Dispatcher for the "threshold" detection method. For scalar methods
    ("otsu", "triangle", "li", "li_log", "median", "ave", "multiotsu") the
    computed threshold is scaled by ``threshold_factor`` (e.g. infer-subc's
    lipid-droplet recipe uses Otsu * 0.8). For "multiotsu", ``multiotsu_classes``
    and ``multiotsu_level`` select the number of classes and which boundary is
    the foreground cutoff. For "masked_object" / "mo", the per-object MO mask is
    returned directly (``threshold_factor`` ignored).
    """
    method = (method or "otsu").lower()
    if method in ("masked_object", "mo"):
        return masked_object_threshold(
            image,
            global_method=mo_global_method,
            local_adjust=mo_local_adjust,
            object_min_area_px=mo_object_min_area_px,
        )
    t = _scalar_intensity_threshold(
        image, method,
        multiotsu_classes=multiotsu_classes, multiotsu_level=multiotsu_level,
    )
    return np.asarray(image) > (t * threshold_factor)
