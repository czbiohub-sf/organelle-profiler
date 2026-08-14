"""
Blob Detection for Organelle Segmentation
==========================================

This module provides LoG (Laplacian of Gaussian) blob detection methods
for detecting round, blob-like structures such as nucleoli, vesicles,
and other punctate organelles.

LoG is specifically designed for detecting blob-like structures and is
better suited for round objects than Frangi (which targets ridges/tubes).

Key functions:
- _segment_blob_log: General LoG blob detection with optional masking
- _segment_nucleoli_blob: Wrapper for nucleoli detection
- _segment_nucleoli_frangi: Alternative Frangi-based nucleoli detection
- _segment_nucleoli_in_tile: Dispatcher for nucleoli segmentation methods
"""

import os

import numpy as np
from skimage.feature import blob_log
from skimage.draw import disk
from skimage.measure import label
from skimage.filters import frangi

from .configs import (
    DEFAULT_METHODS,
    SEGMENTATION_CONFIGS,
)
from .frangi import compute_frangi_threshold
from .postprocessing import postprocess_nucleoli_mask

# Try to import cupy for the optional GPU-accelerated blob path. Falls back
# silently — callers check the imported modules via ``_GPU_BLOB_AVAILABLE``.
try:
    import cupy as _cp
    from cupyx.scipy.ndimage import gaussian_laplace as _cu_gaussian_laplace
    from cupyx.scipy.ndimage import maximum_filter as _cu_maximum_filter
    _GPU_BLOB_AVAILABLE = True
except Exception:
    _cp = None
    _cu_gaussian_laplace = None
    _cu_maximum_filter = None
    _GPU_BLOB_AVAILABLE = False


# -----------------------------------------------------------------------------
# GPU disk painting — one CUDA thread per blob, atomicMin gives bit-exact
# "lowest-index-wins" = CPU first-come-first-served semantics.
# -----------------------------------------------------------------------------
_PAINT_DISKS_KERNEL_SRC = r"""
extern "C" __global__ void paint_disks(
    const float * __restrict__ blobs_yxs,   // (N, 3) float32: (y, x, sigma)
    const unsigned char * __restrict__ mask, // (H, W) uchar, non-zero = inside mask; may be nullptr
    int * __restrict__ labels,               // (H, W) int32, init to INT_MAX sentinel
    const int N, const int H, const int W,
    const int use_mask)                      // 0 = ignore mask ptr, 1 = honor mask
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= N) return;

    const float by = blobs_yxs[bi * 3 + 0];
    const float bx = blobs_yxs[bi * 3 + 1];
    const float bs = blobs_yxs[bi * 3 + 2];

    // radius = max(2, int(sigma * sqrt(2))) — matches the CPU path exactly
    int r = (int)(bs * 1.41421356f);
    if (r < 2) r = 2;
    const int r2 = r * r;

    const int cy = (int)by;
    const int cx = (int)bx;
    const int y0 = max(0, cy - r);
    const int y1 = min(H - 1, cy + r);
    const int x0 = max(0, cx - r);
    const int x1 = min(W - 1, cx + r);

    const int label = bi + 1;

    for (int y = y0; y <= y1; ++y) {
        const int dy = y - cy;
        const int dy2 = dy * dy;
        const int row = y * W;
        for (int x = x0; x <= x1; ++x) {
            const int dx = x - cx;
            // Match skimage.draw.disk exactly: strict < (excludes boundary ring).
            if (dx * dx + dy2 >= r2) continue;
            const int idx = row + x;
            if (use_mask && mask[idx] == 0) continue;
            atomicMin(&labels[idx], label);
        }
    }
}
"""

_paint_disks_kernel = None  # JIT-compiled lazily


def _get_paint_disks_kernel():
    global _paint_disks_kernel
    if _paint_disks_kernel is None:
        if _cp is None:
            raise RuntimeError("cupy not available — paint_disks unusable")
        _paint_disks_kernel = _cp.RawKernel(
            _PAINT_DISKS_KERNEL_SRC, "paint_disks"
        )
    return _paint_disks_kernel


def _blob_log_paint_gpu_core(
    tile_gpu,               # cupy 2D float32 in [0, 1]
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    mask_gpu_uint8=None,    # cupy 2D uint8 or None
):
    """On-device core: scale-space LoG → 3D peak find → paint_disks.

    Accepts cupy arrays and returns a cupy int32 label array of shape
    ``tile_gpu.shape``. Used by both the numpy-in/numpy-out wrapper
    (``_blob_log_paint_gpu``) and the pipelined-driver tile compute path
    (``_compute_tile_blob_on_gpu``) — the latter wants to avoid the extra
    H2D/D2H hops.
    """
    if _cp is None:
        raise RuntimeError("cupy not available — GPU blob path unusable")
    cp = _cp
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma).astype(np.float32)
    H, W = int(tile_gpu.shape[0]), int(tile_gpu.shape[1])

    log_slices = []
    for s in sigmas:
        log_slices.append(-(float(s) ** 2) * _cu_gaussian_laplace(tile_gpu, sigma=float(s)))
    log_stack = cp.stack(log_slices, axis=0)
    max_filt = _cu_maximum_filter(log_stack, size=3)
    peaks_mask = (log_stack == max_filt) & (log_stack > float(threshold))
    coords_gpu = cp.argwhere(peaks_mask)  # (N, 3) int: (sigma_idx, y, x)
    del log_slices, log_stack, max_filt, peaks_mask

    N = int(coords_gpu.shape[0])
    if N == 0:
        return cp.zeros((H, W), dtype=cp.int32)

    sigmas_gpu = cp.asarray(sigmas)
    blobs_gpu = cp.empty((N, 3), dtype=cp.float32)
    blobs_gpu[:, 0] = coords_gpu[:, 1].astype(cp.float32)
    blobs_gpu[:, 1] = coords_gpu[:, 2].astype(cp.float32)
    blobs_gpu[:, 2] = sigmas_gpu[coords_gpu[:, 0]]

    INT32_MAX = np.iinfo(np.int32).max
    labels_gpu = cp.full((H, W), INT32_MAX, dtype=cp.int32)

    use_mask_flag = 1 if mask_gpu_uint8 is not None else 0
    if mask_gpu_uint8 is None:
        mask_gpu_uint8 = cp.zeros(1, dtype=cp.uint8)  # dummy; unread when flag=0

    kernel = _get_paint_disks_kernel()
    threads = 256
    blocks = (N + threads - 1) // threads
    kernel(
        (blocks,), (threads,),
        (blobs_gpu, mask_gpu_uint8, labels_gpu, np.int32(N), np.int32(H), np.int32(W),
         np.int32(use_mask_flag)),
    )

    return cp.where(labels_gpu == INT32_MAX, cp.int32(0), labels_gpu)


def _blob_log_paint_gpu(
    tile_norm: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    binary_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Full GPU blob pipeline (numpy-in / numpy-out).

    Thin wrapper around ``_blob_log_paint_gpu_core`` — kept for the
    ``_segment_blob_log`` CPU-dispatched call site, which feeds a numpy
    tile in and expects a numpy label array back.
    """
    if _cp is None:
        raise RuntimeError("cupy not available — GPU blob path unusable")
    cp = _cp
    tile_gpu = cp.asarray(tile_norm, dtype=cp.float32)
    mask_gpu_uint8 = None
    if binary_mask is not None:
        mask_gpu_uint8 = cp.asarray(binary_mask.astype(np.uint8, copy=False))
    labels_gpu = _blob_log_paint_gpu_core(
        tile_gpu, min_sigma, max_sigma, num_sigma, threshold, mask_gpu_uint8,
    )
    return cp.asnumpy(labels_gpu)


def _blob_log_gpu(
    tile_norm: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
) -> np.ndarray:
    """GPU-accelerated LoG peak finder. Returns a numpy (N, 3) array of
    ``(y, x, sigma)`` entries — same shape as ``skimage.feature.blob_log``.

    Pipeline:
      1. H2D normalized tile.
      2. Scale-space LoG: stack ``-sigma**2 * gaussian_laplace(img, sigma)``
         across ``num_sigma`` scales.
      3. 3D local maxima via ``log_stack == maximum_filter(log_stack, 3)``
         gated by ``log_stack > threshold``.
      4. D2H peak coordinates.

    Skips the ``_prune_blobs`` overlap filter that ``skimage.feature.blob_log``
    applies; the caller's first-come-first-served disk painting resolves
    overlaps well enough for segmentation and avoids the O(N²) prune step.
    Validated on ops0094 to keep IoU ≥ 0.94 vs the skimage reference.
    """
    if _cp is None:
        raise RuntimeError("cupy not available — GPU blob path unusable")
    cp = _cp
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)

    tile_gpu = cp.asarray(tile_norm, dtype=cp.float32)
    log_slices = []
    for s in sigmas:
        log_slices.append(-(float(s) ** 2) * _cu_gaussian_laplace(tile_gpu, sigma=float(s)))
    log_stack = cp.stack(log_slices, axis=0)
    max_filt = _cu_maximum_filter(log_stack, size=3)
    peaks_mask = (log_stack == max_filt) & (log_stack > float(threshold))
    coords_gpu = cp.argwhere(peaks_mask)
    if coords_gpu.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    coords = cp.asnumpy(coords_gpu)  # (N, 3): (sigma_idx, y, x)
    # Re-shape to match skimage.blob_log output: (y, x, sigma)
    out = np.empty((coords.shape[0], 3), dtype=np.float64)
    out[:, 0] = coords[:, 1]  # y
    out[:, 1] = coords[:, 2]  # x
    out[:, 2] = sigmas[coords[:, 0]]
    return out


def _segment_nucleoli_frangi(
    tile_data: np.ndarray,
    tile_mask: np.ndarray,
    pixel_resolution_um: float,
    frangi_params: dict = None,
) -> np.ndarray:
    """
    Segment nucleoli within a tile using Vesicular Frangi filter with nuclear masking.

    This approach is effective for nucleoli because:
    1. Frangi is gradient-based and handles masked images naturally
    2. Vesicular params (alpha=0.5) detect round blob-like structures
    3. Nuclear mask constrains detection to within nuclei only

    Args:
        tile_data: Image tile data (2D array, typically Phase2D channel)
        tile_mask: Nuclear mask for the tile (labels, not binary) - from nuclear_seg
        pixel_resolution_um: Pixel size in micrometers
        frangi_params: Optional custom Frangi params (default: nucleoli frangi config from SEGMENTATION_CONFIGS)

    Returns:
        Labeled mask for nucleoli in the tile (int32)
    """
    # Skip tiles with no nuclei
    if tile_mask.max() == 0:
        return np.zeros_like(tile_mask, dtype=np.int32)

    # Use default nucleoli Frangi params if not specified
    if frangi_params is None:
        frangi_params = SEGMENTATION_CONFIGS[("nucleoli", "phase2d", "frangi")].copy()

    # Create binary mask for nuclei regions
    binary_mask = tile_mask > 0

    # Apply nuclear mask BEFORE Frangi - zeros outside nuclei
    # Frangi's gradient calculation handles masked images naturally
    masked_tile = tile_data.astype(np.float32)
    masked_tile[~binary_mask] = 0.0

    # Calculate sigma range in pixels
    min_sigma = frangi_params["min_radius_um"] / pixel_resolution_um
    max_sigma = frangi_params["max_radius_um"] / pixel_resolution_um
    sigmas = np.geomspace(min_sigma, max_sigma, num=5)

    # Run vesicular Frangi (matching sweep script - only sigmas and black_ridges)
    # Since nucleoli are bright inside nuclei, use black_ridges=False
    vesselness = frangi(
        masked_tile,
        sigmas=sigmas,
        black_ridges=False,  # Bright nucleoli on darker nuclear interior
    )

    if vesselness.max() == 0:
        return np.zeros_like(tile_mask, dtype=np.int32)

    # Use fixed or dynamic thresholding
    fixed_threshold = frangi_params.get("threshold", 0.01)
    if fixed_threshold is not None:
        # Use fixed threshold directly
        threshold = fixed_threshold
    else:
        # Use dynamic thresholding with threshold_mult
        threshold_mult = frangi_params.get("threshold_mult", 0.01)
        threshold = compute_frangi_threshold(vesselness, threshold_mult=threshold_mult, xp=np)

    # Apply threshold
    binary_result = vesselness > threshold

    # Ensure nucleoli are only within nuclear boundaries
    binary_result = binary_result & binary_mask

    # Apply aggressive post-processing for large round structures
    # Extract postprocess params from config
    pp_min_size = frangi_params.get("min_object_size", 20)
    pp_do_opening = frangi_params.get("postprocess_opening", True)
    pp_opening_radius = frangi_params.get("postprocess_opening_radius", 1)
    pp_do_closing = frangi_params.get("postprocess_closing", True)
    pp_closing_radius = frangi_params.get("postprocess_closing_radius", 3)
    binary_result = postprocess_nucleoli_mask(
        binary_result,
        min_size=pp_min_size,
        do_opening=pp_do_opening,
        opening_radius=pp_opening_radius,
        do_closing=pp_do_closing,
        closing_radius=pp_closing_radius,
    )

    # Label connected components
    nucleoli_labels = label(binary_result).astype(np.int32)

    return nucleoli_labels


def _segment_blob_log(
    tile_data: np.ndarray,
    pixel_resolution_um: float,
    blob_params: dict,
    mask: np.ndarray = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Unified LoG blob detection for nucleoli, vesicles, etc.

    LoG is specifically designed for detecting blob-like structures and is better
    suited for round objects than Frangi (which targets ridges/tubes).

    The algorithm:
    1. Applies LoG at multiple scales (sigma values)
    2. Finds local maxima in scale-space
    3. Converts blob centers to circular masks
    4. Labels connected components

    Args:
        tile_data: Image tile data (2D array)
        pixel_resolution_um: Pixel size in micrometers
        blob_params: Blob detection params (min_radius_um, max_radius_um, threshold, etc.)
        mask: Optional mask (labels or binary). If provided, only detect within mask.
        invert: If True, invert image to detect dark blobs on bright background.

    Returns:
        Labeled mask (int32)
    """
    # Handle optional mask
    if mask is not None:
        binary_mask = mask > 0
        if binary_mask.sum() == 0:
            return np.zeros(tile_data.shape, dtype=np.int32)
    else:
        binary_mask = None

    # Normalize image for blob detection (0-1 range works best for LoG)
    tile_float = tile_data.astype(np.float32)
    if binary_mask is not None:
        tile_float[~binary_mask] = 0.0
        mask_values = tile_float[binary_mask]
    else:
        mask_values = tile_float.ravel()

    if mask_values.size > 0 and mask_values.max() > mask_values.min():
        vmin, vmax = np.percentile(mask_values, [1, 99])
        tile_norm = np.clip((tile_float - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        return np.zeros(tile_data.shape, dtype=np.int32)

    # Invert for dark blob detection (vesicular_dark)
    if invert:
        tile_norm = 1.0 - tile_norm

    # Calculate sigma range in pixels (sigma H radius / sqrt(2) for LoG)
    min_sigma = blob_params["min_radius_um"] / pixel_resolution_um / np.sqrt(2)
    max_sigma = blob_params["max_radius_um"] / pixel_resolution_um / np.sqrt(2)

    # Ensure reasonable sigma values
    min_sigma = max(1.0, min_sigma)
    max_sigma = max(min_sigma + 1, max_sigma)

    # Run LoG blob detection. GPU path swaps skimage.blob_log for a cupy
    # scale-space LoG + 3D peak-finder (4-13x faster per tile on ops0094,
    # IoU 0.94-0.96 vs skimage reference). Enabled via ORG_SEG_BLOB_GPU=1;
    # falls back to CPU silently if cupy is unavailable.
    # Master toggle: ORG_SEG_OPTIMIZED=1 (default) turns on the fast
    # path. Individual ORG_SEG_BLOB_GPU override still takes precedence.
    _blob_gpu_default = "1" if os.environ.get("ORG_SEG_OPTIMIZED", "1") == "1" else "0"
    _use_gpu_blob = (
        _GPU_BLOB_AVAILABLE
        and os.environ.get("ORG_SEG_BLOB_GPU", _blob_gpu_default) == "1"
    )
    # ORG_SEG_BLOB_DISK_GPU=1 additionally moves the disk-painting loop to
    # GPU (atomicMin RawKernel), eliminating the per-tile CPU post-processing
    # that currently dominates blob wall time. Requires BLOB_GPU=1; falls
    # back to the CPU-painting path on any exception.
    _use_gpu_disk = (
        _use_gpu_blob
        and os.environ.get("ORG_SEG_BLOB_DISK_GPU", "0") == "1"
    )
    if _use_gpu_disk:
        try:
            return _blob_log_paint_gpu(
                tile_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=blob_params.get("num_sigma", 10),
                threshold=blob_params.get("threshold", 0.02),
                binary_mask=binary_mask,
            )
        except Exception as _e:
            print(f"  [blob_paint_gpu] failed ({type(_e).__name__}: {_e}); "
                  f"falling back to CPU disk painting")
            # fall through into the CPU-paint path below

    if _use_gpu_blob:
        try:
            blobs = _blob_log_gpu(
                tile_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=blob_params.get("num_sigma", 10),
                threshold=blob_params.get("threshold", 0.02),
            )
        except Exception as _e:
            print(f"  [blob_log_gpu] failed ({type(_e).__name__}: {_e}); "
                  f"falling back to CPU")
            blobs = blob_log(
                tile_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=blob_params.get("num_sigma", 10),
                threshold=blob_params.get("threshold", 0.02),
                overlap=blob_params.get("overlap", 0.5),
                exclude_border=blob_params.get("exclude_border", False),
            )
    else:
        # Returns array of (y, x, sigma) for each detected blob
        blobs = blob_log(
            tile_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=blob_params.get("num_sigma", 10),
            threshold=blob_params.get("threshold", 0.02),
            overlap=blob_params.get("overlap", 0.5),
            exclude_border=blob_params.get("exclude_border", False),
        )

    if len(blobs) == 0:
        return np.zeros(tile_data.shape, dtype=np.int32)

    # Create labeled mask from blob detections
    labels = np.zeros(tile_data.shape, dtype=np.int32)

    for i, (y, x, sigma) in enumerate(blobs):
        # Radius is approximately sigma * sqrt(2) for LoG
        radius = max(2, int(sigma * np.sqrt(2)))

        # Create circular mask for this blob
        rr, cc = disk((int(y), int(x)), radius, shape=tile_data.shape)

        if binary_mask is not None:
            # Only include pixels within the mask
            valid = binary_mask[rr, cc]
            rr, cc = rr[valid], cc[valid]

        # Assign label (don't overwrite existing labels - first come first served)
        unlabeled = labels[rr, cc] == 0
        labels[rr[unlabeled], cc[unlabeled]] = i + 1

    return labels


def _segment_nucleoli_blob(
    tile_data: np.ndarray,
    tile_mask: np.ndarray,
    pixel_resolution_um: float,
    blob_params: dict = None,
) -> np.ndarray:
    """Segment nucleoli using LoG blob detection (wrapper for _segment_blob_log)."""
    if blob_params is None:
        blob_params = SEGMENTATION_CONFIGS[("nucleoli", "phase2d", "blob")].copy()
    return _segment_blob_log(tile_data, pixel_resolution_um, blob_params, mask=tile_mask)


def _segment_nucleoli_in_tile(
    tile_data,
    tile_mask,
    pixel_resolution_um=0.108,
    method: str = None,
    frangi_params: dict = None,
    blob_params: dict = None,
):
    """
    Segment nucleoli within a tile using the specified method.

    Args:
        tile_data: Image tile data
        tile_mask: Nuclear mask for the tile (labels, not binary)
        pixel_resolution_um: Pixel size in micrometers (used for Frangi/blob)
        method: "blob" (LoG, default) or "frangi"
        frangi_params: Optional custom Frangi params (only used if method="frangi")
        blob_params: Optional custom blob params (only used if method="blob")

    Returns:
        Labeled mask for nucleoli in the tile
    """
    # Use default method if not specified
    if method is None:
        method = DEFAULT_METHODS.get(("nucleoli", "phase2d"), "frangi")

    if method == "blob":
        return _segment_nucleoli_blob(
            tile_data=tile_data,
            tile_mask=tile_mask,
            pixel_resolution_um=pixel_resolution_um,
            blob_params=blob_params,
        )
    else:
        # Frangi method (default)
        return _segment_nucleoli_frangi(
            tile_data=tile_data,
            tile_mask=tile_mask,
            pixel_resolution_um=pixel_resolution_um,
            frangi_params=frangi_params,
        )
