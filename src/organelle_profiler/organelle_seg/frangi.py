"""
Frangi Vesselness Filter for Organelle Segmentation
====================================================

This module provides the FrangiFilter class for detecting tubular and
vesicular structures in microscopy images using the Frangi vesselness
filter based on Hessian eigenvalue analysis.

The filter can operate on both 2D and 3D images and supports GPU
acceleration via CuPy.

Key features:
- Multi-scale vesselness detection with configurable sigma range
- GPU acceleration for large images
- Adaptive gamma calculation for automatic thresholding
- Support for both dark ridges (mitochondria) and bright ridges (vesicles)
"""

import sys
import numpy as np
import scipy.ndimage as scipy_ndi
from itertools import combinations_with_replacement

from organelle_profiler.organelle_seg.thresholding import (
    otsu_threshold,
    triangle_threshold,
)
from .geometry import get_bbox


def compute_frangi_threshold(
    vesselness_map: np.ndarray,
    threshold_mult: float = 0.1,
    xp=None,
) -> float:
    """
    Compute dynamic threshold for Frangi vesselness map using min(triangle, otsu).

    This implements the standard Frangi thresholding approach used throughout the
    organelle segmentation pipeline:
    1. Take log10 of positive vesselness values
    2. Compute triangle and otsu thresholds on log scale
    3. Convert back to linear scale
    4. Return threshold_mult * min(triangle, otsu)

    Args:
        vesselness_map: Frangi vesselness output (2D or 3D array)
        threshold_mult: Multiplier for the threshold (default: 0.1)
        xp: Array module (numpy or cupy). If None, uses numpy.

    Returns:
        Computed threshold value in linear scale

    Examples:
        >>> threshold = compute_frangi_threshold(vesselness_map)
        >>> binary_mask = vesselness_map > threshold

        >>> # With custom multiplier
        >>> threshold = compute_frangi_threshold(vesselness_map, threshold_mult=0.001)
    """
    if xp is None:
        xp = np

    # CRITICAL: Only compute threshold on POSITIVE vesselness values (matches org_seg_old2.py)
    # Including zeros/negatives drastically lowers the threshold, causing weak detections
    positive_vesselness = vesselness_map[vesselness_map > 0]

    if len(positive_vesselness) == 0:
        return 0.0

    log_vesselness = xp.log10(positive_vesselness)

    # Compute thresholds on log scale
    tri_thresh_log = triangle_threshold(log_vesselness, xp=xp)
    otsu_thresh_log, _ = otsu_threshold(log_vesselness, xp=xp)

    # Convert back to linear scale
    triangle_thresh_linear = 10**tri_thresh_log
    otsu_thresh_linear = 10**otsu_thresh_log

    # Use min of the two thresholds, scaled by multiplier
    threshold = threshold_mult * min(float(triangle_thresh_linear), float(otsu_thresh_linear))

    return threshold


class FrangiFilter:
    """
    A class that applies the Frangi vesselness filter to 3D image data.

    Parameters
    ----------
    image_data : array
        Input image (2D or 3D)
    pixel_resolution : dict
        Dict with "X", "Y", "Z" keys for pixel size in um
    min_radius_um : float
        Minimum structure radius in micrometers
    max_radius_um : float
        Maximum structure radius in micrometers
    alpha : float
        Frangi filter sensitivity to elongation (higher = more tubular)
    beta : float
        Frangi filter sensitivity to blob-like structures (lower = more blobs)
    gamma : float or None
        Sensitivity to overall structure. If None, calculated dynamically.
        Lower values = more sensitive to faint structures.
    black_ridges : bool
        If True, detect dark ridges on light background.
        If False, detect bright ridges on dark background (for vesicles).
    num_sigma : int
        Number of sigma values to sample (more = better scale coverage but slower)
    verbose : bool
        Print debug info
    use_gpu : bool
        Use GPU acceleration via CuPy
    remove_edges : bool
        Remove edge artifacts from mask
    """

    def __init__(
        self,
        image_data,
        pixel_resolution: dict,
        min_radius_um=0.2,
        max_radius_um=1.5,
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=True,
        num_sigma=5,
        verbose=False,
        use_gpu=True,
        remove_edges=True,
    ):

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.xp = sys.modules["cupy"]
            self.ndi = sys.modules["cupyx.scipy.ndimage"]
        else:
            self.xp = np
            self.ndi = scipy_ndi

        self.image_data = image_data
        self.pixel_resolution = pixel_resolution
        self.verbose = verbose
        self.is_2d = self.image_data.ndim == 2

        if not self.is_2d:
            self.z_ratio = self.pixel_resolution.get(
                "Z", 1.0
            ) / self.pixel_resolution.get("X", 1.0)
        else:
            self.z_ratio = 1.0

        # No clamping - use the radius values directly as specified
        self.min_radius_um = min_radius_um
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.pixel_resolution.get("X", 1.0)
        self.max_radius_px = self.max_radius_um / self.pixel_resolution.get("X", 1.0)

        self.alpha_sq = alpha**2
        self.beta_sq = beta**2
        self.gamma_override = gamma  # User-specified gamma (None = dynamic)
        self.black_ridges = black_ridges
        self.num_sigma = num_sigma
        self.sigmas = self._set_default_sigmas()
        self.remove_edges = remove_edges

    def _get_frob_mask(self, hessian_elements, frob_thresh=None):
        """Creates a Frobenius norm mask for the Hessian matrix based on a threshold."""
        max_h = self.xp.max(self.xp.abs(hessian_elements))
        if max_h == 0:
            return self.xp.zeros(hessian_elements.shape[1:], dtype="bool")
        rescaled_hessian = hessian_elements / max_h
        frobenius_norm = self.xp.linalg.norm(rescaled_hessian, axis=0)
        if self.xp.any(self.xp.isinf(frobenius_norm)):
            non_inf_max = self.xp.max(frobenius_norm[~self.xp.isinf(frobenius_norm)])
            frobenius_norm[self.xp.isinf(frobenius_norm)] = non_inf_max
        if frob_thresh is None:
            non_zero_frobenius = frobenius_norm[frobenius_norm > 0]
            if len(non_zero_frobenius) == 0:
                return self.xp.zeros_like(frobenius_norm, dtype="bool")
            frob_otsu_thresh, _ = otsu_threshold(non_zero_frobenius, xp=self.xp)
            frob_triangle_thresh = triangle_threshold(non_zero_frobenius, xp=self.xp)
            frobenius_threshold = min(frob_otsu_thresh, frob_triangle_thresh)
        else:
            frobenius_threshold = frob_thresh
        return frobenius_norm > frobenius_threshold

    def _remove_edges_from_mask(self, mask):
        """Removes a 15-pixel border from the top/bottom of the mask's bounding box."""
        if not self.xp.any(mask):
            return mask

        if self.is_2d:
            ymin, ymax, _, _ = get_bbox(mask, self.xp)
            if ymin is not None:
                mask[ymin : min(ymin + 15, ymax), :] = 0
                mask[max(ymax - 15, ymin) : ymax + 1, :] = 0
        else:  # 3D
            for z_idx in range(mask.shape[0]):
                z_slice = mask[z_idx]
                if not self.xp.any(z_slice):
                    continue
                ymin, ymax, _, _ = get_bbox(z_slice, self.xp)
                if ymin is not None:
                    mask[z_idx, ymin : min(ymin + 15, ymax), :] = 0
                    mask[z_idx, max(ymax - 15, ymin) : ymax + 1, :] = 0
        return mask

    def _calculate_gamma(self, gauss_volume):
        """Calculates gamma value for vesselness thresholding."""
        if not self.xp.any(gauss_volume > 0):
            return 1.0
        non_zero_gauss = gauss_volume[gauss_volume > 0]
        gamma_otsu, _ = otsu_threshold(non_zero_gauss, xp=self.xp)
        gamma_tri = triangle_threshold(non_zero_gauss, xp=self.xp)
        return min(gamma_otsu, gamma_tri)

    def _set_default_sigmas(self):
        """Sets the default sigma values for the Frangi filter.

        Uses radius directly as sigma (no division), matching the sweep script behavior.
        This means min_radius_um and max_radius_um in the config directly control
        the sigma range in physical units.
        """
        # Use radius directly as sigma (matches sweep script um_to_sigmas)
        sigma_min = self.min_radius_px
        sigma_max = self.max_radius_px

        # Use geomspace for better coverage of scale space (logarithmic spacing)
        # This is better than linear spacing for detecting structures of varying sizes
        if sigma_max > sigma_min and self.num_sigma > 1:
            sigmas = list(np.geomspace(sigma_min, sigma_max, num=self.num_sigma))
        else:
            sigmas = [sigma_min]

        if self.verbose:
            print(f"Using {len(sigmas)} sigmas for Frangi filter: {sigmas}")
        return sigmas

    def _get_sigma_vec(self, sigma):
        """Generates the sigma vector for Gaussian filtering."""
        if self.is_2d:
            return (sigma, sigma)
        return (sigma / self.z_ratio, sigma, sigma)

    def _compute_hessian(self, image):
        """Computes the Hessian matrix of the input image."""
        gradients = self.xp.gradient(image)
        axes = range(image.ndim)
        h_elems = self.xp.array(
            [
                self.xp.gradient(gradients[ax0], axis=ax1).astype("float16")
                for ax0, ax1 in combinations_with_replacement(axes, 2)
            ]
        )

        h_mask = self._get_frob_mask(h_elems)
        if self.remove_edges:
            h_mask = self._remove_edges_from_mask(h_mask)

        if self.use_gpu:
            masked_h_elems_cpu = h_elems[:, h_mask].get()
        else:
            masked_h_elems_cpu = h_elems[:, h_mask]

        if self.is_2d:
            hxx, hxy, hyy = [
                elem[..., np.newaxis, np.newaxis] for elem in masked_h_elems_cpu
            ]
            hessian_matrices = np.concatenate(
                [
                    np.concatenate([hxx, hxy], axis=-1),
                    np.concatenate([hxy, hyy], axis=-1),
                ],
                axis=-2,
            )
        else:
            hxx, hxy, hxz, hyy, hyz, hzz = [
                elem[..., np.newaxis, np.newaxis] for elem in masked_h_elems_cpu
            ]
            hessian_matrices = np.concatenate(
                [
                    np.concatenate([hxx, hxy, hxz], axis=-1),
                    np.concatenate([hxy, hyy, hyz], axis=-1),
                    np.concatenate([hxz, hyz, hzz], axis=-1),
                ],
                axis=-2,
            )

        return h_mask, hessian_matrices

    def _compute_chunkwise_eigenvalues(self, hessian_matrices, chunk_size=1e5):
        """Computes eigenvalues of the Hessian matrix in chunks."""
        chunk_size = int(chunk_size)
        total_voxels = len(hessian_matrices)
        eigenvalues_list = []

        for start_idx in range(0, total_voxels, chunk_size):
            end_idx = min(start_idx + chunk_size, total_voxels)

            if self.use_gpu:
                cpu_chunk = hessian_matrices[start_idx:end_idx]
                gpu_chunk = self.xp.array(cpu_chunk)
            else:
                # If on CPU, the "chunk" is just a numpy array slice
                gpu_chunk = hessian_matrices[start_idx:end_idx]

            try:
                chunk_eigenvalues = self.xp.linalg.eigvalsh(gpu_chunk)
                eigenvalues_list.append(chunk_eigenvalues)
            except Exception as e:
                print(
                    f"Warning: A CUDA error of type {type(e).__name__} occurred during eigenvalue calculation: {e}. "
                    f"This often manifests as a CUSOLVER_STATUS_INTERNAL_ERROR. "
                    f"Skipping a chunk of {len(gpu_chunk)} pixels."
                )
                eigenvalues_list.append(
                    self.xp.zeros(
                        (len(gpu_chunk), gpu_chunk.shape[1]), dtype=gpu_chunk.dtype
                    )
                )

        eigenvalues_flat = self.xp.concatenate(eigenvalues_list, axis=0)
        sort_order = self.xp.argsort(self.xp.abs(eigenvalues_flat), axis=1)
        eigenvalues_flat = self.xp.take_along_axis(eigenvalues_flat, sort_order, axis=1)

        return eigenvalues_flat

    def _filter_hessian(self, eigenvalues, gamma_sq):
        """Applies the Frangi filter to the Hessian eigenvalues.

        The eigenvalue sign filtering depends on black_ridges:
        - black_ridges=True: detect dark ridges (eigenvalues > 0 are background)
        - black_ridges=False: detect bright ridges (eigenvalues < 0 are background)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.is_2d:
                rb_sq = (
                    self.xp.abs(eigenvalues[:, 0]) / self.xp.abs(eigenvalues[:, 1])
                ) ** 2
                s_sq = (eigenvalues[:, 0] ** 2) + (eigenvalues[:, 1] ** 2)
                filtered_im = (self.xp.exp(-rb_sq / self.beta_sq)) * (
                    1 - self.xp.exp(-s_sq / gamma_sq)
                )
            else:
                ra_sq = (
                    self.xp.abs(eigenvalues[:, 1]) / self.xp.abs(eigenvalues[:, 2])
                ) ** 2
                rb_sq = (
                    self.xp.abs(eigenvalues[:, 1])
                    / self.xp.sqrt(self.xp.abs(eigenvalues[:, 1] * eigenvalues[:, 2]))
                ) ** 2
                s_sq = (
                    self.xp.sqrt(
                        (eigenvalues[:, 0] ** 2)
                        + (eigenvalues[:, 1] ** 2)
                        + (eigenvalues[:, 2] ** 2)
                    )
                ) ** 2
                filtered_im = (
                    (1 - self.xp.exp(-ra_sq / self.alpha_sq))
                    * (self.xp.exp(-rb_sq / self.beta_sq))
                    * (1 - self.xp.exp(-s_sq / gamma_sq))
                )

        # Apply eigenvalue sign filtering based on black_ridges
        # For black_ridges=True (dark structures): eigenvalues > 0 are zeroed
        # For black_ridges=False (bright structures): eigenvalues < 0 are zeroed
        if self.black_ridges:
            if not self.is_2d:
                filtered_im[eigenvalues[:, 1] > 0] = 0
                filtered_im[eigenvalues[:, 2] > 0] = 0
            else:
                filtered_im[eigenvalues[:, 1] > 0] = 0
        else:
            # Bright ridges: filter out negative eigenvalues
            if not self.is_2d:
                filtered_im[eigenvalues[:, 1] < 0] = 0
                filtered_im[eigenvalues[:, 2] < 0] = 0
            else:
                filtered_im[eigenvalues[:, 1] < 0] = 0

        return self.xp.nan_to_num(filtered_im, nan=1.0)

    def run(self):
        """Main method to execute the Frangi filter process."""
        vesselness = self.xp.zeros_like(self.image_data, dtype="float64")
        temp_vesselness_img = self.xp.zeros_like(self.image_data, dtype="float64")
        final_mask = self.xp.ones_like(self.image_data, dtype="bool")

        for sigma in self.sigmas:
            sigma_vec = self._get_sigma_vec(sigma)
            gauss_volume = self.ndi.gaussian_filter(
                self.image_data, sigma=sigma_vec, mode="reflect", cval=0.0, truncate=3
            ).astype("double")

            # Use user-specified gamma if provided, otherwise calculate dynamically
            if self.gamma_override is not None:
                gamma = self.gamma_override
            else:
                gamma = self._calculate_gamma(gauss_volume)
            gamma_sq = 2 * (gamma**2)

            h_mask, hessian_matrices = self._compute_hessian(gauss_volume)
            if len(hessian_matrices) == 0:
                continue

            final_mask = self.xp.where(~h_mask, False, final_mask)

            eigenvalues = self._compute_chunkwise_eigenvalues(
                hessian_matrices.astype("float")
            )

            temp_vesselness_scores = self._filter_hessian(eigenvalues, gamma_sq)

            temp_vesselness_img.fill(0)
            temp_vesselness_img[h_mask] = temp_vesselness_scores

            max_indices = temp_vesselness_img > vesselness
            vesselness[max_indices] = temp_vesselness_img[max_indices]

        vesselness *= final_mask
        return vesselness
