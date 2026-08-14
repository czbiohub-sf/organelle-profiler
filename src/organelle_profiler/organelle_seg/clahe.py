"""
CLAHE Preprocessing for Organelle Segmentation
===============================================

This module provides memory-efficient tiled CLAHE (Contrast Limited Adaptive
Histogram Equalization) for large microscopy images.

The tiled approach:
- Processes images in overlapping tiles to limit memory usage
- Uses linear blending in overlap regions to avoid visible tile boundaries
- Supports debug output for quality verification
"""

import numpy as np
from pathlib import Path
from skimage.exposure import equalize_adapthist

from .visualizations import save_debug_jpeg


def _tiled_clahe(
    image: np.ndarray,
    clip_limit: float = 0.03,
    tile_size: int = 8192,
    overlap: int = 512,
    kernel_size: tuple = None,
    debug_output_path: str = None,
) -> np.ndarray:
    """
    Memory-efficient tiled CLAHE for large images.

    Processes the image in overlapping tiles to avoid loading the entire image
    into memory at once for CLAHE. Uses linear blending in overlap regions to
    avoid visible tile boundaries.

    Args:
        image: Input 2D image (uint16 or will be converted)
        clip_limit: CLAHE clip limit (default: 0.03)
        tile_size: Size of each tile to process (default: 8192 pixels)
        overlap: Overlap between adjacent tiles for blending (default: 512 pixels)
        kernel_size: Kernel size for CLAHE within each tile (default: None, uses skimage default)
        debug_output_path: Optional path to save debug JPEG images showing:
            - First few tiles before/after CLAHE
            - Blending regions between adjacent tiles
            - Final blended result sample

    Returns:
        CLAHE-enhanced image as float32 in [0, 1] range (same as equalize_adapthist but float32)

    Memory usage:
        - Processes one tile at a time: ~(tile_size + 2*overlap)^2 * 4 bytes per tile
        - For 8192 tile + 512 overlap: ~(9216)^2 * 4 = ~340 MB per tile operation
        - Output array: ~44 GB for 105k x 105k (float32)
        - Much more efficient than loading full 105k x 105k image (~85 GB for uint16)
    """
    if image.ndim != 2:
        raise ValueError(f"_tiled_clahe expects 2D image, got shape {image.shape}")

    height, width = image.shape

    # Setup debug output directory
    debug_dir = None
    if debug_output_path:
        debug_dir = Path(debug_output_path)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"  DEBUG: Saving CLAHE intermediates to {debug_dir}")

    # Normalize to [0, 1] float32 BEFORE CLAHE (matching sweep script exactly)
    # This is critical - equalize_adapthist behaves differently with uint16 vs float input
    image = image.astype(np.float32)
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    # For small images, just use regular CLAHE
    if height <= tile_size * 1.5 and width <= tile_size * 1.5:
        result = equalize_adapthist(image, clip_limit=clip_limit, kernel_size=kernel_size).astype(np.float32)
        return result

    # Create output array (float32 [0,1] - saves 50% memory vs float64)
    output = np.zeros((height, width), dtype=np.float32)
    # Weight accumulator for blending
    weights = np.zeros((height, width), dtype=np.float32)

    # Calculate tile grid
    step = tile_size - overlap  # Step size between tiles (with overlap)
    n_tiles_y = max(1, int(np.ceil((height - overlap) / step)))
    n_tiles_x = max(1, int(np.ceil((width - overlap) / step)))

    total_tiles = n_tiles_y * n_tiles_x
    print(f"  Processing CLAHE in {n_tiles_y} x {n_tiles_x} = {total_tiles} tiles "
          f"(tile_size={tile_size}, overlap={overlap})...")

    # Track tiles for debug blending visualization
    debug_tiles_saved = 0
    debug_max_tiles = 4  # Save first 4 tiles for debugging

    tile_idx = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_idx += 1

            # Calculate tile boundaries
            y_start = ty * step
            x_start = tx * step
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Extend to include overlap on the edges (except image boundaries)
            y_start_ext = max(0, y_start - overlap // 2)
            x_start_ext = max(0, x_start - overlap // 2)
            y_end_ext = min(height, y_end + overlap // 2)
            x_end_ext = min(width, x_end + overlap // 2)

            # Extract tile (extended region for better CLAHE context)
            tile = image[y_start_ext:y_end_ext, x_start_ext:x_end_ext]

            # Apply CLAHE to tile (returns float64, convert to float32 to save memory)
            tile_clahe = equalize_adapthist(tile, clip_limit=clip_limit, kernel_size=kernel_size).astype(np.float32)

            # Debug: Save first few tiles before and after CLAHE
            if debug_dir and debug_tiles_saved < debug_max_tiles:
                # Downsample for reasonable file size (max 2048x2048)
                max_debug_size = 2048
                scale = min(1.0, max_debug_size / max(tile.shape))
                if scale < 1.0:
                    from scipy.ndimage import zoom
                    tile_small = zoom(tile, scale, order=1)
                    clahe_small = zoom(tile_clahe, scale, order=1)
                else:
                    tile_small = tile
                    clahe_small = tile_clahe

                save_debug_jpeg(
                    tile_small,
                    debug_dir / f"tile_{tile_idx:03d}_ty{ty}_tx{tx}_before_clahe.jpg",
                    mode="normalize",
                )
                save_debug_jpeg(
                    clahe_small,
                    debug_dir / f"tile_{tile_idx:03d}_ty{ty}_tx{tx}_after_clahe.jpg",
                    mode="clip",  # Already in [0,1]
                )
                debug_tiles_saved += 1

            # Create weight mask for blending (linear ramp at edges)
            tile_h, tile_w = tile_clahe.shape
            weight_y = np.ones(tile_h, dtype=np.float32)
            weight_x = np.ones(tile_w, dtype=np.float32)

            # Linear ramp for blending in overlap regions
            blend_size = overlap // 2

            # Y-direction blending (top and bottom edges)
            if y_start_ext > 0:  # Not at top edge
                ramp_len = min(blend_size, tile_h // 2)
                weight_y[:ramp_len] = np.linspace(0, 1, ramp_len)
            if y_end_ext < height:  # Not at bottom edge
                ramp_len = min(blend_size, tile_h // 2)
                weight_y[-ramp_len:] = np.linspace(1, 0, ramp_len)

            # X-direction blending (left and right edges)
            if x_start_ext > 0:  # Not at left edge
                ramp_len = min(blend_size, tile_w // 2)
                weight_x[:ramp_len] = np.linspace(0, 1, ramp_len)
            if x_end_ext < width:  # Not at right edge
                ramp_len = min(blend_size, tile_w // 2)
                weight_x[-ramp_len:] = np.linspace(1, 0, ramp_len)

            # 2D weight mask
            weight_mask = np.outer(weight_y, weight_x)

            # Accumulate weighted result
            output[y_start_ext:y_end_ext, x_start_ext:x_end_ext] += tile_clahe * weight_mask
            weights[y_start_ext:y_end_ext, x_start_ext:x_end_ext] += weight_mask

            if tile_idx % 10 == 0 or tile_idx == total_tiles:
                print(f"    CLAHE tile {tile_idx}/{total_tiles} done", end="\r")

    print()  # New line after progress

    # Normalize by total weights
    # Avoid division by zero (shouldn't happen if tiles cover the image)
    weights = np.maximum(weights, 1e-10)
    output /= weights

    # Debug: Save blending region samples and final result
    if debug_dir:
        from scipy.ndimage import zoom

        # Save a sample of the blending region (center of image, around tile boundaries)
        # Find a tile boundary region (where two tiles meet)
        blend_region_size = 2048  # Size of debug region to save
        center_y, center_x = height // 2, width // 2

        # Extract region around center (likely to contain tile boundaries)
        y1 = max(0, center_y - blend_region_size // 2)
        y2 = min(height, center_y + blend_region_size // 2)
        x1 = max(0, center_x - blend_region_size // 2)
        x2 = min(width, center_x + blend_region_size // 2)

        blend_region_output = output[y1:y2, x1:x2]
        blend_region_input = image[y1:y2, x1:x2]
        blend_region_weights = weights[y1:y2, x1:x2]

        save_debug_jpeg(
            blend_region_input,
            debug_dir / "blend_region_input.jpg",
            mode="normalize",
        )
        save_debug_jpeg(
            blend_region_output,
            debug_dir / "blend_region_output_clahe.jpg",
            mode="clip",
        )
        save_debug_jpeg(
            blend_region_weights,
            debug_dir / "blend_region_weights.jpg",
            mode="normalize",
        )

        # Save a low-res overview of the entire output
        overview_max_size = 4096
        overview_scale = min(1.0, overview_max_size / max(height, width))
        if overview_scale < 1.0:
            output_overview = zoom(output, overview_scale, order=1)
        else:
            output_overview = output
        save_debug_jpeg(
            output_overview,
            debug_dir / "full_output_overview.jpg",
            mode="clip",
        )

    return output
