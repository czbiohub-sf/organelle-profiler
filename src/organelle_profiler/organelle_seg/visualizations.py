"""
Visualization Utilities for Organelle Segmentation
===================================================

This module provides debug visualization utilities for organelle segmentation,
including saving debug images, overlays, and combined canvas displays.

Functions:
- save_debug_image: Save array as PNG with various normalization modes
- save_debug_jpeg: Alias for save_debug_image (backwards compatibility)
- save_debug_overlay: Save RGB overlay of mask on raw image
- create_combined_canvas: Create matplotlib grid of results
- _save_tiled_debug_images: Save debug images for tiled segmentation
- save_segmentation_params_yaml: Save segmentation params to YAML file
"""

import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from datetime import datetime


def save_debug_image(
    arr: np.ndarray,
    path,
    mode: str = "normalize",
    verbose: bool = True,
    percentile_clip: tuple = None,
) -> None:
    """
    Save array as PNG for debug visualization at full resolution.

    This is a centralized helper for saving debug images throughout the
    organelle segmentation pipeline. Uses PNG format for lossless full-resolution output.

    Args:
        arr: 2D numpy array to save
        path: Output path (str or Path) - will be saved as PNG regardless of extension
        mode: How to convert array to uint8:
            - "normalize": Min-max normalize to 0-255 (default, for continuous data)
            - "clip": Clip to [0,1] then scale to 0-255 (for data already in [0,1])
            - "labels": Use modulo 256 coloring (for label masks)
            - "percentile": Use percentile_clip for contrast (requires percentile_clip param)
        verbose: If True, print message when saving (default: True)
        percentile_clip: Tuple (low_percentile, high_percentile) for percentile-based
                        contrast. E.g., (1, 99) clips to 1st-99th percentile.
                        Only used when mode="percentile".

    Examples:
        >>> save_debug_image(vesselness_map, "debug/vesselness.png")  # normalize mode
        >>> save_debug_image(clahe_output, "debug/clahe.png", mode="clip")  # [0,1] data
        >>> save_debug_image(labels, "debug/labels.png", mode="labels")  # label mask
        >>> save_debug_image(raw, "debug/raw.png", mode="percentile", percentile_clip=(1, 99))
    """
    from PIL import Image
    from pathlib import Path as P

    if mode == "labels":
        # For labels, create RGB image with distinct colors per object
        # Use HSV-style coloring: hue varies by label, saturation/value fixed
        # Background (label 0) is black
        vis = np.zeros((*arr.shape, 3), dtype=np.uint8)
        mask = arr > 0
        if mask.any():
            # Use prime multiplier for good hue distribution
            hue = ((arr * 37) % 180).astype(np.uint8)  # OpenCV HSV hue is 0-179
            # Convert HSV to RGB manually (simplified)
            # H in [0,180], S=255, V=255 -> distinct bright colors
            h_norm = hue.astype(np.float32) / 30.0  # 0-6 range for sector
            sector = np.floor(h_norm).astype(np.int32) % 6
            f = h_norm - np.floor(h_norm)
            # p = 0 (since S=1), q = 1-f, t = f
            r = np.zeros_like(arr, dtype=np.uint8)
            g = np.zeros_like(arr, dtype=np.uint8)
            b = np.zeros_like(arr, dtype=np.uint8)
            # Sector 0: R=255, G=t*255, B=0
            s0 = sector == 0
            r[s0] = 255; g[s0] = (f[s0] * 255).astype(np.uint8); b[s0] = 0
            # Sector 1: R=q*255, G=255, B=0
            s1 = sector == 1
            r[s1] = ((1-f[s1]) * 255).astype(np.uint8); g[s1] = 255; b[s1] = 0
            # Sector 2: R=0, G=255, B=t*255
            s2 = sector == 2
            r[s2] = 0; g[s2] = 255; b[s2] = (f[s2] * 255).astype(np.uint8)
            # Sector 3: R=0, G=q*255, B=255
            s3 = sector == 3
            r[s3] = 0; g[s3] = ((1-f[s3]) * 255).astype(np.uint8); b[s3] = 255
            # Sector 4: R=t*255, G=0, B=255
            s4 = sector == 4
            r[s4] = (f[s4] * 255).astype(np.uint8); g[s4] = 0; b[s4] = 255
            # Sector 5: R=255, G=0, B=q*255
            s5 = sector == 5
            r[s5] = 255; g[s5] = 0; b[s5] = ((1-f[s5]) * 255).astype(np.uint8)
            # Apply only to labeled pixels
            vis[..., 0] = np.where(mask, r, 0)
            vis[..., 1] = np.where(mask, g, 0)
            vis[..., 2] = np.where(mask, b, 0)
    elif mode == "clip":
        # For data already in [0,1], clip and scale
        vis = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif mode == "percentile" and percentile_clip is not None:
        # Percentile-based contrast for better visibility
        low_val = np.percentile(arr, percentile_clip[0])
        high_val = np.percentile(arr, percentile_clip[1])
        if high_val > low_val:
            clipped = np.clip(arr, low_val, high_val)
            vis = ((clipped - low_val) / (high_val - low_val) * 255).astype(np.uint8)
        else:
            vis = np.zeros_like(arr, dtype=np.uint8)
    else:  # mode == "normalize" (default)
        # Min-max normalize to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            vis = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            vis = np.zeros_like(arr, dtype=np.uint8)

    # Ensure PNG extension for lossless output
    path = P(path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    Image.fromarray(vis).save(str(path), format="PNG")
    if verbose:
        print(f"    DEBUG: Saved {path}")


# Keep old name as alias for backwards compatibility
save_debug_jpeg = save_debug_image


def save_debug_overlay(
    raw_image: np.ndarray,
    mask: np.ndarray,
    path,
    mask_alpha: float = 0.5,
    raw_percentile: tuple = (1, 99),
    verbose: bool = True,
) -> None:
    """
    Save RGB overlay of mask on raw image for debug visualization.

    Args:
        raw_image: 2D array of raw image data
        mask: 2D array of segmentation mask (labels or binary)
        path: Output path
        mask_alpha: Opacity of mask overlay (0-1)
        raw_percentile: Percentile clip for raw image contrast
        verbose: Print when saving
    """
    from PIL import Image
    from pathlib import Path as P

    # Normalize raw image with percentile clipping for good contrast
    low_val = np.percentile(raw_image, raw_percentile[0])
    high_val = np.percentile(raw_image, raw_percentile[1])
    if high_val > low_val:
        raw_norm = np.clip(raw_image, low_val, high_val)
        raw_norm = ((raw_norm - low_val) / (high_val - low_val) * 255).astype(np.uint8)
    else:
        raw_norm = np.zeros_like(raw_image, dtype=np.uint8)

    # Create RGB image from grayscale raw
    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

    # Create mask colors (use hue rotation for distinct label colors)
    if mask.max() > 1:
        # Labels - use hue rotation for distinct colors
        hue = ((mask * 37) % 256).astype(np.uint8)  # Prime for variety
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[..., 0] = np.where(mask > 0, hue, 0)  # Red channel varies
        mask_rgb[..., 1] = np.where(mask > 0, 180, 0)  # Green where mask
        mask_rgb[..., 2] = np.where(mask > 0, 255 - hue, 0)  # Blue inverse
    else:
        # Binary - use bright green
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[..., 1] = np.where(mask > 0, 255, 0)

    # Blend
    mask_binary = mask > 0
    blended = rgb.copy()
    blended[mask_binary] = (
        (1 - mask_alpha) * rgb[mask_binary] +
        mask_alpha * mask_rgb[mask_binary]
    ).astype(np.uint8)

    path = P(path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    Image.fromarray(blended).save(str(path), format="PNG")
    if verbose:
        print(f"    DEBUG: Saved overlay {path}")


def create_combined_canvas(
    images: list,
    labels: list = None,
    output_path=None,
    title: str = "Organelle Segmentation Preview",
    raw_percentile: tuple = (1, 99),
    fluorescence_gamma: float = 0.25,
    show_outlines: bool = True,
    channel_params: dict = None,
) -> None:
    """
    Create a combined canvas with all channel images in a grid using matplotlib.

    Creates four rows:
    1. Raw images - grayscale intensity maps
    2. Labeled segmentation masks - colorized by object ID
    3. Overlay - labels overlaid on raw image (filled regions)
    4. Outlines - segmentation contours on raw image (cyan outlines only)

    Uses matplotlib GridSpec for proper spacing and professional labeling.

    Args:
        images: List of dicts with keys:
            - 'name': Channel name (e.g., "Phase2D_tubular")
            - 'raw': 2D raw image array (optional, first one used for all overlays)
            - 'vesselness': 2D vesselness array (optional)
            - 'labels': 2D label array
        labels: (deprecated, use images[i]['labels'] instead)
        output_path: Path to save combined canvas PNG
        title: Title for the canvas
        raw_percentile: Percentile clip for raw image contrast
        fluorescence_gamma: Gamma correction for fluorescence channels (default: 0.25).
            Values < 1 brighten dim structures. Applied to GFP, mCherry, etc.
        show_outlines: If True, add a 4th row with outline segmentations (default: True).
        channel_params: Dict mapping channel name to params dict for subtitle display.
            Example: {"GFP_tubular": {"frangi": {...}, "clahe": {...}}}
    """
    import time
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pathlib import Path as P
    from skimage.segmentation import find_boundaries

    canvas_start = time.time()
    print(f"  Creating combined canvas...")

    if not images:
        print(f"    WARNING: No images to combine for canvas")
        return

    # Get dimensions from first image with labels
    first_labels = None
    for img in images:
        if img.get('labels') is not None:
            first_labels = img['labels']
            break

    if first_labels is None:
        print(f"    WARNING: No labels found in any image")
        return

    n_cols = len(images)
    n_rows = 4 if show_outlines else 3  # Raw, Labels, Overlay, [Outlines]

    # Create figure with GridSpec for proper spacing
    fig_width = 6 * n_cols
    fig_height = 6 * n_rows
    print(f"    Figure size: {fig_width}x{fig_height} inches, {n_cols} columns x {n_rows} rows")

    fig_create_start = time.time()
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.02)
    print(f"    [TIMING] Figure creation: {time.time() - fig_create_start:.2f}s")

    row_labels = ["Raw", "Labels", "Overlay"]
    if show_outlines:
        row_labels.append("Outlines")

    # Fluorescence channel detection (case-insensitive)
    FLUORESCENCE_MARKERS = ['gfp', 'mcherry', 'rfp', 'cfp', 'yfp', 'bfp', 'tomato', 'cherry', 'alexa', 'cy3', 'cy5', 'fitc', 'dapi']

    def is_fluorescence_channel(channel_name: str) -> bool:
        """Check if channel name indicates a fluorescence channel."""
        name_lower = channel_name.lower()
        return any(marker in name_lower for marker in FLUORESCENCE_MARKERS)

    def apply_gamma(img_norm: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to normalized [0,1] image."""
        return np.power(img_norm, gamma)

    render_start = time.time()
    for col_idx, img_data in enumerate(images):
        name = img_data.get('name', f'Channel {col_idx}')
        labels_arr = img_data.get('labels')
        vesselness = img_data.get('vesselness')
        channel_raw = img_data.get('raw')  # Use this channel's own raw image
        n_objects = int(labels_arr.max()) if labels_arr is not None else 0

        # Check if this is a fluorescence channel (needs gamma correction)
        is_fluor = is_fluorescence_channel(name)

        # --- Row 0: Raw image ---
        ax = fig.add_subplot(gs[0, col_idx])
        if channel_raw is not None:
            vmin, vmax = np.percentile(channel_raw, raw_percentile)
            # Normalize to [0, 1]
            raw_norm = np.clip(channel_raw, vmin, vmax)
            raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)
            # Apply gamma for fluorescence channels
            if is_fluor and fluorescence_gamma != 1.0:
                raw_norm = apply_gamma(raw_norm, fluorescence_gamma)
            ax.imshow(raw_norm, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros_like(first_labels), cmap='gray')

        # Column title (only on top row)
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray',
                             edgecolor='black', alpha=0.9, linewidth=1.5))
        ax.axis('off')

        # Row label (only on first column)
        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[0], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 1: Labels (colorized) ---
        ax = fig.add_subplot(gs[1, col_idx])
        if labels_arr is not None and n_objects > 0:
            # Create random colors for each label
            np.random.seed(42)
            colored_mask = np.zeros((*labels_arr.shape, 3))
            for obj_id in range(1, n_objects + 1):
                color = np.random.rand(3)
                colored_mask[labels_arr == obj_id] = color
            ax.imshow(colored_mask)
            # Object count badge
            ax.text(0.02, 0.98, f'N={n_objects}', transform=ax.transAxes,
                   fontsize=11, color='white', fontweight='bold',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        else:
            ax.imshow(np.zeros((*first_labels.shape, 3)))
            ax.text(0.02, 0.98, 'N=0', transform=ax.transAxes,
                   fontsize=11, color='white', fontweight='bold',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row label (only on first column)
        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[1], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 2: Overlay (use THIS channel's raw image, not shared) ---
        ax = fig.add_subplot(gs[2, col_idx])
        if channel_raw is not None:
            # Normalize this channel's raw image for display
            vmin, vmax = np.percentile(channel_raw, raw_percentile)
            raw_norm = np.clip(channel_raw, vmin, vmax)
            raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)

            # Apply gamma for fluorescence channels
            if is_fluor and fluorescence_gamma != 1.0:
                raw_norm = apply_gamma(raw_norm, fluorescence_gamma)

            # Create RGB from grayscale
            rgb_overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

            if labels_arr is not None and n_objects > 0:
                # Blend with label colors
                np.random.seed(42)
                label_mask = labels_arr > 0
                label_colors = np.zeros((*labels_arr.shape, 3))
                for obj_id in range(1, n_objects + 1):
                    color = np.random.rand(3)
                    label_colors[labels_arr == obj_id] = color
                alpha = 0.4
                rgb_overlay[label_mask] = (1 - alpha) * rgb_overlay[label_mask] + alpha * label_colors[label_mask]

            ax.imshow(rgb_overlay)
        else:
            ax.imshow(np.zeros((*first_labels.shape, 3)))
        ax.axis('off')

        # Row label (only on first column)
        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[2], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 3: Outlines (cyan contours on raw image) ---
        if show_outlines:
            ax = fig.add_subplot(gs[3, col_idx])
            if channel_raw is not None:
                # Normalize raw image for display
                vmin, vmax = np.percentile(channel_raw, raw_percentile)
                raw_norm = np.clip(channel_raw, vmin, vmax)
                raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)

                # Apply gamma for fluorescence channels
                if is_fluor and fluorescence_gamma != 1.0:
                    raw_norm = apply_gamma(raw_norm, fluorescence_gamma)

                # Create RGB from grayscale
                rgb_outline = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

                if labels_arr is not None and n_objects > 0:
                    # Dilate labels first to push outline outward from objects
                    # This gives more space to see the objects between outline and interior
                    from scipy.ndimage import binary_dilation, grey_dilation
                    dilated_labels = grey_dilation(labels_arr, size=2)  # 2px dilation
                    # Find outer boundary of dilated labels (thin 1px line)
                    boundaries = find_boundaries(dilated_labels, mode='outer')
                    # Solid cyan outline (no alpha blending for visibility)
                    rgb_outline[boundaries, 0] = 0.0  # Red = 0
                    rgb_outline[boundaries, 1] = 1.0  # Green = 1
                    rgb_outline[boundaries, 2] = 1.0  # Blue = 1

                ax.imshow(rgb_outline)
            else:
                ax.imshow(np.zeros((*first_labels.shape, 3)))
            ax.axis('off')

            # Row label (only on first column)
            if col_idx == 0:
                ax.text(-0.05, 0.5, row_labels[3], transform=ax.transAxes,
                       fontsize=14, fontweight='bold', rotation=90,
                       va='center', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                edgecolor='orange', alpha=0.9))

        print(f"      Column {col_idx + 1}/{n_cols} ({name}): rendered")

    print(f"    [TIMING] Rendering all columns: {time.time() - render_start:.2f}s")

    # Add main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

    # Add subtitle with params if provided
    if channel_params:
        subtitle_lines = []
        for ch_name, params in channel_params.items():
            # Format frangi params (most important)
            frangi = params.get('frangi_params', params.get('frangi', {}))
            clahe = params.get('clahe_params', params.get('clahe', {}))

            # Extract key frangi values
            pixel_size = frangi.get('pixel_size_um', 'N/A')
            threshold = frangi.get('threshold', 'N/A')
            alpha = frangi.get('alpha', 'N/A')
            beta = frangi.get('beta', 'N/A')
            min_r = frangi.get('min_radius_um', 'N/A')
            max_r = frangi.get('max_radius_um', 'N/A')

            # Extract key CLAHE values
            clip_limit = clahe.get('clip_limit', 'N/A')
            kernel = clahe.get('kernel_size', 'N/A')
            if isinstance(kernel, (list, tuple)):
                kernel = f"{kernel[0]}x{kernel[1]}"

            # Format line
            line = (f"{ch_name}: pixel_size={pixel_size}um, "
                   f"sigma=[{min_r}-{max_r}]um, alpha={alpha}, beta={beta}, "
                   f"thresh={threshold}, CLAHE(clip={clip_limit}, kernel={kernel})")
            subtitle_lines.append(line)

        subtitle = '\n'.join(subtitle_lines)
        fig.text(0.5, 0.98, subtitle, fontsize=9, ha='center', va='top',
                 family='monospace', wrap=True,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8))

    # Save at high DPI for full resolution
    output_path = P(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_start = time.time()
    print(f"    Saving to disk (150 DPI)...")
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"    [TIMING] savefig: {time.time() - save_start:.2f}s")

    plt.close(fig)

    total_time = time.time() - canvas_start
    print(f"  Saved combined canvas: {output_path}")
    print(f"    Grid: {n_rows} rows x {n_cols} columns, 150 DPI")
    print(f"    [TIMING] Total canvas time: {total_time:.2f}s")


def create_sweep_canvas(
    images: list,
    output_path=None,
    title: str = "Parameter Sweep",
    sweep_var: str = "pixel_size_um",
    channel_name: str = "Channel",
    raw_percentile: tuple = (1, 99),
    fluorescence_gamma: float = 0.25,
    show_outlines: bool = True,
) -> None:
    """
    Create a canvas for parameter sweep visualization.

    Similar to create_combined_canvas, but optimized for showing the same channel
    with varying parameter values. Column headers show the sweep parameter value.

    Creates four rows:
    1. Raw images - grayscale intensity maps (same raw for all columns)
    2. Labeled segmentation masks - colorized by object ID
    3. Overlay - labels overlaid on raw image (filled regions)
    4. Outlines - segmentation contours on raw image (cyan outlines only)

    Args:
        images: List of dicts with keys:
            - 'name': Parameter value label (e.g., "0.1625")
            - 'raw': 2D raw image array
            - 'vesselness': 2D vesselness array (optional)
            - 'labels': 2D label array
        output_path: Path to save combined canvas PNG
        title: Title for the canvas
        sweep_var: Name of the swept parameter (e.g., "pixel_size_um")
        channel_name: Channel name being swept (e.g., "GFP")
        raw_percentile: Percentile clip for raw image contrast
        fluorescence_gamma: Gamma correction for fluorescence channels (default: 0.25).
        show_outlines: If True, add a 4th row with outline segmentations (default: True).
    """
    import time
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pathlib import Path as P
    from skimage.segmentation import find_boundaries

    canvas_start = time.time()
    print(f"  Creating sweep canvas...")

    if not images:
        print(f"    WARNING: No images to combine for canvas")
        return

    # Get dimensions from first image with labels
    first_labels = None
    for img in images:
        if img.get('labels') is not None:
            first_labels = img['labels']
            break

    if first_labels is None:
        print(f"    WARNING: No labels found in any image")
        return

    n_cols = len(images)
    n_rows = 4 if show_outlines else 3  # Raw, Labels, Overlay, [Outlines]

    # Create figure with GridSpec for proper spacing
    fig_width = 6 * n_cols
    fig_height = 6 * n_rows
    print(f"    Figure size: {fig_width}x{fig_height} inches, {n_cols} columns x {n_rows} rows")

    fig_create_start = time.time()
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.02)
    print(f"    [TIMING] Figure creation: {time.time() - fig_create_start:.2f}s")

    row_labels = ["Raw", "Labels", "Overlay"]
    if show_outlines:
        row_labels.append("Outlines")

    # Fluorescence channel detection (case-insensitive)
    FLUORESCENCE_MARKERS = ['gfp', 'mcherry', 'rfp', 'cfp', 'yfp', 'bfp', 'tomato', 'cherry', 'alexa', 'cy3', 'cy5', 'fitc', 'dapi']

    def is_fluorescence_channel(ch_name: str) -> bool:
        """Check if channel name indicates a fluorescence channel."""
        name_lower = ch_name.lower()
        return any(marker in name_lower for marker in FLUORESCENCE_MARKERS)

    def apply_gamma(img_norm: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to normalized [0,1] image."""
        return np.power(img_norm, gamma)

    is_fluor = is_fluorescence_channel(channel_name)

    render_start = time.time()
    for col_idx, img_data in enumerate(images):
        name = img_data.get('name', f'{col_idx}')
        labels_arr = img_data.get('labels')
        vesselness = img_data.get('vesselness')
        channel_raw = img_data.get('raw')
        n_objects = int(labels_arr.max()) if labels_arr is not None else 0

        # --- Row 0: Raw image ---
        ax = fig.add_subplot(gs[0, col_idx])
        if channel_raw is not None:
            vmin, vmax = np.percentile(channel_raw, raw_percentile)
            raw_norm = np.clip(channel_raw, vmin, vmax)
            raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)
            if is_fluor and fluorescence_gamma != 1.0:
                raw_norm = apply_gamma(raw_norm, fluorescence_gamma)
            ax.imshow(raw_norm, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros_like(first_labels), cmap='gray')

        # Column title shows sweep parameter value
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                             edgecolor='darkblue', alpha=0.9, linewidth=1.5))
        ax.axis('off')

        # Row label (only on first column)
        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[0], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 1: Labels (colorized) ---
        ax = fig.add_subplot(gs[1, col_idx])
        if labels_arr is not None and n_objects > 0:
            np.random.seed(42)
            colored_mask = np.zeros((*labels_arr.shape, 3))
            for obj_id in range(1, n_objects + 1):
                color = np.random.rand(3)
                colored_mask[labels_arr == obj_id] = color
            ax.imshow(colored_mask)
            ax.text(0.02, 0.98, f'N={n_objects}', transform=ax.transAxes,
                   fontsize=11, color='white', fontweight='bold',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        else:
            ax.imshow(np.zeros((*first_labels.shape, 3)))
            ax.text(0.02, 0.98, 'N=0', transform=ax.transAxes,
                   fontsize=11, color='white', fontweight='bold',
                   va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        ax.axis('off')

        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[1], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 2: Overlay ---
        ax = fig.add_subplot(gs[2, col_idx])
        if channel_raw is not None:
            vmin, vmax = np.percentile(channel_raw, raw_percentile)
            raw_norm = np.clip(channel_raw, vmin, vmax)
            raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)
            if is_fluor and fluorescence_gamma != 1.0:
                raw_norm = apply_gamma(raw_norm, fluorescence_gamma)
            rgb_overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

            if labels_arr is not None and n_objects > 0:
                np.random.seed(42)
                label_mask = labels_arr > 0
                label_colors = np.zeros((*labels_arr.shape, 3))
                for obj_id in range(1, n_objects + 1):
                    color = np.random.rand(3)
                    label_colors[labels_arr == obj_id] = color
                alpha = 0.4
                rgb_overlay[label_mask] = (1 - alpha) * rgb_overlay[label_mask] + alpha * label_colors[label_mask]

            ax.imshow(rgb_overlay)
        else:
            ax.imshow(np.zeros((*first_labels.shape, 3)))
        ax.axis('off')

        if col_idx == 0:
            ax.text(-0.05, 0.5, row_labels[2], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', rotation=90,
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

        # --- Row 3: Outlines ---
        if show_outlines:
            ax = fig.add_subplot(gs[3, col_idx])
            if channel_raw is not None:
                vmin, vmax = np.percentile(channel_raw, raw_percentile)
                raw_norm = np.clip(channel_raw, vmin, vmax)
                raw_norm = (raw_norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_norm)
                if is_fluor and fluorescence_gamma != 1.0:
                    raw_norm = apply_gamma(raw_norm, fluorescence_gamma)
                rgb_outline = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

                if labels_arr is not None and n_objects > 0:
                    from scipy.ndimage import grey_dilation
                    dilated_labels = grey_dilation(labels_arr, size=2)
                    boundaries = find_boundaries(dilated_labels, mode='outer')
                    rgb_outline[boundaries, 0] = 0.0
                    rgb_outline[boundaries, 1] = 1.0
                    rgb_outline[boundaries, 2] = 1.0

                ax.imshow(rgb_outline)
            else:
                ax.imshow(np.zeros((*first_labels.shape, 3)))
            ax.axis('off')

            if col_idx == 0:
                ax.text(-0.05, 0.5, row_labels[3], transform=ax.transAxes,
                       fontsize=14, fontweight='bold', rotation=90,
                       va='center', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                edgecolor='orange', alpha=0.9))

        print(f"      Column {col_idx + 1}/{n_cols} ({name}): rendered")

    print(f"    [TIMING] Rendering all columns: {time.time() - render_start:.2f}s")

    # Add main title with channel and sweep info
    fig.suptitle(f"{title}\nChannel: {channel_name}, Varying: {sweep_var}",
                 fontsize=18, fontweight='bold', y=0.995)

    # Save at high DPI
    output_path = P(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_start = time.time()
    print(f"    Saving to disk (150 DPI)...")
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"    [TIMING] savefig: {time.time() - save_start:.2f}s")

    plt.close(fig)

    total_time = time.time() - canvas_start
    print(f"  Saved sweep canvas: {output_path}")
    print(f"    Grid: {n_rows} rows x {n_cols} columns, 150 DPI")
    print(f"    [TIMING] Total canvas time: {total_time:.2f}s")


def _save_tiled_debug_images(
    source_zarr_path: str,
    pos_path: str,
    output_label_name: str,
    method_name: str,
    center_tile_result: dict,
    center_ty: int,
    center_tx: int,
    tile_size: int,
    tile_overlap: int,
    height: int,
    width: int,
    extra_arrays: dict = None,
    channel_index: int = 0,
):
    """
    Save enhanced debug images for tiled segmentation (before/after stitching).

    This is a shared helper used by both Frangi and CellPose tiled segmentation.
    Saves images to: {3-assembly}/organelle_seg_debug/{method}_{pos_path}_{label}/

    Args:
        source_zarr_path: Path to the v3 zarr store
        pos_path: Position path like "A/1/0"
        output_label_name: Name of the output label (e.g., "mitoc_tomm20_seg")
        method_name: Segmentation method ("frangi" or "cellpose")
        center_tile_result: Dict with at least "labels_before_stitch" key
        center_ty: Y index of center tile
        center_tx: X index of center tile
        tile_size: Size of each tile
        tile_overlap: Overlap between tiles
        height: Total image height
        width: Total image width
        extra_arrays: Optional dict of additional arrays to save (e.g., {"vesselness": arr})
        channel_index: Channel index in source data for loading raw tile (default: 0)
    """
    # Build debug directory path in 3-assembly folder
    assembly_dir = Path(source_zarr_path).parent  # 3-assembly folder (parent of phenotyping_v3.zarr)
    debug_dir = assembly_dir / "organelle_seg_debug" / f"{method_name}_{pos_path.replace('/', '_')}_{output_label_name}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    step = tile_size - tile_overlap
    y_start_center = center_ty * step
    x_start_center = center_tx * step
    y_end_center = min(y_start_center + tile_size, height)
    x_end_center = min(x_start_center + tile_size, width)

    # Load raw input tile for comparison
    raw_tile = None
    try:
        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            raw_tile = np.asarray(
                ds[pos_path]["0"][0, channel_index, 0, y_start_center:y_end_center, x_start_center:x_end_center]
            )
    except Exception as e:
        print(f"  Warning: Could not load raw tile for debug: {e}")

    # 1. Save raw input tile with percentile contrast
    if raw_tile is not None:
        save_debug_image(
            raw_tile,
            debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_01_raw_input.png",
            mode="percentile",
            percentile_clip=(1, 99),
            verbose=False,
        )

    if center_tile_result is not None and "labels_before_stitch" in center_tile_result:
        labels_before = center_tile_result["labels_before_stitch"]

        # 2. Save labels before stitching
        save_debug_image(
            labels_before,
            debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_02_before_stitch.png",
            mode="labels",
            verbose=False,
        )

        # 3. Save vesselness with PERCENTILE contrast (much better visibility)
        if extra_arrays and "vesselness" in extra_arrays and extra_arrays["vesselness"] is not None:
            vesselness = extra_arrays["vesselness"]

            # Save with percentile contrast (excludes extreme outliers)
            save_debug_image(
                vesselness,
                debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_03_vesselness_pct.png",
                mode="percentile",
                percentile_clip=(5, 99),
                verbose=False,
            )

            # Also save log-scale vesselness for better dynamic range visibility
            log_vesselness = np.log10(vesselness + 1e-10)
            save_debug_image(
                log_vesselness,
                debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_04_vesselness_log.png",
                mode="percentile",
                percentile_clip=(5, 99),
                verbose=False,
            )

        # 4. Save overlay of mask on raw (before stitch)
        if raw_tile is not None:
            save_debug_overlay(
                raw_image=raw_tile,
                mask=labels_before,
                path=debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_05_overlay_before.png",
                mask_alpha=0.4,
                raw_percentile=(1, 99),
                verbose=False,
            )

    # 5. Save after-stitch labels
    with open_ome_zarr(source_zarr_path, mode="r") as ds:
        final_labels_center = np.asarray(
            ds[pos_path].zgroup["labels"][output_label_name]["0"][
                0, 0, 0, y_start_center:y_end_center, x_start_center:x_end_center
            ]
        )

    save_debug_image(
        final_labels_center,
        debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_06_after_stitch.png",
        mode="labels",
        verbose=False,
    )

    # 6. Save overlay of after-stitch mask on raw
    if raw_tile is not None:
        save_debug_overlay(
            raw_image=raw_tile,
            mask=final_labels_center,
            path=debug_dir / f"{method_name}_tile_{center_ty}_{center_tx}_07_overlay_after.png",
            mask_alpha=0.4,
            raw_percentile=(1, 99),
            verbose=False,
        )

    print(f"  Debug images saved to: {debug_dir}")
    print(f"    Files: 01_raw_input, 02_before_stitch, 03_vesselness_pct, 04_vesselness_log,")
    print(f"           05_overlay_before, 06_after_stitch, 07_overlay_after")


def save_segmentation_params_yaml(
    channel_params: dict,
    output_path,
    experiment: str = None,
    position: str = None,
    preview_mode: bool = False,
) -> Path:
    """
    Save segmentation parameters to a YAML file for reproducibility.

    In preview/debug mode, saves to the debug subfolder.
    In normal mode, saves to the 3-assembly folder.

    Args:
        channel_params: Dict mapping channel name to params dict.
            Example: {"GFP_tubular": {"frangi_params": {...}, "clahe_params": {...}}}
        output_path: Path to save YAML file.
        experiment: Experiment name (for metadata).
        position: Position path (for metadata).
        preview_mode: If True, indicates this is a preview run (adds note to file).

    Returns:
        Path to saved YAML file.
    """
    import yaml

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output structure with metadata
    output_data = {
        "metadata": {
            "experiment": experiment,
            "position": position,
            "generated_at": datetime.now().isoformat(),
            "preview_mode": preview_mode,
        },
        "channels": {},
    }

    # Process each channel's params
    for ch_name, params in channel_params.items():
        frangi = params.get('frangi_params', params.get('frangi', {}))
        clahe = params.get('clahe_params', params.get('clahe', {}))
        structure_type = params.get('structure_type')

        # Convert numpy types to Python native types for YAML serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj

        output_data["channels"][ch_name] = {
            "structure_type": structure_type,
            "frangi_params": convert_to_native(frangi),
            "clahe_params": convert_to_native(clahe),
        }

    # Write YAML with nice formatting
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"  Saved segmentation params to: {output_path}")
    return output_path
