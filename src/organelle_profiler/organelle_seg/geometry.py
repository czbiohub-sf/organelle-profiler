"""
Geometry Utilities for Organelle Segmentation
==============================================

This module provides geometric utility functions for coordinate calculations
and bounding box operations.

Main functions:
- calculate_center_crop_bbox(): Calculate center crop bounding box
- get_bbox(): Find bounding box of binary image (2D or 3D)
"""

import numpy as np


def calculate_center_crop_bbox(full_shape: tuple, crop_fraction: float) -> tuple:
    """
    Calculate bounding box for center crop of an image.

    Args:
        full_shape: Full image shape (T, C, Z, Y, X) or (Y, X) or (Z, Y, X)
        crop_fraction: Fraction of image to keep (e.g., 0.01 for 1%, 0.1 for 10%)

    Returns:
        Tuple of (y_start, y_end, x_start, x_end) for the center crop region

    Examples:
        >>> calculate_center_crop_bbox((1, 1, 10, 1000, 1000), 0.01)
        (450, 550, 450, 550)  # 100x100 center crop (sqrt(0.01) = 0.1 = 10% linear)
        >>> calculate_center_crop_bbox((2048, 2048), 0.25)
        (512, 1536, 512, 1536)  # 1024x1024 center crop (sqrt(0.25) = 0.5 = 50% linear)
    """
    if len(full_shape) == 5:  # (T, C, Z, Y, X)
        height, width = full_shape[3], full_shape[4]
    elif len(full_shape) == 3:  # (Z, Y, X)
        height, width = full_shape[1], full_shape[2]
    elif len(full_shape) == 2:  # (Y, X)
        height, width = full_shape[0], full_shape[1]
    else:
        raise ValueError(f"Unexpected shape: {full_shape}")

    # Calculate crop dimensions (linear scale, not area)
    # For 1% area, we want sqrt(0.01) = 0.1 = 10% linear
    linear_fraction = np.sqrt(crop_fraction)
    crop_height = int(height * linear_fraction)
    crop_width = int(width * linear_fraction)

    # Ensure minimum size
    crop_height = max(crop_height, 512)
    crop_width = max(crop_width, 512)

    # Calculate center
    center_y = height // 2
    center_x = width // 2

    y_start = center_y - crop_height // 2
    y_end = y_start + crop_height
    x_start = center_x - crop_width // 2
    x_end = x_start + crop_width

    # Clamp to image bounds
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(height, y_end)
    x_end = min(width, x_end)

    return (y_start, y_end, x_start, x_end)


def get_bbox(im, xp):
    """
    Find the bounding box of a binary image.

    Args:
        im: Binary image (2D or 3D numpy/cupy array)
        xp: Array module (numpy or cupy)

    Returns:
        For 2D: (ymin, ymax, xmin, xmax) or (None, None, None, None) if empty
        For 3D: (zmin, zmax, ymin, ymax, xmin, xmax) or tuple of Nones if empty

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((10, 10), dtype=bool)
        >>> img[2:5, 3:7] = True
        >>> get_bbox(img, np)
        (2, 4, 3, 6)

        >>> img_3d = np.zeros((5, 10, 10), dtype=bool)
        >>> img_3d[1:3, 2:5, 3:7] = True
        >>> get_bbox(img_3d, np)
        (1, 2, 2, 4, 3, 6)
    """
    if im.ndim == 3:
        coords = xp.where(im)
        if len(coords[0]) == 0:
            return None, None, None, None, None, None
        zmin, zmax = coords[0].min(), coords[0].max()
        ymin, ymax = coords[1].min(), coords[1].max()
        xmin, xmax = coords[2].min(), coords[2].max()
        return zmin, zmax, ymin, ymax, xmin, xmax
    else:  # 2D
        coords = xp.where(im)
        if len(coords[0]) == 0:
            return None, None, None, None
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()
        return ymin, ymax, xmin, xmax
