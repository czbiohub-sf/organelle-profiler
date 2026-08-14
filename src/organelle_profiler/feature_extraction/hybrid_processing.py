"""
Hybrid GPU/CPU Processing for Feature Extraction.

This module implements a complexity-based routing strategy:
- Simple cells -> GPU batch processing (fast for basic regionprops)
- Complex cells -> CPU parallel processing (better for network analysis)

Complexity is determined by:
1. Bbox type: CP-only (simple) vs dual-pass (complex)
2. Network organelles: cells requiring network analysis are complex
3. Bbox size: larger bboxes = more complex
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def compute_cell_complexity(
    cells_df: pd.DataFrame,
    network_organelles: List[str],
) -> pd.DataFrame:
    """
    Compute complexity score for each cell to enable GPU/CPU routing.

    Complexity factors:
    - has_dual_pass: +50 (requires processing both standard and CP pipelines)
    - bbox_area: normalized size contribution
    - network_organelle_count: +10 per network organelle present

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell metadata with bbox columns
    network_organelles : list
        List of organelle names requiring network analysis

    Returns
    -------
    pd.DataFrame
        Input dataframe with added '_complexity_score' column
    """
    df = cells_df.copy()

    # Initialize complexity score
    df['_complexity_score'] = 0.0

    # Factor 1: Dual-pass requirement (both standard and CP bboxes)
    # Check for valid bboxes
    has_std_bbox = df['bbox'].notna() if 'bbox' in df.columns else pd.Series(False, index=df.index)
    has_cp_bbox = df['cp_bbox'].notna() if 'cp_bbox' in df.columns else pd.Series(False, index=df.index)

    # Handle string 'nan' values
    if 'bbox' in df.columns:
        has_std_bbox = has_std_bbox & (df['bbox'].astype(str) != 'nan')
    if 'cp_bbox' in df.columns:
        has_cp_bbox = has_cp_bbox & (df['cp_bbox'].astype(str) != 'nan')

    # Check for valid seg_ids
    has_std_seg = df['seg_id'].notna() if 'seg_id' in df.columns else pd.Series(False, index=df.index)
    has_cp_seg = df['cp_seg_id'].notna() if 'cp_seg_id' in df.columns else pd.Series(False, index=df.index)

    # Valid standard processing requires both bbox and seg_id
    valid_std = has_std_bbox & has_std_seg
    valid_cp = has_cp_bbox & has_cp_seg

    # Dual-pass cells are the most complex
    is_dual_pass = valid_std & valid_cp
    df.loc[is_dual_pass, '_complexity_score'] += 50

    # Factor 2: Bbox area (larger = more complex)
    # Parse bbox strings to compute area
    def bbox_area(bbox_val):
        try:
            # Handle None/NaN
            if bbox_val is None:
                return 0

            # Handle numpy arrays
            if isinstance(bbox_val, np.ndarray):
                if bbox_val.size == 0:
                    return 0
                if len(bbox_val) >= 4:
                    y_min, x_min, y_max, x_max = bbox_val[:4]
                    return float((y_max - y_min) * (x_max - x_min))
                return 0

            # Handle scalar NaN
            if pd.isna(bbox_val):
                return 0

            # Handle string 'nan'
            if str(bbox_val).lower() == 'nan':
                return 0

            # Handle string representation
            if isinstance(bbox_val, str):
                bbox_val = bbox_val.strip('[]() ')
                parts = [float(x) for x in bbox_val.replace(',', ' ').split()]
                if len(parts) >= 4:
                    y_min, x_min, y_max, x_max = parts[:4]
                    return (y_max - y_min) * (x_max - x_min)

            # Handle list/tuple
            elif hasattr(bbox_val, '__iter__') and hasattr(bbox_val, '__len__'):
                parts = list(bbox_val)
                if len(parts) >= 4:
                    y_min, x_min, y_max, x_max = parts[:4]
                    return float((y_max - y_min) * (x_max - x_min))
        except Exception:
            pass
        return 0

    # Compute bbox areas
    if 'bbox' in df.columns:
        std_areas = df['bbox'].apply(bbox_area)
    else:
        std_areas = pd.Series(0, index=df.index)

    if 'cp_bbox' in df.columns:
        cp_areas = df['cp_bbox'].apply(bbox_area)
    else:
        cp_areas = pd.Series(0, index=df.index)

    # Use max of the two bbox areas, normalized
    max_areas = np.maximum(std_areas, cp_areas)
    if max_areas.max() > 0:
        normalized_areas = max_areas / max_areas.max() * 20  # Scale to 0-20
        df['_complexity_score'] += normalized_areas

    # Factor 3: Network organelles (each adds significant computation)
    # For now, assume all cells with valid segmentation will have network analysis
    # This adds a constant factor based on number of network organelles
    n_network = len(network_organelles) if network_organelles else 0
    df.loc[valid_std | valid_cp, '_complexity_score'] += n_network * 5

    # Store component flags for debugging/analysis
    df['_is_dual_pass'] = is_dual_pass
    df['_is_cp_only'] = valid_cp & ~valid_std
    df['_is_std_only'] = valid_std & ~valid_cp

    return df


def partition_cells_by_complexity(
    cells_df: pd.DataFrame,
    network_organelles: List[str],
    gpu_threshold_percentile: float = 40,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition cells into GPU (simple) and CPU (complex) groups.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell metadata
    network_organelles : list
        List of organelle names requiring network analysis
    gpu_threshold_percentile : float
        Percentile threshold - cells below this complexity go to GPU
        Default 40 means bottom 40% of cells go to GPU

    Returns
    -------
    tuple of (gpu_cells_df, cpu_cells_df)
        Partitioned cell DataFrames
    """
    # Compute complexity scores
    scored_df = compute_cell_complexity(cells_df, network_organelles)

    # Determine threshold
    threshold = np.percentile(scored_df['_complexity_score'], gpu_threshold_percentile)

    # Partition
    gpu_mask = scored_df['_complexity_score'] <= threshold

    gpu_cells = scored_df[gpu_mask].copy()
    cpu_cells = scored_df[~gpu_mask].copy()

    # Print partition stats
    n_total = len(scored_df)
    n_gpu = len(gpu_cells)
    n_cpu = len(cpu_cells)

    print(f"\n  Cell complexity partitioning:")
    print(f"    Total cells: {n_total}")
    print(f"    GPU (simple, score <= {threshold:.1f}): {n_gpu} ({100*n_gpu/n_total:.1f}%)")
    print(f"    CPU (complex, score > {threshold:.1f}): {n_cpu} ({100*n_cpu/n_total:.1f}%)")

    # Show breakdown by type
    n_dual = scored_df['_is_dual_pass'].sum()
    n_cp_only = scored_df['_is_cp_only'].sum()
    n_std_only = scored_df['_is_std_only'].sum()
    print(f"    Breakdown: {n_dual} dual-pass, {n_cp_only} CP-only, {n_std_only} std-only")

    return gpu_cells, cpu_cells


def get_hybrid_processing_config(
    cells_df: pd.DataFrame,
    network_organelles: List[str],
    n_cpu_workers: int = 31,
    n_gpu_workers: int = 8,
    gpu_batch_size: int = 256,
    cpu_batch_size: int = 64,
) -> Dict:
    """
    Get configuration for hybrid GPU/CPU processing.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell metadata
    network_organelles : list
        Network organelles list
    n_cpu_workers : int
        Number of CPU workers for complex cells
    n_gpu_workers : int
        Number of GPU workers (Dask) for simple cells
    gpu_batch_size : int
        Batch size for GPU processing (larger = better GPU utilization)
    cpu_batch_size : int
        Batch size for CPU processing

    Returns
    -------
    dict
        Configuration for hybrid processing
    """
    gpu_cells, cpu_cells = partition_cells_by_complexity(cells_df, network_organelles)

    return {
        'gpu_cells': gpu_cells,
        'cpu_cells': cpu_cells,
        'n_gpu_workers': n_gpu_workers,
        'n_cpu_workers': n_cpu_workers,
        'gpu_batch_size': gpu_batch_size,
        'cpu_batch_size': cpu_batch_size,
        'gpu_threshold': np.percentile(
            compute_cell_complexity(cells_df, network_organelles)['_complexity_score'],
            40
        ),
    }
