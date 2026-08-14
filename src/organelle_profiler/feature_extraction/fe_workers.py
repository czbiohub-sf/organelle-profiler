import warnings
# Suppress CUDA deprecation warnings BEFORE any CUDA-related imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cuda.cudart.*deprecated.*")

import numpy as np
import os
import time

import pandas as pd
from iohub import open_ome_zarr
from pathlib import Path
from tqdm import tqdm
from skimage.measure import regionprops_table, label
import tensorstore as ts

# GPU-dependent imports (torch/CUDA) — guarded so SPMD workers can run on CPU-only nodes.
try:
    from cyclops_utils.data.bbox_utils import BaseDataset
    from organelle_profiler.feature_extraction.localization_features import (
        precompute_boundary_kdtrees,
        compute_localization_kdtree,
        compute_cell_level_localization_summary,
    )
    from cyclops_utils.hpc.resource_manager import get_optimal_workers
    from organelle_profiler.feature_extraction.hybrid_processing import (
        compute_cell_complexity,
        partition_cells_by_complexity,
    )
    from organelle_profiler.feature_extraction.fe_metadata import create_global_cell_id
except ImportError:
    pass  # CPU-only node: GPU functions won't be called

# Dask for GPU-compatible parallelization (uses spawn instead of fork)
try:
    from dask.distributed import LocalCluster, Client, WorkerPlugin
    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False
    print("[WARNING] Dask not available, falling back to ProcessPoolExecutor (no GPU support)")


class CUDALibraryPlugin(WorkerPlugin):
    """Dask worker plugin to set CUDA library paths at startup.

    This plugin runs when a worker starts (before any tasks), allowing us to
    set LD_LIBRARY_PATH properly in spawned processes.

    NOTE: GPU device assignment is handled inside the task function via
    cupy.cuda.Device(gpu_id).use(), NOT via CUDA_VISIBLE_DEVICES in this plugin.
    Setting CUDA_VISIBLE_DEVICES in setup() is too late — the CUDA runtime may
    already be initialized during module imports before setup() runs.
    """
    name = "cuda-library-plugin"

    def __init__(self, available_gpus=None, conda_prefix=None):
        self.available_gpus = available_gpus or []
        self.conda_prefix = conda_prefix or os.environ.get('CONDA_PREFIX', '')

    def setup(self, worker):
        """Called when the worker starts, before any tasks run."""
        # Set CUDA library paths
        if self.conda_prefix:
            import sys
            pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site_pkgs = f"{self.conda_prefix}/lib/{pyver}/site-packages"
            cuda_lib_paths = [
                f"{site_pkgs}/nvidia/cuda_nvrtc/lib",
                f"{site_pkgs}/nvidia/cuda_runtime/lib",
                f"{site_pkgs}/nvidia/cublas/lib",
                f"{site_pkgs}/nvidia/cusparse/lib",
                f"{self.conda_prefix}/lib",
            ]
            new_ld_path = ":".join(cuda_lib_paths)
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if new_ld_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = new_ld_path + ":" + current_ld_path

    def teardown(self, worker):
        """Called when the worker shuts down."""
        pass


# Per-worker registry of in-flight background pickle-write threads spawned by
# gpu_phase_worker batches. Drained via client.run(_drain_pending_writes) right
# before cluster.close(), otherwise Nanny SIGKILLs the worker ~4s after the last
# task and any in-flight pickle write is lost (silently dropping partial files).
_PENDING_WRITES: list = []


def _drain_pending_writes():
    """Block until all background pickle-write threads in this worker finish.

    Invoked via client.run() before cluster.close(). Joining a finished thread
    is a no-op, so this is safe to call even if all writes have already drained.
    """
    for t in list(_PENDING_WRITES):
        if t.is_alive():
            t.join()
    _PENDING_WRITES.clear()


# GPU-accelerated feature extraction with CPU fallback
# HYBRID MODE: Use GPU for simple cells, CPU for complex cells
_FORCE_CPU_MODE = False   # Set to True to disable GPU entirely
_USE_HYBRID_MODE = False  # Disabled - run ALL cells through GPU

_USE_GPU_FEATURES = False
# Stash the GPU import/init failure so gpu_phase_worker can hard-fail with a clear
# message. The module must still load on CPU-only nodes (for cpu_network_worker),
# so we fall back to CPU versions silently here and raise later at the GPU entry point.
_GPU_IMPORT_ERROR = None
_GPU_INIT_ERROR = None

if not _FORCE_CPU_MODE:
    try:
        from organelle_profiler.feature_extraction.morphology_features_gpu import (
            batch_extract_organelle_features_gpu as batch_extract_organelle_features,
            is_gpu_available as _morphology_gpu_available,
            get_gpu_init_error as _morphology_gpu_init_error,
        )
        from organelle_profiler.feature_extraction.network_analysis_gpu import (
            calculate_network_features_gpu as calculate_network_features,
        )
        if _morphology_gpu_available():
            _USE_GPU_FEATURES = True
            print("[GPU] Using GPU-accelerated feature extraction")
        else:
            _GPU_INIT_ERROR = _morphology_gpu_init_error() or "cp.cuda.Device(0).compute_capability failed"
    except ImportError as e:
        _GPU_IMPORT_ERROR = f"{type(e).__name__}: {e}"

if not _USE_GPU_FEATURES:
    from organelle_profiler.feature_extraction.morphology_features import (
        batch_extract_organelle_features,
    )
    from organelle_profiler.feature_extraction.network_analysis import calculate_network_features
    if _FORCE_CPU_MODE:
        print("[CPU] _FORCE_CPU_MODE=True — using CPU feature extraction")
    else:
        print("[CPU] Using CPU feature extraction (GPU stack unavailable — fine for CPU-only jobs; GPU jobs will hard-fail)")


def _require_gpu_for_feature_extraction() -> None:
    """Hard-fail if the GPU feature-extraction stack is not usable in this process.

    Called at the start of gpu_phase_worker so GPU jobs abort in seconds instead of
    silently falling back to CPU for hours.
    """
    if _FORCE_CPU_MODE:
        return  # explicit opt-out
    if _USE_GPU_FEATURES:
        return
    if _GPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            "GPU feature extraction requires cucim/cupy but import failed: "
            f"{_GPU_IMPORT_ERROR}\n"
            "  Install with: uv sync (from the repo root)\n"
            "  (or: uv pip install 'cucim-cu12>=25.6')\n"
            "  To run on CPU only, set _FORCE_CPU_MODE=True in fe_workers.py."
        )
    raise RuntimeError(
        "GPU is not available for feature extraction.\n"
        f"  cucim/cupy init error: {_GPU_INIT_ERROR}\n"
        f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
        "  Ensure this process runs on a GPU node with CUDA accessible.\n"
        "  To run on CPU only, set _FORCE_CPU_MODE=True in fe_workers.py."
    )


def _open_tensorstore_label(store_path: str, well: str, label_name: str) -> ts.TensorStore:
    """
    Open a zarr v3 label array using tensorstore (thread-safe).
    
    Parameters
    ----------
    store_path : str
        Path to the zarr store root
    well : str
        Well/position key (e.g., 'A/1/0')
    label_name : str
        Label name (e.g., 'cell_seg', 'cp1_mitochondria_tomm20_seg')
    
    Returns
    -------
    ts.TensorStore
        Tensorstore handle for the label array
    """
    label_path = f"{store_path}/{well}/labels/{label_name}/0"
    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': label_path},
    }
    return ts.open(spec).result()


def _load_label_crop_ts(ts_array: ts.TensorStore, bbox: tuple) -> np.ndarray:
    """
    Load a label crop using tensorstore (thread-safe).
    
    Parameters
    ----------
    ts_array : ts.TensorStore
        Tensorstore handle for the label array
    bbox : tuple
        Bounding box as (y_min, x_min, y_max, x_max)
    
    Returns
    -------
    np.ndarray
        2D label crop
    """
    y_min, x_min, y_max, x_max = bbox
    ndim = len(ts_array.shape)
    
    # Handle various array dimensions (labels can be 2D, 3D, 4D, or 5D)
    if ndim == 5:
        # 5D: (T, C, Z, Y, X) - zarr v3 format
        return ts_array[0, 0, 0, y_min:y_max, x_min:x_max].read().result()
    elif ndim == 4:
        # 4D: (T, C, Y, X) or (T, Z, Y, X)
        return ts_array[0, 0, y_min:y_max, x_min:x_max].read().result()
    elif ndim == 3:
        # 3D: (Z, Y, X) or (C, Y, X)
        return ts_array[0, y_min:y_max, x_min:x_max].read().result()
    else:
        # 2D: (Y, X)
        return ts_array[y_min:y_max, x_min:x_max].read().result()


# =============================================================================
# Process Pool Worker with Persistent Tensorstore Handles
# =============================================================================

# Global handles for ProcessPoolExecutor workers (initialized once per process)
_WORKER_TS_IMAGE = None
_WORKER_TS_LABELS = None


def _init_worker_tensorstore(store_path: str, well: str, available_labels: dict):
    """
    Initializer for ProcessPoolExecutor workers.
    Opens tensorstore handles once per process - reused for all cells.
    """
    global _WORKER_TS_IMAGE, _WORKER_TS_LABELS
    
    # Open main image array
    _WORKER_TS_IMAGE = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f"{store_path}/{well}/0"},
    }).result()
    
    # Open label arrays
    _WORKER_TS_LABELS = {}
    for internal_name, zarr_label_name in available_labels.items():
        try:
            _WORKER_TS_LABELS[internal_name] = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': f"{store_path}/{well}/labels/{zarr_label_name}/0"},
            }).result()
        except Exception:
            pass


def _process_cell_with_global_handles(
    cell_index: int,
    cell_dict: dict,
    static_args: dict,
    debug: bool = False,
):
    """
    Process a single cell using globally initialized tensorstore handles.
    Called by ProcessPoolExecutor workers after initialization.
    
    Supports dual-bbox system:
    - Standard cells: use y_global_pheno/x_global_pheno + cell_seg
    - CP-only cells: use cp_bbox + cp_cell_seg
    """
    global _WORKER_TS_IMAGE, _WORKER_TS_LABELS
    
    try:
        # Extract static args
        available_labels = static_args['available_labels']
        organelles_to_process = static_args['organelles_to_process']
        network_organelles = static_args['network_organelles']
        spacing = static_args['spacing']
        channel_names = static_args['channel_names']
        organelle_map = static_args['organelle_map']
        full_features = static_args['full_features']
        initial_yx_patch_size = static_args['initial_yx_patch_size']
        max_objects_per_organelle = static_args['max_objects_per_organelle']
        half_h, half_w = initial_yx_patch_size[0] // 2, initial_yx_patch_size[1] // 2
        
        # Determine which bbox system to use
        # Standard bbox: y_global_pheno/x_global_pheno (for standard organelles)
        # CP bbox: cp_bbox tuple (for CP-only cells)
        y_pheno = cell_dict.get('y_global_pheno')
        x_pheno = cell_dict.get('x_global_pheno')
        
        # Check for valid standard bbox (handle scalar vs array values)
        def _is_valid_scalar(val):
            if val is None:
                return False
            if isinstance(val, (np.ndarray, list)):
                return False  # Should be scalar
            if isinstance(val, float) and pd.isna(val):
                return False
            return True
        
        has_standard_bbox = _is_valid_scalar(y_pheno) and _is_valid_scalar(x_pheno)
        
        # Parse cp_bbox if available
        cp_bbox_raw = cell_dict.get('cp_bbox')
        cp_bbox = None
        
        # Check if cp_bbox_raw is valid (handle various types: None, NaN, string, tuple, array)
        cp_bbox_valid = False
        if cp_bbox_raw is not None:
            if isinstance(cp_bbox_raw, float):
                cp_bbox_valid = not pd.isna(cp_bbox_raw)
            elif isinstance(cp_bbox_raw, str):
                cp_bbox_valid = cp_bbox_raw not in ('None', 'nan', '')
            elif isinstance(cp_bbox_raw, (tuple, list, np.ndarray)):
                cp_bbox_valid = len(cp_bbox_raw) == 4
            else:
                cp_bbox_valid = True
        
        if cp_bbox_valid:
            if isinstance(cp_bbox_raw, (tuple, list)):
                cp_bbox = tuple(int(v) for v in cp_bbox_raw)
            elif isinstance(cp_bbox_raw, np.ndarray):
                cp_bbox = tuple(int(v) for v in cp_bbox_raw.flatten())
            elif isinstance(cp_bbox_raw, str):
                try:
                    # Parse string like "(y_min, x_min, y_max, x_max)"
                    cp_bbox = tuple(int(float(v.strip())) for v in cp_bbox_raw.strip('()[]').split(','))
                except:
                    pass
        
        has_cp_bbox = cp_bbox is not None and len(cp_bbox) == 4
        
        # Get segmentation IDs
        seg_id = cell_dict.get('segmentation_id')
        cp_seg_id = cell_dict.get('cp_cell_seg_id')
        
        # Validate seg IDs (use same helper function)
        has_seg_id = _is_valid_scalar(seg_id)
        has_cp_seg_id = _is_valid_scalar(cp_seg_id)
        
        if has_seg_id:
            seg_id = int(seg_id)
        if has_cp_seg_id:
            cp_seg_id = int(cp_seg_id)
        
        # Decide which path to use
        use_standard = has_standard_bbox and has_seg_id
        use_cp = has_cp_bbox and has_cp_seg_id
        
        # Unconditional debug for first cell
        if cell_index == 0:
            print(f"  [TRACE] Cell 0: has_std_bbox={has_standard_bbox}, has_cp_bbox={has_cp_bbox}")
            print(f"  [TRACE] Cell 0: has_seg_id={has_seg_id}, has_cp_seg_id={has_cp_seg_id}")
            print(f"  [TRACE] Cell 0: use_standard={use_standard}, use_cp={use_cp}")
            print(f"  [TRACE] Cell 0: seg_id={seg_id}, cp_seg_id={cp_seg_id}")
            print(f"  [TRACE] Cell 0: cp_bbox_raw={cp_bbox_raw}, cp_bbox={cp_bbox}")
        
        if not use_standard and not use_cp:
            if cell_index == 0:
                print(f"  [TRACE] Cell 0: RETURNING NONE - no valid bbox system")
            return None  # No valid bbox system
        
        # Primary bbox for image loading (prefer standard, fall back to CP)
        if use_standard:
            y = int(y_pheno)
            x = int(x_pheno)
            y_min = max(0, y - half_h)
            x_min = max(0, x - half_w)
            y_max = y_min + initial_yx_patch_size[0]
            x_max = x_min + initial_yx_patch_size[1]
            pheno_bbox = (y_min, x_min, y_max, x_max)
        else:
            pheno_bbox = None
        
        # Load image data at primary bbox (or CP bbox if no standard)
        if pheno_bbox is not None:
            data = _WORKER_TS_IMAGE[0, :, 0, pheno_bbox[0]:pheno_bbox[2], pheno_bbox[1]:pheno_bbox[3]].read().result()
            data = np.array(data)
            bbox = pheno_bbox
        elif cp_bbox is not None:
            data = _WORKER_TS_IMAGE[0, :, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
            data = np.array(data)
            bbox = cp_bbox
        else:
            return None
        
        # Get cell masks for both bbox systems (initialize to None first)
        cell_mask = None
        
        # Standard cell mask (at pheno_bbox)
        if use_standard and pheno_bbox is not None:
            for label_key in ['cell_mask', 'cell_seg']:
                ts_label = _WORKER_TS_LABELS.get(label_key)
                if ts_label is not None:
                    try:
                        mask_data = ts_label[0, 0, 0, pheno_bbox[0]:pheno_bbox[2], pheno_bbox[1]:pheno_bbox[3]].read().result()
                        cell_mask = (np.array(mask_data) == seg_id).astype(np.uint8)
                        if np.any(cell_mask):
                            break
                    except Exception:
                        pass
        
        # CP cell mask (at cp_bbox)
        cp_cell_mask = None
        cp_intensity = None
        if use_cp and cp_bbox is not None:
            ts_label = _WORKER_TS_LABELS.get('cp_cell_mask') or _WORKER_TS_LABELS.get('cp_cell_seg')
            if ts_label is not None:
                try:
                    mask_data = ts_label[0, 0, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
                    cp_cell_mask = (np.array(mask_data) == cp_seg_id).astype(np.uint8)
                except Exception:
                    pass
            
            # Load CP intensity at cp_bbox (cell may have moved between imaging sessions)
            try:
                cp_intensity = _WORKER_TS_IMAGE[0, :, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
                cp_intensity = np.array(cp_intensity)
            except Exception:
                pass
        
        # Must have at least one valid cell mask
        has_cell_mask = cell_mask is not None and np.any(cell_mask)
        has_cp_cell_mask = cp_cell_mask is not None and np.any(cp_cell_mask)
        
        if cell_index == 0:
            print(f"  [TRACE] Cell 0: has_cell_mask={has_cell_mask}, has_cp_cell_mask={has_cp_cell_mask}")
            print(f"  [TRACE] Cell 0: cell_mask is None={cell_mask is None}, cp_cell_mask is None={cp_cell_mask is None}")
        
        if not has_cell_mask and not has_cp_cell_mask:
            if cell_index == 0:
                print(f"  [TRACE] Cell 0: RETURNING NONE - no valid cell mask")
            return None
        
        # Read organelle labels with dual-bbox support
        organelle_mask_arrays_crop = {}
        for internal_name in available_labels.keys():
            if internal_name in ('cell_mask', 'cp_cell_mask'):
                continue
            ts_label = _WORKER_TS_LABELS.get(internal_name)
            if ts_label is None:
                continue
            
            # Determine which bbox and mask to use based on organelle type
            is_cp_organelle = internal_name.lower().startswith('cp')
            
            if is_cp_organelle:
                if not has_cp_cell_mask or cp_bbox is None:
                    continue  # Skip CP organelles if no CP mask
                bbox_to_use = cp_bbox
                mask_to_use = cp_cell_mask
            else:
                if not has_cell_mask or pheno_bbox is None:
                    continue  # Skip standard organelles if no standard mask
                bbox_to_use = pheno_bbox
                mask_to_use = cell_mask
            
            try:
                label_data = ts_label[0, 0, 0, bbox_to_use[0]:bbox_to_use[2], bbox_to_use[1]:bbox_to_use[3]].read().result()
                label_crop = np.array(label_data) * mask_to_use
                if np.any(label_crop > 0):
                    organelle_mask_arrays_crop[internal_name] = label_crop
            except Exception:
                pass
        
        # Extract nuclear mask (standard)
        nuclear_mask = None
        for nuc_key in ("nuclei", "nuclear_seg"):
            if nuc_key in organelle_mask_arrays_crop:
                nuclear_mask = (organelle_mask_arrays_crop[nuc_key] > 0).astype(np.uint8)
                break
        
        # Extract CP nuclear mask (for CP organelles)
        cp_nuclear_mask = None
        for key in organelle_mask_arrays_crop:
            if key.lower().startswith('cp') and 'nucl' in key.lower():
                cp_nuclear_mask = (organelle_mask_arrays_crop[key] > 0).astype(np.uint8)
                break
        
        # Build crop_info
        crop_info = dict(cell_dict)
        crop_info['bbox'] = bbox
        
        # Use appropriate cell mask for feature extraction
        # Prefer standard mask, fall back to CP mask for CP-only cells
        primary_cell_mask = cell_mask if has_cell_mask else cp_cell_mask
        
        if debug:
            print(f"  [DEBUG] Cell {cell_index}: organelle_mask_arrays_crop keys={list(organelle_mask_arrays_crop.keys())}")
            print(f"  [DEBUG] Cell {cell_index}: primary_cell_mask shape={primary_cell_mask.shape if primary_cell_mask is not None else None}, has_pixels={np.any(primary_cell_mask) if primary_cell_mask is not None else False}")
            print(f"  [DEBUG] Cell {cell_index}: data shape={data.shape if data is not None else None}")
        
        # Call feature extraction with dual-bbox support
        cell_features, object_features, network_features = process_single_cell(
            crop_info,
            primary_cell_mask,
            organelle_mask_arrays_crop,
            data,
            {},
            organelles_to_process,
            network_organelles,
            spacing,
            channel_names,
            organelle_map,
            full_features,
            nuclear_mask=nuclear_mask,
            cp_intensity_image=cp_intensity,
            cp_cell_mask=cp_cell_mask,
            cp_nuclear_mask=cp_nuclear_mask,
            max_objects_per_organelle=max_objects_per_organelle,
            debug=debug,
        )
        
        if cell_index == 0:
            print(f"  [TRACE] Cell 0: cell_features is None={cell_features is None}")
            if cell_features:
                print(f"  [TRACE] Cell 0: cell_features keys={list(cell_features.keys())[:5]}...")
        
        if cell_features is not None:
            return (cell_features, object_features, network_features)
        if cell_index == 0:
            print(f"  [TRACE] Cell 0: RETURNING NONE - cell_features is None")
        return None
        
    except Exception as e:
        if cell_index == 0:
            import traceback
            print(f"  [TRACE] Cell 0: Exception occurred: {e}")
            traceback.print_exc()
        return None


def _process_cell_batch_with_global_handles(
    start_idx: int,
    end_idx: int,
    cells_dict: list,
    static_args: dict,
    debug: bool = False,
):
    """
    Process a BATCH of contiguous cells using globally initialized tensorstore handles.

    This preserves cache locality - consecutive cells share zarr chunks, so
    processing them sequentially within one worker maximizes cache hits.

    Parameters
    ----------
    start_idx : int
        Starting index in cells_dict (inclusive)
    end_idx : int
        Ending index in cells_dict (exclusive)
    cells_dict : list
        List of cell dictionaries (full list, we slice it)
    static_args : dict
        Static arguments for cell processing
    debug : bool
        Enable debug output for first cell

    Returns
    -------
    list
        List of (cell_features, object_features, network_features) tuples
    """
    results = []
    for i in range(start_idx, end_idx):
        if i >= len(cells_dict):
            break
        result = _process_cell_with_global_handles(
            cell_index=i,
            cell_dict=cells_dict[i],
            static_args=static_args,
            debug=debug and i == start_idx,  # Debug only first cell in batch
        )
        if result is not None:
            results.append(result)
    return results


# =============================================================================
# Module-level network analysis function (for ProcessPoolExecutor pickling)
# =============================================================================

def _run_network_analysis_single(args):
    """
    Process a single (cell, organelle) network analysis task on CPU.
    Must be at module level for ProcessPoolExecutor pickling.
    Uses CPU-only network analysis to avoid 32 processes competing for GPU.

    Args: tuple of (local_idx, organelle_name, organelle_mask, crop_info, spacing)
    Returns: (local_idx, organelle_name, branch_df, network_summary_dict, per_object_network_df, crop_info) or None
    """
    local_idx, organelle_name, organelle_mask, crop_info, spacing = args
    try:
        from organelle_profiler.feature_extraction.network_analysis import (
            calculate_network_features,
        )
        result = calculate_network_features(
            organelle_mask,
            spacing=spacing,
            intensity_image=None,
        )
        if result:
            branch_df, network_summary_dict, per_object_network_df, task_timings = result
            return (local_idx, organelle_name, branch_df, network_summary_dict, per_object_network_df, crop_info, task_timings)
    except Exception:
        pass
    return None


def _run_network_analysis_from_zarr(args):
    """
    Process a single network analysis task by re-reading the mask from zarr.
    Used by CPU-only jobs in the split GPU/CPU mode.

    Args: dict with keys: global_cell_id, well, organelle_name, seg_label_name,
          cell_seg_label_name, bbox, seg_id, spacing_y, spacing_x, store_path
    Returns: dict with network results or None
    """
    try:
        from organelle_profiler.feature_extraction.network_analysis import (
            calculate_network_features,
        )

        store_path = args['store_path']
        well = args['well']
        organelle_name = args['organelle_name']
        seg_label_name = args['seg_label_name']
        cell_seg_label_name = args['cell_seg_label_name']
        bbox = args['bbox']
        seg_id = args['seg_id']
        spacing = (args['spacing_y'], args['spacing_x'])
        global_cell_id = args['global_cell_id']
        task_idx = args.get('task_idx', 0)

        if bbox is None or seg_id is None or seg_label_name is None:
            return None

        import numpy as _np
        y0, x0, y1, x1 = bbox

        # Read organelle segmentation mask (cached handle)
        seg_handle, seg_ndim = _get_ts_handle(store_path, well, seg_label_name)
        seg_data = _np.array(_ts_read_2d(seg_handle, seg_ndim, y0, y1, x0, x1))

        # Read cell mask to isolate this cell's organelles (cached handle)
        if cell_seg_label_name is not None:
            cell_handle, cell_ndim = _get_ts_handle(store_path, well, cell_seg_label_name)
            cell_data = _ts_read_2d(cell_handle, cell_ndim, y0, y1, x0, x1)
            cell_mask = (_np.array(cell_data) == seg_id).astype(_np.uint8)
            organelle_mask = seg_data * cell_mask
        else:
            organelle_mask = seg_data

        if not _np.any(organelle_mask > 0):
            return None

        result = calculate_network_features(
            organelle_mask,
            spacing=spacing,
            intensity_image=None,
        )
        if result:
            branch_df, network_summary_dict, per_object_network_df, task_timings = result
            return {
                'task_idx': task_idx,
                'global_cell_id': global_cell_id,
                'organelle_name': organelle_name,
                'branch_df': branch_df,
                'network_summary_dict': network_summary_dict,
                'per_object_network_df': per_object_network_df,
                'task_timings': task_timings,
            }
    except Exception:
        pass
    return None


_morph_ts_cache = {}  # Cache tensorstore handles: (store_path, well, label) -> (ts_handle, ndim)


def _get_ts_handle(store_path, well, label_name):
    """Get or create a cached tensorstore handle."""
    import tensorstore as ts
    key = (store_path, well, label_name)
    if key not in _morph_ts_cache:
        handle = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': f"{store_path}/{well}/labels/{label_name}/0"},
        }).result()
        _morph_ts_cache[key] = (handle, len(handle.shape))
    return _morph_ts_cache[key]


def _ts_read_2d(handle, ndim, y0, y1, x0, x1):
    """Read a 2D slice from a tensorstore handle of any dimensionality."""
    import numpy as _np
    if ndim == 5:
        data = handle[0, 0, 0, y0:y1, x0:x1].read().result()
    elif ndim == 4:
        data = handle[0, 0, y0:y1, x0:x1].read().result()
    elif ndim == 3:
        data = handle[0, y0:y1, x0:x1].read().result()
    else:
        data = handle[y0:y1, x0:x1].read().result()
    return _np.array(data)


def _run_morph_supplement_from_zarr(args):
    """
    Compute 3 morphology properties that the GPU phase skips (too slow on GPU):
    perimeter, perimeter_crofton, euler_number.

    Uses cached tensorstore handles to avoid repeated ts.open() overhead.

    Args: dict with keys: global_cell_id, well, organelle_name, seg_label_name,
          cell_seg_label_name, bbox, seg_id, spacing_y, spacing_x, store_path
    Returns: dict with global_cell_id, organelle_name, morph_supplement_df,
             and timing breakdown (t_io, t_compute, bbox_area) or None
    """
    try:
        import time as _time
        import numpy as _np
        from skimage.measure import regionprops_table

        store_path = args['store_path']
        well = args['well']
        organelle_name = args['organelle_name']
        seg_label_name = args['seg_label_name']
        cell_seg_label_name = args['cell_seg_label_name']
        bbox = args['bbox']
        seg_id = args['seg_id']
        spacing = (args['spacing_y'], args['spacing_x'])
        global_cell_id = args['global_cell_id']

        if bbox is None or seg_id is None or seg_label_name is None:
            return None

        y0, x0, y1, x1 = bbox
        bbox_area = (y1 - y0) * (x1 - x0)

        # Read organelle segmentation mask (cached handle)
        t_io_start = _time.time()
        seg_handle, seg_ndim = _get_ts_handle(store_path, well, seg_label_name)
        seg_data = _ts_read_2d(seg_handle, seg_ndim, y0, y1, x0, x1)

        # Read cell mask to isolate this cell's organelles (cached handle)
        if cell_seg_label_name is not None:
            cell_handle, cell_ndim = _get_ts_handle(store_path, well, cell_seg_label_name)
            cell_data = _ts_read_2d(cell_handle, cell_ndim, y0, y1, x0, x1)
            cell_mask = (cell_data == seg_id).astype(_np.uint8)
            organelle_mask = seg_data * cell_mask
        else:
            organelle_mask = seg_data
        t_io = _time.time() - t_io_start

        nonzero = _np.nonzero(organelle_mask)
        if len(nonzero[0]) == 0:
            return None

        # Crop to tight bbox of this cell's organelle pixels
        organelle_mask = organelle_mask[
            nonzero[0].min():nonzero[0].max() + 1,
            nonzero[1].min():nonzero[1].max() + 1,
        ]

        # Compute the 3 properties that GPU skips (convex hull props removed for speed)
        t_compute_start = _time.time()
        props = regionprops_table(
            organelle_mask.astype(_np.int32),
            spacing=spacing,
            properties=('label', 'perimeter', 'perimeter_crofton', 'euler_number'),
        )
        t_compute = _time.time() - t_compute_start

        if not props or len(props.get('label', [])) == 0:
            return None

        import pandas as _pd
        morph_df = _pd.DataFrame(props)

        return {
            'global_cell_id': global_cell_id,
            'organelle_name': organelle_name,
            'morph_supplement_df': morph_df,
            't_io': t_io,
            't_compute': t_compute,
            'bbox_area': bbox_area,
        }
    except Exception:
        pass
    return None


def _dask_process_cell_batch(
    batch_start: int,
    batch_end: int,
    cells_dict: list,
    store_path: str,
    well: str,
    available_labels: dict,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    channel_names: list,
    organelle_map: dict,
    full_features: bool,
    initial_yx_patch_size: tuple,
    max_objects_per_organelle: int,
    debug: bool = False,
    gpu_only: bool = False,
    available_gpus: list = None,
    partials_dir: str = None,
    batch_idx: int = 0,
):
    """
    Dask-compatible worker function to process a batch of cells.

    This function is designed for Dask LocalCluster which uses spawn mode,
    properly initializing CUDA contexts in each worker process.

    Each worker opens its own tensorstore handles (no global state).

    Parameters
    ----------
    batch_start : int
        Starting index in cells_dict (inclusive)
    batch_end : int
        Ending index in cells_dict (exclusive)
    cells_dict : list
        List of cell dictionaries (full list, we slice it)
    store_path : str
        Path to the zarr store
    well : str
        Well identifier
    available_labels : dict
        Mapping of internal organelle names to zarr label names
    organelles_to_process : list
        List of organelle names to extract features for
    network_organelles : list
        List of organelle names for network analysis
    spacing : tuple
        Pixel spacing (y, x)
    channel_names : list
        List of channel names
    organelle_map : dict
        Mapping of organelle names to channel names
    full_features : bool
        Whether to compute expensive features
    initial_yx_patch_size : tuple
        Size of patches to extract around cells
    max_objects_per_organelle : int
        Maximum objects per organelle for downsampling
    debug : bool
        Enable debug output

    Returns
    -------
    list
        List of (cell_features, object_features, network_features) tuples
    """
    # Suppress warnings in worker
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*cuda.cudart.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*regions with <=1 background pixel spacing.*")
    warnings.filterwarnings("ignore", message="Input image is entirely zero")

    import os
    import sys

    # Use stderr for worker messages - it's unbuffered and bypasses Dask's stdout capture
    # Only print routine messages when debug=True; always print warnings/errors
    def log(msg, important=False):
        if debug or important:
            print(msg, file=sys.stderr)

    def warn(msg):
        """Always print warnings/errors regardless of debug setting."""
        print(msg, file=sys.stderr)

    worker_pid = os.getpid()

    # CRITICAL: Set LD_LIBRARY_PATH for CUDA libraries in conda environment
    # Spawned workers don't inherit this, causing "libnvrtc.so.12 not found" errors
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    cuda_lib_paths = [
        f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cuda_nvrtc/lib",
        f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cuda_runtime/lib",
        f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cublas/lib",
        f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cusparse/lib",
        f"{conda_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cufft/lib",
        f"{conda_prefix}/lib",
    ]
    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = ':'.join([p for p in cuda_lib_paths if os.path.isdir(p)])
    if new_paths:
        os.environ['LD_LIBRARY_PATH'] = f"{new_paths}:{existing_ld_path}" if existing_ld_path else new_paths

    # Import feature extraction functions - GPU detection happens fresh in each worker
    # This is critical: Dask workers are spawned (not forked), so CUDA contexts
    # are properly initialized in each worker process

    import time as time_module
    worker_start_time = time_module.time()

    # Always log worker startup for debugging
    log(f"  [Worker {worker_pid}] Starting batch {batch_start}-{batch_end} ({batch_end - batch_start} cells)")

    try:
        t_import_start = time_module.time()
        from organelle_profiler.feature_extraction.morphology_features_gpu import (
            batch_extract_organelle_features_gpu,
            batch_extract_features_gpu_multicell,
            compute_localization_gpu_batched,
            is_gpu_available,
        )
        from organelle_profiler.feature_extraction.network_analysis_gpu import (
            calculate_network_features_gpu,
        )
        t_import = time_module.time() - t_import_start

        t_gpu_check_start = time_module.time()
        use_gpu = is_gpu_available()
        t_gpu_check = time_module.time() - t_gpu_check_start

        if use_gpu:
            batch_extract_func = batch_extract_organelle_features_gpu
            multicell_batch_func = batch_extract_features_gpu_multicell
            localization_gpu_func = compute_localization_gpu_batched
            network_func = calculate_network_features_gpu
            log(f"  [Worker {worker_pid}] GPU enabled with multicell batching + GPU localization (import: {t_import:.2f}s, check: {t_gpu_check:.2f}s)")
        else:
            from organelle_profiler.feature_extraction.morphology_features import (
                batch_extract_organelle_features,
            )
            from organelle_profiler.feature_extraction.network_analysis import (
                calculate_network_features,
            )
            batch_extract_func = batch_extract_organelle_features
            multicell_batch_func = None
            localization_gpu_func = None
            network_func = calculate_network_features
            warn(f"  [Worker {worker_pid}] GPU NOT available, using CPU (import: {t_import:.2f}s)")
    except ImportError as e:
        from organelle_profiler.feature_extraction.morphology_features import (
            batch_extract_organelle_features,
        )
        from organelle_profiler.feature_extraction.network_analysis import (
            calculate_network_features,
        )
        batch_extract_func = batch_extract_organelle_features
        multicell_batch_func = None
        localization_gpu_func = None
        network_func = calculate_network_features
        use_gpu = False
        warn(f"  [Worker {worker_pid}] GPU import FAILED: {e}")

    # Import other required modules
    from organelle_profiler.feature_extraction.localization_features import (
        precompute_boundary_kdtrees,
        compute_localization_kdtree,
        compute_cell_level_localization_summary,
    )

    # Select GPU device for this worker (multi-GPU support)
    # This must happen BEFORE any cupy allocations but AFTER imports.
    # We use cupy.cuda.Device().use() instead of CUDA_VISIBLE_DEVICES because
    # the env var must be set before CUDA runtime init (which happens at import time).
    if use_gpu and available_gpus and len(available_gpus) > 1:
        try:
            from dask.distributed import get_worker
            _worker = get_worker()
            _worker_name = _worker.name
            if isinstance(_worker_name, int):
                _gpu_id = _worker_name % len(available_gpus)
                import cupy as _cp
                _cp.cuda.Device(_gpu_id).use()
                log(f"  [Worker {worker_pid}] Selected GPU {_gpu_id} (worker_name={_worker_name}, {len(available_gpus)} GPUs available)")
            else:
                warn(f"  [Worker {worker_pid}] Worker name not int ({_worker_name}), using default GPU")
        except Exception as _gpu_err:
            warn(f"  [Worker {worker_pid}] GPU device selection failed: {_gpu_err}, using default GPU")

    # Open tensorstore handles for this worker
    t_ts_start = time_module.time()
    ts_image = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f"{store_path}/{well}/0"},
    }).result()

    ts_labels = {}
    for internal_name, zarr_label_name in available_labels.items():
        try:
            ts_labels[internal_name] = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': f"{store_path}/{well}/labels/{zarr_label_name}/0"},
            }).result()
        except Exception:
            pass
    t_ts_open = time_module.time() - t_ts_start
    log(f"  [Worker {worker_pid}] Opened tensorstore ({len(ts_labels)} labels) in {t_ts_open:.2f}s")

    half_h, half_w = initial_yx_patch_size[0] // 2, initial_yx_patch_size[1] // 2
    results = []

    # Timing accumulators for the batch
    total_io_time = 0.0
    total_gpu_time = 0.0
    total_loc_time = 0.0
    total_net_time = 0.0
    cells_processed = 0
    cells_skipped = 0

    batch_loop_start = time_module.time()
    n_cells_in_batch = min(batch_end, len(cells_dict)) - batch_start

    # Helper function for scalar validation
    def _is_valid_scalar(val):
        if val is None:
            return False
        if isinstance(val, (np.ndarray, list)):
            return False
        if isinstance(val, float) and pd.isna(val):
            return False
        return True

    # ==========================================================================
    # PHASE 1: Load all cell data upfront (I/O phase) - PARALLEL
    # ==========================================================================
    t_io_start = time_module.time()

    # Helper function to load a single cell (for parallel execution)
    def _load_single_cell(cell_idx):
        """Load all data for a single cell. Returns dict or None on failure."""
        cell_dict = cells_dict[cell_idx]
        try:
            # Determine which bbox system to use
            y_pheno = cell_dict.get('y_global_pheno')
            x_pheno = cell_dict.get('x_global_pheno')
            has_standard_bbox = _is_valid_scalar(y_pheno) and _is_valid_scalar(x_pheno)

            # Parse cp_bbox if available
            cp_bbox_raw = cell_dict.get('cp_bbox')
            cp_bbox = None
            cp_bbox_valid = False
            if cp_bbox_raw is not None:
                if isinstance(cp_bbox_raw, float):
                    cp_bbox_valid = not pd.isna(cp_bbox_raw)
                elif isinstance(cp_bbox_raw, str):
                    cp_bbox_valid = cp_bbox_raw not in ('None', 'nan', '')
                elif isinstance(cp_bbox_raw, (tuple, list, np.ndarray)):
                    cp_bbox_valid = len(cp_bbox_raw) == 4
                else:
                    cp_bbox_valid = True

            if cp_bbox_valid:
                if isinstance(cp_bbox_raw, (tuple, list)):
                    cp_bbox = tuple(int(v) for v in cp_bbox_raw)
                elif isinstance(cp_bbox_raw, np.ndarray):
                    cp_bbox = tuple(int(v) for v in cp_bbox_raw.flatten())
                elif isinstance(cp_bbox_raw, str):
                    try:
                        cp_bbox = tuple(int(float(v.strip())) for v in cp_bbox_raw.strip('()[]').split(','))
                    except:
                        pass

            has_cp_bbox = cp_bbox is not None and len(cp_bbox) == 4

            # Get segmentation IDs
            seg_id = cell_dict.get('segmentation_id')
            cp_seg_id = cell_dict.get('cp_cell_seg_id')
            has_seg_id = _is_valid_scalar(seg_id)
            has_cp_seg_id = _is_valid_scalar(cp_seg_id)

            if has_seg_id:
                seg_id = int(seg_id)
            if has_cp_seg_id:
                cp_seg_id = int(cp_seg_id)

            use_standard = has_standard_bbox and has_seg_id
            use_cp = has_cp_bbox and has_cp_seg_id

            if not use_standard and not use_cp:
                return None

            # Calculate bboxes
            if use_standard:
                y = int(y_pheno)
                x = int(x_pheno)
                y_min = max(0, y - half_h)
                x_min = max(0, x - half_w)
                y_max = y_min + initial_yx_patch_size[0]
                x_max = x_min + initial_yx_patch_size[1]
                pheno_bbox = (y_min, x_min, y_max, x_max)
            else:
                pheno_bbox = None

            # Load image data
            if pheno_bbox is not None:
                data = ts_image[0, :, 0, pheno_bbox[0]:pheno_bbox[2], pheno_bbox[1]:pheno_bbox[3]].read().result()
                data = np.array(data)
                bbox = pheno_bbox
            elif cp_bbox is not None:
                data = ts_image[0, :, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
                data = np.array(data)
                bbox = cp_bbox
            else:
                return None

            # Get cell masks
            cell_mask = None
            if use_standard and pheno_bbox is not None:
                for label_key in ['cell_mask', 'cell_seg']:
                    ts_label = ts_labels.get(label_key)
                    if ts_label is not None:
                        try:
                            mask_data = ts_label[0, 0, 0, pheno_bbox[0]:pheno_bbox[2], pheno_bbox[1]:pheno_bbox[3]].read().result()
                            cell_mask = (np.array(mask_data) == seg_id).astype(np.uint8)
                            if np.any(cell_mask):
                                break
                        except Exception:
                            pass

            # CP cell mask
            cp_cell_mask = None
            cp_intensity = None
            if use_cp and cp_bbox is not None:
                ts_label = ts_labels.get('cp_cell_mask') or ts_labels.get('cp_cell_seg')
                if ts_label is not None:
                    try:
                        mask_data = ts_label[0, 0, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
                        cp_cell_mask = (np.array(mask_data) == cp_seg_id).astype(np.uint8)
                    except Exception:
                        pass

                try:
                    cp_intensity = ts_image[0, :, 0, cp_bbox[0]:cp_bbox[2], cp_bbox[1]:cp_bbox[3]].read().result()
                    cp_intensity = np.array(cp_intensity)
                except Exception:
                    pass

            has_cell_mask = cell_mask is not None and np.any(cell_mask)
            has_cp_cell_mask = cp_cell_mask is not None and np.any(cp_cell_mask)

            if not has_cell_mask and not has_cp_cell_mask:
                return None

            # Read organelle labels
            organelle_mask_arrays_crop = {}
            for internal_name in available_labels.keys():
                if internal_name in ('cell_mask', 'cp_cell_mask'):
                    continue
                ts_label = ts_labels.get(internal_name)
                if ts_label is None:
                    continue

                is_cp_organelle = internal_name.lower().startswith('cp')
                if is_cp_organelle:
                    if not has_cp_cell_mask or cp_bbox is None:
                        continue
                    bbox_to_use = cp_bbox
                    mask_to_use = cp_cell_mask
                else:
                    if not has_cell_mask or pheno_bbox is None:
                        continue
                    bbox_to_use = pheno_bbox
                    mask_to_use = cell_mask

                try:
                    label_data = ts_label[0, 0, 0, bbox_to_use[0]:bbox_to_use[2], bbox_to_use[1]:bbox_to_use[3]].read().result()
                    label_crop = np.array(label_data) * mask_to_use
                    if np.any(label_crop > 0):
                        organelle_mask_arrays_crop[internal_name] = label_crop
                except Exception:
                    pass

            # Extract nuclear masks
            nuclear_mask = None
            for nuc_key in ("nuclei", "nuclear_seg"):
                if nuc_key in organelle_mask_arrays_crop:
                    nuclear_mask = (organelle_mask_arrays_crop[nuc_key] > 0).astype(np.uint8)
                    break

            cp_nuclear_mask = None
            for key in organelle_mask_arrays_crop:
                if key.lower().startswith('cp') and 'nucl' in key.lower():
                    cp_nuclear_mask = (organelle_mask_arrays_crop[key] > 0).astype(np.uint8)
                    break

            # Build crop_info
            crop_info = dict(cell_dict)
            crop_info['bbox'] = bbox
            primary_cell_mask = cell_mask if has_cell_mask else cp_cell_mask

            # Return all data for this cell
            return {
                'cell_idx': cell_idx,
                'crop_info': crop_info,
                'cell_mask': cell_mask,
                'cp_cell_mask': cp_cell_mask,
                'primary_cell_mask': primary_cell_mask,
                'nuclear_mask': nuclear_mask,
                'cp_nuclear_mask': cp_nuclear_mask,
                'organelle_masks': organelle_mask_arrays_crop,
                'data': data,
                'cp_intensity': cp_intensity,
                'pheno_bbox': pheno_bbox,
                'cp_bbox': cp_bbox,
            }

        except Exception as e:
            if debug:
                import traceback
                warn(f"  [Worker {worker_pid}] Cell {cell_idx} I/O error: {e}")
                traceback.print_exc()
            return None

    # Parallel I/O using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cell_indices = list(range(batch_start, min(batch_end, len(cells_dict))))
    _io_thread_cap = int(os.environ.get('_FE_IO_THREADS', '4'))
    n_io_threads = min(_io_thread_cap, len(cell_indices))  # Configurable I/O threads per worker

    loaded_cells = []
    with ThreadPoolExecutor(max_workers=n_io_threads) as executor:
        # Submit all cell loading tasks
        futures = {executor.submit(_load_single_cell, idx): idx for idx in cell_indices}

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                loaded_cells.append(result)
            else:
                cells_skipped += 1

    # Sort by cell_idx to maintain order (optional but good for reproducibility)
    loaded_cells.sort(key=lambda x: x['cell_idx'])

    total_io_time = time_module.time() - t_io_start
    log(f"  [Worker {worker_pid}] I/O phase: loaded {len(loaded_cells)} cells in {total_io_time:.1f}s (parallel, {n_io_threads} threads)")

    if not loaded_cells:
        return results

    # ==========================================================================
    # PHASE 2: GPU morphology extraction (multicell batched if available)
    # ==========================================================================
    t_gpu_start = time_module.time()

    # Organize masks by organelle for multicell batching
    # Structure: {organelle_name: [cell0_mask, cell1_mask, ...]}
    cell_masks_by_organelle = {}
    cell_intensities_by_organelle = {}

    for organelle_name in organelles_to_process:
        masks_for_organelle = []
        intensities_for_organelle = []

        for cell_data in loaded_cells:
            organelle_masks = cell_data['organelle_masks']
            data = cell_data['data']
            cp_intensity = cell_data['cp_intensity']

            # Get mask for this organelle
            if organelle_name == "cell_membrane":
                mask = cell_data['primary_cell_mask']
            else:
                mask = organelle_masks.get(organelle_name)

            masks_for_organelle.append(mask)

            # Get intensity image
            is_cp_organelle = organelle_name.lower().startswith("cp")
            if is_cp_organelle and cp_intensity is not None:
                if cp_intensity.ndim >= 3 and cp_intensity.shape[0] > 1:
                    intensity = np.mean(cp_intensity, axis=0)
                elif cp_intensity.ndim >= 3:
                    intensity = cp_intensity[0]
                else:
                    intensity = cp_intensity
            elif data is not None:
                if data.ndim >= 3 and data.shape[0] > 1:
                    intensity = np.mean(data, axis=0)
                elif data.ndim >= 3:
                    intensity = data[0]
                else:
                    intensity = data
            else:
                intensity = None

            intensities_for_organelle.append(intensity)

        cell_masks_by_organelle[organelle_name] = masks_for_organelle
        cell_intensities_by_organelle[organelle_name] = intensities_for_organelle

    # Downsample if needed (apply to each cell's masks)
    if max_objects_per_organelle is not None and max_objects_per_organelle > 0:
        for org_name in cell_masks_by_organelle:
            for i, mask in enumerate(cell_masks_by_organelle[org_name]):
                if mask is not None:
                    n_objects = len(np.unique(mask)) - 1
                    if n_objects > max_objects_per_organelle:
                        seed = loaded_cells[i]['crop_info'].get("total_index", 0)
                        cell_masks_by_organelle[org_name][i] = _downsample_labeled_mask_simple(
                            mask, max_objects_per_organelle, seed=seed
                        )

    # Try multicell batch processing if GPU available
    multicell_results = None
    multicell_timing = {}

    log(f"  [Worker {worker_pid}] Phase 2: Starting GPU morphology extraction (use_gpu={use_gpu}, multicell_func={multicell_batch_func is not None})")

    if use_gpu and multicell_batch_func is not None:
        try:
            log(f"  [Worker {worker_pid}] Calling multicell_batch_func with {len(organelles_to_process)} organelles, {len(loaded_cells)} cells")
            multicell_results, multicell_timing = multicell_batch_func(
                cell_masks_by_organelle,
                cell_intensities_by_organelle,
                spacing,
                device_id=0,
            )
            if multicell_results is not None:
                log(f"  [Worker {worker_pid}] Multicell GPU batch: {len(loaded_cells)} cells processed in {time_module.time() - t_gpu_start:.1f}s")
            else:
                warn(f"  [Worker {worker_pid}] Multicell GPU batch returned None")
        except Exception as e:
            import traceback
            warn(f"  [Worker {worker_pid}] Multicell batch failed: {e}")
            traceback.print_exc()
            multicell_results = None

    # GPU-batched localization (distance transforms on montage)
    gpu_localization_results = None
    if use_gpu and localization_gpu_func is not None and multicell_results is not None:
        try:
            # Collect cell masks and nuclear masks for all loaded cells
            loc_cell_masks = [cd['primary_cell_mask'] for cd in loaded_cells]
            loc_nuclear_masks = [cd['nuclear_mask'] for cd in loaded_cells]

            gpu_localization_results, loc_timing = localization_gpu_func(
                cell_masks=loc_cell_masks,
                nuclear_masks=loc_nuclear_masks,
                organelle_masks_by_organelle=cell_masks_by_organelle,
                spacing=spacing,
                organelles_to_process=organelles_to_process,
                device_id=0,
            )
            if gpu_localization_results is not None:
                log(f"  [Worker {worker_pid}] GPU localization: {len(loaded_cells)} cells in {loc_timing.get('total_ms', 0):.0f}ms")
        except Exception as e:
            import traceback
            warn(f"  [Worker {worker_pid}] GPU localization failed: {e}")
            traceback.print_exc()
            gpu_localization_results = None

    total_gpu_time = time_module.time() - t_gpu_start

    # ==========================================================================
    # PHASE 3a: Per-cell feature assembly (fast, sequential)
    # ==========================================================================
    t_percell_start = time_module.time()

    # Pre-allocate per-cell containers
    all_cell_features = []
    all_object_features = []
    all_network_features = []  # Will be populated by parallel network phase

    # Collect network work items for parallel processing
    network_work_items = []  # (local_idx, organelle_name, organelle_mask)
    # Collect morph supplement work items (all organelles, not just network)
    morph_supplement_work_items = []

    # Per-step timing accumulators for Phase 3a profiling
    _p3a_cell_morph_time = 0.0
    _p3a_obj_features_time = 0.0
    _p3a_localization_time = 0.0
    _p3a_network_collect_time = 0.0
    _p3a_loc_gpu_count = 0
    _p3a_loc_cpu_count = 0
    _p3a_slow_cells = []  # Track cells taking > 1s

    for local_idx, cell_data in enumerate(loaded_cells):
        try:
            crop_info = cell_data['crop_info']
            primary_cell_mask = cell_data['primary_cell_mask']
            nuclear_mask = cell_data['nuclear_mask']
            cp_cell_mask = cell_data['cp_cell_mask']
            cp_nuclear_mask = cell_data['cp_nuclear_mask']
            organelle_masks = cell_data['organelle_masks']
            data = cell_data['data']
            cp_intensity = cell_data['cp_intensity']

            _t_cell_start = time_module.time()

            # Base features
            base_features = {
                "cell_id": crop_info.get("global_cell_id"),
                "well": crop_info.get("well"),
            }

            # Cell morphology (simple CPU extraction)
            _t_step = time_module.time()
            cell_morphology = _extract_cell_features_simple(primary_cell_mask, spacing)
            cell_features = {**base_features, **cell_morphology}

            # CP cell morphology if available
            if cp_cell_mask is not None and np.any(cp_cell_mask > 0):
                cp_cell_morphology = _extract_cell_features_simple(cp_cell_mask, spacing)
                for key, value in cp_cell_morphology.items():
                    cp_key = key.replace("cell_", "cp_cell_", 1)
                    cell_features[cp_key] = value
            _p3a_cell_morph_time += time_module.time() - _t_step

            object_features_for_cell = {}

            # Get morphology features from multicell batch or fallback to per-cell
            _t_step = time_module.time()
            if multicell_results is not None and local_idx < len(multicell_results):
                batch_features = multicell_results[local_idx]
                for organelle_name, df in batch_features.items():
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["cell_id"] = crop_info.get("global_cell_id")
                        df["total_index"] = crop_info.get("total_index")
                        object_features_for_cell[organelle_name] = df
            else:
                # Fallback: per-cell GPU/CPU extraction
                organelles_with_masks = {}
                intensity_images_dict = {}

                for organelle_name in organelles_to_process:
                    if organelle_name == "cell_membrane":
                        organelles_with_masks[organelle_name] = primary_cell_mask
                    else:
                        mask = organelle_masks.get(organelle_name)
                        if mask is not None:
                            organelles_with_masks[organelle_name] = mask

                    is_cp_organelle = organelle_name.lower().startswith("cp")
                    if is_cp_organelle and cp_intensity is not None:
                        if cp_intensity.ndim >= 3 and cp_intensity.shape[0] > 1:
                            intensity_images_dict[organelle_name] = np.mean(cp_intensity, axis=0)
                        elif cp_intensity.ndim >= 3:
                            intensity_images_dict[organelle_name] = cp_intensity[0]
                        else:
                            intensity_images_dict[organelle_name] = cp_intensity
                    elif data is not None:
                        if data.ndim >= 3 and data.shape[0] > 1:
                            intensity_images_dict[organelle_name] = np.mean(data, axis=0)
                        elif data.ndim >= 3:
                            intensity_images_dict[organelle_name] = data[0]
                        else:
                            intensity_images_dict[organelle_name] = data

                if organelles_with_masks:
                    batch_features, _ = batch_extract_func(
                        organelles_with_masks,
                        spacing,
                        intensity_images=intensity_images_dict,
                        profile_properties=False,
                        profile_organelle=None,
                    )
                    for organelle_name, df in batch_features.items():
                        if not df.empty:
                            df["cell_id"] = crop_info.get("global_cell_id")
                            df["total_index"] = crop_info.get("total_index")
                            object_features_for_cell[organelle_name] = df
            _p3a_obj_features_time += time_module.time() - _t_step

            # Localization: use GPU-batched results if available, else fall back to KDTree
            _t_step = time_module.time()
            has_gpu_loc = (gpu_localization_results is not None
                           and local_idx < len(gpu_localization_results)
                           and gpu_localization_results[local_idx])

            if has_gpu_loc:
                cell_features.update(gpu_localization_results[local_idx])
                _p3a_loc_gpu_count += 1
            else:
                _p3a_loc_cpu_count += 1
                tree_cache_standard = precompute_boundary_kdtrees(
                    cell_mask=primary_cell_mask,
                    nuclear_mask=nuclear_mask,
                    spacing=spacing,
                )
                tree_cache_cp = None
                if cp_cell_mask is not None:
                    cp_nuc = cp_nuclear_mask if cp_nuclear_mask is not None else nuclear_mask
                    tree_cache_cp = precompute_boundary_kdtrees(
                        cell_mask=cp_cell_mask,
                        nuclear_mask=cp_nuc,
                        spacing=spacing,
                    )

                for organelle_name in organelles_to_process:
                    if organelle_name in ("nuclei", "nuclear_seg", "cell_membrane"):
                        continue
                    if organelle_name == "cell_membrane":
                        organelle_mask = primary_cell_mask
                    else:
                        organelle_mask = organelle_masks.get(organelle_name)
                    if organelle_mask is None or not np.any(organelle_mask > 0):
                        continue

                    is_cp_organelle = organelle_name.lower().startswith("cp")
                    tree_cache = tree_cache_cp if (is_cp_organelle and tree_cache_cp is not None) else tree_cache_standard

                    try:
                        localization_df = compute_localization_kdtree(
                            organelle_mask=organelle_mask,
                            tree_cache=tree_cache,
                            spacing=spacing,
                        )
                        if not localization_df.empty:
                            loc_summary = compute_cell_level_localization_summary(
                                localization_df, organelle_name
                            )
                            cell_features.update(loc_summary)
                    except Exception:
                        pass
            _p3a_localization_time += time_module.time() - _t_step

            # Inter-organelle contact features (cheap pairwise mask overlap;
            # radial distribution is emitted for free by the localization summary).
            # NOT guarded: a failure here must surface, not silently drop the group.
            from organelle_profiler.feature_extraction.spatial_features import (
                compute_spatial_features,
            )
            cell_features.update(compute_spatial_features(organelle_masks, spacing))

            # Collect network work items (will be processed in parallel below)
            _t_step = time_module.time()
            for organelle_name in network_organelles:
                organelle_mask = organelle_masks.get(organelle_name)
                if organelle_mask is not None and np.any(organelle_mask > 0):
                    if gpu_only:
                        # GPU-only mode: emit metadata for CPU job to re-read masks from zarr
                        # Determine which bbox was used for this organelle
                        is_cp_organelle = organelle_name.lower().startswith('cp')
                        if is_cp_organelle:
                            bbox_to_use = cell_data.get('cp_bbox')
                            seg_id_to_use = crop_info.get('cp_cell_seg_id')
                        else:
                            bbox_to_use = cell_data.get('pheno_bbox')
                            seg_id_to_use = crop_info.get('segmentation_id')
                        network_work_items.append({
                            'local_idx': local_idx,
                            'organelle_name': organelle_name,
                            'global_cell_id': crop_info.get('global_cell_id'),
                            'well': well,
                            'position': crop_info.get('position', well),
                            'bbox': bbox_to_use,
                            'seg_id': int(seg_id_to_use) if seg_id_to_use is not None else None,
                            'spacing_y': spacing[0],
                            'spacing_x': spacing[1],
                            'seg_label_name': available_labels.get(organelle_name),
                            'cell_seg_label_name': available_labels.get('cp_cell_mask' if is_cp_organelle else 'cell_mask',
                                                                        available_labels.get('cp_cell_seg' if is_cp_organelle else 'cell_seg')),
                        })
                    else:
                        # Combined mode: pass masks directly (current behavior)
                        network_work_items.append((local_idx, organelle_name, organelle_mask, crop_info, spacing))

            # Collect morph supplement work items for ALL organelles (GPU skips these 5 props)
            # Only collect morph supplement tasks when full_features=True
            # These compute expensive properties: perimeter, perimeter_crofton, euler_number
            if gpu_only and full_features:
                for organelle_name in organelles_to_process:
                    organelle_mask = organelle_masks.get(organelle_name)
                    if organelle_name == "cell_membrane":
                        organelle_mask = cell_data['primary_cell_mask']
                    if organelle_mask is not None and np.any(organelle_mask > 0):
                        is_cp_organelle = organelle_name.lower().startswith('cp')
                        if is_cp_organelle:
                            bbox_to_use = cell_data.get('cp_bbox')
                            seg_id_to_use = crop_info.get('cp_cell_seg_id')
                        else:
                            bbox_to_use = cell_data.get('pheno_bbox')
                            seg_id_to_use = crop_info.get('segmentation_id')
                        morph_supplement_work_items.append({
                            'local_idx': local_idx,
                            'organelle_name': organelle_name,
                            'global_cell_id': crop_info.get('global_cell_id'),
                            'well': well,
                            'position': crop_info.get('position', well),
                            'bbox': bbox_to_use,
                            'seg_id': int(seg_id_to_use) if seg_id_to_use is not None else None,
                            'spacing_y': spacing[0],
                            'spacing_x': spacing[1],
                            'seg_label_name': available_labels.get(organelle_name),
                            'cell_seg_label_name': available_labels.get('cp_cell_mask' if is_cp_organelle else 'cell_mask',
                                                                        available_labels.get('cp_cell_seg' if is_cp_organelle else 'cell_seg')),
                        })

            _p3a_network_collect_time += time_module.time() - _t_step

            # Track slow cells (> 1s total)
            _t_cell_elapsed = time_module.time() - _t_cell_start
            if _t_cell_elapsed > 1.0:
                _p3a_slow_cells.append((local_idx, crop_info.get("global_cell_id"), _t_cell_elapsed))

            all_cell_features.append(cell_features)
            all_object_features.append(object_features_for_cell)
            all_network_features.append({})  # Placeholder, filled by network phase
            cells_processed += 1

        except Exception as e:
            cells_skipped += 1
            all_cell_features.append({"cell_id": cell_data.get('crop_info', {}).get("global_cell_id"), "well": cell_data.get('crop_info', {}).get("well")})
            all_object_features.append({})
            all_network_features.append({})
            if debug:
                import traceback
                warn(f"  [Worker {worker_pid}] Cell processing error: {e}")
                traceback.print_exc()
            continue

    t_phase3a = time_module.time() - t_percell_start
    log(f"  [Worker {worker_pid}] Phase 3a (feature assembly): {cells_processed} cells in {t_phase3a:.1f}s")
    log(f"    cell_morph={_p3a_cell_morph_time:.1f}s  obj_features={_p3a_obj_features_time:.1f}s  localization={_p3a_localization_time:.1f}s  network_collect={_p3a_network_collect_time:.1f}s")
    # Only warn about CPU fallbacks (GPU fallback indicates potential issue)
    if _p3a_loc_cpu_count > 0:
        warn(f"  [Worker {worker_pid}] localization: {_p3a_loc_gpu_count} GPU, {_p3a_loc_cpu_count} CPU fallback")
    else:
        log(f"    localization: {_p3a_loc_gpu_count} GPU, {_p3a_loc_cpu_count} CPU fallback")
    if _p3a_slow_cells:
        _p3a_slow_cells.sort(key=lambda x: x[2], reverse=True)
        top_slow = _p3a_slow_cells[:5]
        warn(f"  [Worker {worker_pid}] slow cells (>{1.0}s): {len(_p3a_slow_cells)} total, top 5: {[(cid, f'{t:.1f}s') for _, cid, t in top_slow]}")

    # Build results list (network analysis will be done in main process with full CPU parallelism)
    for local_idx in range(len(all_cell_features)):
        results.append((all_cell_features[local_idx], all_object_features[local_idx], all_network_features[local_idx]))

    total_percell_time = time_module.time() - t_percell_start

    # Final summary for this batch
    batch_elapsed = time_module.time() - batch_loop_start
    total_elapsed = time_module.time() - worker_start_time
    rate = cells_processed / batch_elapsed if batch_elapsed > 0 else 0
    io_pct = 100 * total_io_time / batch_elapsed if batch_elapsed > 0 else 0
    gpu_pct = 100 * total_gpu_time / batch_elapsed if batch_elapsed > 0 else 0
    net_pct = 0  # Network done in main process now
    log(f"  [Worker {worker_pid}] DONE: {cells_processed} cells in {batch_elapsed:.1f}s ({rate:.1f} cells/s), IO={total_io_time:.1f}s ({io_pct:.0f}%), GPU={total_gpu_time:.1f}s ({gpu_pct:.0f}%), percell={total_percell_time:.1f}s, network_items={len(network_work_items)}, morph_supp_items={len(morph_supplement_work_items)}, skipped={cells_skipped}")

    # Write results to disk for crash safety (if partials_dir provided)
    if partials_dir is not None:
        import pickle
        import tempfile
        import threading
        n_network = len(network_work_items)
        n_morph = len(morph_supplement_work_items)

        partial_path = f"{partials_dir}/gpu_partial_{batch_idx:06d}.pkl"

        # Fire-and-forget: write pickle in background thread so the worker
        # can immediately start the next batch.  The thread owns the data
        # references; they'll be freed when the thread finishes.
        data_to_write = {
            'results': results,
            'network_items': network_work_items,
            'morph_items': morph_supplement_work_items,
        }

        def _bg_write(data, path, directory):
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
            try:
                with os.fdopen(fd, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.rename(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                # Log but don't crash — the batch will be retried on resume
                import traceback
                traceback.print_exc()

        t = threading.Thread(target=_bg_write, args=(data_to_write, partial_path, partials_dir), daemon=True)
        t.start()
        _PENDING_WRITES.append(t)  # drained by _drain_pending_writes before cluster.close()

        # Drop local references — the thread holds its own via data_to_write
        del results, network_work_items, morph_supplement_work_items, data_to_write

        return (partial_path, cells_processed, n_network, n_morph)

    return results, network_work_items, morph_supplement_work_items


def _process_cell_features(
    crop_info: dict,
    cell_specific_mask: np.ndarray,
    organelle_mask_arrays_crop: dict,
    data: np.ndarray,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    channel_names: list,
    organelle_map: dict,
    full_features: bool,
    nuclear_mask: np.ndarray,
    cp_intensity_image: np.ndarray,
    cp_cell_mask: np.ndarray,
    cp_nuclear_mask: np.ndarray,
    max_objects_per_organelle: int,
    batch_extract_func,
    network_func,
    debug: bool = False,
):
    """
    Process features for a single cell using provided extraction functions.

    This is a helper for Dask workers that takes the feature extraction
    functions as parameters (allowing GPU or CPU versions to be passed in).
    """
    import time
    from organelle_profiler.feature_extraction.localization_features import (
        precompute_boundary_kdtrees,
        compute_localization_kdtree,
        compute_cell_level_localization_summary,
    )
    from skimage.measure import regionprops_table, label

    cell_start_time = time.time()
    timing_stats = {"cell_morph": 0, "batch_extract": 0, "localization": {}, "network": {}}

    base_features = {
        "cell_id": crop_info.get("global_cell_id"),
        "well": crop_info.get("well"),
    }

    # Extract cell morphology
    t0 = time.time()
    cell_morphology = _extract_cell_features_simple(cell_specific_mask, spacing)
    timing_stats["cell_morph"] = time.time() - t0
    cell_features = {**base_features, **cell_morphology}

    # CP cell morphology if available
    if cp_cell_mask is not None and np.any(cp_cell_mask > 0):
        t0 = time.time()
        cp_cell_morphology = _extract_cell_features_simple(cp_cell_mask, spacing)
        timing_stats["cp_cell_morph"] = time.time() - t0
        for key, value in cp_cell_morphology.items():
            cp_key = key.replace("cell_", "cp_cell_", 1)
            cell_features[cp_key] = value

    object_features_for_cell = {}
    network_features_for_cell = {}

    # Build organelle masks and intensity images
    organelles_with_masks = {}
    intensity_images_dict = {}
    for organelle_name in organelles_to_process:
        if organelle_name == "cell_membrane":
            organelles_with_masks[organelle_name] = cell_specific_mask
        else:
            mask = organelle_mask_arrays_crop.get(organelle_name)
            if mask is not None:
                organelles_with_masks[organelle_name] = mask

        is_cp_organelle = organelle_name.lower().startswith("cp")
        if is_cp_organelle and cp_intensity_image is not None:
            if cp_intensity_image.ndim >= 3 and cp_intensity_image.shape[0] > 1:
                intensity_images_dict[organelle_name] = np.mean(cp_intensity_image, axis=0)
            elif cp_intensity_image.ndim >= 3:
                intensity_images_dict[organelle_name] = cp_intensity_image[0]
            else:
                intensity_images_dict[organelle_name] = cp_intensity_image
        elif data is not None:
            if data.ndim >= 3 and data.shape[0] > 1:
                intensity_images_dict[organelle_name] = np.mean(data, axis=0)
            elif data.ndim >= 3:
                intensity_images_dict[organelle_name] = data[0]
            else:
                intensity_images_dict[organelle_name] = data

    # Downsample if needed
    if max_objects_per_organelle is not None and max_objects_per_organelle > 0:
        seed = crop_info.get("total_index", 0)
        for org_name in list(organelles_with_masks.keys()):
            org_mask = organelles_with_masks[org_name]
            n_objects = len(np.unique(org_mask)) - 1
            if n_objects > max_objects_per_organelle:
                organelles_with_masks[org_name] = _downsample_labeled_mask_simple(
                    org_mask, max_objects_per_organelle, seed=seed
                )

    # Batch extract morphological features
    if organelles_with_masks:
        t0 = time.time()
        batch_features, batch_timing = batch_extract_func(
            organelles_with_masks,
            spacing,
            intensity_images=intensity_images_dict,
            profile_properties=False,
            profile_organelle=None,
        )
        timing_stats["batch_extract"] = time.time() - t0
        timing_stats["batch_extract_per_org"] = batch_timing

        for organelle_name, df in batch_features.items():
            if not df.empty:
                df["cell_id"] = crop_info.get("global_cell_id")
                df["total_index"] = crop_info.get("total_index")
                object_features_for_cell[organelle_name] = df

    # Precompute KDTrees
    tree_cache_standard = precompute_boundary_kdtrees(
        cell_mask=cell_specific_mask,
        nuclear_mask=nuclear_mask,
        spacing=spacing,
    )
    tree_cache_cp = None
    if cp_cell_mask is not None:
        cp_nuc = cp_nuclear_mask if cp_nuclear_mask is not None else nuclear_mask
        tree_cache_cp = precompute_boundary_kdtrees(
            cell_mask=cp_cell_mask,
            nuclear_mask=cp_nuc,
            spacing=spacing,
        )

    # Process localization and network features
    for organelle_name, organelle_mask in organelles_with_masks.items():
        is_cp_organelle = organelle_name.lower().startswith("cp")
        if is_cp_organelle and tree_cache_cp is not None:
            tree_cache = tree_cache_cp
        else:
            tree_cache = tree_cache_standard

        # Localization features
        if organelle_name not in ("nuclei", "nuclear_seg", "cell_membrane"):
            try:
                t0 = time.time()
                localization_df = compute_localization_kdtree(
                    organelle_mask=organelle_mask,
                    tree_cache=tree_cache,
                    spacing=spacing,
                )
                if not localization_df.empty:
                    loc_summary = compute_cell_level_localization_summary(
                        localization_df, organelle_name
                    )
                    cell_features.update(loc_summary)
                timing_stats["localization"][organelle_name] = time.time() - t0
            except Exception:
                pass

        # Network analysis
        if organelle_name in network_organelles:
            t0 = time.time()
            network_analysis_results = network_func(
                organelle_mask,
                spacing=spacing,
                intensity_image=None,
            )

            if network_analysis_results:
                branch_df, network_summary_dict, per_object_network_df = network_analysis_results[:3]   # calculate_network_features returns (…, timings); take the first 3
                if not branch_df.empty:
                    branch_df["cell_id"] = crop_info.get("global_cell_id")
                    branch_df["total_index"] = crop_info.get("total_index")
                    network_features_for_cell[organelle_name] = {
                        "branch_df": branch_df,
                        "per_object_df": per_object_network_df,
                    }
                if network_summary_dict:
                    for key, value in network_summary_dict.items():
                        cell_features[f"network_{organelle_name}_{key}"] = value
            timing_stats["network"][organelle_name] = time.time() - t0

    # Inter-organelle contact features (cheap pairwise mask overlap; radial
    # distribution is emitted for free by the localization summary).
    # NOT guarded: a failure here must surface, not silently drop the group.
    from organelle_profiler.feature_extraction.spatial_features import (
        compute_spatial_features,
    )
    cell_features.update(compute_spatial_features(organelles_with_masks, spacing))

    # Store timing
    cell_total_time = time.time() - cell_start_time
    cell_features["_timing_total_ms"] = cell_total_time * 1000
    cell_features["_timing_batch_extract_ms"] = timing_stats["batch_extract"] * 1000
    cell_features["_timing_cell_morph_ms"] = timing_stats["cell_morph"] * 1000
    cell_features["_timing_localization_ms"] = sum(timing_stats["localization"].values()) * 1000
    cell_features["_timing_network_ms"] = sum(timing_stats["network"].values()) * 1000

    return cell_features, object_features_for_cell, network_features_for_cell


def _extract_cell_features_simple(cell_mask: np.ndarray, spacing: tuple) -> dict:
    """Simple cell feature extraction (duplicate of extract_cell_features for Dask workers)."""
    from skimage.measure import regionprops_table, label

    try:
        if not np.any(cell_mask > 0):
            return {}

        cell_mask_labeled = label(cell_mask > 0)

        base_properties = (
            "area",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "extent",
            "orientation",
        )
        props = regionprops_table(
            cell_mask_labeled, properties=base_properties, spacing=spacing
        )

        hu_props = regionprops_table(
            cell_mask_labeled,
            properties=("moments_hu",),
            spacing=(1, 1),
        )
        props.update(hu_props)

        if "area" in props and props["area"].size > 0:
            minor_axis = props["axis_minor_length"][0] if props["axis_minor_length"][0] > 0 else 1.0
            major_axis = props["axis_major_length"][0]
            perimeter = props["perimeter"][0]
            circularity = (4 * np.pi * props["area"][0]) / (perimeter**2) if perimeter > 0 else 1.0

            features = {
                "cell_area": props["area"][0],
                "cell_perimeter": perimeter,
                "cell_axis_major_length": major_axis,
                "cell_axis_minor_length": minor_axis,
                "cell_aspect_ratio": major_axis / minor_axis,
                "cell_solidity": props["solidity"][0],
                "cell_extent": props["extent"][0],
                "cell_orientation": props["orientation"][0],
                "cell_circularity": circularity,
            }
            for i in range(7):
                features[f"cell_hu_moment_{i}"] = props[f"moments_hu-{i}"][0]
            return features

        return {}
    except Exception:
        return {}


def _downsample_labeled_mask_simple(labeled_mask: np.ndarray, max_objects: int, seed: int = None) -> np.ndarray:
    """Simple downsampling (duplicate for Dask workers)."""
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) <= max_objects:
        return labeled_mask

    rng = np.random.default_rng(seed)
    selected_labels = rng.choice(unique_labels, size=max_objects, replace=False)
    return np.where(np.isin(labeled_mask, selected_labels), labeled_mask, 0)


# =============================================================================
# Cell Feature Extraction Functions
# =============================================================================

def _downsample_labeled_mask(labeled_mask: np.ndarray, max_objects: int, seed: int = None) -> np.ndarray:
    """Randomly downsample a labeled mask to at most max_objects.

    Returns a new mask with only the selected labels, preserving original label IDs.
    For reproducibility, use a deterministic seed (e.g., cell index).

    Parameters
    ----------
    labeled_mask : np.ndarray
        2D labeled segmentation mask where each unique value > 0 is an object.
    max_objects : int
        Maximum number of objects to keep.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Downsampled mask with at most max_objects unique labels.
    """
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background

    if len(unique_labels) <= max_objects:
        return labeled_mask  # No downsampling needed

    # Random selection with deterministic seed
    rng = np.random.default_rng(seed)
    selected_labels = rng.choice(unique_labels, size=max_objects, replace=False)

    # Create new mask with only selected labels
    return np.where(np.isin(labeled_mask, selected_labels), labeled_mask, 0)


def extract_cell_features(cell_mask: np.ndarray, spacing: tuple) -> dict:
    """Extracts basic morphological features from the cell mask.

    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask for the cell (H, W).
    spacing : tuple
        Pixel spacing (y, x) for morphological calculations.

    Returns
    -------
    dict
        Dictionary of cell-level morphological features.
    """
    try:
        if not np.any(cell_mask > 0):
            return {}

        # Ensure mask is integer-typed for regionprops
        cell_mask_labeled = label(cell_mask > 0)

        base_properties = (
            "area",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "extent",
            "orientation",
        )
        props = regionprops_table(
            cell_mask_labeled, properties=base_properties, spacing=spacing
        )

        # Calculate Hu moments separately without spacing
        hu_props = regionprops_table(
            cell_mask_labeled,
            properties=("moments_hu",),
            spacing=(1, 1),  # Explicitly set spacing to (1, 1) for Hu moments
        )
        props.update(hu_props)

        if "area" in props and props["area"].size > 0:
            minor_axis = (
                props["axis_minor_length"][0]
                if props["axis_minor_length"][0] > 0
                else 1.0
            )
            major_axis = props["axis_major_length"][0]
            perimeter = props["perimeter"][0]
            # Sphericity calculation
            circularity = (
                (4 * np.pi * props["area"][0]) / (perimeter**2)
                if perimeter > 0
                else 1.0
            )

            features = {
                "cell_area": props["area"][0],
                "cell_perimeter": perimeter,
                "cell_axis_major_length": major_axis,
                "cell_axis_minor_length": minor_axis,
                "cell_aspect_ratio": major_axis / minor_axis,
                "cell_solidity": props["solidity"][0],
                "cell_extent": props["extent"][0],
                "cell_orientation": props["orientation"][0],
                "cell_circularity": circularity,
            }
            for i in range(7):
                features[f"cell_hu_moment_{i}"] = props[f"moments_hu-{i}"][0]
            return features

        return {}
    except Exception as e:
        print(f"Error extracting cell features: {e}")
        return {}




def process_single_cell(
    cell_info: dict,
    cell_specific_mask: np.ndarray,
    organelle_mask_arrays: dict,
    intensity_image: np.ndarray,
    frangi_image_arrays: dict,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    channel_names: list,
    organelle_map: dict,
    full_features: bool,
    nuclear_mask: np.ndarray = None,
    cp_intensity_image: np.ndarray = None,
    cp_cell_mask: np.ndarray = None,
    cp_nuclear_mask: np.ndarray = None,
    max_objects_per_organelle: int = None,
    debug: bool = False,
):
    """
    Process a single cell and extract all features.

    Parameters
    ----------
    cell_info : dict
        Cell metadata from BaseDataset crop_info (or pd.Series.to_dict()).
    cell_specific_mask : np.ndarray
        Binary mask for this specific cell (H, W).
    organelle_mask_arrays : dict
        Dict of organelle name -> cropped label array.
    intensity_image : np.ndarray
        Cropped intensity image (C, H, W) at phenotyping bbox location.
    frangi_image_arrays : dict
        Dict of organelle name -> cropped Frangi vesselness map.
    organelles_to_process : list
        List of organelle names to process.
    network_organelles : list
        List of organelle names for network analysis.
    spacing : tuple
        Pixel spacing (y, x).
    channel_names : list
        List of channel names.
    organelle_map : dict
        Mapping of organelle name -> channel name.
    full_features : bool
        Whether to compute expensive features.
    nuclear_mask : np.ndarray, optional
        Binary mask for the nucleus (for localization features).
    cp_intensity_image : np.ndarray, optional
        Cropped intensity image (C, H, W) at cp_bbox location for CP organelles.
        The cell has moved between CellPainting and phenotyping imaging sessions.
    cp_cell_mask : np.ndarray, optional
        Binary mask for the cell at cp_bbox location (from cp_cell_seg).
    cp_nuclear_mask : np.ndarray, optional
        Binary nuclear mask at cp_bbox location for CP organelles.
    max_objects_per_organelle : int, optional
        Maximum objects per organelle (for downsampling).

    Returns
    -------
    tuple
        (cell_features, object_features_for_cell, network_features_for_cell)
    """
    import time
    cell_start_time = time.time()
    timing_stats = {"cell_morph": 0, "batch_extract": 0, "localization": {}, "network": {}}

    # IMPORTANT: Only include essential ID columns for joining - NOT all metadata.
    # Metadata will be joined back at the very end using cell_id to avoid
    # metadata columns being incorrectly aggregated as features.
    base_features = {
        "cell_id": cell_info.get("global_cell_id"),
        "well": cell_info.get("well"),  # Needed for grouping during processing
    }

    t0 = time.time()
    cell_morphology = extract_cell_features(cell_specific_mask, spacing)
    timing_stats["cell_morph"] = time.time() - t0
    cell_features = {**base_features, **cell_morphology}

    # Also extract cell morphology from cp_cell_mask if available (CellPainting cell boundary)
    # These features are prefixed with "cp_cell_" to distinguish from phenotyping "cell_" features
    if cp_cell_mask is not None and np.any(cp_cell_mask > 0):
        t0 = time.time()
        cp_cell_morphology = extract_cell_features(cp_cell_mask, spacing)
        timing_stats["cp_cell_morph"] = time.time() - t0
        # Rename keys from "cell_" to "cp_cell_"
        for key, value in cp_cell_morphology.items():
            cp_key = key.replace("cell_", "cp_cell_", 1)
            cell_features[cp_key] = value

    object_features_for_cell = {}
    network_features_for_cell = {}

    # Filter organelles to process (skip if no mask available)
    organelles_with_masks = {}
    intensity_images_dict = {}
    for organelle_name in organelles_to_process:
        if organelle_name == "cell_membrane":
            organelles_with_masks[organelle_name] = cell_specific_mask
        else:
            mask = organelle_mask_arrays.get(organelle_name)
            if mask is not None:
                organelles_with_masks[organelle_name] = mask

        # Use mean intensity across all channels for each organelle
        # This gives us intensity features for every organelle without complex channel mapping
        is_cp_organelle = organelle_name.lower().startswith("cp")
        if is_cp_organelle and cp_intensity_image is not None:
            # CP organelles use cp_intensity_image (at cp_bbox location)
            if cp_intensity_image.ndim >= 3 and cp_intensity_image.shape[0] > 1:
                # Mean across channels for multi-channel image
                intensity_images_dict[organelle_name] = np.mean(cp_intensity_image, axis=0)
            elif cp_intensity_image.ndim >= 3:
                intensity_images_dict[organelle_name] = cp_intensity_image[0]
            else:
                intensity_images_dict[organelle_name] = cp_intensity_image
        elif intensity_image is not None:
            # Standard organelles use intensity_image (at bbox location)
            if intensity_image.ndim >= 3 and intensity_image.shape[0] > 1:
                # Mean across channels for multi-channel image
                intensity_images_dict[organelle_name] = np.mean(intensity_image, axis=0)
            elif intensity_image.ndim >= 3:
                intensity_images_dict[organelle_name] = intensity_image[0]
            else:
                intensity_images_dict[organelle_name] = intensity_image

    # OPTIMIZATION: Downsample high object count organelles to speed up feature extraction
    # This keeps all organelle types but limits the number of objects per type
    if max_objects_per_organelle is not None and max_objects_per_organelle > 0:
        seed = cell_info.get("total_index", 0)  # Deterministic for reproducibility
        for org_name in list(organelles_with_masks.keys()):
            org_mask = organelles_with_masks[org_name]
            n_objects = len(np.unique(org_mask)) - 1  # Exclude background
            if n_objects > max_objects_per_organelle:
                organelles_with_masks[org_name] = _downsample_labeled_mask(
                    org_mask, max_objects_per_organelle, seed=seed
                )

    # CPU BATCH EXTRACTION: Extract morphological features for all organelles at once
    timing_stats["batch_extract_per_org"] = {}  # Per-organelle batch extraction timing
    if organelles_with_masks:
        t0 = time.time()
        # Profile per-property timing on first cell only (total_index == 0) in debug/preview mode
        # This provides representative timing data without log spam in production runs
        total_idx = cell_info.get("total_index", -1)
        profile_this_cell = debug and total_idx == 0
        if profile_this_cell:
            print(f"[PROFILE DEBUG] Cell total_index={total_idx}, profiling ENABLED")
        batch_features, batch_timing = batch_extract_organelle_features(
            organelles_with_masks,
            spacing,
            intensity_images=intensity_images_dict,
            profile_properties=profile_this_cell,
            profile_organelle="focus3d_vesicular",  # Profile slowest organelle
        )
        timing_stats["batch_extract"] = time.time() - t0
        timing_stats["batch_extract_per_org"] = batch_timing  # Per-organelle timing
        if profile_this_cell:
            # Debug: show what timing keys we got
            prop_keys = [k for k in batch_timing.keys() if k.startswith("_property_")]
            print(f"[PROFILE DEBUG] batch_timing property keys: {prop_keys}")

        # Add cell_id and total_index to each result
        for organelle_name, df in batch_features.items():
            if not df.empty:
                df["cell_id"] = cell_info.get("global_cell_id")
                df["total_index"] = cell_info.get("total_index")
                object_features_for_cell[organelle_name] = df

    # OPTIMIZATION: Pre-compute boundary KDTrees ONCE for this cell
    # KDTree queries are ~175x faster than distance transforms for localization
    # Separate caches for standard and CP organelles (different cell masks)
    tree_cache_standard = precompute_boundary_kdtrees(
        cell_mask=cell_specific_mask,
        nuclear_mask=nuclear_mask,
        spacing=spacing,
    )
    tree_cache_cp = None
    if cp_cell_mask is not None:
        # Use cp_nuclear_mask for CP organelles if available, otherwise fall back to nuclear_mask
        cp_nuc = cp_nuclear_mask if cp_nuclear_mask is not None else nuclear_mask
        tree_cache_cp = precompute_boundary_kdtrees(
            cell_mask=cp_cell_mask,
            nuclear_mask=cp_nuc,
            spacing=spacing,
        )

    # Process localization and network features (these need per-organelle handling)
    for organelle_name, organelle_mask in organelles_with_masks.items():
        # Determine which KDTree cache to use
        is_cp_organelle = organelle_name.lower().startswith("cp")
        if is_cp_organelle and tree_cache_cp is not None:
            tree_cache = tree_cache_cp
        else:
            tree_cache = tree_cache_standard

        # Compute localization features using KDTree queries (vectorized, ~175x faster)
        if organelle_name not in ("nuclei", "nuclear_seg", "cell_membrane"):
            try:
                t0 = time.time()
                # Use KDTree for massive speedup over distance transforms
                localization_df = compute_localization_kdtree(
                    organelle_mask=organelle_mask,
                    tree_cache=tree_cache,
                    spacing=spacing,
                )
                if not localization_df.empty:
                    loc_summary = compute_cell_level_localization_summary(
                        localization_df, organelle_name
                    )
                    cell_features.update(loc_summary)
                timing_stats["localization"][organelle_name] = time.time() - t0
            except Exception:
                pass

        # Network analysis for specified organelles
        if organelle_name in network_organelles:
            t0 = time.time()
            network_analysis_results = calculate_network_features(
                organelle_mask,
                spacing=spacing,
                intensity_image=None,
            )

            if network_analysis_results:
                branch_df, network_summary_dict, per_object_network_df = network_analysis_results[:3]   # calculate_network_features returns (…, timings); take the first 3
                if not branch_df.empty:
                    branch_df["cell_id"] = cell_info.get("global_cell_id")
                    branch_df["total_index"] = cell_info.get("total_index")
                    network_features_for_cell[organelle_name] = {
                        "branch_df": branch_df,
                        "per_object_df": per_object_network_df,
                    }
                if network_summary_dict:
                    for key, value in network_summary_dict.items():
                        cell_features[f"network_{organelle_name}_{key}"] = value
            timing_stats["network"][organelle_name] = time.time() - t0

    # Inter-organelle contact features (cheap pairwise mask overlap; radial
    # distribution is emitted for free by the localization summary).
    # NOT guarded: a failure here must surface, not silently drop the group.
    from organelle_profiler.feature_extraction.spatial_features import (
        compute_spatial_features,
    )
    cell_features.update(compute_spatial_features(organelles_with_masks, spacing))

    # Log timing stats for first few cells (to avoid log spam)
    cell_total_time = time.time() - cell_start_time
    cell_id = cell_info.get("global_cell_id", "?")

    # Store timing in cell_features for aggregation
    cell_features["_timing_total_ms"] = cell_total_time * 1000
    cell_features["_timing_batch_extract_ms"] = timing_stats["batch_extract"] * 1000
    cell_features["_timing_cell_morph_ms"] = timing_stats["cell_morph"] * 1000
    cell_features["_timing_localization_ms"] = sum(timing_stats["localization"].values()) * 1000
    cell_features["_timing_network_ms"] = sum(timing_stats["network"].values()) * 1000

    # Store per-organelle timing for detailed analysis
    batch_timing_data = timing_stats.get("batch_extract_per_org", {})
    for key, val in batch_timing_data.items():
        if key.startswith("_property_"):
            # Property profiling data (already in ms from morphology_features.py)
            cell_features[f"_timing{key}"] = val
        else:
            # Per-organelle timing (convert from seconds to ms)
            cell_features[f"_timing_batch_{key}_ms"] = val * 1000
    for org, t in timing_stats["localization"].items():
        cell_features[f"_timing_loc_{org}_ms"] = t * 1000
    for org, t in timing_stats["network"].items():
        cell_features[f"_timing_net_{org}_ms"] = t * 1000

    return cell_features, object_features_for_cell, network_features_for_cell

def _worker_process_features_for_cells(
    cell_indices: list,
    labels_df: pd.DataFrame,
    stores: dict,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    organelle_map: dict,
    full_features: bool,
    channel_names: list,
    initial_yx_patch_size: tuple,
    available_labels: dict,
    debug: bool = False,
    max_objects_per_organelle: int = None,
):
    """
    Joblib worker function to process a batch of cells using BaseDataset.

    This function creates a BaseDataset instance and uses its __getitem__ method
    to load cropped cell data and masks, then extracts features from each cell.

    Each worker opens its own zarr store to avoid pickling issues with joblib's
    loky backend, enabling true multiprocessing parallelism.

    Parameters
    ----------
    cell_indices : list
        List of indices into labels_df for cells to process in this worker.
    labels_df : pd.DataFrame
        Full DataFrame with cell metadata (will be subset by indices).
    stores : dict
        Dictionary mapping store_key to either:
        - A path string (str/Path) to the zarr store (for parallel workers)
        - An already-open zarr store object (for sequential mode)
    organelles_to_process : list
        List of organelle names to extract features for.
    network_organelles : list
        List of organelle names for network analysis.
    spacing : tuple
        Pixel spacing (y, x) for morphological calculations.
    organelle_map : dict
        Mapping of organelle names to channel names.
    full_features : bool
        Whether to compute expensive features.
    channel_names : list
        List of channel names in the store.
    initial_yx_patch_size : tuple
        Size of patches to extract around cells.
    available_labels : dict
        Mapping of internal organelle names to zarr label names.

    Returns
    -------
    list
        List of (cell_features, object_features, network_features) tuples.
    """
    # Suppress warnings in worker process (workers have their own interpreter)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress CUDA deprecation warnings
    warnings.filterwarnings("ignore", message=".*regions with <=1 background pixel spacing.*")
    warnings.filterwarnings("ignore", message="Found.*regions with <=1 background pixel spacing")
    warnings.filterwarnings("ignore", message="Input image is entirely zero")

    results_list = []

    # Subset the labels_df to only the cells for this worker
    worker_labels_df = labels_df.iloc[cell_indices].reset_index(drop=True)

    # Open stores if paths are provided (for parallel workers)
    # This enables joblib's loky backend to pickle arguments correctly
    opened_stores = {}
    stores_to_close = []
    for key, value in stores.items():
        if isinstance(value, (str, Path)):
            # It's a path - open it
            opened_store = open_ome_zarr(str(value), mode="r")
            opened_stores[key] = opened_store
            stores_to_close.append(opened_store)
        else:
            # Already an open store
            opened_stores[key] = value

    try:
        # Create BaseDataset for this worker's cells
        # Use simple iohub reads - no tensorstore (per-read overhead too slow)
        base_dataset = BaseDataset(
            stores=opened_stores,
            labels_df=worker_labels_df,
            initial_yx_patch_size=initial_yx_patch_size,
            final_yx_patch_size=initial_yx_patch_size,
            out_channels="all",
            mask_cell=False,  # We need the raw mask for feature extraction
            use_original_crop_size=False,
        )
        

        # Get the pheno store for loading organelle labels
        pheno_store = opened_stores.get("pheno_assembled_v3")

        for i in range(len(worker_labels_df)):
            try:
                # Use BaseDataset.__getitem__ to load cell data and mask
                batch = base_dataset[i]

                # Extract data from batch
                data = batch["data"].numpy() if hasattr(batch["data"], 'numpy') else np.array(batch["data"])
                mask = batch["mask"].numpy() if hasattr(batch["mask"], 'numpy') else np.array(batch["mask"])
                crop_info = batch["crop_info"]

                # Get cell-specific binary mask (BaseDataset already isolates the cell)
                cell_specific_mask = mask[0].astype(np.uint8)  # (H, W)

                if not np.any(cell_specific_mask):
                    continue

                # Load organelle segmentation labels for this cell's crop region
                # Dual bbox system: cp_bbox for CP organelles, bbox for standard organelles
                organelle_mask_arrays_crop = {}
                frangi_image_arrays_crop = {}
                cp_intensity = None  # Intensity at cp_bbox for CP organelle feature extraction
                cp_cell_mask = None  # Cell mask at cp_bbox for CP organelles

                if pheno_store is not None and available_labels:
                    well = crop_info.get("well")
                    # Get both bboxes
                    pheno_bbox = batch.get("bbox")  # Standard bbox from phenotyping
                    cp_bbox = crop_info.get("cp_bbox")  # CP bbox from cell painting

                    # Get both segmentation IDs
                    seg_id = crop_info.get("segmentation_id")  # For cell_seg (pheno)
                    cp_seg_id = crop_info.get("cp_cell_seg_id")  # For cp_cell_seg (CP)

                    if pd.notna(seg_id):
                        seg_id = int(seg_id)
                    else:
                        seg_id = None
                    if cp_seg_id is not None and pd.notna(cp_seg_id):
                        cp_seg_id = int(cp_seg_id)
                    else:
                        cp_seg_id = seg_id  # Fallback

                    if well:
                        try:
                            position = pheno_store[well]

                            if "labels" in position.zgroup:
                                labels_group = position.zgroup["labels"]
                                
                                # Tensorstore disabled - per-read handle opening too slow
                                # Using iohub for all label reads
                                store_path_str = None

                                # Load cp_cell_mask using cp_bbox (for CP organelles)
                                cp_cell_mask = None
                                cp_intensity = None  # Intensity at cp_bbox for CP organelle feature extraction
                                if cp_bbox is not None and "cp_cell_seg" in labels_group and cp_seg_id is not None:
                                    try:
                                        # Use iohub for label reads
                                        cp_label_array = labels_group["cp_cell_seg"]["0"]
                                        y_min, x_min, y_max, x_max = cp_bbox
                                        cp_cell_mask_raw = np.array(cp_label_array[0, 0, 0, y_min:y_max, x_min:x_max])
                                        
                                        cp_cell_mask = (cp_cell_mask_raw == cp_seg_id).astype(np.uint8)

                                        # Load intensity at cp_bbox location (different from phenotyping bbox!)
                                        # This is critical: the cell has moved between CP and pheno imaging
                                        fov = position["0"]
                                        cp_y_min, cp_x_min, cp_y_max, cp_x_max = cp_bbox
                                        cp_intensity = np.array(fov[0, :, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                                    except Exception:
                                        cp_cell_mask = None
                                        cp_intensity = None

                                # Process each organelle with appropriate bbox
                                for internal_name, zarr_label_name in available_labels.items():
                                    if internal_name == "cell_mask":
                                        continue

                                    if zarr_label_name not in labels_group:
                                        continue

                                    # Determine which bbox to use based on organelle type
                                    # Case-insensitive check for CP organelles (cp*, CP*)
                                    is_cp_organelle = internal_name.lower().startswith("cp")

                                    if is_cp_organelle:
                                        bbox_to_use = cp_bbox
                                        mask_to_use = cp_cell_mask
                                    else:
                                        bbox_to_use = pheno_bbox
                                        mask_to_use = cell_specific_mask

                                    if bbox_to_use is None or mask_to_use is None:
                                        continue

                                    try:
                                        # Use iohub for label reads
                                        label_array = labels_group[zarr_label_name]["0"]
                                        y_min, x_min, y_max, x_max = bbox_to_use
                                        ndim = label_array.ndim
                                        if ndim == 5:
                                            label_crop = np.array(label_array[0, 0, 0, y_min:y_max, x_min:x_max])
                                        elif ndim == 4:
                                            label_crop = np.array(label_array[0, 0, y_min:y_max, x_min:x_max])
                                        elif ndim == 3:
                                            label_crop = np.array(label_array[0, y_min:y_max, x_min:x_max])
                                        else:
                                            label_crop = np.array(label_array[y_min:y_max, x_min:x_max])

                                        # Ensure shapes match
                                        if label_crop.shape != mask_to_use.shape:
                                            matched_crop = np.zeros(mask_to_use.shape, dtype=label_crop.dtype)
                                            h = min(label_crop.shape[0], matched_crop.shape[0])
                                            w = min(label_crop.shape[1], matched_crop.shape[1])
                                            matched_crop[:h, :w] = label_crop[:h, :w]
                                            label_crop = matched_crop

                                        # Mask to only include labels within the cell boundary
                                        label_crop = label_crop * (mask_to_use > 0).astype(label_crop.dtype)

                                        if np.any(label_crop > 0):
                                            organelle_mask_arrays_crop[internal_name] = label_crop
                                            if debug and i == 0:
                                                print(f"      [DEBUG] Loaded {internal_name}: shape={label_crop.shape}, unique_labels={len(np.unique(label_crop))}")
                                        elif debug and i == 0:
                                            print(f"      [DEBUG] {internal_name}: empty after cell masking")

                                    except Exception as e:
                                        if debug and i == 0:
                                            print(f"      [DEBUG] {internal_name}: load error - {e}")

                        except Exception as e:
                            if debug and i == 0:
                                print(f"    [DEBUG] Error accessing position: {e}")

                # Debug: print summary of loaded organelle masks for first cell
                if debug and i == 0:
                    print(f"  [DEBUG] Cell 0 summary:")
                    print(f"    - available_labels: {list(available_labels.keys())}")
                    print(f"    - organelle_mask_arrays_crop: {list(organelle_mask_arrays_crop.keys())}")
                    print(f"    - pheno_bbox: {pheno_bbox}")
                    print(f"    - cp_bbox: {cp_bbox}")
                    print(f"    - cell_specific_mask has pixels: {np.any(cell_specific_mask)}")

                # Extract nuclear mask for localization features
                # Look for nuclei or nuclear_seg in the organelle arrays (standard bbox)
                nuclear_mask = None
                for nuc_key in ("nuclei", "nuclear_seg"):
                    if nuc_key in organelle_mask_arrays_crop:
                        nuc_arr = organelle_mask_arrays_crop[nuc_key]
                        # Convert to binary mask
                        nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                        break

                # Extract CP nuclear mask for CP organelles (cp_bbox)
                # Look for cp*_nucl* patterns (e.g., cp1_nuclei_hoechst)
                cp_nuclear_mask = None
                for key in organelle_mask_arrays_crop:
                    if key.lower().startswith("cp") and "nucl" in key.lower():
                        nuc_arr = organelle_mask_arrays_crop[key]
                        cp_nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                        break

                # Call processing function
                cell_features, object_features, network_features = (
                    process_single_cell(
                        crop_info,
                        cell_specific_mask,
                        organelle_mask_arrays_crop,
                        data,  # (C, H, W) at phenotyping bbox
                        frangi_image_arrays_crop,
                        organelles_to_process,
                        network_organelles,
                        spacing,
                        channel_names,
                        organelle_map,
                        full_features,
                        nuclear_mask=nuclear_mask,
                        cp_intensity_image=cp_intensity,  # (C, H, W) at cp_bbox (different location!)
                        cp_cell_mask=cp_cell_mask,  # Cell mask at cp_bbox
                        cp_nuclear_mask=cp_nuclear_mask,  # Nuclear mask at cp_bbox for CP organelles
                        max_objects_per_organelle=max_objects_per_organelle,
                        debug=debug,
                    )
                )

                # Debug: print extracted features for first cell
                if debug and i == 0:
                    print(f"  [DEBUG] Extracted features for cell 0:")
                    print(f"    - cell_features keys: {list(cell_features.keys()) if cell_features else 'None'}")
                    print(f"    - object_features organelles: {list(object_features.keys()) if object_features else 'None'}")
                    for org, df in (object_features or {}).items():
                        print(f"      - {org}: {len(df)} objects, columns={list(df.columns)[:5]}...")
                    print(f"    - network_features organelles: {list(network_features.keys()) if network_features else 'None'}")

                if cell_features is not None:
                    results_list.append(
                        (cell_features, object_features, network_features)
                    )
                else:
                    # Debug: why is cell_features None?
                    if i == 0:
                        print(f"  [DEBUG WORKER] Cell 0: cell_features is None after process_single_cell")

            except Exception as e:
                import traceback
                cell_id = worker_labels_df.iloc[i].get("global_cell_id", i)
                print(f"Error processing cell {cell_id}: {e}\n{traceback.format_exc()}")
                continue

        return results_list

    finally:
        # Close any stores we opened
        for store in stores_to_close:
            try:
                store.close()
            except Exception:
                pass




def _worker_process_well(
    well: str,
    well_cells_df: pd.DataFrame,
    store_path: str,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    organelle_map: dict,
    full_features: bool,
    channel_names: list,
    initial_yx_patch_size: tuple,
    available_labels: dict,
    show_progress: bool = False,
    n_jobs: int = None,
    debug: bool = False,
    max_objects_per_organelle: int = None,
):
    """
    Joblib worker function to process all cells in a single well.

    This function is optimized for data locality - it processes all cells
    from one well, which means:
    - Only one zarr position is accessed
    - Better chunk caching efficiency
    - Smaller DataFrame passed to worker (only cells from this well)

    When show_progress=True (single-well SLURM mode), uses parallel processing
    with cell batches to utilize all available CPUs.

    Parameters
    ----------
    well : str
        Well identifier (e.g., 'A/1/0').
    well_cells_df : pd.DataFrame
        DataFrame with cell metadata for only this well (already filtered).
    store_path : str
        Path to the pheno zarr store (picklable string, not zarr object).
    organelles_to_process : list
        List of organelle names to extract features for.
    network_organelles : list
        List of organelle names for network analysis.
    spacing : tuple
        Pixel spacing (y, x) for morphological calculations.
    organelle_map : dict
        Mapping of organelle names to channel names.
    full_features : bool
        Whether to compute expensive features.
    channel_names : list
        List of channel names in the store.
    initial_yx_patch_size : tuple
        Size of patches to extract around cells.
    available_labels : dict
        Mapping of internal organelle names to zarr label names.
    show_progress : bool
        If True, display tqdm progress bar and use parallel cell processing.
    n_jobs : int
        Number of parallel jobs for cell processing (only used when show_progress=True).

    Returns
    -------
    list
        List of (cell_features, object_features, network_features) tuples for all cells in well.
    """
    # Suppress warnings in worker process (workers have their own interpreter)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress CUDA deprecation warnings
    warnings.filterwarnings("ignore", message=".*regions with <=1 background pixel spacing.*")
    warnings.filterwarnings("ignore", message="Found.*regions with <=1 background pixel spacing")
    warnings.filterwarnings("ignore", message="Input image is entirely zero")

    # Cells should already be chunk-sorted from SLURM batching
    # Re-sort by chunk for safety (handles cases where cells weren't pre-sorted)
    if 'y_global_pheno' in well_cells_df.columns and 'x_global_pheno' in well_cells_df.columns:
        CHUNK_SIZE = 512  # Zarr chunk size for phenotyping data
        # Handle NaN values - fill with large value so they sort to the end
        y_coords = well_cells_df['y_global_pheno'].fillna(999999)
        x_coords = well_cells_df['x_global_pheno'].fillna(999999)
        well_cells_df["_chunk_y"] = (y_coords // CHUNK_SIZE).astype(int)
        well_cells_df["_chunk_x"] = (x_coords // CHUNK_SIZE).astype(int)
        well_cells_df = well_cells_df.sort_values(
            ['_chunk_y', '_chunk_x', 'y_global_pheno', 'x_global_pheno'],
            na_position="last"
        ).reset_index(drop=True)
        well_cells_df = well_cells_df.drop(columns=["_chunk_y", "_chunk_x"])

    # Open the store - each worker opens its own to avoid pickling issues
    pheno_store = open_ome_zarr(store_path, mode="r")
    stores = {"pheno_assembled_v3": pheno_store}

    try:
        # Create BaseDataset for this well's cells
        # Use simple iohub reads - no tensorstore (per-read overhead too slow)
        base_dataset = BaseDataset(
            stores=stores,
            labels_df=well_cells_df,
            initial_yx_patch_size=initial_yx_patch_size,
            final_yx_patch_size=initial_yx_patch_size,
            out_channels="all",
            mask_cell=False,  # We need the raw mask for feature extraction
            use_original_crop_size=False,
        )

        # Pre-load the position once for this well (data locality optimization)
        labels_group = None
        position_fov = None
        try:
            position = pheno_store[well]
            if "labels" in position.zgroup:
                labels_group = position.zgroup["labels"]
            # Get FOV for loading cp_intensity (needed for CP organelles)
            position_fov = position["0"]
        except Exception:
            pass

        n_cells = len(well_cells_df)

        if show_progress and n_jobs is None:
            # Determine optimal workers based on GPU availability
            # GPU mode: MORE workers to saturate GPU (workers spend most time on I/O,
            # briefly use GPU - need many workers to keep GPU busy)
            # This pattern was proven successful in reconstruct_pheno-2d (64 workers, ~100% GPU util)
            # CPU mode: workers to utilize all cores
            if _USE_GPU_FEATURES and _DASK_AVAILABLE:
                n_jobs = 64  # GPU mode: high parallelism - workers mostly do I/O, briefly use GPU
                print(f"GPU mode: using {n_jobs} Dask workers (high parallelism for GPU saturation)")
            else:
                n_jobs = get_optimal_workers(use_gpu=False, model_ram_gb=0.1, data_ram_gb=0.2)
                print(f"CPU mode: using {n_jobs} Joblib workers")

        if show_progress and n_jobs > 1:
            results_list = []

            # HYBRID MODE: Partition cells by complexity and process GPU/CPU in parallel
            if _USE_HYBRID_MODE and _USE_GPU_FEATURES and _DASK_AVAILABLE:
                print(f"\n  === HYBRID GPU/CPU MODE ===")

                # Partition cells by complexity
                gpu_cells_df, cpu_cells_df = partition_cells_by_complexity(
                    well_cells_df, network_organelles, gpu_threshold_percentile=40
                )

                n_gpu = len(gpu_cells_df)
                n_cpu = len(cpu_cells_df)

                # Convert to dicts for processing
                gpu_cells_dict = gpu_cells_df.to_dict('records') if n_gpu > 0 else []
                cpu_cells_dict = cpu_cells_df.to_dict('records') if n_cpu > 0 else []

                # Configure batch sizes
                GPU_BATCH_SIZE = 256  # Larger batches for GPU (better utilization)
                CPU_BATCH_SIZE = 64   # Smaller batches for CPU (better load balancing)

                n_gpu_workers = 8     # Dask workers for GPU
                n_cpu_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.1, data_ram_gb=0.2)

                print(f"  GPU path: {n_gpu} simple cells, {n_gpu_workers} workers, batch size {GPU_BATCH_SIZE}")
                print(f"  CPU path: {n_cpu} complex cells, {n_cpu_workers} workers, batch size {CPU_BATCH_SIZE}")

                # Shared static args
                static_args = {
                    'store_path': store_path,
                    'well': well,
                    'available_labels': available_labels,
                    'organelles_to_process': organelles_to_process,
                    'network_organelles': network_organelles,
                    'spacing': spacing,
                    'channel_names': channel_names,
                    'organelle_map': organelle_map,
                    'full_features': full_features,
                    'initial_yx_patch_size': initial_yx_patch_size,
                    'max_objects_per_organelle': max_objects_per_organelle,
                }

                # Process GPU and CPU cells concurrently using threading
                from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
                import threading

                gpu_results = []
                cpu_results = []
                gpu_progress = [0]
                cpu_progress = [0]
                progress_lock = threading.Lock()

                def process_gpu_cells():
                    """Process simple cells on GPU using Dask."""
                    nonlocal gpu_results
                    if n_gpu == 0:
                        return

                    _require_gpu_for_feature_extraction()

                    import multiprocessing
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                    except RuntimeError:
                        pass

                    # Get available GPUs and conda prefix for the plugin
                    import torch
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    available_gpus = list(range(num_gpus)) if num_gpus > 0 else []
                    conda_prefix = os.environ.get('CONDA_PREFIX', '')
                    print(f"  [GPU] Available GPUs: {available_gpus}, CONDA_PREFIX: {conda_prefix}")

                    cluster = LocalCluster(
                        n_workers=n_gpu_workers,
                        threads_per_worker=1,
                        processes=True,
                        memory_limit='8GB',
                    )
                    client = Client(cluster)

                    # Register plugin to set LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES in each worker
                    plugin = CUDALibraryPlugin(available_gpus, conda_prefix)
                    client.register_worker_plugin(plugin)
                    print(f"  [GPU] Registered CUDALibraryPlugin for {len(available_gpus)} GPUs")

                    try:
                        # Submit GPU batches
                        n_gpu_batches = (n_gpu + GPU_BATCH_SIZE - 1) // GPU_BATCH_SIZE
                        futures = []
                        for batch_idx in range(n_gpu_batches):
                            start_idx = batch_idx * GPU_BATCH_SIZE
                            end_idx = min(start_idx + GPU_BATCH_SIZE, n_gpu)
                            batch_cells = gpu_cells_dict[start_idx:end_idx]
                            future = client.submit(
                                _dask_process_cell_batch,
                                batch_start=0,
                                batch_end=len(batch_cells),
                                cells_dict=batch_cells,
                                store_path=store_path,
                                well=well,
                                available_labels=available_labels,
                                organelles_to_process=organelles_to_process,
                                network_organelles=network_organelles,
                                spacing=spacing,
                                channel_names=channel_names,
                                organelle_map=organelle_map,
                                full_features=full_features,
                                initial_yx_patch_size=initial_yx_patch_size,
                                max_objects_per_organelle=max_objects_per_organelle,
                                debug=False,
                                available_gpus=available_gpus,
                            )
                            futures.append((future, len(batch_cells)))

                        # Collect GPU results
                        from dask.distributed import as_completed as dask_as_completed
                        for completed_future in dask_as_completed([f for f, _ in futures]):
                            batch_size_done = dict(futures)[completed_future]
                            try:
                                batch_results, _, _ = completed_future.result()
                                if batch_results:
                                    gpu_results.extend(batch_results)
                                with progress_lock:
                                    gpu_progress[0] += batch_size_done
                            except Exception as e:
                                print(f"  GPU batch failed: {e}")
                                with progress_lock:
                                    gpu_progress[0] += batch_size_done
                    finally:
                        try:
                            client.close()
                            cluster.close(timeout=60)
                        except:
                            pass

                def process_cpu_cells():
                    """Process complex cells on CPU using ProcessPoolExecutor."""
                    nonlocal cpu_results
                    if n_cpu == 0:
                        return

                    n_cpu_batches = (n_cpu + CPU_BATCH_SIZE - 1) // CPU_BATCH_SIZE

                    with ProcessPoolExecutor(
                        max_workers=n_cpu_workers,
                        initializer=_init_worker_tensorstore,
                        initargs=(store_path, well, available_labels),
                    ) as executor:
                        futures = {}
                        for batch_idx in range(n_cpu_batches):
                            start_idx = batch_idx * CPU_BATCH_SIZE
                            end_idx = min(start_idx + CPU_BATCH_SIZE, n_cpu)
                            # Create static args with the CPU cells
                            batch_static_args = static_args.copy()
                            future = executor.submit(
                                _process_cell_batch_with_global_handles,
                                start_idx=0,
                                end_idx=end_idx - start_idx,
                                cells_dict=cpu_cells_dict[start_idx:end_idx],
                                static_args=batch_static_args,
                                debug=False,
                            )
                            futures[future] = end_idx - start_idx

                        # Collect CPU results
                        for future in as_completed(futures):
                            batch_size_done = futures[future]
                            try:
                                batch_results = future.result()
                                if batch_results:
                                    cpu_results.extend(batch_results)
                                with progress_lock:
                                    cpu_progress[0] += batch_size_done
                            except Exception as e:
                                print(f"  CPU batch failed: {e}")
                                with progress_lock:
                                    cpu_progress[0] += batch_size_done

                # Run GPU and CPU processing in parallel threads
                print(f"\n  Starting hybrid processing...")
                with ThreadPoolExecutor(max_workers=2) as thread_executor:
                    gpu_future = thread_executor.submit(process_gpu_cells)
                    cpu_future = thread_executor.submit(process_cpu_cells)

                    # Progress monitoring
                    import time
                    with tqdm(total=n_cells, desc="Processing cells (hybrid)", unit="cell") as pbar:
                        last_total = 0
                        while not (gpu_future.done() and cpu_future.done()):
                            with progress_lock:
                                current_total = gpu_progress[0] + cpu_progress[0]
                            if current_total > last_total:
                                pbar.update(current_total - last_total)
                                last_total = current_total
                            time.sleep(0.5)

                        # Final update
                        with progress_lock:
                            current_total = gpu_progress[0] + cpu_progress[0]
                        if current_total > last_total:
                            pbar.update(current_total - last_total)

                    # Wait for both to complete and check for exceptions
                    gpu_future.result()
                    cpu_future.result()

                # Merge results
                results_list = gpu_results + cpu_results
                print(f"  GPU processed: {len(gpu_results)} cells")
                print(f"  CPU processed: {len(cpu_results)} cells")
                print(f"  Total: {len(results_list)} cells")

            # PIPELINED GPU MODE: Multiple workers with small batches for natural overlap
            elif _DASK_AVAILABLE and _USE_GPU_FEATURES:
                import multiprocessing
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                # Scale workers based on available GPU VRAM
                # Each worker uses ~3GB peak VRAM for morphology + localization montages
                # Reserve 10GB for CuPy overhead, CUDA context, etc.
                CELLS_PER_BATCH = 512  # Larger batches = fewer GPU kernel launches, better utilization
                _PER_WORKER_VRAM_GB = 3.0
                _RESERVED_VRAM_GB = 10.0
                try:
                    import cupy as _cp
                    _free_mem, _total_mem = _cp.cuda.Device(0).mem_info
                    _total_gb = _total_mem / (1024**3)
                    _usable_gb = _total_gb - _RESERVED_VRAM_GB
                    N_WORKERS = max(8, min(64, int(_usable_gb / _PER_WORKER_VRAM_GB)))
                    print(f"  [GPU] VRAM: {_total_gb:.0f}GB total, {_usable_gb:.0f}GB usable -> {N_WORKERS} workers ({_PER_WORKER_VRAM_GB}GB/worker)")
                except Exception as _e:
                    N_WORKERS = 24  # Fallback
                    print(f"  [GPU] Could not query VRAM ({_e}), using {N_WORKERS} workers")

                # Convert DataFrame to dict
                cells_dict = well_cells_df.to_dict('records')
                n_batches = (n_cells + CELLS_PER_BATCH - 1) // CELLS_PER_BATCH

                print(f"Processing {n_cells} cells in {well} with PIPELINED architecture...")
                print(f"  Workers: {N_WORKERS} (natural pipeline overlap)")
                print(f"  Batch size: {CELLS_PER_BATCH} cells")
                print(f"  Total batches: {n_batches}")

                # Get available GPUs and conda prefix for the plugin
                import torch
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                available_gpus = list(range(num_gpus)) if num_gpus > 0 else []
                conda_prefix = os.environ.get('CONDA_PREFIX', '')
                print(f"  [GPU] Available GPUs: {available_gpus}, CONDA_PREFIX: {conda_prefix}")

                # Dynamic memory limit: use system RAM minus overhead, divided among workers
                import psutil
                _total_ram_gb = psutil.virtual_memory().total / (1024**3)
                _reserved_ram_gb = 40  # Main process, OS, CUDA driver overhead
                _mem_per_worker_gb = max(4, int((_total_ram_gb - _reserved_ram_gb) / N_WORKERS))
                _memory_limit = f'{_mem_per_worker_gb}GB'
                print(f"  [RAM] {_total_ram_gb:.0f}GB total, {_reserved_ram_gb}GB reserved -> {_mem_per_worker_gb}GB/worker Dask limit")

                cluster = LocalCluster(
                    n_workers=N_WORKERS,
                    threads_per_worker=1,
                    processes=True,
                    memory_limit=_memory_limit,
                )
                client = Client(cluster)

                # Register plugin to set LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES in each worker
                plugin = CUDALibraryPlugin(available_gpus, conda_prefix)
                client.register_worker_plugin(plugin)
                print(f"  [GPU] Registered CUDALibraryPlugin for {len(available_gpus)} GPUs")

                try:
                    # Submit ALL batches upfront - Dask will manage parallelism
                    futures = []
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * CELLS_PER_BATCH
                        end_idx = min(start_idx + CELLS_PER_BATCH, n_cells)
                        if start_idx < n_cells:
                            batch_cells = cells_dict[start_idx:end_idx]
                            future = client.submit(
                                _dask_process_cell_batch,
                                batch_start=0,
                                batch_end=len(batch_cells),
                                cells_dict=batch_cells,
                                store_path=store_path,
                                well=well,
                                available_labels=available_labels,
                                organelles_to_process=organelles_to_process,
                                network_organelles=network_organelles,
                                spacing=spacing,
                                channel_names=channel_names,
                                organelle_map=organelle_map,
                                full_features=full_features,
                                initial_yx_patch_size=initial_yx_patch_size,
                                max_objects_per_organelle=max_objects_per_organelle,
                                debug=debug and batch_idx == 0,
                                available_gpus=available_gpus,
                            )
                            futures.append((future, end_idx - start_idx))

                    # ============================================================
                    # OVERLAPPED GPU + CPU: Start network analysis as GPU batches complete
                    # ============================================================
                    from dask.distributed import as_completed as dask_as_completed
                    import time as time_mod
                    from concurrent.futures import ProcessPoolExecutor, as_completed as cpu_as_completed

                    future_to_size = {f: size for f, size in futures}

                    # Start CPU pool for overlapped network analysis
                    n_cpu_workers = min(64, os.cpu_count() or 64)
                    net_executor = ProcessPoolExecutor(max_workers=n_cpu_workers)
                    net_futures = []  # List of Future objects
                    net_work_count = 0
                    t_net_start = time_mod.time()

                    cells_processed = 0
                    with tqdm(total=n_cells, desc="GPU phases", unit="cell") as pbar:
                        for completed_future in dask_as_completed([f for f, _ in futures]):
                            n_batch_cells = future_to_size[completed_future]
                            try:
                                batch_results, batch_network_items, _ = completed_future.result()
                                if batch_results:
                                    # Remap local_idx to global result index
                                    base_idx = len(results_list)
                                    results_list.extend(batch_results)
                                    # Immediately submit network tasks to CPU pool (overlapped)
                                    for item in batch_network_items:
                                        local_idx, org_name, org_mask, crop_info, sp = item
                                        work_item = (base_idx + local_idx, org_name, org_mask, crop_info, sp)
                                        nf = net_executor.submit(_run_network_analysis_single, work_item)
                                        net_futures.append(nf)
                                        net_work_count += 1
                                pbar.update(n_batch_cells)
                                cells_processed += n_batch_cells
                            except Exception as e:
                                print(f"  Batch failed: {e}")
                                import traceback
                                traceback.print_exc()
                                pbar.update(n_batch_cells)

                finally:
                    try:
                        client.close()
                        cluster.close(timeout=120)
                    except Exception:
                        pass

                # ============================================================
                # PHASE 4: Collect network results (many already done from overlap)
                # ============================================================
                if net_futures:
                    already_done = sum(1 for nf in net_futures if nf.done())
                    print(f"  Network analysis: {net_work_count} tasks across {n_cpu_workers} CPU workers (overlapped)")
                    print(f"    {already_done}/{net_work_count} tasks already completed during GPU phase")

                    # Collect all results - early futures return instantly (already done)
                    n_net_completed = 0
                    from collections import defaultdict
                    timing_accum = defaultdict(list)

                    with tqdm(total=len(net_futures), desc="Network analysis", unit="task",
                              initial=already_done) as net_pbar:
                        collected = 0
                        for nf in net_futures:
                            net_result = nf.result()
                            collected += 1
                            if collected > already_done:
                                net_pbar.update(1)
                            if net_result is None:
                                continue
                            global_idx, organelle_name, branch_df, network_summary_dict, per_object_network_df, crop_info, task_timings = net_result
                            if global_idx < len(results_list):
                                cell_features, object_features, network_features = results_list[global_idx]
                                if not branch_df.empty:
                                    branch_df = branch_df.copy()
                                    branch_df["cell_id"] = crop_info.get("global_cell_id")
                                    branch_df["total_index"] = crop_info.get("total_index")
                                    network_features[organelle_name] = {
                                        "branch_df": branch_df,
                                        "per_object_df": per_object_network_df,
                                    }
                                if network_summary_dict:
                                    for key, value in network_summary_dict.items():
                                        cell_features[f"network_{organelle_name}_{key}"] = value
                                n_net_completed += 1
                            # Collect timings
                            if task_timings:
                                for k, v in task_timings.items():
                                    timing_accum[k].append(v)

                    net_executor.shutdown(wait=True)
                    t_net_total = time_mod.time() - t_net_start
                    print(f"  Network analysis complete: {n_net_completed}/{net_work_count} tasks in {t_net_total:.1f}s (includes GPU overlap)")

                    # Print profiling summary
                    if timing_accum:
                        print(f"  Network analysis profiling ({n_net_completed} tasks):")
                        step_order = [
                            "label_regionprops_euler", "skeletonize_clean", "skeleton_label",
                            "skan_analysis", "network_wide_features", "distance_transform",
                            "branch_thickness", "tortuosity", "per_object_features",
                        ]
                        total_cpu_time = 0.0
                        for step in step_order:
                            if step in timing_accum:
                                vals = timing_accum[step]
                                total = sum(vals)
                                total_cpu_time += total
                                mean = total / len(vals)
                                mx = max(vals)
                                print(f"    {step:30s}: total={total:8.1f}s  mean={mean*1000:7.1f}ms  max={mx*1000:7.1f}ms  n={len(vals)}")
                        print(f"    {'TOTAL (sum across tasks)':30s}: {total_cpu_time:8.1f}s")
                        print(f"    {'Wall time (parallel)':30s}: {t_net_total:8.1f}s")
                        print(f"    {'Parallelism efficiency':30s}: {total_cpu_time/t_net_total:.1f}x across {n_cpu_workers} workers")
                        if "num_branches" in timing_accum:
                            branches = timing_accum["num_branches"]
                            print(f"    Branches per task: mean={np.mean(branches):.0f}  max={max(branches):.0f}  total={sum(branches):.0f}")

            else:
                # Fall back to ProcessPoolExecutor (CPU mode or Dask not available)
                from concurrent.futures import ProcessPoolExecutor, as_completed

                # Convert DataFrame to dict for pickling
                cells_dict = well_cells_df.to_dict('records')
                TARGET_CELLS_PER_BATCH = 64
                batch_size = min(TARGET_CELLS_PER_BATCH, max(1, (n_cells + n_jobs - 1) // n_jobs))
                n_batches = (n_cells + batch_size - 1) // batch_size
                print(f"Processing {n_cells} chunk-sorted cells in {well} with {n_jobs} workers...")
                print(f"  Batching into {n_batches} contiguous batches of ~{batch_size} cells each (cache-optimized)")

                # Prepare static args that don't change per cell
                static_args = {
                    'store_path': store_path,
                    'well': well,
                    'available_labels': available_labels,
                    'organelles_to_process': organelles_to_process,
                    'network_organelles': network_organelles,
                    'spacing': spacing,
                    'channel_names': channel_names,
                    'organelle_map': organelle_map,
                    'full_features': full_features,
                    'initial_yx_patch_size': initial_yx_patch_size,
                    'max_objects_per_organelle': max_objects_per_organelle,
                }

                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_init_worker_tensorstore,
                    initargs=(store_path, well, available_labels),
                ) as executor:
                    # Submit batch tasks - each worker gets a contiguous range of cells
                    futures = {}
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, n_cells)
                        if start_idx < n_cells:
                            future = executor.submit(
                                _process_cell_batch_with_global_handles,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                cells_dict=cells_dict,
                                static_args=static_args,
                                debug=debug and batch_idx == 0,
                            )
                            futures[future] = (batch_idx, end_idx - start_idx)

                    # Collect results with progress tracking
                    cells_processed = 0
                    with tqdm(total=n_cells, desc="Processing cells", unit="cell") as pbar:
                        for future in as_completed(futures):
                            batch_idx, batch_cells = futures[future]
                            try:
                                batch_results = future.result()
                                if batch_results:
                                    results_list.extend(batch_results)
                                pbar.update(batch_cells)
                                cells_processed += batch_cells
                            except Exception as e:
                                print(f"  Batch {batch_idx} failed: {e}")
                                pbar.update(batch_cells)

            print(f"  Extracted features from {len(results_list)}/{n_cells} cells")
        else:
            # Sequential processing (when n_jobs=1 for benchmarking or multi-well parallel mode)
            import time
            results_list = []
            start_time = time.time()

            print(f"Processing {n_cells} cells SEQUENTIALLY in {well}...")
            with tqdm(total=n_cells, desc="Sequential cells", unit="cell") as pbar:
                for i in range(n_cells):
                    result = _process_single_cell_in_well(
                        cell_index=i,
                        base_dataset=base_dataset,
                        labels_group=labels_group,
                        available_labels=available_labels,
                        organelles_to_process=organelles_to_process,
                        network_organelles=network_organelles,
                        spacing=spacing,
                        channel_names=channel_names,
                        organelle_map=organelle_map,
                        full_features=full_features,
                        well=well,
                        position_fov=position_fov,
                        max_objects_per_organelle=max_objects_per_organelle,
                        debug=debug,
                    )
                    if result is not None:
                        results_list.append(result)

                    pbar.update(1)
                    # Update cells/sec in postfix
                    elapsed = time.time() - start_time
                    cells_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"cells/sec": f"{cells_per_sec:.2f}"})

        # Debug: print total results collected
        print(f"  [DEBUG] _worker_process_well returning {len(results_list)} results for {n_cells} cells")
        return results_list

    finally:
        # Always close the store
        pheno_store.close()

def _process_single_cell_in_well(
    cell_index: int,
    base_dataset,
    labels_group,
    available_labels: dict,
    organelles_to_process: list,
    network_organelles: list,
    spacing: tuple,
    channel_names: list,
    organelle_map: dict,
    full_features: bool,
    well: str,
    position_fov=None,
    max_objects_per_organelle: int = None,
    debug: bool = False,
):
    """
    Process a single cell within a well.

    This is the innermost processing function called by batch workers.

    Parameters
    ----------
    position_fov : zarr array, optional
        The FOV intensity data for this well (position["0"]).
        Used to load cp_intensity for CP organelles at cp_bbox location.
    """
    try:
        # Use BaseDataset.__getitem__ to load cell data and mask
        batch = base_dataset[cell_index]

        # Extract data from batch
        data = batch["data"].numpy() if hasattr(batch["data"], 'numpy') else np.array(batch["data"])
        mask = batch["mask"].numpy() if hasattr(batch["mask"], 'numpy') else np.array(batch["mask"])
        crop_info = batch["crop_info"]

        # Get cell-specific binary mask (BaseDataset already isolates the cell)
        cell_specific_mask = mask[0].astype(np.uint8)  # (H, W)

        if not np.any(cell_specific_mask):
            return None

        # Load organelle segmentation labels for this cell's crop region
        # Dual bbox system: cp_bbox for CP organelles, bbox for standard organelles
        organelle_mask_arrays_crop = {}
        frangi_image_arrays_crop = {}

        if labels_group is not None and available_labels:
            # Get both bboxes from crop_info
            # cp_bbox: Cell location at CellPainting imaging time (for cp* organelles)
            # bbox: Cell location at phenotyping imaging time (for standard organelles)
            pheno_bbox = batch.get("bbox")  # Standard bbox from phenotyping
            cp_bbox = crop_info.get("cp_bbox")  # CP bbox from cell painting

            # Get both segmentation IDs
            seg_id = crop_info.get("segmentation_id")  # For cell_seg (pheno)
            cp_seg_id = crop_info.get("cp_cell_seg_id")  # For cp_cell_seg (CP)

            if pd.notna(seg_id):
                seg_id = int(seg_id)
            else:
                seg_id = None
            if pd.notna(cp_seg_id) if cp_seg_id is not None else False:
                cp_seg_id = int(cp_seg_id)
            else:
                cp_seg_id = seg_id  # Fallback to seg_id if cp_seg_id not available

            # Helper to load label crop at specific bbox coordinates
            def _load_label_at_bbox(label_array, bbox_coords):
                y_min, x_min, y_max, x_max = bbox_coords
                # Handle various array dimensions (labels can be 2D, 3D, 4D, or 5D)
                if label_array.ndim == 5:
                    # 5D: (T, C, Z, Y, X) - zarr v3 format
                    return np.array(label_array[0, 0, 0, y_min:y_max, x_min:x_max])
                elif label_array.ndim == 4:
                    # 4D: (T, C, Y, X) or (T, Z, Y, X)
                    return np.array(label_array[0, 0, y_min:y_max, x_min:x_max])
                elif label_array.ndim == 3:
                    # 3D: (Z, Y, X) or (C, Y, X)
                    return np.array(label_array[0, y_min:y_max, x_min:x_max])
                else:
                    # 2D: (Y, X)
                    return np.array(label_array[y_min:y_max, x_min:x_max])

            # Load cp_cell_mask and cp_intensity using cp_bbox (for CP organelles)
            cp_cell_mask = None
            cp_intensity = None
            if cp_bbox is not None and "cp_cell_seg" in labels_group and cp_seg_id is not None:
                try:
                    cp_label_array = labels_group["cp_cell_seg"]["0"]
                    cp_cell_mask_raw = _load_label_at_bbox(cp_label_array, cp_bbox)
                    cp_cell_mask = (cp_cell_mask_raw == cp_seg_id).astype(np.uint8)

                    # Load intensity at cp_bbox location (different from phenotyping bbox!)
                    # This is critical: the cell has moved between CP and pheno imaging
                    if position_fov is not None:
                        cp_y_min, cp_x_min, cp_y_max, cp_x_max = cp_bbox
                        cp_intensity = np.array(position_fov[0, :, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                except Exception:
                    cp_cell_mask = None
                    cp_intensity = None

            # Process each organelle with appropriate bbox
            for internal_name, zarr_label_name in available_labels.items():
                # Skip cell_mask - we already have it
                if internal_name == "cell_mask":
                    continue

                if zarr_label_name not in labels_group:
                    continue

                # Determine which bbox to use based on organelle type
                # CP organelles (prefixed with "cp" or "CP") use cp_bbox
                # Standard organelles use pheno_bbox
                is_cp_organelle = internal_name.lower().startswith("cp")

                if is_cp_organelle:
                    bbox_to_use = cp_bbox
                    mask_to_use = cp_cell_mask
                    target_shape = cp_cell_mask.shape if cp_cell_mask is not None else None
                else:
                    bbox_to_use = pheno_bbox
                    mask_to_use = cell_specific_mask
                    target_shape = cell_specific_mask.shape

                if bbox_to_use is None:
                    continue

                try:
                    label_array = labels_group[zarr_label_name]["0"]
                    label_crop = _load_label_at_bbox(label_array, bbox_to_use)

                    # Use the appropriate cell mask for this organelle type
                    if mask_to_use is None:
                        continue

                    # Ensure shapes match (resize if needed due to bbox size differences)
                    if label_crop.shape != mask_to_use.shape:
                        matched_crop = np.zeros(mask_to_use.shape, dtype=label_crop.dtype)
                        h = min(label_crop.shape[0], matched_crop.shape[0])
                        w = min(label_crop.shape[1], matched_crop.shape[1])
                        matched_crop[:h, :w] = label_crop[:h, :w]
                        label_crop = matched_crop

                    # Mask to only include labels within the cell boundary
                    label_crop = label_crop * (mask_to_use > 0).astype(label_crop.dtype)

                    if np.any(label_crop > 0):
                        organelle_mask_arrays_crop[internal_name] = label_crop

                except Exception:
                    pass

        # Extract nuclear mask for localization features
        # Look for nuclei or nuclear_seg in the organelle arrays (standard bbox)
        nuclear_mask = None
        for nuc_key in ("nuclei", "nuclear_seg"):
            if nuc_key in organelle_mask_arrays_crop:
                nuc_arr = organelle_mask_arrays_crop[nuc_key]
                # Convert to binary mask
                nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                break

        # Extract CP nuclear mask for CP organelles (cp_bbox)
        # Look for cp*_nucl* patterns (e.g., cp1_nuclei_hoechst)
        cp_nuclear_mask = None
        for key in organelle_mask_arrays_crop:
            if key.lower().startswith("cp") and "nucl" in key.lower():
                nuc_arr = organelle_mask_arrays_crop[key]
                cp_nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                break

        # Call processing function
        cell_features, object_features, network_features = (
            process_single_cell(
                crop_info,
                cell_specific_mask,
                organelle_mask_arrays_crop,
                data,  # (C, H, W) at phenotyping bbox
                frangi_image_arrays_crop,
                organelles_to_process,
                network_organelles,
                spacing,
                channel_names,
                organelle_map,
                full_features,
                nuclear_mask=nuclear_mask,
                cp_intensity_image=cp_intensity,  # (C, H, W) at cp_bbox (different location!)
                cp_cell_mask=cp_cell_mask,  # Cell mask at cp_bbox
                cp_nuclear_mask=cp_nuclear_mask,  # Nuclear mask at cp_bbox for CP organelles
                max_objects_per_organelle=max_objects_per_organelle,
                debug=debug,
            )
        )
        if cell_features is not None:
            return (cell_features, object_features, network_features)
        return None

    except Exception as e:
        import traceback
        print(f"Error processing cell {cell_index} in well {well}: {e}")
        return None


# =============================================================================
# Parallel partial loading (module-level for ProcessPoolExecutor pickling)
# =============================================================================

def _load_gpu_partials_chunk(chunk_info):
    """Load a chunk of GPU partial pickle files, aggregate, write as parquet.

    Runs in subprocess via ProcessPoolExecutor. Returns only file paths
    (no large Python objects through IPC). pickle.load() is CPU-bound and
    GIL-limited, so processes (not threads) are required for parallelism.
    """
    partial_paths, chunk_id, output_dir = chunk_info
    import pickle
    import os
    import pandas as pd

    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_id:04d}")
    os.makedirs(chunk_dir, exist_ok=True)

    cell_features_rows = []
    object_features = {}  # {organelle: [df, ...]}
    network_meta = []
    morph_meta = []
    result_idx = 0

    for path in partial_paths:
        with open(path, 'rb') as f:
            partial = pickle.load(f)

        for item in partial['network_items']:
            item['result_idx'] = result_idx + item['local_idx']
            network_meta.append(item)
        for item in partial['morph_items']:
            item['result_idx'] = result_idx + item['local_idx']
            morph_meta.append(item)

        for cell_feat, obj_feat, _net_feat in partial['results']:
            if cell_feat is not None:
                cell_features_rows.append(cell_feat)
            if obj_feat:
                for org, df in obj_feat.items():
                    if not df.empty:
                        object_features.setdefault(org, []).append(df)
            result_idx += 1
        del partial

    n_cells = len(cell_features_rows)

    # Write cell features
    if cell_features_rows:
        pd.DataFrame(cell_features_rows).to_parquet(
            os.path.join(chunk_dir, "cell_features.parquet"), index=False)
    del cell_features_rows

    # Write per-organelle object features
    orgs_written = []
    for org, dfs in object_features.items():
        pd.concat(dfs, ignore_index=True).to_parquet(
            os.path.join(chunk_dir, f"obj_{org}.parquet"), index=False)
        orgs_written.append(org)
    del object_features

    # Write network/morph metadata (expand bbox tuples to columns for parquet)
    n_net = len(network_meta)
    n_morph = len(morph_meta)

    if network_meta:
        ndf = pd.DataFrame(network_meta)
        if 'bbox' in ndf.columns:
            bbox_df = pd.DataFrame(
                ndf['bbox'].tolist(),
                columns=['bbox_y0', 'bbox_x0', 'bbox_y1', 'bbox_x1'])
            ndf = pd.concat([ndf.drop(columns=['bbox']), bbox_df], axis=1)
        ndf.to_parquet(os.path.join(chunk_dir, "network_meta.parquet"), index=False)

    if morph_meta:
        mdf = pd.DataFrame(morph_meta)
        if 'bbox' in mdf.columns:
            bbox_df = pd.DataFrame(
                mdf['bbox'].tolist(),
                columns=['bbox_y0', 'bbox_x0', 'bbox_y1', 'bbox_x1'])
            mdf = pd.concat([mdf.drop(columns=['bbox']), bbox_df], axis=1)
        mdf.to_parquet(os.path.join(chunk_dir, "morph_meta.parquet"), index=False)

    return (chunk_dir, n_cells, n_net, n_morph, orgs_written)


# =============================================================================
# Split-mode workers: GPU-only and CPU-only phases
# =============================================================================

def gpu_phase_worker(
    experiment: str,
    well: str,
    batch_idx: int,
    batch_cells_df: pd.DataFrame,
    output_dir: str,
    full_features: bool = True,
    n_workers_override: int = None,
    io_threads_override: int = None,
    max_cells: int = None,
):
    """
    GPU-only phase: runs morphology + localization, saves features and network task metadata.

    This is the first half of the split GPU/CPU pipeline. It:
    1. Runs Dask GPU workers for morphology batching + localization
    2. Saves GPU features (cell + object) to parquet
    3. Saves network task metadata (bbox coordinates, not masks) to parquet

    The CPU job can then read the metadata and re-crop masks from zarr.

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0')
    batch_idx : int
        Batch index within the well
    batch_cells_df : pd.DataFrame
        DataFrame with cell metadata for this batch
    output_dir : str
        Directory to save intermediate results
    full_features : bool
        Whether to compute expensive features

    Returns
    -------
    dict
        Result with status, paths to output files, cell count
    """
    _require_gpu_for_feature_extraction()

    import time as time_mod
    import multiprocessing
    import sys as _sys

    output_dir = Path(output_dir)
    well_safe = well.replace("/", "_")
    batch_id = f"{well_safe}_{batch_idx:04d}"
    batch_results_dir = output_dir / "_batch_results"
    batch_results_dir.mkdir(parents=True, exist_ok=True)
    gpu_partials_dir = batch_results_dir / f"_gpu_partials_{batch_id}"
    gpu_partials_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GPU Phase: {batch_id}")
    print(f"Experiment: {experiment}")
    print(f"Well: {well}")
    print(f"Cells: {len(batch_cells_df)} (batch {batch_idx})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    t_start = time_mod.time()

    # Initialize experiment data
    from cyclops_utils.data.experiment import OpsDataset
    ds = OpsDataset(experiment)
    store_path = str(ds.store_paths["pheno_assembled_v3"])

    from organelle_profiler.feature_extraction.fe_metadata import _discover_available_labels
    from organelle_profiler.feature_extraction.feature_extraction import is_network_organelle

    available_labels = _discover_available_labels(Path(store_path))
    organelles_to_process = [name for name in available_labels.keys()
                             if name not in ('cell_mask', 'cp_cell_mask', 'CP2_nuclear')]
    network_organelles = [name for name in available_labels.keys() if is_network_organelle(name)]
    print(f"Organelles: {len(organelles_to_process)}, Network: {len(network_organelles)}")

    # Get channel names and spacing
    with open_ome_zarr(store_path, mode="r") as store:
        channel_names = store.channel_names
        position = store[well]
        spacing = tuple(position.scale[-2:])

    organelle_map = {}  # Empty for now
    initial_yx_patch_size = (512, 512)

    # Truncate cells for benchmarking if requested
    if max_cells and len(batch_cells_df) > max_cells:
        batch_cells_df = batch_cells_df.head(max_cells).copy()
        print(f"  [Benchmark] Truncated to {max_cells} cells (from {len(batch_cells_df) + (len(batch_cells_df) - max_cells)})")

    # Sort cells by chunk for zarr locality
    n_cells = len(batch_cells_df)
    if 'y_global_pheno' in batch_cells_df.columns:
        CHUNK_SIZE = 512
        y_coords = batch_cells_df['y_global_pheno'].fillna(999999)
        x_coords = batch_cells_df['x_global_pheno'].fillna(999999)
        batch_cells_df = batch_cells_df.copy()
        batch_cells_df["_chunk_y"] = (y_coords // CHUNK_SIZE).astype(int)
        batch_cells_df["_chunk_x"] = (x_coords // CHUNK_SIZE).astype(int)
        batch_cells_df = batch_cells_df.sort_values(
            ['_chunk_y', '_chunk_x', 'y_global_pheno', 'x_global_pheno'],
            na_position="last"
        ).reset_index(drop=True)
        batch_cells_df = batch_cells_df.drop(columns=["_chunk_y", "_chunk_x"])

    # Run GPU phase via Dask
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Ramping batch sizes to stagger workers: small batches first to desynchronize,
    # then large batches for sustained GPU throughput
    BATCH_RAMP = [32, 64, 128, 256]  # Ramp up to stagger workers, then stay at 256
    CELLS_PER_BATCH = BATCH_RAMP[-1]  # Steady-state size (used for estimate)
    # GPU-only mode: all workers are on GPU simultaneously (no CPU network phase to stagger them)
    # Combined mode uses 3.0GB/worker because mask assembly naturally staggers GPU access
    # 3.0GB/worker: on the 141GB H200 this gives ~43 workers/GPU (matching the
    # old combined run that hit ~90 cells/sec). Peak per-worker is ~3-4GB, so
    # this is VRAM-tight in GPU-only mode (43×3≈129/130 usable) — the batch ramp
    # staggers allocations to absorb peaks. (Was 4.0 -> only 17 on an 80GB card.)
    _PER_WORKER_VRAM_GB = 3.0
    _RESERVED_VRAM_GB = 10.0
    # Cap is 24 by default. 4i experiments hit synchronized
    # "Event loop unresponsive in Nanny for ~6s" pile-ups across all
    # 24 workers + scheduler — GPFS I/O bandwidth saturates when 24
    # workers all pull multi-GB 4i image chunks at once. The submitter
    # exports `OPS_FE_4I_EXPERIMENT=1` for 4i runs to lower the cap to
    # 16, which cuts parallel I/O pressure by 33%. Live-cell experiments
    # keep the 24-worker default — their feature stacks are smaller and
    # 24 saturates the GPU without triggering I/O storms.
    _is_4i_experiment = os.environ.get("OPS_FE_4I_EXPERIMENT", "").strip() == "1"
    _MAX_WORKERS_PER_GPU = 16 if _is_4i_experiment else 44
    if _is_4i_experiment:
        print(f"  [GPU] 4i experiment detected (OPS_FE_4I_EXPERIMENT=1) — "
              f"capping workers at {_MAX_WORKERS_PER_GPU}/GPU to ease I/O contention")
    try:
        import cupy as _cp
        num_gpus = _cp.cuda.runtime.getDeviceCount()
        # Use smallest GPU's VRAM for worker calculation (conservative)
        _total_gb = min(
            _cp.cuda.Device(i).mem_info[1] / (1024**3) for i in range(num_gpus)
        )
        _usable_gb = _total_gb - _RESERVED_VRAM_GB
        _workers_per_gpu = max(8, min(_MAX_WORKERS_PER_GPU, int(_usable_gb / _PER_WORKER_VRAM_GB)))
        N_WORKERS = _workers_per_gpu * num_gpus
        print(f"  [GPU] {num_gpus} GPU(s), {_total_gb:.0f}GB VRAM each, {_usable_gb:.0f}GB usable -> {_workers_per_gpu}/GPU × {num_gpus} = {N_WORKERS} total ({_PER_WORKER_VRAM_GB}GB/worker, cap={_MAX_WORKERS_PER_GPU}/GPU)")
    except Exception as _e:
        num_gpus = 1
        N_WORKERS = 24
        print(f"  [GPU] Could not query VRAM ({_e}), using {N_WORKERS} workers")

    # Cap workers by RAM: each worker needs ~13GB for I/O threads + montage processing
    _MIN_RAM_PER_WORKER_GB = 13
    _slurm_mem_mb_str = os.environ.get('SLURM_MEM_PER_NODE', '')
    if _slurm_mem_mb_str:
        _slurm_ram_gb = int(_slurm_mem_mb_str) / 1024
    else:
        import psutil as _psutil
        _slurm_ram_gb = _psutil.virtual_memory().total / (1024**3)
    _ram_reserved = 40  # Main process, OS, CUDA driver
    _max_workers_by_ram = max(4, int((_slurm_ram_gb - _ram_reserved) / _MIN_RAM_PER_WORKER_GB))
    if N_WORKERS > _max_workers_by_ram:
        print(f"  [RAM] Capping workers {N_WORKERS} -> {_max_workers_by_ram} "
              f"({_slurm_ram_gb:.0f}GB alloc, {_MIN_RAM_PER_WORKER_GB}GB/worker min)")
        N_WORKERS = _max_workers_by_ram

    # Apply overrides for benchmarking
    if n_workers_override is not None:
        N_WORKERS = n_workers_override
        print(f"  [Benchmark] Overriding workers: {N_WORKERS}")

    # Store I/O thread override as env var so task function can read it
    if io_threads_override is not None:
        os.environ['_FE_IO_THREADS'] = str(io_threads_override)
        print(f"  [Benchmark] Overriding I/O threads: {io_threads_override}")

    cells_dict = batch_cells_df.to_dict('records')

    print(f"GPU phase: {n_cells} cells, {N_WORKERS} Dask workers, ramp {BATCH_RAMP} -> {BATCH_RAMP[-1]} steady")

    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    available_gpus = list(range(num_gpus)) if num_gpus > 0 else []
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    print(f"  [GPU] Available GPUs: {available_gpus}, CONDA_PREFIX: {conda_prefix}")

    # Dynamic memory limit: use SLURM allocation (not total system RAM) divided among workers
    import psutil
    _total_ram_gb = psutil.virtual_memory().total / (1024**3)
    # Use SLURM memory allocation if available, otherwise fall back to system RAM
    _slurm_mem_mb = os.environ.get('SLURM_MEM_PER_NODE', '')
    if _slurm_mem_mb:
        _available_ram_gb = int(_slurm_mem_mb) / 1024
    else:
        _available_ram_gb = _total_ram_gb
    _reserved_ram_gb = 40  # Main process, OS, CUDA driver overhead
    _mem_per_worker_gb = max(4, int((_available_ram_gb - _reserved_ram_gb) / N_WORKERS))
    _memory_limit = f'{_mem_per_worker_gb}GB'
    print(f"  [RAM] {_available_ram_gb:.0f}GB SLURM alloc ({_total_ram_gb:.0f}GB system), {_reserved_ram_gb}GB reserved -> {_mem_per_worker_gb}GB/worker Dask limit")

    # Workers inherit CUDA_VISIBLE_DEVICES from SLURM so they see the allocated GPUs.
    # GPU device selection is done inside the task function via cupy.cuda.Device(gpu_id).use()
    # (NOT via CUDA_VISIBLE_DEVICES in the plugin, which is too late after CUDA runtime init)
    print(f"  [GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')} (workers inherit this)")

    # Check for existing partials (resume support)
    import glob as glob_mod
    existing_partials = sorted(glob_mod.glob(str(gpu_partials_dir / "gpu_partial_*.pkl")))
    existing_indices = set()
    for p in existing_partials:
        fname = os.path.basename(p)
        try:
            idx = int(fname.replace("gpu_partial_", "").replace(".pkl", ""))
            existing_indices.add(idx)
        except ValueError:
            pass

    # Build batch index → cursor mapping so we can skip completed batches
    batch_ranges = []  # (bi, cursor_start, cursor_end)
    cursor = 0
    bi = 0
    while cursor < n_cells:
        if bi < len(BATCH_RAMP):
            batch_size = BATCH_RAMP[bi]
        else:
            batch_size = BATCH_RAMP[-1]
        end_idx = min(cursor + batch_size, n_cells)
        batch_ranges.append((bi, cursor, end_idx))
        cursor = end_idx
        bi += 1
    n_batches = len(batch_ranges)
    n_existing = len(existing_indices)

    if n_existing > 0:
        print(f"  [Resume] Found {n_existing}/{n_batches} existing partials, submitting {n_batches - n_existing} remaining")

    if n_existing >= n_batches:
        # All partials already exist — skip Dask entirely
        print(f"  [Resume] All {n_batches} partials exist, skipping GPU compute")
    else:
        import dask
        dask.config.set({
            "distributed.worker.memory.pause": False,  # Don't pause — unmanaged memory (CuPy pools) can't be freed
            # terminate left at default (0.95) — nanny restarts worker with fresh memory;
            # only 1 in-flight batch is lost since completed results are on disk
        })
        # Start all workers - Dask scheduler assigns work as workers become ready
        # No need to wait - ready workers will pick up batches immediately
        cluster = LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=1,
            processes=True,
            memory_limit=_memory_limit,
        )
        client = Client(cluster)

        plugin = CUDALibraryPlugin(available_gpus, conda_prefix)
        client.register_worker_plugin(plugin)
        print(f"  [GPU] Cluster started, workers initializing in background")

        try:
            futures = []
            n_submitted = 0
            for bi, start, end in batch_ranges:
                if bi in existing_indices:
                    continue  # Already have this partial on disk
                batch_cells = cells_dict[start:end]
                actual_size = end - start
                future = client.submit(
                    _dask_process_cell_batch,
                    batch_start=0,
                    batch_end=len(batch_cells),
                    cells_dict=batch_cells,
                    store_path=store_path,
                    well=well,
                    available_labels=available_labels,
                    organelles_to_process=organelles_to_process,
                    network_organelles=network_organelles,
                    spacing=spacing,
                    channel_names=channel_names,
                    organelle_map=organelle_map,
                    full_features=full_features,
                    initial_yx_patch_size=initial_yx_patch_size,
                    max_objects_per_organelle=None,
                    debug=False,
                    gpu_only=True,  # Emit metadata, not masks
                    available_gpus=available_gpus,
                    partials_dir=str(gpu_partials_dir),
                    batch_idx=bi,
                )
                futures.append((future, actual_size))
                n_submitted += 1
            n_total = n_existing + n_submitted
            print(f"  Submitted {n_submitted} batches (skipped {n_existing} existing)")
            print(f"  Progress: {n_existing}/{n_total} ({100*n_existing/n_total:.1f}%) complete at start")

            from dask.distributed import as_completed as dask_as_completed
            import time as _progress_time

            n_failed = 0
            _progress_start = _progress_time.time()
            # Stall detection: warn if no batches complete in 2 minutes
            STALL_TIMEOUT_SEC = 120
            _stall_warned = False

            # Calculate average batch size for cells/sec estimate
            _avg_batch_size = sum(size for _, size in futures) / len(futures) if futures else 256
            _completed_count = 0
            for completed_future in dask_as_completed([f for f, _ in futures]):
                # Stall detection: warn once if first batch takes too long
                _now = _progress_time.time()
                if _completed_count == 0 and not _stall_warned and (_now - _progress_start) > STALL_TIMEOUT_SEC:
                    print(f"  ⚠️ STALL WARNING: No batches completed in {STALL_TIMEOUT_SEC}s - workers may be stuck initializing", file=_sys.stderr, flush=True)
                    _stall_warned = True

                i = _completed_count
                _completed_count += 1
                try:
                    completed_future.result()
                    n_done = n_existing + i + 1
                    cells_done = (i + 1) * _avg_batch_size
                    elapsed = _progress_time.time() - _progress_start
                    batches_per_min = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    cells_per_sec = cells_done / elapsed if elapsed > 0 else 0
                    pct = 100 * n_done / n_total
                    remaining_batches = n_submitted - (i + 1)
                    eta_min = remaining_batches / batches_per_min if batches_per_min > 0 else 0
                    print(f"  Batch {i+1}/{n_submitted}: {n_done}/{n_total} ({pct:.1f}%) | {batches_per_min:.1f} batch/min | {cells_per_sec:.1f} cells/sec | ETA: {eta_min:.0f}min", file=_sys.stderr, flush=True)
                except Exception as e:
                    n_failed += 1
                    print(f"  Batch failed ({n_failed} total): {e}", file=_sys.stderr)
                    import traceback
                    traceback.print_exc()
                finally:
                    completed_future.release()

            if n_failed:
                print(f"  WARNING: {n_failed}/{n_submitted} batches failed", file=_sys.stderr)

        finally:
            # Drain in-flight background pickle writes on every worker BEFORE close.
            # Without this, Nanny SIGKILLs workers ~4s after the last task and any
            # daemon thread still pickling its partial gets killed mid-write,
            # silently dropping that batch's output file.
            try:
                client.run(_drain_pending_writes)
            except Exception:
                pass
            try:
                client.close()
                cluster.close(timeout=120)
            except Exception:
                pass

    t_gpu = time_mod.time() - t_start
    # Discover all partials on disk (existing + newly written)
    all_partial_paths = sorted(glob_mod.glob(str(gpu_partials_dir / "gpu_partial_*.pkl")))
    print(f"\nGPU compute: {len(all_partial_paths)}/{n_batches} partials in {t_gpu:.1f}s")

    # Load all partials via parallel processes (pickle.load is CPU-bound & GIL-limited)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    t_load_start = time_mod.time()

    n_load_cpus = min(os.cpu_count() or 16, max(1, len(all_partial_paths)))
    chunk_size = max(1, -(-len(all_partial_paths) // n_load_cpus))  # ceil division
    load_chunks_dir = gpu_partials_dir / "_load_chunks"
    load_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for ci in range(0, len(all_partial_paths), chunk_size):
        chunk_paths = all_partial_paths[ci:ci + chunk_size]
        chunks.append((chunk_paths, len(chunks), str(load_chunks_dir)))

    print(f"  Loading {len(all_partial_paths)} partials with {len(chunks)} processes...")
    ctx = mp.get_context('forkserver')
    with ProcessPoolExecutor(max_workers=n_load_cpus, mp_context=ctx) as pool:
        chunk_results = list(pool.map(_load_gpu_partials_chunk, chunks))

    # Stream-merge chunk parquet files using PyArrow (memory-efficient)
    # Instead of loading all chunks into memory then concat, we stream directly
    import pyarrow.parquet as pq
    import pyarrow as pa

    total_cells = 0
    n_net_total = 0
    n_morph_total = 0

    # Collect paths for streaming merge
    cell_parquet_paths = []
    object_parquet_paths = {org: [] for org in organelles_to_process}
    network_parquet_paths = []
    morph_parquet_paths = []

    for chunk_dir, n_cells, n_net, n_morph, orgs_written in chunk_results:
        total_cells += n_cells
        n_net_total += n_net
        n_morph_total += n_morph

        cell_path = os.path.join(chunk_dir, "cell_features.parquet")
        if os.path.exists(cell_path):
            cell_parquet_paths.append(cell_path)

        for org in orgs_written:
            obj_path = os.path.join(chunk_dir, f"obj_{org}.parquet")
            if os.path.exists(obj_path):
                if org in object_parquet_paths:
                    object_parquet_paths[org].append(obj_path)
                else:
                    object_parquet_paths[org] = [obj_path]

        net_path = os.path.join(chunk_dir, "network_meta.parquet")
        if os.path.exists(net_path):
            network_parquet_paths.append(net_path)

        morph_path = os.path.join(chunk_dir, "morph_meta.parquet")
        if os.path.exists(morph_path):
            morph_parquet_paths.append(morph_path)

    t_load = time_mod.time() - t_load_start
    print(f"  Indexed {len(all_partial_paths)} partials in {t_load:.1f}s ({len(chunks)} chunks)")
    print(f"  Cells: {total_cells}, Network tasks: {n_net_total}, Morph tasks: {n_morph_total}")

    # Stream-read cell features using PyArrow dataset (memory-efficient)
    print(f"  -> Step 1/3: Stream-loading cell features from {len(cell_parquet_paths)} chunks...")
    if cell_parquet_paths:
        # Use PyArrow dataset to read multiple parquet files as one logical table
        # This streams data and doesn't load all into memory at once
        cell_dataset = pq.ParquetDataset(cell_parquet_paths)
        cell_features_list = cell_dataset.read_pandas().to_pandas()
        del cell_dataset
    else:
        cell_features_list = pd.DataFrame()

    # DON'T load object features here - they stay in chunk parquets
    # CPU merge phase will do streaming aggregation to avoid OOM
    # Just track the paths for the CPU merge job
    obj_parquet_dir = batch_results_dir / f"batch_{batch_id}_obj_chunks"
    obj_parquet_dir.mkdir(parents=True, exist_ok=True)

    # Copy object parquets to a consolidated location for CPU merge
    import shutil as _shutil_obj
    n_obj_files = 0
    for org, paths in object_parquet_paths.items():
        if paths:
            org_dir = obj_parquet_dir / org
            org_dir.mkdir(parents=True, exist_ok=True)
            for i, src_path in enumerate(paths):
                dst_path = org_dir / f"chunk_{i:04d}.parquet"
                _shutil_obj.copy2(src_path, dst_path)
                n_obj_files += 1
    print(f"  -> Step 2/3: Copied {n_obj_files} object parquets to {obj_parquet_dir}")

    # Stream-load network/morph metadata
    if network_parquet_paths:
        net_dataset = pq.ParquetDataset(network_parquet_paths)
        network_meta_dfs = [net_dataset.read_pandas().to_pandas()]
        del net_dataset
    else:
        network_meta_dfs = []

    if morph_parquet_paths:
        morph_dataset = pq.ParquetDataset(morph_parquet_paths)
        morph_meta_dfs = [morph_dataset.read_pandas().to_pandas()]
        del morph_dataset
    else:
        morph_meta_dfs = []

    # Clean up load chunks (keep data in memory!)
    import shutil as _shutil_load
    try:
        _shutil_load.rmtree(str(load_chunks_dir))
    except Exception:
        pass

    # Write task metadata parquets (concat chunk metadata, add store_path)
    network_tasks_path = batch_results_dir / f"batch_{batch_id}_network_tasks.parquet"
    morph_supplement_tasks_path = batch_results_dir / f"batch_{batch_id}_morph_supplement_tasks.parquet"

    def _write_concat_meta(meta_dfs, output_path):
        if meta_dfs:
            df = pd.concat(meta_dfs, ignore_index=True)
            df['store_path'] = store_path
            df.to_parquet(output_path, index=False)
            return len(df)
        else:
            pd.DataFrame().to_parquet(output_path, index=False)
            return 0

    task_write_pool = ThreadPoolExecutor(max_workers=2)
    net_write_future = task_write_pool.submit(_write_concat_meta, network_meta_dfs, network_tasks_path)
    morph_write_future = task_write_pool.submit(_write_concat_meta, morph_meta_dfs, morph_supplement_tasks_path)

    # Write cell features directly (no aggregation here - CPU merge handles object aggregation)
    # cell_features_list contains per-cell measurements (cell morphology + localization summaries)
    print(f"  -> Step 3/3: Writing cell features...")

    gpu_features_path = batch_results_dir / f"batch_{batch_id}_gpu_features.parquet"

    if isinstance(cell_features_list, pd.DataFrame) and not cell_features_list.empty:
        # Join with metadata
        if batch_cells_df is not None and not batch_cells_df.empty:
            meta_df = batch_cells_df.copy()
            create_global_cell_id(meta_df)
            meta_df = meta_df.rename(columns={"global_cell_id": "cell_id"})

            # Match types for merge
            if "cell_id" in cell_features_list.columns:
                cell_features_list["cell_id"] = cell_features_list["cell_id"].astype(str)
                meta_df["cell_id"] = meta_df["cell_id"].astype(str)
                # Drop 'well' from cell_features_list if present - meta_df has authoritative metadata
                # This prevents well_x/well_y suffix issues from pd.merge
                if "well" in cell_features_list.columns:
                    cell_features_list = cell_features_list.drop(columns=["well"])
                cell_df = pd.merge(cell_features_list, meta_df, on="cell_id", how="left")
            else:
                cell_df = cell_features_list
        else:
            cell_df = cell_features_list

        cell_df.to_parquet(gpu_features_path, index=False)
        print(f"  Saved {len(cell_df)} cell GPU features to {gpu_features_path}")
    else:
        print("  WARNING: No cell features generated")
        pd.DataFrame().to_parquet(gpu_features_path, index=False)
        cell_df = pd.DataFrame()

    # Wait for task metadata writes to finish
    n_net_tasks = net_write_future.result()
    n_morph_tasks = morph_write_future.result()
    task_write_pool.shutdown(wait=False)
    if n_net_tasks > 0:
        print(f"  Saved {n_net_tasks} network tasks to {network_tasks_path}")
    else:
        print(f"  No network tasks to save")
    if n_morph_tasks > 0:
        print(f"  Saved {n_morph_tasks} morph supplement tasks to {morph_supplement_tasks_path}")

    # Keep GPU partials for now (in case we need to re-run)
    # Object chunk parquets are in batch_{id}_obj_chunks/ for CPU merge
    print(f"  Keeping GPU partials for safety: {gpu_partials_dir}")

    t_total = time_mod.time() - t_start
    print(f"\nGPU phase total: {t_total:.1f}s")

    return {
        "status": "success",
        "batch_id": batch_id,
        "n_cells": len(cell_df),
        "n_network_tasks": n_net_total,
        "n_morph_supplement_tasks": n_morph_total,
        "gpu_features_path": str(gpu_features_path),
        "network_tasks_path": str(network_tasks_path),
        "morph_supplement_tasks_path": str(morph_supplement_tasks_path),
        "obj_chunks_dir": str(obj_parquet_dir),  # For CPU merge to aggregate
    }



def cpu_network_worker(
    experiment: str,
    well: str,
    batch_idx: int,
    output_dir: str,
):
    """
    CPU-only phase: reads GPU outputs, runs network analysis, merges and saves final parquet.

    This is the second half of the split GPU/CPU pipeline. It:
    1. Loads network task metadata from parquet
    2. Re-reads organelle masks from zarr using saved bbox coordinates
    3. Runs network analysis via MPI (multi-node) or ProcessPoolExecutor (single-node)
    4. Loads GPU features parquet
    5. Merges network features into the GPU features
    6. Saves final cells parquet

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0')
    batch_idx : int
        Batch index within the well
    output_dir : str
        Directory with GPU phase outputs

    Returns
    -------
    dict
        Result with status, path to final output, cell count
    """
    import time as time_mod
    from concurrent.futures import ProcessPoolExecutor
    from collections import defaultdict

    output_dir = Path(output_dir)
    well_safe = well.replace("/", "_")
    batch_id = f"{well_safe}_{batch_idx:04d}"
    batch_results_dir = output_dir / "_batch_results"

    print(f"\n{'='*60}")
    print(f"CPU Network Phase: {batch_id}")
    print(f"Experiment: {experiment}")
    print(f"Well: {well}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    t_start = time_mod.time()

    # Load GPU features
    gpu_features_path = batch_results_dir / f"batch_{batch_id}_gpu_features.parquet"
    if not gpu_features_path.exists():
        print(f"ERROR: GPU features not found: {gpu_features_path}")
        return {"status": "failed", "error": f"GPU features not found: {gpu_features_path}"}

    cell_df = pd.read_parquet(gpu_features_path)
    print(f"Loaded {len(cell_df)} cells from GPU features")

    # Load network task metadata
    network_tasks_path = batch_results_dir / f"batch_{batch_id}_network_tasks.parquet"
    if not network_tasks_path.exists():
        print(f"No network tasks file found, saving GPU features as final output")
        final_path = batch_results_dir / f"batch_{batch_id}_cells.parquet"
        cell_df.to_parquet(final_path, index=False)
        return {"status": "success", "batch_id": batch_id, "n_cells": len(cell_df),
                "output_path": str(final_path)}

    net_tasks_df = pd.read_parquet(network_tasks_path)
    if len(net_tasks_df) == 0:
        print(f"No network tasks to process, saving GPU features as final output")
        final_path = batch_results_dir / f"batch_{batch_id}_cells.parquet"
        cell_df.to_parquet(final_path, index=False)
        return {"status": "success", "batch_id": batch_id, "n_cells": len(cell_df),
                "output_path": str(final_path)}

    print(f"Network tasks: {len(net_tasks_df)}")

    # Build work items for ProcessPoolExecutor
    work_items = []
    for idx, row in net_tasks_df.iterrows():
        work_items.append({
            'task_idx': idx,
            'global_cell_id': row['global_cell_id'],
            'well': row['well'],
            'organelle_name': row['organelle_name'],
            'seg_label_name': row['seg_label_name'],
            'cell_seg_label_name': row.get('cell_seg_label_name'),
            'bbox': (int(row['bbox_y0']), int(row['bbox_x0']), int(row['bbox_y1']), int(row['bbox_x1'])),
            'seg_id': int(row['seg_id']) if pd.notna(row.get('seg_id')) else None,
            'spacing_y': row['spacing_y'],
            'spacing_x': row['spacing_x'],
            'store_path': row['store_path'],
            'result_idx': int(row['result_idx']),
        })

    t_net_start = time_mod.time()
    n_cpu_workers = min(64, os.cpu_count() or 64)
    print(f"Running network analysis: {len(work_items)} tasks across {n_cpu_workers} CPU workers...")
    with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
        net_results = list(executor.map(
            _run_network_analysis_from_zarr,
            work_items,
            chunksize=8,
        ))

    # Collect network results: build dict-of-dicts for cell-level summaries (vectorized),
    # and collect branch/per-object DataFrames grouped by organelle for aggregation.
    n_completed = 0
    timing_accum = defaultdict(list)

    # Cell-level network summaries: {cell_id -> {col_name: value}}
    cell_network_summaries = {}
    # Branch-level and per-object DataFrames grouped by organelle
    network_features_dict = defaultdict(list)   # org -> [branch_df, ...]
    per_object_network_dict = defaultdict(list)  # org -> [per_object_df, ...]

    t_collect_start = time_mod.time()
    for net_result in net_results:
        if net_result is None:
            continue

        global_cell_id = net_result['global_cell_id']
        organelle_name = net_result['organelle_name']
        branch_df = net_result.get('branch_df')
        network_summary_dict = net_result.get('network_summary_dict')
        per_object_network_df = net_result.get('per_object_network_df')
        task_timings = net_result.get('task_timings', {})

        # Collect cell-level network summaries into dict (vectorized merge later)
        if network_summary_dict:
            if global_cell_id not in cell_network_summaries:
                cell_network_summaries[global_cell_id] = {}
            for key, value in network_summary_dict.items():
                cell_network_summaries[global_cell_id][f"network_{organelle_name}_{key}"] = value

        # Collect branch-level features for aggregation (same as combined mode)
        if branch_df is not None and not branch_df.empty:
            branch_df = branch_df.copy()
            branch_df["cell_id"] = global_cell_id
            network_features_dict[organelle_name].append(branch_df)

        # Collect per-object features for aggregation
        if per_object_network_df is not None and not per_object_network_df.empty:
            per_obj = per_object_network_df.copy()
            per_obj["cell_id"] = global_cell_id
            per_object_network_dict[organelle_name].append(per_obj)

        n_completed += 1

        if task_timings:
            for k, v in task_timings.items():
                timing_accum[k].append(v)

    t_collect = time_mod.time() - t_collect_start

    t_net_total = time_mod.time() - t_net_start
    print(f"Network analysis complete: {n_completed}/{len(work_items)} tasks in {t_net_total:.1f}s (collect: {t_collect:.1f}s)")

    # Print profiling summary
    if timing_accum:
        print(f"  Network analysis profiling ({n_completed} tasks):")
        step_order = [
            "label_regionprops_euler", "skeletonize_clean", "skeleton_label",
            "skan_analysis", "network_wide_features", "distance_transform",
            "branch_thickness", "tortuosity", "per_object_features",
        ]
        total_cpu_time = 0.0
        for step in step_order:
            if step in timing_accum:
                vals = timing_accum[step]
                total = sum(vals)
                total_cpu_time += total
                mean = total / len(vals)
                mx = max(vals)
                print(f"    {step:30s}: total={total:8.1f}s  mean={mean*1000:7.1f}ms  max={mx*1000:7.1f}ms  n={len(vals)}")
        print(f"    {'TOTAL (sum across tasks)':30s}: {total_cpu_time:8.1f}s")
        print(f"    {'Wall time (parallel)':30s}: {t_net_total:8.1f}s")
        print(f"    {'Parallelism efficiency':30s}: {total_cpu_time/t_net_total:.1f}x across {n_cpu_workers} workers")

    # Vectorized merge of cell-level network summaries (replaces row-by-row updates)
    t_merge_start = time_mod.time()
    if cell_network_summaries and 'cell_id' in cell_df.columns:
        summary_df = pd.DataFrame.from_dict(cell_network_summaries, orient='index')
        summary_df.index.name = 'cell_id'
        cell_df = cell_df.set_index('cell_id').join(summary_df).reset_index()
        print(f"  Merged cell-level network summaries: {len(cell_network_summaries)} cells, "
              f"{len(summary_df.columns)} columns in {time_mod.time() - t_merge_start:.1f}s")

    # Aggregate branch-level and per-object features in parallel (one process per organelle)
    from organelle_profiler.feature_extraction.fe_anndata import _aggregate_one_organelle, AGG_FUNCS
    from concurrent.futures import ProcessPoolExecutor as _AggPool, as_completed as _agg_as_completed
    t_agg_start = time_mod.time()

    network_orgs = list(set(list(network_features_dict.keys()) + list(per_object_network_dict.keys())))
    agg_frames = []
    agg_metadata = {}

    organelle_work = [
        (org, None, network_features_dict.get(org), per_object_network_dict.get(org), True, None, AGG_FUNCS)
        for org in network_orgs
    ]
    n_agg_workers = min(len(organelle_work), 8)

    with _AggPool(max_workers=n_agg_workers) as pool:
        futures = {pool.submit(_aggregate_one_organelle, work): work[0] for work in organelle_work}
        for future in _agg_as_completed(futures):
            _, org_frames, org_meta, _ = future.result()
            agg_frames.extend(org_frames)
            agg_metadata.update(org_meta)

    if agg_frames:
        combined_agg = pd.concat(agg_frames, axis=1)
        # Join on cell_id
        if 'cell_id' in cell_df.columns:
            cell_df = cell_df.set_index('cell_id').join(combined_agg).reset_index()
        print(f"  Aggregated {len(network_orgs)} network organelles ({len(combined_agg.columns)} columns) "
              f"in {time_mod.time() - t_agg_start:.1f}s ({n_agg_workers} parallel workers)")

    # Save final parquet
    final_path = batch_results_dir / f"batch_{batch_id}_cells.parquet"
    cell_df.to_parquet(final_path, index=False)
    print(f"\nSaved {len(cell_df)} cells to {final_path}")

    t_total = time_mod.time() - t_start
    print(f"CPU phase total: {t_total:.1f}s")

    return {
        "status": "success",
        "batch_id": batch_id,
        "n_cells": len(cell_df),
        "n_network_completed": n_completed,
        "output_path": str(final_path),
    }
