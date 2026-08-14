"""
SLURM batch submission for Feature Extraction.

Parallelizes feature extraction by per-well batches (~3500 cells per batch) for:
- ~1 hour job duration (fits in standard SLURM limits)
- Fault tolerance (failed batches can be retried independently)
- Multi-node scaling (batches run on different nodes)
- Efficient zarr chunk caching (each batch is from one well, cells spatially sorted)

Key design:
- Each batch contains cells from a SINGLE well (no cross-well batches)
- Cells within each batch are sorted by (y, x) coordinates for zarr locality
- Output files: _batch_results/batch_{well}_{idx}_cells.parquet

Usage:
------
# Preview mode: test pipeline with 5 batches from different wells
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --preview

# Preview with custom batch count
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --preview 3

# Submit feature extraction for an experiment
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e ops0094

# Dry run (show what would be submitted)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --dry-run

# Process specific wells only
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --wells A/1/0 A/2/0

# Submit and don't wait for completion
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --no-wait

# Resume from checkpoint (skip completed batches, process remaining)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --checkpoint

# Force reprocess (delete existing results and start fresh)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --force

# Check if all batches completed
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --check-complete

# Aggregate only (skip extraction, just combine existing parquets into AnnData)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --aggregate-only

# Custom cells per batch (default 3500 for ~1hr runtime)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --cells-per-batch 5000

# Runs GPU phase on GPU nodes, then CPU phase on cheap CPU nodes (always split)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94

# With preview
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --preview 3

# Resume from a specific phase - runs that phase + downstream
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --resume-from spmd
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --resume-from merge
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --resume-from aggregate

# Submit to specific partition
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --partition cpu

# Auto-escalate: gpu -> cpu -> preempted (50s each before escalating)
python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --escalate
"""

import argparse
import gc
import multiprocessing
import sys
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.getcwd())

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from organelle_profiler.feature_extraction.feature_extraction import (
    dry_run_discovery,
    is_network_organelle,
)
from iohub import open_ome_zarr
import pandas as pd
import numpy as np
import anndata as ad
from datetime import datetime

from organelle_profiler.feature_extraction.fe_metadata import _discover_available_labels, _load_cells_metadata, validate_segmentation_labels, create_global_cell_id
from organelle_profiler.feature_extraction.fe_constants import (
    AGGREGATION_FUNCTIONS,
    CATEGORY_CELL_MORPHOLOGY,
    CATEGORY_MORPHOLOGY,
    CATEGORY_INTENSITY,
    CATEGORY_LOCALIZATION,
    CATEGORY_NETWORK,
    CATEGORY_CONTACT,
    CATEGORY_DISTRIBUTION,
    get_unit_for_metric,
)


def _get_category_for_metric(metric: str) -> str:
    """Determine feature category from metric name."""
    if "intensity" in metric:
        return CATEGORY_INTENSITY
    if "distance_from" in metric or "normalized_radial" in metric:
        return CATEGORY_LOCALIZATION
    if "radial_frac_bin" in metric or "radial_anisotropy" in metric:
        return CATEGORY_DISTRIBUTION
    return CATEGORY_MORPHOLOGY


def parse_feature_name(feature_name: str, organelle_names: list) -> dict:
    """
    Parse a feature column name to extract metadata.
    
    Uses discovered organelle names (not hardcoded) to properly split feature names.
    
    Parameters
    ----------
    feature_name : str
        The feature column name (e.g., "cp1_mitochondria_tomm20_area_mean")
    organelle_names : list
        List of organelle names discovered from zarr store
        
    Returns
    -------
    dict with keys: organelle, metric, category, aggregation
    """
    agg_funcs = ["sum", "mean", "median", "std", "min", "max", "count"]
    
    # Cell morphology features (no aggregation)
    if feature_name.startswith("cell_"):
        return {
            "organelle": "cell", 
            "metric": feature_name[5:], 
            "category": CATEGORY_CELL_MORPHOLOGY, 
            "aggregation": None
        }
    if feature_name.startswith("cp_cell_"):
        return {
            "organelle": "cp_cell", 
            "metric": feature_name[8:],
            "category": CATEGORY_CELL_MORPHOLOGY, 
            "aggregation": None
        }
    
    # Network features (start with "network_")
    if feature_name.startswith("network_"):
        rest = feature_name[8:]  # Remove "network_" prefix
        
        # Check for aggregation suffix first
        agg = None
        for a in agg_funcs:
            if rest.endswith(f"_{a}"):
                agg = a
                rest = rest[:-(len(a)+1)]
                break
        
        # Find organelle by matching against discovered names (sorted by length for greedy match)
        organelle = None
        metric = rest
        for org in sorted(organelle_names, key=len, reverse=True):
            if rest.startswith(org + "_"):
                organelle = org
                metric = rest[len(org)+1:]
                break
            elif rest == org:
                organelle = org
                metric = ""
                break
        
        return {
            "organelle": organelle, 
            "metric": metric,
            "category": CATEGORY_NETWORK, 
            "aggregation": agg
        }
    
    # Inter-organelle contact features: contact_{A}__{B}_{metric} (cell-level scalar,
    # no aggregation). organelle = the "{A}__{B}" pair; metric = the trailing suffix.
    if feature_name.startswith("contact_"):
        rest = feature_name[len("contact_"):]
        for cm in ("overlap_area", "overlap_frac_a", "overlap_frac_b", "n_contacts"):
            if rest.endswith(f"_{cm}"):
                return {
                    "organelle": rest[:-(len(cm) + 1)],
                    "metric": cm,
                    "category": CATEGORY_CONTACT,
                    "aggregation": None,
                }
        return {"organelle": rest, "metric": "", "category": CATEGORY_CONTACT, "aggregation": None}

    # Object features: {organelle}_{metric}_{agg} or {organelle}_count
    # Check for aggregation suffix first
    agg = None
    name_without_agg = feature_name
    for a in agg_funcs:
        if feature_name.endswith(f"_{a}"):
            agg = a
            name_without_agg = feature_name[:-(len(a)+1)]
            break
    
    # Find organelle by matching against discovered names
    organelle = None
    metric = name_without_agg
    for org in sorted(organelle_names, key=len, reverse=True):
        if name_without_agg.startswith(org + "_"):
            organelle = org
            metric = name_without_agg[len(org)+1:]
            break
        elif name_without_agg == org:
            # Feature is exactly "{organelle}_count" - the organelle name with count suffix
            organelle = org
            metric = "count" if agg == "count" else ""
            agg = None  # count is the metric, not aggregation in this case
            break
    
    category = _get_category_for_metric(metric) if organelle and metric else "unknown"
    
    return {
        "organelle": organelle, 
        "metric": metric,
        "category": category, 
        "aggregation": agg
    }


# Default cells per batch for large-scale feature extraction
# Based on empirical data: ~750K cells fits within 8hr GPU timeout
# Results in ~1 batch per well for typical experiments (~2-3M cells)
# Each batch is well-specific and chunk-sorted for optimal zarr cache hits
DEFAULT_CELLS_PER_BATCH = 750000


def _get_cuda_ld_library_path() -> str:
    """Build LD_LIBRARY_PATH for CUDA libraries in the current conda environment."""
    import sys
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_base = f"{conda_prefix}/lib/python{py_version}/site-packages/nvidia"
    cuda_libs = [
        f"{nvidia_base}/cuda_nvrtc/lib",
        f"{nvidia_base}/cuda_runtime/lib",
        f"{nvidia_base}/cublas/lib",
        f"{nvidia_base}/cusparse/lib",
        f"{conda_prefix}/lib",
    ]
    return ":".join(cuda_libs) + ":$LD_LIBRARY_PATH"


# Base SLURM resource configuration for cell-batch feature extraction jobs
# Each job processes ~3500 cells from a single well in ~1 hour
# Note: submitit v1.5+ requires slurm_ prefix for most params
# Note: chunk-sorted cells reduce I/O contention, letting SLURM manage parallelism
FE_SLURM_PARAMS_BASE = {
    "slurm_mem": "800GB",           # match the validated split config: 43 workers @ ~18GB on H200 (the prior "60GB" was a typo for 600; dense wells OOM'd at ~440GB)
    "cpus_per_task": 64,             # 64 CPUs for hybrid GPU/CPU processing
    "timeout_min": 360,             # combined mode runs morphology+network in one job; a 700k-cell well is >1hr, so match the GPU-phase 6hr budget
    "slurm_partition": "gpu",       # GPU partition for hybrid processing
    "slurm_ntasks_per_node": 1,     # 1 task per node
    "slurm_array_parallelism": 2,   # Max concurrent jobs (filesystem I/O limited - fewer = faster per-job)
    "slurm_gres": "gpu:1",          # Request 1 GPU for hybrid mode (simple cells on GPU)
    "slurm_constraint": "h200",     # pin H200 (141GB): 43 workers @ 3.0GB/worker -> ~140 cells/sec; removes node-luck variance
    # CRITICAL: Set LD_LIBRARY_PATH for CUDA libraries in conda environment
    # Dynamically built from current conda environment at submission time
    "slurm_setup": [
        f"export LD_LIBRARY_PATH={_get_cuda_ld_library_path()}",
    ] + ([f"export OPS_OUTPUT_BASE_DIR={os.environ['OPS_OUTPUT_BASE_DIR']}"] if os.environ.get('OPS_OUTPUT_BASE_DIR') else []),
}





# SLURM params for GPU-only phase (split mode)
# Same CPU/memory as combined mode - workers are CPU-bound during I/O
FE_GPU_SLURM_PARAMS = {
    "slurm_mem": "800GB",            # 1 GPU -> ~43 dask workers @ ~18GB each (RAM scales with worker count, not GPU count; dense-well montages OOM'd at ~15GB/worker). H200 has 2TB.
    "cpus_per_task": 64,             # ~43 workers + I/O threads
    "timeout_min": 360,             # 700K cells ~136min on A6000; 8hr for safety (partials load + aggregation add ~1hr)
    "slurm_partition": "gpu",
    "slurm_ntasks_per_node": 1,
    "slurm_array_parallelism": 8,    # Let all GPU batches run concurrently
    # 1 GPU/job on H200 (141GB): with the 3.0GB/worker budget + raised cap the
    # split GPU phase gets ~43 workers on one GPU (same as the old combined run
    # that hit ~90 cells/sec), keeping GPU usage proportional (1/8 of the node).
    "slurm_gres": "gpu:1",
    # Pin H200 (141GB): the worker count scales with usable VRAM
    # (usable/4.0GB, capped 24). On an 80GB card that's 17 workers; on the
    # 141GB H200 it's the full 24-worker cap. Pinning removes the node-luck
    # variance where a run lands on an 80GB card and runs ~1.4x slower.
    "slurm_constraint": "h200",
    "slurm_setup": FE_SLURM_PARAMS_BASE["slurm_setup"],
}

# 4i-specific overrides applied at submission time when `submit_feature_extraction_jobs`
# detects a 4i experiment (any zarr label or organelle key starts with "4i_").
# 4i workloads carry ~5× the feature stacks of pure live-cell exps (every IF
# round contributes its own organelle features), so workers need more RAM
# headroom (16 dask workers × ~54 GB Dask limit each instead of 17 GB at
# 450 GB / 24 workers). Wall clock is unchanged from the default — with
# the right memory + worker-cap settings, 4i runs complete in well under
# 3h on H200, inside the default 480-min budget. H200 constraint kept
# until cucim init is stable on a100_80 / h100. Live-cell experiments
# fall through to FE_GPU_SLURM_PARAMS unchanged.
FE_GPU_SLURM_PARAMS_4I_OVERRIDES = {
    "slurm_mem": "500GB",
    "slurm_constraint": "h100|h200",
}

# SLURM params for CPU-only network analysis phase (split mode)
# No GPU needed, uses cheap CPU-only nodes
FE_CPU_SLURM_PARAMS = {
    "slurm_mem": "128GB",
    "cpus_per_task": 64,             # All CPU for network analysis
    "timeout_min": 30,
    "slurm_partition": "cpu",        # Cheap CPU-only nodes
    "slurm_ntasks_per_node": 1,
    "slurm_array_parallelism": 4,
    "slurm_setup": FE_SLURM_PARAMS_BASE["slurm_setup"],
}

# Partition escalation order: try each in sequence if pending too long
# gpu -> cpu -> preempted
PARTITION_ESCALATION = ["gpu", "cpu", "preempted"]
PENDING_TIMEOUT_SECS = 50  # Resubmit to next partition after 50s pending


def get_batch_output_path(output_dir: Path, well: str, batch_idx: int) -> Path:
    """Get the output path for a batch's parquet file."""
    well_safe = well.replace("/", "_")
    return output_dir / "_batch_results" / f"batch_{well_safe}_{batch_idx:04d}_cells.parquet"


def estimate_timeout_minutes(
    n_cells: int,
    n_segmentations: int = 17,
    n_network_organelles: int = 7,
    n_workers: int = 64,
    safety_factor: float = 12.0,
) -> int:
    """
    Estimate SLURM timeout based on cell count and segmentation complexity.

    Based on empirical data from ops0094 (Cell Painting, 17 segs, 7 network):
    - ~15 cells/second EFFECTIVE rate (with 44 parallel workers)
    - Chunk-sorted cells maximize zarr cache hits and reduce I/O contention

    Parameters
    ----------
    n_cells : int
        Number of cells in this batch
    n_segmentations : int
        Total number of segmentations to process (default 17 for CP experiments)
    n_network_organelles : int
        Number of organelles with network analysis (default 7)
    n_workers : int
        Number of parallel workers (default 44)
    safety_factor : float
        Multiplier for safety margin (default 2.0x)

    Returns
    -------
    int
        Estimated timeout in minutes
    """
    # Base effective rate (empirical): ~7.5 cells/second (conservative 2x buffer)
    # Measured on ops0094 with 17 segmentations and 7 network organelles
    # Rate scales roughly linearly with worker count
    BASE_CELLS_PER_SEC = 7.5  # Conservative estimate with 2x safety buffer
    BASE_WORKERS = 44
    BASE_SEGMENTATIONS = 17
    BASE_NETWORK_ORGS = 7
    
    # Scale by worker count (more workers = faster)
    worker_scale = n_workers / BASE_WORKERS
    
    # Scale by segmentation count (more segs = slower)
    seg_scale = BASE_SEGMENTATIONS / n_segmentations
    
    # Scale by network organelles (network analysis is expensive, ~10% each)
    network_scale = 1.0 - (n_network_organelles - BASE_NETWORK_ORGS) * 0.1
    network_scale = max(0.5, network_scale)  # Don't go below 50%
    
    # Effective cells per second
    cells_per_sec = BASE_CELLS_PER_SEC * worker_scale * seg_scale * network_scale
    
    # Total processing time
    processing_seconds = n_cells / cells_per_sec
    processing_minutes = processing_seconds / 60
    
    # Overhead: import time + worker spawning + store opening + metadata loading + saving
    # PyTorch/MONAI imports can take 1-2 minutes on cold starts
    # Scale overhead based on batch size (smaller batches = less overhead)
    import_overhead = 3  # PyTorch/MONAI import time buffer
    batch_overhead = 1 if n_cells <= 500 else 3 if n_cells <= 2000 else 5
    overhead_minutes = import_overhead + batch_overhead

    estimated = (processing_minutes + overhead_minutes) * safety_factor

    # Return estimated time (minimum 5 min to account for import overhead)
    return max(5, int(estimated))


def check_jobs_pending(job_ids: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which jobs are pending vs running.
    
    Returns:
        (pending_ids, running_ids): Lists of job IDs in each state
    """
    import subprocess
    
    if not job_ids:
        return [], []
    
    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i %T"],
            capture_output=True, text=True, timeout=10
        )
        
        pending = []
        running = []
        
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                job_id, state = parts[0], parts[1]
                if state == "PENDING":
                    pending.append(job_id)
                elif state == "RUNNING":
                    running.append(job_id)
        
        return pending, running
    except Exception as e:
        print(f"Warning: Could not check job status: {e}")
        return [], []


def cancel_jobs(job_ids: list[str]) -> bool:
    """Cancel SLURM jobs by ID."""
    import subprocess
    
    if not job_ids:
        return True
    
    try:
        result = subprocess.run(
            ["scancel"] + job_ids,
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Warning: Could not cancel jobs: {e}")
        return False


def submit_with_escalation(
    jobs_to_submit: list,
    experiment: str,
    slurm_params: dict,
    log_dir: str,
    args,
    output_dir: Path,
    on_completion_callback,
    print_resource_summary: bool = True,
) -> dict:
    """
    Submit jobs with automatic partition escalation if pending too long.
    
    Escalation order: gpu -> cpu -> preempted
    If jobs are pending for >60 seconds, cancel and resubmit to next partition.
    """
    import time
    
    escalation_order = PARTITION_ESCALATION.copy()
    current_partition = slurm_params["slurm_partition"]
    
    # Find starting position in escalation order
    try:
        start_idx = escalation_order.index(current_partition)
    except ValueError:
        # Not in escalation order, just use as-is
        start_idx = 0
        escalation_order = [current_partition] + escalation_order
    
    for partition_idx in range(start_idx, len(escalation_order)):
        partition = escalation_order[partition_idx]
        current_slurm_params = {**slurm_params, "slurm_partition": partition}
        
        print(f"\n{'='*60}")
        print(f"Submitting to partition: {partition}")
        print(f"{'='*60}")
        
        # Submit jobs
        result = submit_parallel_jobs(
            jobs_to_submit=jobs_to_submit,
            experiment=experiment,
            slurm_params=current_slurm_params,
            log_dir=log_dir,
            manifest_prefix="feature_extraction",
            dry_run=args.dry_run,
            wait_for_completion=False,  # Don't wait yet, we'll monitor
            verbose=not getattr(args, 'quiet', False),
        )
        
        if args.dry_run:
            return result
        
        if not result.get("success"):
            return result
        
        base_job_id = result.get("base_job_id")
        submitted_jobs = result.get("submitted_jobs", [])
        
        if not submitted_jobs:
            return result
        
        # Get all job IDs for monitoring
        n_jobs = len(submitted_jobs)
        job_ids = [f"{base_job_id}_{i}" for i in range(n_jobs)]
        
        # Monitor for pending status
        print(f"\nMonitoring jobs for {PENDING_TIMEOUT_SECS}s...")
        start_time = time.time()
        
        while time.time() - start_time < PENDING_TIMEOUT_SECS:
            pending, running = check_jobs_pending(job_ids)
            
            if running:
                # Some jobs started running, we're good!
                print(f"\n✓ {len(running)} jobs now running on {partition}")
                
                # Now wait for completion if requested
                if not args.no_wait:
                    from cyclops_utils.hpc.slurm_batch_utils import _wait_for_jobs
                    completed, failed = _wait_for_jobs(
                        submitted_jobs=submitted_jobs,
                        base_job_id=base_job_id,
                        slurm_params=current_slurm_params,
                        experiment=experiment,
                        verbose=not getattr(args, 'quiet', False),
                        print_resource_summary=print_resource_summary,
                    )
                    
                    # Run callback
                    if on_completion_callback:
                        try:
                            on_completion_callback(submitted_jobs, experiment)
                        except Exception as e:
                            print(f"\n⚠️  Post-completion callback failed: {e}")
                    
                    result["completed"] = completed
                    result["failed"] = failed
                    result["all_completed"] = len(failed) == 0
                
                result["partition_used"] = partition
                return result
            
            if not pending:
                # No jobs pending or running - they completed or failed already
                break
            
            # Still pending, wait a bit
            elapsed = int(time.time() - start_time)
            print(f"  {elapsed}s: {len(pending)} jobs still pending...", end="\r")
            time.sleep(5)
        
        # Check final state after timeout
        pending, running = check_jobs_pending(job_ids)
        
        if running:
            # Started running during final check
            print(f"\n✓ {len(running)} jobs now running on {partition}")
            result["partition_used"] = partition
            
            if not args.no_wait:
                from cyclops_utils.hpc.slurm_batch_utils import _wait_for_jobs
                completed, failed = _wait_for_jobs(
                    submitted_jobs=submitted_jobs,
                    base_job_id=base_job_id,
                    slurm_params=current_slurm_params,
                    experiment=experiment,
                    verbose=not getattr(args, 'quiet', False),
                    print_resource_summary=print_resource_summary,
                )
                if on_completion_callback:
                    try:
                        on_completion_callback(submitted_jobs, experiment)
                    except Exception as e:
                        print(f"\n⚠️  Post-completion callback failed: {e}")
                result["completed"] = completed
                result["failed"] = failed
            return result
        
        # Still pending after timeout, try next partition
        if pending and partition_idx < len(escalation_order) - 1:
            next_partition = escalation_order[partition_idx + 1]
            print(f"\n⚠️  Jobs still pending after {PENDING_TIMEOUT_SECS}s on {partition}")
            print(f"   Cancelling and escalating to {next_partition}...")
            
            cancel_jobs(job_ids)
            
            # Delete the log directory for cancelled jobs to keep things clean
            array_log_dir = result.get("array_log_dir")
            if array_log_dir:
                import shutil
                log_path = Path(array_log_dir)
                if log_path.exists():
                    print(f"   Cleaning up logs: {log_path}")
                    shutil.rmtree(log_path)
            
            time.sleep(2)  # Brief pause for cancellation to propagate
            continue
        
        # No more partitions to try, or jobs completed/failed
        if not args.no_wait:
            from cyclops_utils.hpc.slurm_batch_utils import _wait_for_jobs
            completed, failed = _wait_for_jobs(
                submitted_jobs=submitted_jobs,
                base_job_id=base_job_id,
                slurm_params=current_slurm_params,
                experiment=experiment,
                verbose=not getattr(args, 'quiet', False),
                print_resource_summary=print_resource_summary,
            )
            if on_completion_callback:
                try:
                    on_completion_callback(submitted_jobs, experiment)
                except Exception as e:
                    print(f"\n⚠️  Post-completion callback failed: {e}")
            result["completed"] = completed
            result["failed"] = failed
        
        result["partition_used"] = partition
        return result
    
    return {"success": False, "error": "Escalation exhausted all partitions"}


def feature_extraction_batch_worker(
    experiment: str,
    well: str,
    batch_idx: int,
    batch_cells_path: str,
    output_dir: str,
    full_features: bool = False,
    sequential: bool = False,
) -> dict:
    """
    SLURM-compatible worker for extracting features from a batch of cells.

    Each batch contains cells from a single well, spatially sorted for
    optimal zarr chunk cache locality. Cell metadata is pre-saved to avoid
    loading all 2M+ cells on each worker.

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0') - all cells in batch are from this well
    batch_idx : int
        Batch index within the well (0, 1, 2, ...)
    batch_cells_path : str
        Path to parquet file containing pre-filtered cells for this batch
    output_dir : str
        Directory to save intermediate results
    full_features : bool
        Whether to compute expensive texture features
    sequential : bool
        If True, process cells sequentially (for benchmarking vs parallel)

    Returns
    -------
    dict
        Result with well, batch_idx, status, n_cells, and any errors
    """
    import traceback
    from organelle_profiler.feature_extraction.feature_extraction import (
        extract_features_for_batch_direct
    )

    try:
        # Load pre-filtered batch cells (fast - just this batch's cells)
        batch_cells_df = pd.read_parquet(batch_cells_path)
        n_cells = len(batch_cells_df)
        
        well_safe = well.replace("/", "_")
        batch_id = f"{well_safe}_{batch_idx:04d}"
        
        print(f"\n{'='*60}")
        print(f"Feature Extraction Batch: {batch_id}")
        print(f"Experiment: {experiment}")
        print(f"Well: {well}")
        print(f"Cells: {n_cells} (batch {batch_idx})")
        print(f"Mode: {'SEQUENTIAL (1 worker)' if sequential else 'PARALLEL'}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        # Run feature extraction for this batch (no cell loading needed)
        output_path = extract_features_for_batch_direct(
            experiment=experiment,
            well=well,
            batch_idx=batch_idx,
            batch_cells_df=batch_cells_df,
            output_dir=Path(output_dir),
            full_features=full_features,
            sequential=sequential,
        )

        if output_path and output_path.exists():
            df = pd.read_parquet(output_path)
            n_cells_out = len(df)
            print(f"\n✓ Completed batch {batch_id}: {n_cells_out} cells")
            return {
                "well": well,
                "batch_idx": batch_idx,
                "batch_id": batch_id,
                "status": "success",
                "n_cells": n_cells_out,
                "output_path": str(output_path),
            }
        else:
            print(f"\n⚠ No output for batch {batch_id}")
            return {
                "well": well,
                "batch_idx": batch_idx,
                "batch_id": batch_id,
                "status": "skipped",
                "reason": "no_output",
                "n_cells": 0,
            }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"\n✗ FAILED batch {well_safe}_{batch_idx:04d}: {error_msg}")
        return {
            "well": well,
            "batch_idx": batch_idx,
            "status": "failed",
            "error": str(e),
        }


def gpu_batch_worker(
    experiment: str,
    well: str,
    batch_idx: int,
    batch_cells_path: str,
    output_dir: str,
    full_features: bool = True,
    n_workers: int = None,
    io_threads: int = None,
    max_cells: int = None,
) -> dict:
    """
    SLURM worker for GPU-only feature extraction phase.

    Runs morphology + localization on GPU, saves features and network task metadata.
    The CPU phase (cpu_network_batch_worker) should run after this completes.

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0')
    batch_idx : int
        Batch index within the well
    batch_cells_path : str
        Path to parquet with pre-filtered cells for this batch
    output_dir : str
        Directory to save intermediate results
    full_features : bool
        Whether to compute expensive features

    Returns
    -------
    dict
        Result with status, output paths, cell count
    """
    import traceback

    try:
        batch_cells_df = pd.read_parquet(batch_cells_path)

        from organelle_profiler.feature_extraction.fe_workers import gpu_phase_worker
        result = gpu_phase_worker(
            experiment=experiment,
            well=well,
            batch_idx=batch_idx,
            batch_cells_df=batch_cells_df,
            output_dir=output_dir,
            full_features=full_features,
            n_workers_override=n_workers,
            io_threads_override=io_threads,
            max_cells=max_cells,
        )
        return result

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        well_safe = well.replace("/", "_")
        print(f"\n✗ FAILED GPU phase {well_safe}_{batch_idx:04d}: {error_msg}")
        return {
            "well": well,
            "batch_idx": batch_idx,
            "status": "failed",
            "error": str(e),
        }


def cpu_network_batch_worker(
    experiment: str,
    well: str,
    batch_idx: int,
    output_dir: str,
) -> dict:
    """
    SLURM worker for CPU-only network analysis phase.

    Reads GPU phase outputs, runs network analysis, merges and saves final parquet.
    Should run after gpu_batch_worker completes for the same batch.

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
    import traceback
    from organelle_profiler.feature_extraction.fe_workers import cpu_network_worker

    try:
        result = cpu_network_worker(
            experiment=experiment,
            well=well,
            batch_idx=batch_idx,
            output_dir=output_dir,
        )

        well_safe = well.replace("/", "_")
        batch_id = f"{well_safe}_{batch_idx:04d}"

        if result.get("status") == "success":
            print(f"\n✓ Completed CPU phase {batch_id}: {result.get('n_cells', 0)} cells")
        return result

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        well_safe = well.replace("/", "_")
        print(f"\n✗ FAILED CPU phase {well_safe}_{batch_idx:04d}: {error_msg}")
        return {
            "well": well,
            "batch_idx": batch_idx,
            "status": "failed",
            "error": str(e),
        }


def _merge_worker(
    experiment: str,
    batch_ids: list[str],
    output_dir: str,
) -> dict:
    """Submitit-compatible worker that runs the SPMD-CPU merge step.

    Shells out to the existing ``spmd_cpu_merge`` CLI so the well-tested
    aggregation code path is unchanged; only the SLURM-submission machinery
    is unified with the rest of the pipeline (via ``submit_parallel_jobs``).
    """
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "organelle_profiler.feature_extraction.spmd_cpu_merge",
        "--experiment",
        experiment,
        "--batch-ids",
        ",".join(batch_ids),
        "--output-dir",
        str(output_dir),
    ]
    print(f"[merge_worker] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"spmd_cpu_merge exited with code {result.returncode}")
    return {"experiment": experiment, "batch_ids": batch_ids, "status": "success"}


def _run_spmd_rank(
    rank: int,
    n_ranks: int,
    experiment: str,
    batch_ids: list[str],
    output_dir: str,
) -> dict:
    """
    Run a single SPMD rank as a function (for array job submission).

    This wrapper calls the spmd_cpu_worker main() via subprocess to match
    the original SLURM script behavior exactly.
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-u", "-m", "organelle_profiler.feature_extraction.spmd_cpu_worker",
        "--experiment", experiment,
        "--batch-ids", ",".join(batch_ids),
        "--output-dir", output_dir,
        "--rank", str(rank),
        "--n-ranks", str(n_ranks),
        "--restart",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "rank": rank,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def submit_spmd_cpu_phase(
    batch_ids: list[str],
    experiment: str,
    output_dir: Path,
    log_dir: str,
    n_ranks: int = 1024,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    submit_merge_after: bool = True,
) -> dict:
    """
    Submit SPMD CPU phase as array jobs (one job per rank).

    Uses SLURM array jobs via submitit. Benefits:
    - Each rank is an independent job (stragglers don't block others)
    - Skips already-completed ranks (respects .done markers)
    - Better fault tolerance (failed ranks can be resubmitted)

    Parameters
    ----------
    batch_ids : list[str]
        List of batch IDs to process
    experiment : str
        Experiment name
    output_dir : Path
        Output directory
    log_dir : str
        Base log directory for SLURM logs
    n_ranks : int
        Total number of ranks (default 1024)
    dry_run : bool
        If True, print plan without submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete

    Returns
    -------
    dict
        Result with job IDs and status
    """
    import subprocess

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Partials directory (check for already-completed ranks)
    partials_dir = output_dir / "_batch_results" / "_partials_combined"
    partials_dir.mkdir(parents=True, exist_ok=True)

    # Find incomplete ranks (don't have .done marker)
    completed_ranks = set()
    for done_marker in partials_dir.glob("rank*_done"):
        try:
            rank_num = int(done_marker.stem.replace("rank", "").replace("_done", ""))
            completed_ranks.add(rank_num)
        except ValueError:
            pass

    incomplete_ranks = [r for r in range(n_ranks) if r not in completed_ranks]

    print(f"\n{'='*60}")
    print(f"PHASE 2: SPMD CPU (array job mode)")
    print(f"{'='*60}")
    print(f"  Batches: {len(batch_ids)}")
    print(f"  Total ranks: {n_ranks}")
    print(f"  Already completed: {len(completed_ranks)}")
    print(f"  Ranks to submit: {len(incomplete_ranks)}")

    if not incomplete_ranks:
        print(f"  All ranks already complete!")
        return {"success": True, "skipped": True, "completed_ranks": len(completed_ranks)}

    if dry_run:
        print(f"  [DRY RUN] Would submit {len(incomplete_ranks)} array jobs")
        return {"success": True, "dry_run": True}

    # Build job list for incomplete ranks
    jobs_to_submit = []
    for rank in incomplete_ranks:
        jobs_to_submit.append({
            "name": f"spmd_rank_{rank:04d}",
            "func": _run_spmd_rank,
            "kwargs": {
                "rank": rank,
                "n_ranks": n_ranks,
                "experiment": experiment,
                "batch_ids": batch_ids,
                "output_dir": str(output_dir),
            },
            "metadata": {"rank": rank},
        })

    # SLURM parameters for CPU workers
    slurm_params = {
        "timeout_min": 20,  # 20 minutes per rank
        "slurm_mem": "4G",
        "cpus_per_task": 1,
        "slurm_partition": "cpu",
        "slurm_additional_parameters": {
            "export": "ALL,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1",
        },
    }

    # Submit using shared infrastructure
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=f"slurm_feature_extraction_logs/{experiment}_spmd_array",
        manifest_prefix="spmd_cpu",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        print_success=False,  # Suppress per-rank ✓ lines for 1000+ jobs (failures still print)
        print_resource_summary=False,  # Skip resource summary for large arrays
    )

    if not result.get("success"):
        return result

    if not submit_merge_after:
        # Wave-orchestration path: caller will submit merge separately as Wave 3.
        return {
            "success": True,
            "spmd_result": result,
            "merge_result": None,
        }

    # After SPMD completes, submit merge job
    print(f"\n{'='*60}")
    print(f"PHASE 3: Merge (post-SPMD)")
    print(f"{'='*60}")

    spmd_array_id = result.get("base_job_id")
    merge_slurm_params = {
        "timeout_min": 90,
        "mem": "250G",
        "cpus_per_task": 32,
        "slurm_partition": "cpu",
    }
    # When SPMD didn't block, ensure SLURM holds the merge job until SPMD completes.
    if not wait_for_completion and spmd_array_id:
        merge_slurm_params["slurm_additional_parameters"] = {
            "dependency": f"afterok:{spmd_array_id}",
        }
    merge_jobs = [{
        "name": f"fe_merge_{experiment}",
        "func": _merge_worker,
        "kwargs": {
            "experiment": experiment,
            "batch_ids": list(batch_ids),
            "output_dir": str(output_dir),
        },
        "metadata": {"experiment": experiment, "phase": "merge"},
    }]
    merge_result = submit_parallel_jobs(
        jobs_to_submit=merge_jobs,
        experiment=experiment,
        slurm_params=merge_slurm_params,
        log_dir=f"slurm_feature_extraction_logs/{experiment}_merge",
        manifest_prefix="fe_merge",
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
        verbose=True,
        print_resource_summary=False,
        print_success=False,
    )

    return {
        "success": merge_result.get("success", False),
        "spmd_result": result,
        "merge_result": merge_result,
    }


def submit_merge_job(
    batch_ids: list[str],
    experiment: str,
    output_dir: Path,
    log_dir: str,
    wait_for_completion: bool = True,
) -> dict:
    """
    Submit the merge-only job (used by --resume-from merge and Wave 3 of the
    multi-experiment wave runner).

    Uses ``submit_parallel_jobs`` so the result has the standard shape
    (``submitted_jobs``, ``base_job_id``) that ``wait_for_multiple_job_arrays``
    consumes.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    merge_jobs = [{
        "name": f"fe_merge_{experiment}",
        "func": _merge_worker,
        "kwargs": {
            "experiment": experiment,
            "batch_ids": list(batch_ids),
            "output_dir": str(output_dir),
        },
        "metadata": {"experiment": experiment, "phase": "merge"},
    }]
    return submit_parallel_jobs(
        jobs_to_submit=merge_jobs,
        experiment=experiment,
        slurm_params={
            "timeout_min": 90,
            "mem": "250G",
            "cpus_per_task": 32,
            "slurm_partition": "cpu",
        },
        log_dir=f"slurm_feature_extraction_logs/{experiment}_merge",
        manifest_prefix="fe_merge",
        dry_run=False,
        wait_for_completion=wait_for_completion,
        verbose=True,
        print_resource_summary=False,
        print_success=False,
    )


# ---------------------------------------------------------------------------
# Vectorized groupby aggregation - imported from separate module to avoid
# pickling issues when running via submitit (which uses __main__)
# ---------------------------------------------------------------------------
from organelle_profiler.feature_extraction.fe_aggregation import vectorized_groupby_agg


def aggregate_batch_results(
    submitted_jobs: list,
    experiment: str,
    output_dir: Path,
    available_labels: dict = None,
) -> dict:
    """
    Aggregate per-batch results into final AnnData files.

    This callback is run after all SLURM jobs complete. It:
    1. Loads all per-batch parquet files
    2. Concatenates into single cell-level DataFrame
    3. Aggregates to guide and gene levels
    4. Saves as AnnData files with proper var metadata

    Parameters
    ----------
    submitted_jobs : list
        List of job result dicts from SLURM
    experiment : str
        Experiment name
    output_dir : Path
        Output directory for final files
    available_labels : dict, optional
        Discovered labels from zarr store (organelle_name -> label_name).
        Used to parse feature names and populate var metadata.
        If None, will be discovered from zarr store.
    """
    print(f"\n{'='*60}")
    print("AGGREGATING BATCH RESULTS")
    print(f"{'='*60}\n")

    # Get organelle names from available_labels (discover if not passed)
    if available_labels is None:
        print("Discovering available labels from zarr store...")
        ds = OpsDataset(experiment)
        morphology_path = ds.store_paths["pheno_assembled_v3"]
        available_labels = _discover_available_labels(morphology_path)
    
    # Get list of organelle names for feature parsing
    organelle_names = list(available_labels.keys())
    print(f"Using {len(organelle_names)} organelles for feature metadata parsing")

    # Collect successful job outputs from batch results directory
    batch_results_dir = output_dir / "_batch_results"
    parquet_files = list(batch_results_dir.glob("batch_*_cells.parquet"))

    if not parquet_files:
        print("ERROR: No parquet files found to aggregate!")
        return {"status": "failed", "error": "No parquet files found"}

    print(f"Found {len(parquet_files)} batch result files")

    # Load all parquet files in parallel (I/O bound)
    print("Loading per-batch results...")
    t0 = time.time()

    def _load_one(pf):
        return pf.name, pd.read_parquet(pf)

    dfs = []
    with ThreadPoolExecutor(max_workers=len(parquet_files)) as pool:
        futures = {pool.submit(_load_one, pf): pf for pf in parquet_files}
        for future in as_completed(futures):
            try:
                name, df = future.result()
                dfs.append(df)
                print(f"  {name}: {len(df)} cells")
            except Exception as e:
                print(f"  Warning: Failed to load {futures[future].name}: {e}")

    if not dfs:
        print("ERROR: No valid parquet files could be loaded!")
        return {"status": "failed", "error": "No valid parquet files"}

    cell_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"\nTotal cells: {len(cell_df):,} (loaded in {time.time() - t0:.1f}s)")
    t_prep = time.time()

    # Fix well column: rename well_x to well if needed (backwards compatibility for older batches)
    if "well_x" in cell_df.columns and "well" not in cell_df.columns:
        cell_df = cell_df.rename(columns={"well_x": "well"})
        print(f"  Renamed well_x -> well (backwards compatibility fix)")
    # Drop well_y if it exists (artifact from merge with both having well column)
    if "well_y" in cell_df.columns:
        cell_df = cell_df.drop(columns=["well_y"])

    # Custom-perturbation schema bridge (mirror of fe_metadata._load_cells_metadata):
    # some libraries' per-batch parquets carry barcode + a custom guide column
    # (config ``gene_name_output_column``) + row_type instead of sgRNA + gene_name +
    # NCBI_ID. Project them to the standard schema so the dedup / guide- /
    # gene-aggregation logic below works unchanged; the custom column is preserved.
    guide_col = getattr(OpsDataset(experiment), "gene_name_output_column", None)
    if (
        guide_col
        and guide_col != "gene_name"
        and guide_col in cell_df.columns
        and "sgRNA" not in cell_df.columns
        and "barcode" in cell_df.columns
    ):
        cell_df["sgRNA"] = cell_df["barcode"]
        cell_df["gene_name"] = cell_df[guide_col]
        n_neg = 0
        if "row_type" in cell_df.columns:
            neg_mask = cell_df["row_type"].astype(str) == "neg_ctrl"
            n_neg = int(neg_mask.sum())
            cell_df.loc[neg_mask, "gene_name"] = "neg_ctrl"
        print(f"  Custom-perturbation mode: barcode→sgRNA and "
              f"{guide_col}→gene_name applied; "
              f"{n_neg:,} neg_ctrl rows labeled 'neg_ctrl'")

    # Deduplicate by segmentation_id (multiple tracks can point to same cell after division)
    # This ensures we don't extract features from the same physical cell twice
    if "segmentation_id" in cell_df.columns and "well" in cell_df.columns:
        n_before = len(cell_df)

        # Create composite key for deduplication (faster than sort + drop_duplicates)
        # Only dedupe rows that have segmentation_id
        has_seg_id = cell_df["segmentation_id"].notna()
        n_with_seg = has_seg_id.sum()
        if has_seg_id.any():
            # For rows with segmentation_id, create (well, segmentation_id) key
            seg_key = cell_df.loc[has_seg_id, "well"].astype(str) + "_" + cell_df.loc[has_seg_id, "segmentation_id"].astype(int).astype(str)

            # If sgRNA exists, prefer rows with valid sgRNA when deduplicating
            if "sgRNA" in cell_df.columns:
                sgrna_col = cell_df.loc[has_seg_id, "sgRNA"]
                has_valid_sgrna = sgrna_col.notna() & (sgrna_col != "") & (sgrna_col != "None")
                # Sort by has_valid_sgrna descending so rows with sgRNA come first
                sort_idx = has_valid_sgrna.sort_values(ascending=False).index
                seg_key = seg_key.reindex(sort_idx)

            # Mark duplicates (keep first = the one with sgRNA if available)
            dup_mask = seg_key.duplicated(keep="first")
            rows_to_drop = dup_mask[dup_mask].index
            cell_df = cell_df.drop(rows_to_drop)

        n_after = len(cell_df)
        n_removed = n_before - n_after
        print(f"  Deduplication: {n_before:,} -> {n_after:,} cells ({n_removed:,} duplicates, {n_with_seg:,} with seg_id) [{time.time() - t_prep:.1f}s]")
        t_prep = time.time()

    # Reconstruct cell_id using segmentation_id (unique per cell per well)
    if "segmentation_id" in cell_df.columns and "well" in cell_df.columns:
        has_seg_id = cell_df["segmentation_id"].notna()
        n_with_seg = has_seg_id.sum()
        if n_with_seg > 0:
            cell_df.loc[has_seg_id, "cell_id"] = (
                cell_df.loc[has_seg_id, "well"].astype(str) + "_" +
                cell_df.loc[has_seg_id, "segmentation_id"].astype(int).astype(str)
            )
        # For CP-only cells without segmentation_id, use cp_cell_seg_id if available
        if "cp_cell_seg_id" in cell_df.columns:
            has_cp_only = ~has_seg_id & cell_df["cp_cell_seg_id"].notna()
            n_cp_only = has_cp_only.sum()
            if n_cp_only > 0:
                cell_df.loc[has_cp_only, "cell_id"] = (
                    cell_df.loc[has_cp_only, "well"].astype(str) + "_cp" +
                    cell_df.loc[has_cp_only, "cp_cell_seg_id"].astype(int).astype(str)
                )

    # Aggregate to guide and gene levels
    agg_funcs = ["sum", "mean", "median", "std", "min", "max", "count"]

    # Get numeric columns for aggregation
    # Feature columns start with known prefixes (cell_, cp_cell_, or organelle names)
    # This is much faster than parsing each column individually
    # "contact_" = inter-organelle contact/colocalization scalars (contact_{A}__{B}_{metric});
    # they don't start with an organelle prefix so must be listed explicitly. Radial-distribution
    # features ({org}_radial_frac_bin*/_radial_anisotropy) already match an organelle prefix.
    feature_prefixes = ["cell_", "cp_cell_", "network_", "contact_"] + [f"{org}_" for org in organelle_names]
    metadata_cols = {"cell_id", "segmentation_id", "cp_cell_seg_id", "well", "fov",
                     "barcode", "sgRNA", "gene_name", "gene_effect", "NCBI_ID",
                     "site", "plate", "experiment", "condition"}

    numeric_cols = [
        c for c in cell_df.columns
        if pd.api.types.is_numeric_dtype(cell_df[c])
        and c not in metadata_cols
        and any(c.startswith(prefix) for prefix in feature_prefixes)
    ]
    print(f"  Feature columns: {len(numeric_cols)} [{time.time() - t_prep:.1f}s]")
    t_prep = time.time()

    # Guide-level aggregation. Include geneKO library cols (gene_effect, NCBI_ID)
    # and, for custom-perturbation libraries, the config guide column plus
    # gene_target/row_type — then filter to whichever are actually present in
    # cell_df. Lets one aggregator serve both library types.
    guide_metadata_cols_all = [
        "sgRNA", "barcode", "gene_name",
        "gene_effect", "NCBI_ID",
        "gene_target", "row_type",
    ]
    if guide_col and guide_col not in guide_metadata_cols_all:
        guide_metadata_cols_all.append(guide_col)
    guide_metadata_cols = [c for c in guide_metadata_cols_all if c in cell_df.columns]
    guide_feature_cols = [c for c in numeric_cols if c not in guide_metadata_cols]

    # Normalize NTC gene_name: cells with NCBI_ID = -1 are non-targeting controls
    if "gene_name" in cell_df.columns and "NCBI_ID" in cell_df.columns:
        ntc_mask = cell_df["NCBI_ID"] == -1
        n_ntc = ntc_mask.sum()
        if n_ntc > 0:
            cell_df.loc[ntc_mask, "gene_name"] = "NTC"
            print(f"  Normalized {n_ntc:,} NTC cells (NCBI_ID = -1) [{time.time() - t_prep:.1f}s]")
            t_prep = time.time()

    # Guide-level aggregation: group by sgRNA (not barcode) to avoid duplicates from barcode length inconsistency
    #
    # Why sgRNA over barcode?
    # - Barcodes are truncated to match effective ISS rounds per well (datasets.py lines 415-417, 457-458)
    # - Wells with failed ISS rounds get shorter barcodes (e.g., 9-char vs 10-char)
    # - The SAME guide can have different barcode strings across wells due to truncation
    # - This causes inflated guide counts (e.g., 8,393 guides instead of 4,211 library size)
    # - sgRNA is the 20-char guide sequence from the library (twist1k_pool_CERES.csv) - invariant across wells
    # - sgRNA comes from the gene_index library merge in datasets.py link_ops_experiment()
    #
    print(f"Aggregating to guide level ({len(guide_feature_cols)} features)...")
    t0 = time.time()

    # Require sgRNA column for guide aggregation - no fallback to barcode
    if "sgRNA" not in cell_df.columns:
        raise ValueError(
            "Missing 'sgRNA' column in cell data - cannot aggregate to guide level.\n"
            "sgRNA is required because barcode lengths vary across wells (see comment above).\n"
            "Fix: Re-run linking (link_calls_tracks) to get sgRNA from library merge."
        )

    # Check for valid sgRNA (avoid .astype(str) on 1M+ rows - just check for string "None")
    sgrna_col = cell_df["sgRNA"]
    valid_sgrna_mask = sgrna_col.notna() & (sgrna_col != "") & (sgrna_col != "None")
    cells_for_guide = cell_df[valid_sgrna_mask]
    n_filtered = (~valid_sgrna_mask).sum()
    if n_filtered > 0:
        print(f"  Filtering {n_filtered:,} cells without valid sgRNA from guide aggregation")

    guide_summary = vectorized_groupby_agg(
        cells_for_guide, "sgRNA", guide_feature_cols, agg_funcs,
    )
    # Exclude sgRNA from metadata cols since it's the groupby key
    guide_meta_cols_filtered = [c for c in guide_metadata_cols if c != "sgRNA"]
    guide_metadata = cells_for_guide.groupby("sgRNA")[guide_meta_cols_filtered].first().reset_index()
    guide_metadata = guide_metadata[[c for c in guide_metadata.columns if c in cells_for_guide.columns or c == "sgRNA"]]
    guide_df = pd.merge(guide_summary, guide_metadata, on="sgRNA", how="left")
    del guide_summary
    print(f"  Guides: {len(guide_df)} ({time.time() - t0:.1f}s)")

    # Gene-level aggregation (parallel across column chunks). Same any-of-
    # both-libraries pattern: keep custom-perturbation cols when present.
    gene_metadata_cols_all = [
        "gene_effect", "NCBI_ID",
        "gene_target", "row_type",
    ]
    gene_metadata_cols = [c for c in gene_metadata_cols_all if c in cell_df.columns]
    gene_feature_cols = [c for c in numeric_cols if c not in gene_metadata_cols]

    print(f"Aggregating to gene level ({len(gene_feature_cols)} features)...")
    t0 = time.time()
    gene_summary = vectorized_groupby_agg(
        cell_df, "gene_name", gene_feature_cols, agg_funcs,
    )
    gene_metadata = cell_df.groupby("gene_name")[gene_metadata_cols].first().reset_index()
    gene_metadata = gene_metadata[[c for c in gene_metadata.columns if c in cell_df.columns or c == "gene_name"]]
    gene_df = pd.merge(gene_summary, gene_metadata, on="gene_name", how="left")
    del gene_summary
    print(f"  Genes: {len(gene_df)} ({time.time() - t0:.1f}s)")

    # Pre-compute counts before we start freeing DataFrames
    # (replaces O(n²) lambda lookups with O(n) value_counts)
    n_cells_total = len(cell_df)
    barcode_counts = cell_df["barcode"].value_counts() if "barcode" in cell_df.columns else pd.Series(dtype=int)
    gene_cell_counts = cell_df["gene_name"].value_counts() if "gene_name" in cell_df.columns else pd.Series(dtype=int)
    gene_guide_counts = guide_df["gene_name"].value_counts() if "gene_name" in guide_df.columns else pd.Series(dtype=int)

    feature_cols = numeric_cols
    guide_agg_suffixes = ["_mean", "_median", "_std", "_sum", "_min", "_max"]

    # --- Save guide-level AnnData first (small, frees guide_df) ---
    print("\nSaving AnnData files...")

    # Identify aggregated feature columns by suffix (fast set lookup)
    agg_suffixes = {"_sum", "_mean", "_median", "_std", "_min", "_max", "_count", "_fraction"}
    guide_feature_cols = [
        c for c in guide_df.columns
        if pd.api.types.is_numeric_dtype(guide_df[c])
        and any(c.endswith(s) for s in agg_suffixes)
    ]
    guide_obs_cols = [c for c in guide_df.columns if c not in guide_feature_cols]

    guide_obs_df = guide_df[guide_obs_cols].copy()
    for col in guide_obs_df.select_dtypes(include=['object']).columns:
        guide_obs_df[col] = guide_obs_df[col].fillna("").astype(str)
    # Use sgRNA as index (the true guide identifier), fall back to barcode for legacy
    if "sgRNA" in guide_obs_df.columns:
        guide_obs_df.index = guide_obs_df["sgRNA"].astype(str)
        guide_obs_df.index.name = None
        # Count cells per sgRNA
        sgrna_counts = cell_df["sgRNA"].value_counts() if "sgRNA" in cell_df.columns else pd.Series(dtype=int)
        guide_obs_df["n_cells"] = guide_obs_df["sgRNA"].map(sgrna_counts).fillna(0).astype(int)
    elif "barcode" in guide_obs_df.columns:
        guide_obs_df.index = guide_obs_df["barcode"].astype(str)
        guide_obs_df.index.name = None
        guide_obs_df["n_cells"] = guide_obs_df["barcode"].map(barcode_counts).fillna(0).astype(int)

    X_guide = guide_df[guide_feature_cols].values.astype(np.float32)
    var_guide_df = pd.DataFrame(index=guide_feature_cols)
    var_guide_df["feature_name"] = guide_feature_cols

    # Build var metadata in bulk (avoid per-column parse_feature_name calls)
    organelles, metrics, categories, aggregations, units = [], [], [], [], []
    for gf in guide_feature_cols:
        cell_feat = gf
        for suffix in guide_agg_suffixes:
            if gf.endswith(suffix):
                cell_feat = gf[:-len(suffix)]
                break
        parsed = parse_feature_name(cell_feat, organelle_names)
        organelles.append(parsed["organelle"])
        metrics.append(parsed["metric"])
        categories.append(parsed["category"])
        aggregations.append(parsed["aggregation"])
        units.append(get_unit_for_metric(parsed["metric"]))

    var_guide_df["organelle"] = organelles
    var_guide_df["metric"] = metrics
    var_guide_df["category"] = categories
    var_guide_df["aggregation"] = aggregations
    var_guide_df["unit"] = units

    # Replace inf values with NaN (float32 overflow)
    n_inf_guide = np.isinf(X_guide).sum()
    if n_inf_guide > 0:
        print(f"  Replacing {n_inf_guide:,} inf values with NaN in guide features")
        X_guide = np.where(np.isinf(X_guide), np.nan, X_guide)

    guide_adata = ad.AnnData(X=X_guide, obs=guide_obs_df, var=var_guide_df)
    guide_adata.uns["creation_date"] = datetime.now().isoformat()
    guide_adata.uns["experiment"] = experiment
    guide_adata.uns["level"] = "guide"

    guide_path = output_dir / f"{experiment}_guide_features.h5ad"
    guide_adata.write_h5ad(guide_path)
    print(f"  -> {guide_path}")

    del guide_adata, X_guide, guide_obs_df, var_guide_df, guide_df
    gc.collect()

    # --- Save gene-level AnnData (small, frees gene_df) ---
    gene_feature_cols = [
        c for c in gene_df.columns
        if pd.api.types.is_numeric_dtype(gene_df[c])
        and any(c.endswith(s) for s in agg_suffixes)
    ]
    gene_obs_cols = [c for c in gene_df.columns if c not in gene_feature_cols]

    gene_obs_df = gene_df[gene_obs_cols].copy()
    for col in gene_obs_df.select_dtypes(include=['object']).columns:
        gene_obs_df[col] = gene_obs_df[col].fillna("").astype(str)
    if "gene_name" in gene_obs_df.columns:
        gene_obs_df.index = gene_obs_df["gene_name"].astype(str)
        gene_obs_df.index.name = None
        gene_obs_df["n_cells"] = gene_obs_df["gene_name"].map(gene_cell_counts).fillna(0).astype(int)
        gene_obs_df["n_guides"] = gene_obs_df["gene_name"].map(gene_guide_counts).fillna(0).astype(int)
    else:
        gene_obs_df.index = gene_obs_df.index.astype(str)
        gene_obs_df["n_cells"] = 0
        gene_obs_df["n_guides"] = 0

    X_gene = gene_df[gene_feature_cols].values.astype(np.float32)
    var_gene_df = pd.DataFrame(index=gene_feature_cols)
    var_gene_df["feature_name"] = gene_feature_cols

    # Build var metadata in bulk
    organelles, metrics, categories, aggregations, units = [], [], [], [], []
    for gf in gene_feature_cols:
        cell_feat = gf
        for suffix in guide_agg_suffixes:
            if gf.endswith(suffix):
                cell_feat = gf[:-len(suffix)]
                break
        parsed = parse_feature_name(cell_feat, organelle_names)
        organelles.append(parsed["organelle"])
        metrics.append(parsed["metric"])
        categories.append(parsed["category"])
        aggregations.append(parsed["aggregation"])
        units.append(get_unit_for_metric(parsed["metric"]))

    var_gene_df["organelle"] = organelles
    var_gene_df["metric"] = metrics
    var_gene_df["category"] = categories
    var_gene_df["aggregation"] = aggregations
    var_gene_df["unit"] = units

    # Replace inf values with NaN (float32 overflow)
    n_inf_gene = np.isinf(X_gene).sum()
    if n_inf_gene > 0:
        print(f"  Replacing {n_inf_gene:,} inf values with NaN in gene features")
        X_gene = np.where(np.isinf(X_gene), np.nan, X_gene)

    gene_adata = ad.AnnData(X=X_gene, obs=gene_obs_df, var=var_gene_df)
    gene_adata.uns["creation_date"] = datetime.now().isoformat()
    gene_adata.uns["experiment"] = experiment
    gene_adata.uns["level"] = "gene"

    gene_path = output_dir / f"{experiment}_gene_features.h5ad"
    gene_adata.write_h5ad(gene_path)
    print(f"  -> {gene_path}")

    n_guides = len(guide_feature_cols)
    n_genes = len(gene_feature_cols)
    del gene_adata, X_gene, gene_obs_df, var_gene_df, gene_df
    gc.collect()

    # --- Save cell-level AnnData (large — minimize peak memory) ---
    # Extract obs columns before converting features to float32
    obs_cols = [c for c in cell_df.columns if c not in feature_cols]
    obs_df = cell_df[obs_cols].copy()

    # Convert list/array columns to strings for AnnData compatibility
    # Check dtype instead of sampling values (much faster)
    cols_to_convert = []
    cols_to_drop = []
    for col in obs_df.columns:
        if obs_df[col].dtype == object:
            # Sample first non-null to check type
            first_valid_idx = obs_df[col].first_valid_index()
            if first_valid_idx is not None:
                sample_val = obs_df[col].loc[first_valid_idx]
                if isinstance(sample_val, (list, tuple, np.ndarray)):
                    cols_to_convert.append(col)

    for col in cols_to_convert:
        obs_df[col] = obs_df[col].astype(str).replace("None", "")

    if cols_to_drop:
        print(f"  Dropping incompatible columns from obs: {cols_to_drop}")
        obs_df = obs_df.drop(columns=cols_to_drop)

    if "cell_id" in obs_df.columns:
        obs_df.index = obs_df["cell_id"].astype(str)
        obs_df.index.name = None
    else:
        obs_df.index = obs_df.index.astype(str)

    # Drop columns that are entirely None/NaN (e.g., cp_bbox, cp_cell_seg_id for non-CP experiments)
    all_null_cols = [col for col in obs_df.columns if obs_df[col].isna().all()]
    if all_null_cols:
        print(f"  Dropping {len(all_null_cols)} all-null columns: {all_null_cols}")
        obs_df = obs_df.drop(columns=all_null_cols)

    for bbox_col in ['bbox', 'cp_bbox']:
        if bbox_col in obs_df.columns:
            # Vectorized conversion: astype(str) handles most cases, then fix None/nan
            obs_df[bbox_col] = obs_df[bbox_col].astype(str).replace({"None": "", "nan": ""})
            print(f"  Converted {bbox_col} to string format for HDF5 serialization")

    # Convert cp_cell_seg_id to string (may contain None/NaN/numeric that h5py can't handle)
    if 'cp_cell_seg_id' in obs_df.columns:
        # Vectorized: fillna with empty, convert to int then string
        col = obs_df['cp_cell_seg_id']
        mask = col.notna()
        obs_df['cp_cell_seg_id'] = ""
        if mask.any():
            obs_df.loc[mask, 'cp_cell_seg_id'] = col[mask].astype(int).astype(str)
        print(f"  Converted cp_cell_seg_id to string format for HDF5 serialization")

    # Convert features to float32 in chunks to halve memory (57GB -> 29GB)
    print(f"  Converting {len(feature_cols)} feature columns to float32...", flush=True)
    t0 = time.time()
    CHUNK = 100
    for i in range(0, len(feature_cols), CHUNK):
        cols = feature_cols[i:i + CHUNK]
        cell_df[cols] = cell_df[cols].astype(np.float32)
    print(f"  Float32 conversion: {time.time() - t0:.1f}s", flush=True)

    # Extract X and free cell_df
    X = cell_df[feature_cols].values
    del cell_df
    gc.collect()

    # Replace inf values with NaN (float32 overflow in _sum columns causes inf)
    n_inf = np.isinf(X).sum()
    if n_inf > 0:
        print(f"  Replacing {n_inf:,} inf values with NaN (float32 overflow in _sum columns)")
        X = np.where(np.isinf(X), np.nan, X)

    var_df = pd.DataFrame(index=feature_cols)
    var_df["feature_name"] = feature_cols
    parsed = [parse_feature_name(f, organelle_names) for f in feature_cols]
    var_df["organelle"] = [p["organelle"] for p in parsed]
    var_df["metric"] = [p["metric"] for p in parsed]
    var_df["category"] = [p["category"] for p in parsed]
    var_df["aggregation"] = [p["aggregation"] for p in parsed]
    var_df["unit"] = [get_unit_for_metric(p["metric"]) for p in parsed]

    # Coerce obs columns that h5py/anndata can't serialize natively — object,
    # pandas nullable boolean/string, or category dtypes — to plain strings.
    # Without this, a non-string obs column (e.g. a custom-library
    # 'is_negative_control', a nullable boolean with <NA>) makes write_h5ad raise
    # "Can't implicitly convert non-string objects to strings", which aborts the
    # cell h5ad write and leaves a nameless/positional-var file. Plain numpy
    # numeric/bool columns are left untouched.
    for _col in obs_df.columns:
        _s = obs_df[_col]
        if _s.dtype == object or str(_s.dtype) in ("category", "boolean", "string"):
            obs_df[_col] = _s.astype(str).replace(
                {"None": "", "nan": "", "<NA>": "", "NaN": ""}
            )

    cell_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    cell_adata.obs_names_make_unique()
    cell_adata.uns["creation_date"] = datetime.now().isoformat()
    cell_adata.uns["experiment"] = experiment
    cell_adata.uns["level"] = "cell"
    cell_adata.uns["aggregation_functions"] = agg_funcs

    cell_path = output_dir / f"{experiment}_cell_features.h5ad"
    cell_adata.write_h5ad(cell_path)
    print(f"  -> {cell_path}")

    del cell_adata, X, obs_df
    gc.collect()

    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Cell features:  {n_cells_total:,} cells x {len(feature_cols):,} features")
    print(f"Guide features: {n_guides:,} guides")
    print(f"Gene features:  {n_genes:,} genes")

    # Cleanup intermediate directories after successful aggregation
    import shutil

    # Remove batch metadata (pre-computed cell metadata for workers)
    batch_meta_dir = output_dir / "_batch_metadata"
    if batch_meta_dir.exists():
        n_meta_files = len(list(batch_meta_dir.glob("*.parquet")))
        shutil.rmtree(batch_meta_dir)
        print(f"\n✓ Cleaned up {n_meta_files} batch metadata files")

    print(f"\nNote: Batch results kept at {batch_results_dir}")
    print("      To clean up: rm -rf", batch_results_dir)

    return {
        "status": "success",
        "n_cells": n_cells_total,
        "n_guides": n_guides,
        "n_genes": n_genes,
    }


def submit_aggregation_slurm(
    experiment: str,
    output_dir: Path,
    log_dir: str = None,
    mem: str = "350G",
    cpus: int = 32,
    timeout_min: int = 20,
    dry_run: bool = False,
    wait: bool = True,
) -> dict:
    """
    Submit aggregation as a SLURM job with configurable memory.

    Uses the standard submit_parallel_jobs infrastructure for consistent
    job tracking, manifest generation, and resource reporting.

    Parameters
    ----------
    experiment : str
        Experiment name
    output_dir : Path
        Output directory containing _batch_results
    log_dir : str, optional
        Directory for SLURM logs (default: slurm_feature_extraction_logs/{experiment}_agg)
    mem : str
        Memory allocation (e.g., "256G", "512G")
    cpus : int
        Number of CPUs (default: 32)
    timeout_min : int
        Job timeout in minutes (default: 20)
    dry_run : bool
        If True, only show what would be submitted
    wait : bool
        If True, wait for job completion

    Returns
    -------
    dict
        Result with success status and job info
    """
    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    if log_dir is None:
        log_dir = f"slurm_feature_extraction_logs/{experiment}_agg"

    # Define the aggregation job
    agg_job = {
        "name": f"aggregate_{experiment}",
        "func": aggregate_batch_results,
        "kwargs": {
            "submitted_jobs": [],
            "experiment": experiment,
            "output_dir": output_dir,
        },
        "metadata": {
            "type": "aggregation",
            "experiment": experiment,
            "output_dir": str(output_dir),
        },
    }

    # SLURM parameters for aggregation
    slurm_params = {
        "timeout_min": timeout_min,
        "mem": mem,
        "cpus_per_task": cpus,
        "slurm_partition": "cpu",
        "job_name": f"fe_agg_{experiment}",
    }

    print(f"\n{'='*60}")
    print("AGGREGATION SLURM JOB")
    print(f"{'='*60}")
    print(f"  Experiment: {experiment}")
    print(f"  Memory: {mem}")
    print(f"  CPUs: {cpus}")
    print(f"  Timeout: {timeout_min} min")

    result = submit_parallel_jobs(
        jobs_to_submit=[agg_job],
        experiment=experiment,
        slurm_params=slurm_params,
        log_dir=log_dir,
        manifest_prefix="aggregation",
        dry_run=dry_run,
        wait_for_completion=wait,
        verbose=True,
        print_resource_summary=True,
    )

    return result


def get_wells_for_experiment(experiment: str, wells: Optional[Sequence[str]] = None) -> list[str]:
    """
    Get all well positions from the experiment's zarr store.

    Parameters
    ----------
    experiment : str
        Experiment name
    wells : Optional[Sequence[str]]
        If provided, filter to these specific wells

    Returns
    -------
    list[str]
        List of well identifiers (e.g., ['A/1/0', 'A/2/0', ...])
    """
    ds = OpsDataset(experiment)
    store_path = ds.store_paths["pheno_assembled_v3"]

    if not store_path.exists():
        raise ValueError(f"Zarr store not found: {store_path}")

    with open_ome_zarr(store_path, mode="r") as store:
        all_wells = [f"A/{i}/0" for i in store["A"].group_keys()]

    if wells:
        # Filter to specified wells
        filtered = [w for w in all_wells if w in wells or any(w.startswith(f) for f in wells)]
        print(f"Selected {len(filtered)}/{len(all_wells)} wells matching filter: {wells}")
        return filtered

    print(f"Found {len(all_wells)} wells in {store_path}")
    return all_wells


def _resolve_fe_output_dir(experiment: str, args) -> Path:
    """Resolve the feature-extraction output dir, honoring --output-base if set.

    Default: <experiment>/3-assembly/feature_extraction/ (via OpsDataset.results_fast)
    With --output-base X: X/<experiment>/feature_extraction/
    """
    base = getattr(args, "output_base", None)
    if base:
        return Path(base) / experiment / "feature_extraction"
    return OpsDataset(experiment).results_fast / "feature_extraction"


def _filter_labels_by_modality(available_labels: dict, modality: str, explicit_organelles=None) -> dict:
    """Restrict the discovered-labels dict to the set implied by --modality / --organelles.

    Keeps core segmentation labels (cell_mask, nuclei, cp_cell_mask) intact — they are
    required inputs, not organelles to process. The modality filter only affects the
    organelle labels used for feature extraction.
    """
    if explicit_organelles:
        wanted = {s.strip() for s in explicit_organelles.split(",") if s.strip()}
        core = {"cell_mask", "nuclei", "cp_cell_mask"}
        keep = {k: v for k, v in available_labels.items() if k in wanted or k in core}
        unknown = wanted - set(available_labels.keys())
        if unknown:
            print(f"  [--organelles] WARNING: these names were not found in the zarr and will be skipped: {sorted(unknown)}")
        return keep

    if modality == "all":
        return dict(available_labels)

    core = {"cell_mask", "nuclei", "cp_cell_mask"}
    if modality == "phase":
        return {k: v for k, v in available_labels.items() if k in core or not k.lower().startswith("cp")}
    if modality == "fluorescent":
        return {k: v for k, v in available_labels.items() if k in core or k.lower().startswith("cp")}
    raise ValueError(f"Unknown modality: {modality!r}")


def _apply_cells_csv_filter(cells_df, cells_csv_path: str, modality: str):
    """Restrict cells_df to rows matching a top-cells CSV.

    Joins on (well, segmentation_id) for phase modality, (well, cp_cell_seg_id) for
    fluorescent. For `all`, accepts matches via either column.

    The input CSV uses column `segmentation` holding the cell seg-id (whichever is
    appropriate for the CSV's source model). Wells are normalized via
    canonicalize_well_path + appending '/0' so that 'A3' -> 'A/3/0'.
    """
    from cyclops_utils.data.filesystem import canonicalize_well_path
    subset = pd.read_csv(cells_csv_path)
    required = {"well", "segmentation"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"--cells-csv {cells_csv_path!r} missing columns: {sorted(missing)}")

    # Normalize well format in the subset: "A3" -> "A/3/0", "A/3" -> "A/3/0", "A/3/0" -> "A/3/0"
    def _norm(w):
        canon = canonicalize_well_path(str(w))
        return canon if canon.count("/") >= 2 else f"{canon}/0"

    subset = subset.assign(
        _well=subset["well"].apply(_norm),
        _seg=subset["segmentation"].astype("Int64"),
    )
    print(f"  --cells-csv: loaded {len(subset):,} target cells across {subset['_well'].nunique()} wells")

    subset_keys = set(zip(subset["_well"].tolist(), subset["_seg"].astype("int64").tolist()))

    # Build candidate match keys vectorized per modality.
    mask = pd.Series(False, index=cells_df.index)
    if modality in ("phase", "all") and "segmentation_id" in cells_df.columns:
        seg = pd.to_numeric(cells_df["segmentation_id"], errors="coerce")
        valid = seg.notna()
        phase_keys = list(zip(cells_df.loc[valid, "well"].tolist(), seg.loc[valid].astype("int64").tolist()))
        phase_hit = pd.Series([k in subset_keys for k in phase_keys], index=cells_df.loc[valid].index)
        mask = mask | phase_hit.reindex(cells_df.index, fill_value=False)
    if modality in ("fluorescent", "all") and "cp_cell_seg_id" in cells_df.columns:
        cp_seg = pd.to_numeric(cells_df["cp_cell_seg_id"], errors="coerce")
        valid = cp_seg.notna()
        cp_keys = list(zip(cells_df.loc[valid, "well"].tolist(), cp_seg.loc[valid].astype("int64").tolist()))
        cp_hit = pd.Series([k in subset_keys for k in cp_keys], index=cells_df.loc[valid].index)
        mask = mask | cp_hit.reindex(cells_df.index, fill_value=False)

    filtered = cells_df[mask].copy()
    print(f"  --cells-csv: kept {len(filtered):,} / {len(cells_df):,} cells after subset filter (modality={modality})")
    if len(filtered) == 0:
        missing_keys = list(subset_keys)[:5]
        raise ValueError(
            "--cells-csv produced an empty filter. No cells_df row matched any "
            f"(well, seg_id) in the CSV. First few target keys: {missing_keys}. "
            "Check that --modality matches the CSV's source model, and that wells are "
            "normalized correctly in the experiment's linked metadata."
        )
    return filtered


def _fast_resume(
    experiment: str,
    output_dir: Path,
    args,
    resume_from: str,
    stop_after: "str | None",
) -> dict:
    """Fast path for resuming from spmd/merge/aggregate.

    Recovers batch IDs from existing GPU output files on disk instead of
    re-loading all cell metadata and re-building batches.
    """
    batch_results_dir = output_dir / "_batch_results"

    # Determine which phases to run
    _PHASES = ["gpu", "spmd", "merge", "aggregate"]
    _start_idx = _PHASES.index(resume_from)
    _stop_idx = _PHASES.index(stop_after) if stop_after else len(_PHASES) - 1
    run_spmd = _start_idx <= 1 <= _stop_idx
    run_merge = _start_idx <= 2 <= _stop_idx
    run_aggregate = _start_idx <= 3 <= _stop_idx

    phases_to_run = [p for i, p in enumerate(_PHASES) if _start_idx <= i <= _stop_idx]
    print(f"\n[{experiment}] fast-resume from {resume_from!r} (phases: {' -> '.join(phases_to_run)})")

    # Recover batch IDs from _batch_metadata (written during GPU phase)
    batch_meta_dir = output_dir / "_batch_metadata"
    if batch_meta_dir.exists():
        valid_batch_ids = sorted(
            p.stem.replace("batch_", "").replace("_meta", "")
            for p in batch_meta_dir.glob("batch_*_meta.parquet")
        )
    else:
        valid_batch_ids = []

    if not valid_batch_ids:
        print(f"  ERROR: No batch metadata found in {batch_meta_dir}")
        return {"success": False, "error": "no batch metadata found for fast-resume"}

    print(f"  Found {len(valid_batch_ids)} batches (from _batch_metadata)")

    spmd_ntasks = getattr(args, "spmd_ntasks", 1024)
    gpu_result = {"success": True, "skipped": True}
    spmd_result = None

    # SPMD / Merge
    if run_spmd or run_merge:
        if run_spmd:
            spmd_result = submit_spmd_cpu_phase(
                batch_ids=valid_batch_ids,
                experiment=experiment,
                output_dir=output_dir,
                log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                n_ranks=spmd_ntasks,
                wait_for_completion=not args.no_wait,
                submit_merge_after=run_merge,
            )
        elif run_merge:
            spmd_result = submit_merge_job(
                batch_ids=valid_batch_ids,
                experiment=experiment,
                output_dir=output_dir,
                log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                wait_for_completion=not args.no_wait,
            )
        if spmd_result and not spmd_result.get("success"):
            return {"success": False, "error": spmd_result.get("error"), "spmd_result": spmd_result}
    else:
        spmd_result = {"success": True, "skipped": True}

    # Aggregate
    agg_result = None
    if run_aggregate and not args.dry_run:
        agg_mem = getattr(args, "agg_mem", "350G")
        agg_result = submit_aggregation_slurm(
            experiment=experiment,
            output_dir=output_dir,
            mem=agg_mem,
            dry_run=False,
            wait=not args.no_wait,
        )

    return {
        "success": True,
        "split_mode": True,
        "gpu_result": gpu_result,
        "spmd_result": spmd_result,
        "agg_result": agg_result,
        "gpu_failed": [],
    }


def submit_feature_extraction_jobs(
    experiment: str,
    args,
) -> dict:
    """
    Submit SLURM jobs for per-well batch feature extraction.

    Each job processes ~25k cells from a single well (configurable via --cells-per-batch).
    Cells within each batch are spatially sorted for optimal zarr chunk locality.

    Parameters
    ----------
    experiment : str
        Experiment name
    args : argparse.Namespace
        CLI arguments

    Returns
    -------
    dict
        Job submission results
    """
    ds = OpsDataset(experiment)
    output_dir = _resolve_fe_output_dir(experiment, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fast path: skip expensive cell/label discovery when resuming from
    # a post-GPU phase.  All we need are the batch IDs (from existing files on
    # disk) and the output_dir.
    resume_from = getattr(args, "resume_from", None)
    stop_after = getattr(args, "stop_after", None)
    split_mode = True  # split is the only supported mode (GPU->SPMD CPU->merge)
    if split_mode and resume_from in ("spmd", "merge", "aggregate"):
        return _fast_resume(experiment, output_dir, args, resume_from, stop_after)

    # Preview mode: use separate subdirectory to avoid overwriting real results
    preview_n = getattr(args, 'preview', None)
    if preview_n:
        output_dir = output_dir / "_preview"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get cells per batch setting
    cells_per_batch = getattr(args, 'cells_per_batch', None) or DEFAULT_CELLS_PER_BATCH

    # Get wells to filter (optional)
    wells_filter = getattr(args, 'wells', None)

    # Discover segmentations for info display
    morphology_path = ds.store_paths["pheno_assembled_v3"]
    # Info: which filesystem the data lives on
    print("\n" + "="*60)
    morphology_path_str = str(morphology_path)
    if "icd.fast.ops" in morphology_path_str:
        print("Using icd.fast.ops - VAST filesystem (flash-based)")
    else:
        print(f"Filesystem: {morphology_path_str}")
    print("="*60 + "\n")
    available_labels = _discover_available_labels(morphology_path)
    n_segmentations = len(available_labels)

    # Validate segmentation labels exist for all positions
    print("Validating segmentation labels across all positions...")
    validation = validate_segmentation_labels(morphology_path, wells_filter)
    if not validation["valid"]:
        print(f"\n❌ ERROR: Segmentation validation failed!")
        print(f"   {validation['error']}")
        if validation["missing_positions"]:
            print(f"\n   Missing positions ({len(validation['missing_positions'])}):")
            for mp in validation["missing_positions"][:10]:  # Show first 10
                print(f"     - {mp['position']}: {mp['issue']}")
            if len(validation["missing_positions"]) > 10:
                print(f"     ... and {len(validation['missing_positions']) - 10} more")
        print(f"\n   This indicates segmentation did not complete for all positions.")
        print(f"   Please re-run organelle segmentation before feature extraction.")
        return {"success": False, "error": validation["error"]}
    print(f"  ✓ All {validation['positions_checked']} positions have valid segmentation labels")

    # Restrict to the modality/organelle subset the caller asked for (if any).
    # This must happen *before* network/CP derived lists so they stay in sync.
    if getattr(args, "modality", "all") != "all" or getattr(args, "organelles", None):
        before = set(available_labels.keys())
        available_labels = _filter_labels_by_modality(
            available_labels,
            modality=getattr(args, "modality", "all"),
            explicit_organelles=getattr(args, "organelles", None),
        )
        removed = sorted(before - set(available_labels.keys()))
        print(f"  modality={getattr(args, 'modality', 'all')}, organelles-override={getattr(args, 'organelles', None)}")
        print(f"  Kept {len(available_labels)} labels; removed {len(removed)}: {removed}")

    # Count network organelles
    network_organelles = [name for name in available_labels.keys() if is_network_organelle(name)]

    # Detect CP organelles
    cp_organelles = [name for name in available_labels.keys() if name.lower().startswith("cp")]
    has_cp = len(cp_organelles) > 0

    # Load cell metadata
    print(f"Loading cell metadata...")
    try:
        cells_df = _load_cells_metadata(ds, morphology_path)
        total_cells = len(cells_df)
        print(f"  Total cells: {total_cells:,}")
    except Exception as e:
        print(f"  ERROR: Could not load cell metadata: {e}")
        return {"success": False, "error": str(e)}

    # Create global_cell_id if not present (needed by GPU/CPU split workers)
    create_global_cell_id(cells_df)

    # Filter to specified wells if provided
    if wells_filter:
        cells_df = cells_df[cells_df["well"].isin(wells_filter)].copy()
        print(f"  Filtered to wells {wells_filter}: {len(cells_df):,} cells")

    # Filter to cells listed in an external CSV (top-cells pipeline)
    if getattr(args, "cells_csv", None):
        cells_df = _apply_cells_csv_filter(
            cells_df,
            cells_csv_path=args.cells_csv,
            modality=getattr(args, "modality", "all"),
        )

    if len(cells_df) == 0:
        print("No cells to process")
        return {"success": False, "error": "No cells found"}

    # Group cells by well and create batches within each well
    # Each batch is spatially sorted within its well for optimal zarr locality
    wells = cells_df["well"].unique()
    n_wells = len(wells)
    
    print(f"\nBatch-based feature extraction (per-well):")
    print(f"  Total cells: {len(cells_df):,}")
    print(f"  Wells: {n_wells}")
    print(f"  Cells per batch: {cells_per_batch:,}")
    print(f"  Segmentations: {n_segmentations}")
    print(f"  Network organelles: {len(network_organelles)}")
    if has_cp:
        print(f"  Cell Painting organelles: {len(cp_organelles)}")

    # Calculate timeout based on cells per batch and segmentation complexity
    n_workers = FE_SLURM_PARAMS_BASE["cpus_per_task"]
    timeout_min = estimate_timeout_minutes(
        n_cells=cells_per_batch,
        n_segmentations=n_segmentations,
        n_network_organelles=len(network_organelles),
        n_workers=n_workers,
    )
    
    # CRITICAL: Limit concurrent jobs to 1 per well to avoid zarr I/O contention
    # Multiple jobs hitting the same well's zarr position causes severe filesystem
    # contention on HPC parallel filesystems (Lustre/GPFS), slowing jobs by 10-100x.
    # By limiting to n_wells concurrent jobs, each well is processed by 1 job at a time.
    slurm_params = {
        **FE_SLURM_PARAMS_BASE,
        "timeout_min": timeout_min,
    }

    # Sequential mode: single CPU for benchmarking
    sequential_mode = getattr(args, 'sequential', False)
    if sequential_mode:
        slurm_params["cpus_per_task"] = 1
        slurm_params["slurm_mem"] = "64GB"  # Less memory needed for single worker
        print(f"  🔬 SEQUENTIAL MODE: 1 CPU, processing cells one at a time")

    # Override partition if specified
    if getattr(args, 'partition', None):
        slurm_params["slurm_partition"] = args.partition

    # Override max concurrent jobs if specified
    if getattr(args, 'max_concurrent', None):
        slurm_params["slurm_array_parallelism"] = args.max_concurrent

    print(f"  Timeout per job: {timeout_min} minutes (based on {n_segmentations} segs, {len(network_organelles)} network)")
    print(f"  Partition: {slurm_params['slurm_partition']}")
    print(f"  Max concurrent jobs: {slurm_params.get('slurm_array_parallelism', 'unlimited')}")

    # Always show organelle discovery table
    dry_run_discovery(experiment, preview=bool(preview_n))

    # Check for existing batch results
    batch_results_dir = output_dir / "_batch_results"
    batch_config_file = batch_results_dir / "_batch_config.json"
    existing_batches = set()
    
    # Validate mutually exclusive flags
    if args.force and getattr(args, 'checkpoint', False):
        print("Error: --force and --checkpoint are mutually exclusive")
        return {"success": False, "error": "--force and --checkpoint are mutually exclusive"}
    
    # Check for batch config consistency (critical for --checkpoint)
    if batch_results_dir.exists() and not args.force:
        existing_parquets = list(batch_results_dir.glob("batch_*_cells.parquet"))
        
        if existing_parquets and not batch_config_file.exists():
            # Old run without config - warn user we can't verify consistency
            print(f"\n⚠️  WARNING: Found {len(existing_parquets)} existing batch results but no config file.")
            print(f"   Cannot verify cells_per_batch consistency (config from older run).")
            print(f"   Assuming cells_per_batch={cells_per_batch} is correct.")
            print(f"   Use --force to start fresh if you changed batch settings.\n")
        
        elif batch_config_file.exists():
            import json
            with open(batch_config_file) as f:
                saved_config = json.load(f)
            
            # Check cells_per_batch
            saved_cells_per_batch = saved_config.get("cells_per_batch")
            if saved_cells_per_batch != cells_per_batch:
                print(f"\n⚠️  ERROR: cells_per_batch mismatch!")
                print(f"   Previous run used: {saved_cells_per_batch}")
                print(f"   Current run using: {cells_per_batch}")
                print(f"   This would cause incorrect cell assignments in --checkpoint mode.")
                print(f"   Use --force to start fresh, or use --cells-per-batch {saved_cells_per_batch}")
                return {"success": False, "error": "cells_per_batch mismatch"}
            
            # Check total_cells (source data consistency)
            saved_total_cells = saved_config.get("total_cells")
            if saved_total_cells and saved_total_cells != len(cells_df):
                print(f"\n⚠️  ERROR: Total cells changed!")
                print(f"   Previous run had: {saved_total_cells:,} cells")
                print(f"   Current data has: {len(cells_df):,} cells")
                print(f"   Source data has changed - use --force to start fresh")
                return {"success": False, "error": "total_cells mismatch"}
    
    if args.force and batch_results_dir.exists():
        # Force mode: delete existing batch results and config to ensure fresh processing
        existing = list(batch_results_dir.glob("batch_*_cells.parquet"))
        if existing:
            print(f"\n--force: Deleting {len(existing)} existing batch results...")
            for pf in existing:
                pf.unlink()
            print("  Deleted all existing batch parquets")
        # Also delete config file to reset
        if batch_config_file.exists():
            batch_config_file.unlink()
        # Wipe stale SPMD rank outputs + done-markers so the CPU phase reruns
        # cleanly. Without this, --force leaves _partials_combined behind and the
        # merge picks up stale ranks -> stale h5ad even after source cells change.
        partials_dir = batch_results_dir / "_partials_combined"
        if partials_dir.exists():
            # Rename-then-background-rm: the SPMD rank outputs are many files on
            # NFS; a synchronous rmtree stalls submission. async_delete_path
            # renames to a sibling .trash_* (instant) and rm -rf's detached.
            from cyclops_utils.data.filesystem import async_delete_path
            async_delete_path(partials_dir)
            print("  Deleting stale _partials_combined (SPMD rank outputs, async)")
    elif batch_results_dir.exists():
        existing = list(batch_results_dir.glob("batch_*_cells.parquet"))
        if existing:
            # Extract batch IDs from filenames: batch_{well}_{idx}_cells.parquet
            existing_batches = {p.stem.replace("batch_", "").replace("_cells", "") for p in existing}
            if getattr(args, 'checkpoint', False):
                print(f"\n🔄 CHECKPOINT MODE: Resuming from {len(existing)} completed batches")
            else:
                print(f"\nFound {len(existing)} existing batch results")
                print("Use --checkpoint to resume, or --force to reprocess all")

    # Build job list: create batches per-well with spatial sorting
    # Pre-save batch cell metadata to avoid loading 2M+ cells in each worker
    # Create directory for batch cell metadata
    batch_meta_dir = output_dir / "_batch_metadata"
    batch_meta_dir.mkdir(parents=True, exist_ok=True)

    split_mode = True  # split is the only supported mode (GPU->SPMD CPU->merge)

    # Build jobs per well first, then interleave across wells
    # Interleaving helps spread I/O across different zarr positions
    jobs_per_well = {well: [] for well in wells}
    # In split mode, also build CPU job specs (keyed by same well)
    cpu_jobs_per_well = {well: [] for well in wells} if split_mode else None

    for well in sorted(wells):
        # Get cells for this well and sort by CHUNK location for optimal zarr cache locality
        well_cells = cells_df[cells_df["well"] == well].copy()

        # Determine spatial coordinate columns
        if "y_global_pheno" in well_cells.columns and "x_global_pheno" in well_cells.columns:
            y_col, x_col = "y_global_pheno", "x_global_pheno"
        elif "y_pheno" in well_cells.columns and "x_pheno" in well_cells.columns:
            y_col, x_col = "y_pheno", "x_pheno"
        else:
            raise ValueError(
                f"No global spatial coordinates found for well {well}. "
                f"Expected 'y_global_pheno'/'x_global_pheno' or 'y_pheno'/'x_pheno'. "
                f"Available columns: {list(well_cells.columns)}"
            )

        # Sort by CHUNK index, not raw coordinates
        # This ensures all cells in the same zarr chunk are processed together
        CHUNK_SIZE = 512  # Zarr chunk size for phenotyping data

        # Handle NaN values - fill with large value so they sort to the end
        y_coords = well_cells[y_col].fillna(999999)
        x_coords = well_cells[x_col].fillna(999999)
        well_cells["_chunk_y"] = (y_coords // CHUNK_SIZE).astype(int)
        well_cells["_chunk_x"] = (x_coords // CHUNK_SIZE).astype(int)
        well_cells = well_cells.sort_values(["_chunk_y", "_chunk_x", y_col, x_col], na_position="last").reset_index(drop=True)

        # Report chunk coverage (exclude NaN placeholder chunks)
        valid_chunks_y = well_cells[well_cells["_chunk_y"] < 999]["_chunk_y"].nunique()
        valid_chunks_x = well_cells[well_cells["_chunk_x"] < 999]["_chunk_x"].nunique()
        n_nan = (well_cells["_chunk_y"] >= 999).sum()
        print(f"  Well {well}: chunk-sorted ({valid_chunks_y}x{valid_chunks_x} chunks, {CHUNK_SIZE}px)" +
              (f", {n_nan} cells with missing coords" if n_nan > 0 else ""))

        # Remove temporary columns before saving
        well_cells = well_cells.drop(columns=["_chunk_y", "_chunk_x"])

        n_well_cells = len(well_cells)
        n_batches_for_well = (n_well_cells + cells_per_batch - 1) // cells_per_batch

        # Create batches for this well
        for batch_idx in range(n_batches_for_well):
            well_safe = well.replace("/", "_")
            batch_id = f"{well_safe}_{batch_idx:04d}"

            # Skip if already processed
            if batch_id in existing_batches:
                continue

            # Get cells for this batch
            start_idx = batch_idx * cells_per_batch
            end_idx = min(start_idx + cells_per_batch, n_well_cells)
            batch_cells = well_cells.iloc[start_idx:end_idx].copy()

            # Save batch cells to parquet (fast load in worker)
            batch_cells_path = batch_meta_dir / f"batch_{batch_id}_meta.parquet"
            batch_cells.to_parquet(batch_cells_path, index=False)

            batch_metadata = {
                "experiment": experiment,
                "well": well,
                "batch_idx": batch_idx,
                "batch_id": batch_id,
                "n_cells": len(batch_cells),
            }

            if split_mode:
                # GPU job: morphology + localization only
                gpu_job_spec = {
                    "name": f"FE_GPU_{experiment}_{batch_id}",
                    "func": gpu_batch_worker,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "batch_idx": batch_idx,
                        "batch_cells_path": str(batch_cells_path),
                        "output_dir": str(output_dir),
                        "full_features": getattr(args, 'full_features', False),
                    },
                    "metadata": batch_metadata,
                }
                jobs_per_well[well].append(gpu_job_spec)

                # CPU job: network analysis only (runs after GPU completes)
                cpu_job_spec = {
                    "name": f"FE_CPU_{experiment}_{batch_id}",
                    "func": cpu_network_batch_worker,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "batch_idx": batch_idx,
                        "output_dir": str(output_dir),
                    },
                    "metadata": batch_metadata,
                }
                cpu_jobs_per_well[well].append(cpu_job_spec)
            else:
                # Combined mode: GPU + CPU in single job
                job_spec = {
                    "name": f"FE_{experiment}_{batch_id}",
                    "func": feature_extraction_batch_worker,
                    "kwargs": {
                        "experiment": experiment,
                        "well": well,
                        "batch_idx": batch_idx,
                        "batch_cells_path": str(batch_cells_path),
                        "output_dir": str(output_dir),
                        "full_features": getattr(args, 'full_features', False),
                        "sequential": sequential_mode,
                    },
                    "metadata": batch_metadata,
                }
                jobs_per_well[well].append(job_spec)

    # Interleave jobs across wells (round-robin) so concurrent jobs hit different wells
    # This spreads I/O load across different zarr positions
    jobs_to_submit = []
    well_list = sorted(wells)
    max_batches = max(len(jobs_per_well[w]) for w in well_list) if jobs_per_well else 0

    for batch_round in range(max_batches):
        for well in well_list:
            if batch_round < len(jobs_per_well[well]):
                jobs_to_submit.append(jobs_per_well[well][batch_round])

    # In split mode, also interleave CPU jobs
    cpu_jobs_to_submit = []
    if split_mode and cpu_jobs_per_well:
        max_cpu_batches = max(len(cpu_jobs_per_well[w]) for w in well_list) if cpu_jobs_per_well else 0
        for batch_round in range(max_cpu_batches):
            for well in well_list:
                if batch_round < len(cpu_jobs_per_well[well]):
                    cpu_jobs_to_submit.append(cpu_jobs_per_well[well][batch_round])
    
    total_batches = len(jobs_to_submit)
    print(f"  Saved {total_batches} batch metadata files to {batch_meta_dir}")
    print(f"  Jobs interleaved across {n_wells} wells (concurrent jobs hit different wells)")
    
    # Save batch config for checkpoint consistency validation
    batch_results_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(batch_config_file, 'w') as f:
        json.dump({
            "cells_per_batch": cells_per_batch,
            "total_cells": len(cells_df),
            "n_wells": n_wells,
        }, f)

    if not jobs_to_submit:
        # When resuming from spmd/merge, all GPU batches are already done so
        # jobs_to_submit is empty. Run the requested phase before aggregation.
        resume_from_early = getattr(args, 'resume_from', None)
        if split_mode and resume_from_early in ("spmd", "merge") and existing_batches:
            valid_batch_ids = sorted([
                bid for bid in existing_batches
                if (output_dir / "_batch_results" / f"batch_{bid}_network_tasks.parquet").exists()
            ])
            skipped = len(existing_batches) - len(valid_batch_ids)
            if skipped:
                print(f"\n  Skipping {skipped} batches without network tasks")
            print(f"\nAll GPU batches done. Running {resume_from_early} phase for {len(valid_batch_ids)} batches.")

            if resume_from_early == "spmd":
                result = submit_spmd_cpu_phase(
                    batch_ids=valid_batch_ids, experiment=experiment,
                    output_dir=output_dir,
                    log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                    n_ranks=getattr(args, 'spmd_ntasks', 1024),
                )
            else:
                print(f"\n{'='*60}\nPHASE 3: Merge (resuming - skip SPMD)\n{'='*60}")
                result = submit_merge_job(
                    batch_ids=valid_batch_ids, experiment=experiment,
                    output_dir=output_dir,
                    log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                )
            if not result.get("success"):
                return result
        else:
            print("\nNo batches to process (all already done or skipped)")

        # Run aggregation if we have results
        if not args.dry_run:
            if getattr(args, 'agg_local', False):
                print("Running aggregation locally...")
                aggregate_batch_results([], experiment, output_dir, available_labels)
            else:
                agg_mem = getattr(args, 'agg_mem', '350G')
                agg_result = submit_aggregation_slurm(
                    experiment=experiment, output_dir=output_dir,
                    mem=agg_mem, dry_run=False, wait=not args.no_wait,
                )
                if not agg_result.get("success"):
                    return {"success": False, "error": "Aggregation job failed"}
        return {"success": True, "jobs_submitted": 0, "note": "aggregation_only"}

    # Preview mode: select N batches, spreading across wells as much as possible
    preview_n = getattr(args, 'preview', None)
    if preview_n:
        print(f"\n{'='*60}")
        print(f"PREVIEW MODE: Selecting {preview_n} batches")
        print(f"{'='*60}")

        # Group jobs by well
        from collections import defaultdict
        jobs_by_well = defaultdict(list)
        for job in jobs_to_submit:
            jobs_by_well[job["metadata"]["well"]].append(job)

        # Round-robin select batches from different wells
        preview_jobs = []
        well_list = sorted(jobs_by_well.keys())
        well_idx = 0
        batch_per_well = defaultdict(int)

        while len(preview_jobs) < preview_n:
            well = well_list[well_idx % len(well_list)]
            well_jobs = jobs_by_well[well]
            batch_idx = batch_per_well[well]

            if batch_idx < len(well_jobs):
                preview_jobs.append(well_jobs[batch_idx])
                batch_per_well[well] += 1

            well_idx += 1

            # Stop if we've exhausted all jobs
            if sum(batch_per_well.values()) >= len(jobs_to_submit):
                break

        wells_used = set(j["metadata"]["well"] for j in preview_jobs)
        print(f"Selected {len(preview_jobs)} batches from {len(wells_used)} wells: {sorted(wells_used)}")
        total_preview_cells = sum(j["metadata"]["n_cells"] for j in preview_jobs)
        print(f"Total preview cells: {total_preview_cells:,}")

        jobs_to_submit = preview_jobs

        # In split mode, filter CPU jobs to match the selected preview batches
        if split_mode:
            preview_batch_ids = {j["metadata"]["batch_id"] for j in preview_jobs}
            cpu_jobs_to_submit = [j for j in cpu_jobs_to_submit if j["metadata"]["batch_id"] in preview_batch_ids]

    # Summary
    skipped = total_batches + len(existing_batches) - len(jobs_to_submit)
    print(f"\nTotal batches: {total_batches + len(existing_batches)}")
    if preview_n:
        print(f"Submitting {len(jobs_to_submit)} PREVIEW batch jobs")
    elif len(existing_batches) > 0:
        print(f"Submitting {len(jobs_to_submit)} remaining batch jobs (skipping {len(existing_batches)} completed)")
    else:
        print(f"Submitting {len(jobs_to_submit)} batch jobs")

    spmd_ntasks = getattr(args, 'spmd_ntasks', 1024)

    if split_mode:
        print(f"  Mode: SPLIT (GPU phase -> SPMD CPU phase -> Merge)")
        print(f"  GPU jobs: {len(jobs_to_submit)} (partition: {FE_GPU_SLURM_PARAMS['slurm_partition']}, "
              f"CPUs: {FE_GPU_SLURM_PARAMS['cpus_per_task']}, "
              f"timeout: {FE_GPU_SLURM_PARAMS['timeout_min']}min)")
        print(f"  SPMD: 1 job, {spmd_ntasks} MPI ranks (all {len(cpu_jobs_to_submit)} batches combined)")
        print(f"  Merge: 1 job, 32 CPUs (depends on SPMD)")

    # Define aggregation callback (capture available_labels in closure)
    def on_completion(jobs, exp):
        aggregate_batch_results(jobs, exp, output_dir, available_labels)

    # Only print resource summary in preview mode to avoid clutter with many jobs
    preview_mode = bool(preview_n)

    # --- Split mode: sequential GPU -> CPU -> aggregation ---
    if split_mode:
        # Check --resume-from to skip earlier phases
        resume_from = getattr(args, 'resume_from', None)

        # Determine which phases to run based on resume_from
        # Phase order: gpu -> spmd -> merge -> aggregate
        # `resume_from` sets the FIRST phase to run; `stop_after` sets the LAST.
        # Default behavior (neither set) runs all four phases.
        stop_after = getattr(args, "stop_after", None)
        _PHASES = ["gpu", "spmd", "merge", "aggregate"]
        _start_idx = _PHASES.index(resume_from) if resume_from else 0
        _stop_idx = _PHASES.index(stop_after) if stop_after else len(_PHASES) - 1
        run_gpu = _start_idx <= 0 <= _stop_idx
        run_spmd = _start_idx <= 1 <= _stop_idx
        run_merge = _start_idx <= 2 <= _stop_idx
        run_aggregate = _start_idx <= 3 <= _stop_idx

        if resume_from or stop_after:
            print(f"\n{'='*60}")
            print(f"RESUME MODE: resume_from={resume_from!r}, stop_after={stop_after!r}")
            phases_to_run = []
            if run_gpu: phases_to_run.append("GPU")
            if run_spmd: phases_to_run.append("SPMD")
            if run_merge: phases_to_run.append("Merge")
            if run_aggregate: phases_to_run.append("Aggregate")
            print(f"Phases to run: {' -> '.join(phases_to_run)}")
            print(f"{'='*60}")

        # Override max concurrent if specified
        gpu_slurm_params = {**FE_GPU_SLURM_PARAMS}
        if getattr(args, 'max_concurrent', None):
            gpu_slurm_params["slurm_array_parallelism"] = args.max_concurrent

        # Detect 4i experiment from discovered zarr labels and apply heavier
        # SLURM resources. Trigger: any organelle key OR zarr label starts
        # with "4i_". Sets OPS_FE_4I_EXPERIMENT=1 in the worker's environment
        # so `gpu_phase_worker` lowers _MAX_WORKERS_PER_GPU from 24 to 16.
        # Live-cell-only experiments keep the lighter defaults.
        is_4i_experiment = any(
            str(k).lower().startswith("4i_") or str(v).lower().startswith("4i_")
            for k, v in (available_labels or {}).items()
        )
        if is_4i_experiment:
            print(f"  [4i] 4i experiment detected — overriding GPU SLURM params: "
                  f"{FE_GPU_SLURM_PARAMS_4I_OVERRIDES}")
            gpu_slurm_params.update(FE_GPU_SLURM_PARAMS_4I_OVERRIDES)
            # Append to slurm_setup so the worker's shell env gets the flag
            # before gpu_phase_worker imports any feature-extraction modules.
            base_setup = list(gpu_slurm_params.get("slurm_setup") or [])
            base_setup.append("export OPS_FE_4I_EXPERIMENT=1")
            gpu_slurm_params["slurm_setup"] = base_setup

        gpu_result = None
        gpu_failed = []

        # Phase 1: Submit GPU jobs (skip if resuming from later phase)
        if run_gpu:
            print(f"\n{'='*60}")
            print(f"PHASE 1: GPU (morphology + localization)")
            print(f"{'='*60}")

            gpu_result = submit_parallel_jobs(
                jobs_to_submit=jobs_to_submit,
                experiment=experiment,
                slurm_params=gpu_slurm_params,
                log_dir=f"slurm_feature_extraction_logs/{experiment}_gpu",
                manifest_prefix="feature_extraction_gpu",
                dry_run=args.dry_run,
                wait_for_completion=not args.no_wait,
                verbose=not getattr(args, 'quiet', False),
                print_resource_summary=preview_mode,
            )

            if args.dry_run:
                # In dry run, show SPMD plan
                all_batch_ids = [j["metadata"]["batch_id"] for j in cpu_jobs_to_submit]
                submit_spmd_cpu_phase(
                    batch_ids=all_batch_ids,
                    experiment=experiment,
                    output_dir=output_dir,
                    log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                    n_ranks=spmd_ntasks,
                    dry_run=True,
                )
                return gpu_result

            if not gpu_result.get("success"):
                print(f"\nGPU phase submission failed: {gpu_result.get('error')}")
                return gpu_result

            if args.no_wait:
                print("\n--no-wait: GPU jobs submitted. Run again after GPU completes to submit CPU phase.")
                return gpu_result

            # Check GPU results
            gpu_failed = gpu_result.get("failed", [])
            if gpu_failed:
                print(f"\n{len(gpu_failed)} GPU jobs failed. SPMD phase will proceed for successful batches.")
        else:
            print(f"\n[Skipping GPU phase - resuming from '{resume_from}']")
            gpu_result = {"success": True, "skipped": True}

        spmd_result = None

        # Phase 2+3: SPMD CPU + Merge (single combined submission)
        if run_spmd or run_merge:
            all_batch_ids = [j["metadata"]["batch_id"] for j in cpu_jobs_to_submit]

            # Filter out batches whose GPU phase failed (no network_tasks file)
            batch_results_dir = output_dir / "_batch_results"
            valid_batch_ids = [
                bid for bid in all_batch_ids
                if (batch_results_dir / f"batch_{bid}_network_tasks.parquet").exists()
            ]
            if len(valid_batch_ids) < len(all_batch_ids):
                print(f"\n  Skipping {len(all_batch_ids) - len(valid_batch_ids)} batches without network tasks")

            if run_spmd:
                spmd_result = submit_spmd_cpu_phase(
                    batch_ids=valid_batch_ids,
                    experiment=experiment,
                    output_dir=output_dir,
                    log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                    n_ranks=spmd_ntasks,
                    wait_for_completion=not args.no_wait,
                    submit_merge_after=run_merge,
                )
            elif run_merge:
                # Resume from merge only - submit merge job directly
                # (submit_merge_job is defined in this module)
                print(f"\n{'='*60}")
                print(f"PHASE 3: Merge (resuming - skip SPMD)")
                print(f"{'='*60}")
                spmd_result = submit_merge_job(
                    batch_ids=valid_batch_ids,
                    experiment=experiment,
                    output_dir=output_dir,
                    log_dir=f"slurm_logs/slurm_feature_extraction_logs/{experiment}_cpu",
                    wait_for_completion=not args.no_wait,
                )

            if spmd_result and not spmd_result.get("success"):
                print(f"\nSPMD/Merge phase failed: {spmd_result.get('error')}")
                return spmd_result
        else:
            print(f"\n[Skipping SPMD/Merge phases]")
            spmd_result = {"success": True, "skipped": True}

        # Wave-orchestration short-circuit: if aggregation was excluded by stop_after,
        # return what we have (no_wait path).
        if not run_aggregate:
            return {
                "success": True,
                "split_mode": True,
                "gpu_result": gpu_result,
                "spmd_result": spmd_result,
                "agg_result": None,
                "gpu_failed": gpu_failed,
                "resume_from": resume_from,
                "stop_after": stop_after,
            }

        # Phase 4: Aggregation (SLURM job or local based on --agg-local)
        if getattr(args, 'agg_local', False):
            print("\nRunning aggregation locally...")
            on_completion(cpu_jobs_to_submit, experiment)
            agg_result = {"success": True, "local": True}
        else:
            # Submit aggregation as SLURM job
            agg_mem = getattr(args, 'agg_mem', '256G')
            agg_result = submit_aggregation_slurm(
                experiment=experiment,
                output_dir=output_dir,
                mem=agg_mem,
                dry_run=args.dry_run,
                wait=not args.no_wait,
            )

        result = {
            "success": agg_result.get("success", False),
            "split_mode": True,
            "gpu_result": gpu_result,
            "spmd_result": spmd_result,
            "agg_result": agg_result,
            "gpu_failed": gpu_failed,
            "resume_from": resume_from,
        }
        return result

    # --- Combined mode (default): single job with GPU + CPU ---
    if getattr(args, 'escalate', False):
        print(f"\nEscalation mode enabled: {' -> '.join(PARTITION_ESCALATION)}")
        result = submit_with_escalation(
            jobs_to_submit=jobs_to_submit,
            experiment=experiment,
            slurm_params=slurm_params,
            log_dir=f"slurm_feature_extraction_logs/{experiment}",
            args=args,
            output_dir=output_dir,
            on_completion_callback=on_completion if not args.no_wait else None,
            print_resource_summary=preview_mode,
        )
    else:
        result = submit_parallel_jobs(
            jobs_to_submit=jobs_to_submit,
            experiment=experiment,
            slurm_params=slurm_params,
            log_dir=f"slurm_feature_extraction_logs/{experiment}",
            manifest_prefix="feature_extraction",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not getattr(args, 'quiet', False),
            post_completion_callback=on_completion if not args.no_wait else None,
            print_resource_summary=preview_mode,
        )

    return result


if __name__ == "__main__":
    from cyclops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for cell-chunk based feature extraction."
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        required=True,
        help="Experiment name or shorthand (e.g., '94', 'ops0094_20251217').",
    )
    parser.add_argument(
        "--wells",
        nargs="+",
        help="Process only cells from specific wells (e.g., A/1/0 A/2/0)",
    )
    parser.add_argument(
        "--cells-per-batch",
        type=int,
        default=DEFAULT_CELLS_PER_BATCH,
        help=f"Number of cells per SLURM batch (default: {DEFAULT_CELLS_PER_BATCH} for shorter jobs)",
    )
    parser.add_argument(
        "--full-features",
        action="store_true",
        help="Compute full features including expensive texture features",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and exit without waiting for completion",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of batches that already have results (deletes existing)",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Resume from checkpoint: skip completed batches, only process remaining",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip extraction, just aggregate existing parquet files into AnnData",
    )
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="Check if all batches are complete (compare existing parquets to expected from config)",
    )
    parser.add_argument(
        "--preview",
        nargs="?",
        const=5,
        type=int,
        metavar="N",
        help="Preview mode: only submit N batches from different wells (default: 5) to test pipeline end-to-end",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        choices=["cpu", "gpu", "preempted"],
        help="SLURM partition (default: gpu). Use --escalate for auto gpu->cpu->preempted",
    )
    parser.add_argument(
        "--escalate",
        action="store_true",
        help="Auto-escalate partitions if jobs pending >50s (gpu->cpu->preempted)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent SLURM jobs (default: 2). More jobs = more I/O contention = slower per-job",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process cells sequentially (1 CPU) instead of parallel. For benchmarking.",
    )
    parser.add_argument(
        "--spmd-ntasks",
        type=int,
        default=1024,
        help="Number of MPI ranks for the SPMD CPU phase (default: 1024)",
    )
    parser.add_argument(
        "--agg-mem",
        type=str,
        default="256G",
        help="Memory for aggregation SLURM job (default: 256G). Use higher values for large experiments.",
    )
    parser.add_argument(
        "--agg-local",
        action="store_true",
        help="Run aggregation locally instead of as SLURM job (default: submit to SLURM)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        choices=["gpu", "spmd", "merge", "aggregate"],
        default=None,
        help="Resume from a specific phase (split mode only). "
             "Runs the selected phase and all downstream phases. "
             "Choices: gpu (default start), spmd (skip GPU), merge (skip GPU+SPMD), aggregate (final only)",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        choices=["gpu", "spmd", "merge", "aggregate"],
        default=None,
        help="Stop after a specific phase finishes submitting (split mode only). "
             "Used by the multi-experiment wave runner to fan out phase-by-phase. "
             "Combined with --resume-from, e.g. --resume-from spmd --stop-after spmd "
             "submits only the SPMD CPU array.",
    )
    parser.add_argument(
        "--cells-csv",
        type=str,
        default=None,
        help="Optional per-experiment CSV restricting feature extraction to the listed cells. "
             "Expected columns: well (any format accepted by canonicalize_well_path), "
             "segmentation (int seg-id). Used by the top-cells pipeline; skipped when not provided.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Override the default output location. When set, outputs go to "
             "<output-base>/<experiment>/feature_extraction/ instead of "
             "<experiment>/3-assembly/feature_extraction/.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["phase", "fluorescent", "all"],
        default="all",
        help="Channel modality. 'phase' keeps only non-cp* organelles and joins the "
             "--cells-csv on segmentation_id; 'fluorescent' keeps only cp* organelles "
             "and joins on cp_cell_seg_id; 'all' (default) preserves current behavior.",
    )
    parser.add_argument(
        "--organelles",
        type=str,
        default=None,
        help="Comma-separated list of organelle names (as discovered from the zarr) "
             "to restrict processing to. Overrides the --modality preset when both are set.",
    )

    args = parser.parse_args()

    # Fail fast if GPU stack is missing before submitting any SLURM jobs.
    # Submit node has no GPU, so only check that cucim/cupy are importable.
    # Skip on paths that don't need GPU workers.
    _needs_gpu = not (
        getattr(args, 'check_complete', False)
        or getattr(args, 'aggregate_only', False)
        or getattr(args, 'dry_run', False)
        or getattr(args, 'partition', None) == 'cpu'
    )
    if _needs_gpu:
        try:
            import cupy  # noqa: F401
            import cucim.skimage.measure  # noqa: F401
        except ImportError as _e:
            raise SystemExit(
                f"ERROR: GPU feature extraction requires cucim and cupy in the venv ({type(_e).__name__}: {_e}).\n"
                f"  Fix: uv sync (from the repo root)\n"
                f"  (or: uv pip install 'cucim-cu12>=25.6')\n"
                f"  To run on CPU only, pass --partition cpu or set _FORCE_CPU_MODE=True in fe_workers.py."
            )

    # Resolve experiment name
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True, autoselect=True)
    print(f"\nFeature Extraction SLURM Submission")
    print(f"Experiment: {experiment}")
    print(f"{'='*60}\n")

    # Handle --check-complete mode
    if getattr(args, 'check_complete', False):
        ds = OpsDataset(experiment)
        output_dir = _resolve_fe_output_dir(experiment, args)
        batch_results_dir = output_dir / "_batch_results"
        batch_config_file = batch_results_dir / "_batch_config.json"
        
        if not batch_results_dir.exists():
            print(f"❌ No batch results directory found at {batch_results_dir}")
            sys.exit(1)
        
        if not batch_config_file.exists():
            print(f"⚠️  WARNING: No batch config file found at {batch_config_file}")
            print(f"   Using default cells_per_batch={DEFAULT_CELLS_PER_BATCH}")
            print(f"   Results may be inaccurate if original run used different settings.\n")
            cells_per_batch = DEFAULT_CELLS_PER_BATCH
            config_source = "defaults"
        else:
            import json
            with open(batch_config_file) as f:
                config = json.load(f)
            cells_per_batch = config.get("cells_per_batch", DEFAULT_CELLS_PER_BATCH)
            config_source = "config file"
        
        # Get existing parquet files
        existing_parquets = list(batch_results_dir.glob("batch_*_cells.parquet"))
        existing_batch_ids = {p.stem.replace("batch_", "").replace("_cells", "") for p in existing_parquets}
        
        # Calculate expected batches by loading cell metadata
        morphology_path = ds.store_paths["pheno_assembled_v3"]
        cells_df = _load_cells_metadata(ds, morphology_path)
        n_wells = len(cells_df["well"].unique())
        
        print(f"Batch Config ({config_source}):")
        print(f"  cells_per_batch: {cells_per_batch}")
        print(f"  total_cells: {len(cells_df):,}")
        print(f"  n_wells: {n_wells}")
        
        expected_batch_ids = set()
        for well in cells_df["well"].unique():
            well_safe = well.replace("/", "_")
            n_well_cells = len(cells_df[cells_df["well"] == well])
            n_batches = (n_well_cells + cells_per_batch - 1) // cells_per_batch
            for batch_idx in range(n_batches):
                expected_batch_ids.add(f"{well_safe}_{batch_idx:04d}")
        
        # Compare
        missing = expected_batch_ids - existing_batch_ids
        extra = existing_batch_ids - expected_batch_ids
        complete = existing_batch_ids & expected_batch_ids
        
        print(f"\nCompletion Status:")
        print(f"  Expected batches: {len(expected_batch_ids)}")
        print(f"  Complete batches: {len(complete)}")
        print(f"  Missing batches:  {len(missing)}")
        
        if missing:
            print(f"\n❌ INCOMPLETE - {len(missing)} batches missing:")
            for batch_id in sorted(missing)[:20]:
                print(f"     {batch_id}")
            if len(missing) > 20:
                print(f"     ... and {len(missing) - 20} more")
            print(f"\n   Use --checkpoint to process remaining batches")
            sys.exit(1)
        else:
            print(f"\n✅ COMPLETE - All {len(expected_batch_ids)} batches finished!")
            print(f"   Ready for aggregation: --aggregate-only")
            sys.exit(0)

    # Handle aggregate-only mode
    if args.aggregate_only:
        ds = OpsDataset(experiment)
        output_dir = _resolve_fe_output_dir(experiment, args)
        batch_results_dir = output_dir / "_batch_results"

        if not batch_results_dir.exists():
            print(f"ERROR: No batch results found at {batch_results_dir}")
            print("Run feature extraction first, or check the path.")
            sys.exit(1)

        parquet_files = list(batch_results_dir.glob("batch_*_cells.parquet"))
        if not parquet_files:
            print(f"ERROR: No parquet files found in {batch_results_dir}")
            sys.exit(1)

        print(f"Aggregate-only mode: Found {len(parquet_files)} batch parquet files")

        # Submit as SLURM job (default) or run locally (--agg-local)
        if getattr(args, 'agg_local', False):
            print("Running aggregation locally...")
            result = aggregate_batch_results([], experiment, output_dir)
            if result.get("status") == "success":
                print("\n✓ Aggregation completed successfully")
            else:
                print(f"\n✗ Aggregation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            # Submit to SLURM with configurable memory
            agg_mem = getattr(args, 'agg_mem', '256G')
            result = submit_aggregation_slurm(
                experiment=experiment,
                output_dir=output_dir,
                mem=agg_mem,
                dry_run=args.dry_run,
                wait=not args.no_wait,
            )
            if not result.get("success"):
                print(f"\n✗ Aggregation job failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
    else:
        # Submit jobs
        result = submit_feature_extraction_jobs(experiment, args)

        if result.get("dry_run"):
            # Dry run completed successfully - no error message needed
            pass
        elif result.get("success", False):
            print("\n✓ Feature extraction jobs submitted successfully")
        else:
            print(f"\n✗ Job submission failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
