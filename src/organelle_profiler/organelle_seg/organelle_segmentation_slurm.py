"""
SLURM batch submission for organelle segmentation.

Submits segmentation jobs for all position-channel combinations as parallel SLURM jobs.
Each job runs independently with dedicated resources to maximize throughput.

Dual Frangi Segmentation:
-------------------------
For each Frangi channel (Phase2D, Focus3D, GFP, mCherry, etc.), TWO segmentation jobs
are submitted - one for TUBULAR structures (alpha=4.0, ER/mitochondria networks) and
one for VESICULAR structures (alpha=0.5, lysosomes/lipid droplets). This creates two
output labels per channel, e.g., "phase_2d_tubular_seg" and "phase_2d_vesicular_seg".

Usage:
------
# List available channels for an experiment
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --list-channels

# force reprocess all experiments even if outputs exist
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment 31 --force

# Submit segmentation for all positions and channels
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626

# Submit segmentation for specific positions
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 A/2/0

# Submit segmentation for specific channels (use full channel key with structure type)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --channels Phase2D_tubular GFP_vesicular

# Submit segmentation for specific position-channel combinations
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 --channels nucleoli

# Preview what would be submitted (dry run)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0033_20250429 --dry-run

# Submit without waiting for completion
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --no-wait

# Debug mode with center crop (1% of image) - runs on SLURM
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --debug-crop 0.01

# Process ALL experiments that need segmentation (batch mode)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --all

# Force reprocess all experiments even if outputs exist
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --all --force

Preview Mode:
-------------
Preview mode runs segmentation LOCALLY (no SLURM) on a small 2x2 tile grid (1920x1920 pixels)
to test BOTH segmentation AND tile stitching. Debug images are saved showing raw input,
vesselness map, binary mask, labeled output, and overlay.

# preview all segmentation types for a position
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment 94 --positions A/1/0 --preview-all

# Preview tubular Frangi segmentation on Phase2D channel
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm -e 33 --positions A/1/0 --channels Phase2D --structure-type tubular --preview

# Preview vesicular Frangi segmentation (for round structures like lysosomes)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 --channels Phase2D --structure-type vesicular --preview

# Preview nucleoli segmentation from Phase2D (uses Phase channel + nuclear mask)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 --channels nucleoli_phase2d --preview

# Preview nucleoli segmentation from Focus3D (uses Focus3D channel + nuclear mask)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 --channels nucleoli_focus3d --preview

# Preview GFP fluorescent channel segmentation
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment ops0049_20250626 --positions A/1/0 --channels GFP --structure-type tubular --preview

Preview outputs are saved to: {experiment}/3-assembly/organelle_seg_debug/{position}_{channel}/
  - 01_raw_input.png: Raw input image (for comparison)
  - 06_vesselness.png: Frangi filter response (continuous values)
  - 07_binary_mask.png: Thresholded binary mask
  - 08_labeled_output.png: Connected component labels (colored)
  - 09_overlay.png: Labels overlaid on raw image

Sweep Mode:
-----------
Sweep mode runs parameter sweeps to explore how different values affect segmentation.
Useful for optimizing parameters like pixel_size_um, alpha, beta, threshold, etc.

# Sweep pixel_size_um from 0.1 to 0.5 with 5 samples on GFP channel
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm --experiment 124 --positions A/1/0 --channels gfp  --sweep --sweep-var pixel_size_um --sweep-range "0.1 0.175 5"

# Sweep threshold from 0.001 to 0.1 with logarithmic spacing (10 samples)
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0033_20250429 --positions A/1/0 --channels Phase2D \
    --structure-type tubular --sweep --sweep-var threshold --sweep-range "0.001 0.1 10" --sweep-log

# Sweep alpha (elongation sensitivity) from 0.1 to 1.0 with 5 samples
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0033_20250429 --positions A/1/0 --channels Phase2D \
    --structure-type vesicular --sweep --sweep-var alpha --sweep-range "0.1 1.0 5"

# Sweep CLAHE clip_limit from 0.005 to 0.05
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0033_20250429 --positions A/1/0 --channels GFP \
    --sweep --sweep-var clip_limit --sweep-range "0.005 0.05 5"

Sweep outputs are saved to: {experiment}/3-assembly/organelle_seg_debug/sweep/
  - sweep_{position}_{channel}_{sweep_var}.png: Combined canvas showing all sweep values
  - sweep_{position}_{channel}_{sweep_var}_region2.png: Second crop region (Y+500px offset)
  - sweep_params_{position}_{channel}_{sweep_var}.yaml: Parameters used for each sweep value

Repair Metadata:
-----------------
Update segmentation_metadata on existing label groups without re-running segmentation.
Reads channel labels from ops_channel_maps.yaml and rebuilds biological_annotation fields
(organelle, marker, marker_type, structure_type) that may be missing or incorrect.

# Repair metadata for all labels in an experiment
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm -e 94 --repair-metadata

# Repair metadata for specific positions only
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm -e 94 --positions A/1/0 A/2/0 --repair-metadata
"""

import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from organelle_profiler.organelle_seg.organelle_segmentation import (
    segment_single_position_channel,
    get_available_channels,
)
from cyclops_utils.hpc.slurm_batch_utils import (
    submit_parallel_jobs,
    detect_experiments_needing_processing,
)
from cyclops_utils.data.experiment import OpsDataset
from organelle_profiler.paths import BASE_PATH


def ensure_labels_groups_exist(experiment: str, positions: list[str], verbose: bool = True) -> None:
    """
    Pre-create labels groups for all positions before SLURM job submission.

    This prevents race conditions when multiple parallel jobs try to call
    require_group("labels") on the same position simultaneously. By creating
    the labels group upfront, each job only needs to create its own channel-specific
    subgroup, which won't conflict with other channels.

    Parameters
    ----------
    experiment : str
        Experiment name
    positions : list[str]
        List of positions (e.g., ["A/1/0", "A/2/0", "A/3/0"])
    verbose : bool
        Print progress messages
    """
    dataset = OpsDataset(experiment)
    v3_path = dataset.store_paths.get("pheno_assembled_v3")

    if not v3_path or not v3_path.exists():
        raise FileNotFoundError(f"pheno_assembled_v3 store not found for {experiment}")

    if verbose:
        print(f"Pre-creating labels groups for {len(positions)} positions...")

    import zarr
    store = zarr.open(str(v3_path), mode="r+")

    for pos in positions:
        if pos not in store:
            if verbose:
                print(f"  Warning: Position {pos} not found in store, skipping")
            continue

        pos_group = store[pos]
        # Use require_group to create if not exists, or get existing
        if "labels" not in pos_group:
            pos_group.create_group("labels")
            if verbose:
                print(f"  Created labels group for {pos}")
        elif verbose:
            print(f"  Labels group already exists for {pos}")

    if verbose:
        print("  Done pre-creating labels groups.\n")


def detect_experiments_needing_segmentation(
    positions: list[str] = None,
    channels: list[str] = None,
    force: bool = False,
    verbose: bool = True,
    include_prefixes: list[str] = None,
    exclude_prefixes: list[str] = None,
) -> tuple[list[tuple[str, int, int, dict]], list[tuple[str, int, int, dict]]]:
    """
    Scan the OPS store (``$OPS_BASE_PATH``) to find experiments that need organelle segmentation.

    Parameters
    ----------
    positions : list[str]
        Positions to check (default: all positions in store)
    channels : list[str]
        Channels to check (default: all segmentable channels)
    force : bool
        If True, include all experiments with valid inputs even if outputs exist
    verbose : bool
        Print progress during scan

    Returns
    -------
    tuple[list, list]
        (experiments_to_process, experiments_completed)
        Each list contains tuples of (experiment_name, n_completed, n_expected, extra_data)
    """
    # Define input checker - experiment needs pheno_assembled_v3 store
    def check_segmentation_input(dataset):
        """Check if experiment has v3 phenotyping store."""
        try:
            v3_path = dataset.store_paths.get("pheno_assembled_v3")
            return v3_path is not None and v3_path.exists()
        except (KeyError, AttributeError):
            return False

    # Define output checker - check for segmentation labels
    def get_segmentation_outputs(dataset, wells_list):
        """Get expected segmentation output paths."""
        outputs = []
        try:
            v3_path = dataset.store_paths.get("pheno_assembled_v3")
            if not v3_path or not v3_path.exists():
                return outputs

            # Get available channels and positions. With --force we need ALL channels
            # (not just ones missing outputs) so completed experiments still get queued.
            info = get_available_channels(dataset.experiment, skip_existing=not force)
            pos_list = positions if positions else info["positions"]
            ch_list = channels if channels else info["channels"]

            # For each position-channel combination, check if output label exists
            import zarr
            store = zarr.open(str(v3_path), mode="r")

            for pos in pos_list:
                if pos not in store:
                    continue
                labels_group = store[pos].get("labels", {})

                for ch in ch_list:
                    # Determine expected output label name
                    # This is simplified - actual label names depend on channel type
                    expected_labels = []
                    ch_lower = ch.lower()
                    if "nuclei" in ch_lower and "prediction" in ch_lower:
                        expected_labels.append("nucle_vs_seg")
                    elif "membrane" in ch_lower and "prediction" in ch_lower:
                        expected_labels.append("membr_vs_seg")
                    elif ch == "nucleoli_phase2d":
                        expected_labels.append("nuclo_phase_seg")
                    elif ch == "nucleoli_focus3d":
                        expected_labels.append("nuclo_focus_seg")
                    elif ch == "nucleoli":
                        # Legacy format -> Phase2D
                        expected_labels.append("nuclo_phase_seg")
                    else:
                        # Frangi channels - use channel name prefix
                        ch_prefix = ch[:5].lower()
                        expected_labels.append(f"{ch_prefix}_seg")

                    for label_name in expected_labels:
                        # Check if label exists in this position
                        label_path = v3_path / pos / "labels" / label_name
                        outputs.append(label_path)

        except Exception as e:
            if verbose:
                print(f"  Error checking outputs: {e}")

        return outputs

    # Filter out bad/excluded experiments (date cutoff, do-not-run, iss-only, etc.)
    # Custom-library experiments are NOT excluded — segmentation doesn't depend
    # on the codebook, and they are valid targets.
    try:
        from cyclops_utils.data.bad_experiments import is_excluded as _is_bad_exp
    except ImportError:
        _is_bad_exp = lambda name: False  # noqa: E731

    _include = [p for p in (include_prefixes or []) if p]
    _exclude = [p for p in (exclude_prefixes or []) if p]

    def _experiment_filter(name: str) -> bool:
        # Cheapest checks first — bail before touching bad_experiments/zarr metadata
        if _include and not any(name.startswith(p) for p in _include):
            return False
        if _exclude and any(name.startswith(p) for p in _exclude):
            return False
        try:
            return not _is_bad_exp(name)
        except Exception:
            return True  # on error, let it through

    # Use shared detection utility
    return detect_experiments_needing_processing(
        input_checker=check_segmentation_input,
        output_checker=get_segmentation_outputs,
        wells=[1, 2, 3],  # Positions are A/1/0, A/2/0, A/3/0
        force=force,
        verbose=verbose,
        experiment_filter=_experiment_filter,
    )


def submit_organelle_segmentation_jobs(
    experiment: str,
    positions: list[str] = None,
    channels: list[str] = None,
    slurm_params: dict = None,
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    debug_crop_fraction: float = None,
    force: bool = False,
) -> dict:
    """
    Submit parallel SLURM jobs for organelle segmentation.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., "ops0049_20250626")
    positions : list[str]
        Positions to process (default: all positions in store)
    channels : list[str]
        Channels to process (default: all segmentable channels)
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, etc.)
        If None, uses defaults
    dry_run : bool
        If True, print what would be submitted without actually submitting
    wait_for_completion : bool
        If True, wait for all jobs to complete before returning (default: True)
    verbose : bool
        Print detailed progress
    debug_crop_fraction : float
        If set, only process a center crop (e.g., 0.01 for 1% area)
    force : bool
        If True, include channels that already have segmentation data (default: False)

    Returns
    -------
    dict
        Job submission results with job IDs, metadata, and completion status
    """
    # Get available channels and positions
    # When force=True, include channels that already have data (skip_existing=False)
    try:
        info = get_available_channels(experiment, skip_existing=not force)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}

    # Use provided lists or defaults
    pos_list = positions if positions else info["positions"]
    ch_list = channels if channels else info["channels"]

    # Validate positions
    invalid_positions = [p for p in pos_list if p not in info["positions"]]
    if invalid_positions:
        print(f"Warning: Invalid positions ignored: {invalid_positions}")
        pos_list = [p for p in pos_list if p in info["positions"]]

    # Validate channels - check against ALL channels (not just available ones)
    # User may specify channels that already have data
    all_info = get_available_channels(experiment, skip_existing=False)
    invalid_channels = [c for c in ch_list if c not in all_info["channels"]]
    if invalid_channels:
        print(f"Warning: Invalid channels ignored: {invalid_channels}")
        ch_list = [c for c in ch_list if c in all_info["channels"]]

    if not pos_list:
        print("Error: No valid positions to process")
        return {"success": False, "error": "No valid positions"}

    # If no channels to process after skip_existing filtering, prompt user
    if not ch_list:
        skipped = info.get("skipped_channels", [])
        if skipped:
            print(f"\n{'='*60}")
            print(f"All segmentations already exist for {experiment}")
            print(f"{'='*60}")
            print(f"\nChannels with existing segmentation data:")
            for ch in skipped:
                print(f"  ✓ {ch}")
            print()

            # Prompt for overwrite
            try:
                response = input("Would you like to overwrite existing segmentations? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    # Re-fetch without skipping existing
                    info = get_available_channels(experiment, skip_existing=False)
                    ch_list = info["channels"]
                    print(f"\nProceeding to overwrite {len(ch_list)} channel(s)...")
                else:
                    print("\nNo channels to process. Exiting.")
                    return {"success": True, "skipped": True, "message": "All segmentations already exist"}
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user.")
                return {"success": False, "error": "Cancelled by user"}
        else:
            print("Error: No valid channels to process")
            return {"success": False, "error": "No valid channels"}

    # Use the caller-supplied slurm_params. main() builds these from CLI flags
    # (default: gpu partition, 1 GPU, 256GB, 8 CPUs, [a100|h100|h200] constraint).
    # Fallback only if invoked programmatically without slurm_params.
    if slurm_params is None:
        slurm_params = {
            "timeout_min": 180,
            "mem": "256GB",
            "cpus_per_task": 8,
            "gpus_per_node": 1,
            "slurm_partition": "gpu",
            "slurm_constraint": "[a100|h100|h200]",
        }

    # Prepare job list - one job per position-channel combination
    # For Frangi channels, the channel key includes structure_type (e.g., "Phase2D_tubular")
    # We need to parse this to get the base channel name and structure_type
    jobs_to_submit = []

    for pos in pos_list:
        for ch in ch_list:
            # Create sanitized job name
            pos_safe = pos.replace("/", "_")
            ch_safe = ch.replace("/", "_")
            job_name = f"seg_{pos_safe}_{ch_safe}"

            # Use all_info for method lookup since info may not have methods for skipped channels
            method = all_info["channel_methods"].get(ch, "frangi")

            # Parse structure_type from channel key for Frangi channels
            # Format: "Phase2D_tubular" or "GFP_vesicular"
            structure_type = all_info.get("channel_structure_types", {}).get(ch, None)

            # Extract base channel name (remove _tubular or _vesicular suffix)
            if structure_type and ch.endswith(f"_{structure_type}"):
                base_channel = ch[:-len(f"_{structure_type}")]
            else:
                base_channel = ch

            jobs_to_submit.append({
                "name": job_name,
                "func": segment_single_position_channel,
                "kwargs": {
                    "experiment": experiment,
                    "position": pos,
                    "channel_key": base_channel,
                    "structure_type": structure_type,  # Pass structure_type for dual Frangi
                    "debug_crop_fraction": debug_crop_fraction,
                },
                "metadata": {
                    "experiment": experiment,
                    "position": pos,
                    "channel": ch,  # Keep full channel key for display
                    "base_channel": base_channel,
                    "structure_type": structure_type,
                    "method": method,
                },
                "slurm_params": slurm_params,  # All jobs use same params
            })

    if not jobs_to_submit:
        print("No jobs to submit!")
        return {"success": False, "error": "No jobs to submit"}

    # Print summary
    print(f"\n{'='*60}")
    print(f"Organelle Segmentation Batch Submission")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Positions: {pos_list}")
    print(f"Channels: {ch_list}")
    print(f"Total jobs: {len(jobs_to_submit)} (positions x channels)")
    print(f"  Resources: {slurm_params['mem']}, {slurm_params['cpus_per_task']} CPUs, "
          f"{slurm_params.get('gpus_per_node', 0)} GPUs, partition={slurm_params['slurm_partition']}")
    if debug_crop_fraction:
        print(f"Debug mode: {debug_crop_fraction*100:.1f}% center crop")
    print(f"{'='*60}\n")

    # Pre-create labels groups to avoid race conditions when parallel jobs
    # all try to require_group("labels") simultaneously on the same position
    if not dry_run:
        ensure_labels_groups_exist(experiment, pos_list, verbose=verbose)

    # Submit all jobs as a single job array (all use same SLURM resources)
    print(f"Submitting {len(jobs_to_submit)} segmentation jobs ({slurm_params['slurm_partition']})...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=f"{experiment}_segmentation",
        slurm_params=slurm_params,
        log_dir=f"slurm_segmentation_logs/{experiment}",
        manifest_prefix="organelle_seg",
        dry_run=dry_run,
        wait_for_completion=False,  # Don't wait yet
        verbose=verbose,
        post_completion_callback=None,
    )

    # If user wants to wait, wait for job array completion
    if wait_for_completion and not dry_run:
        from cyclops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

        # Build job_arrays list for wait_for_multiple_job_arrays
        job_arrays = []
        if result.get("success") and "submitted_jobs" in result:
            job_arrays.append({
                "submitted_jobs": result["submitted_jobs"],
                "base_job_id": result["base_job_id"],
                "label": f"Segmentation ({result['base_job_id']})",
                "slurm_params": slurm_params,
            })

        if job_arrays:
            wait_result = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment=experiment,
                verbose=verbose,
            )

            return {
                "success": True,
                "result": result,
                "completed": wait_result.get("completed", []),
                "failed": wait_result.get("failed", []),
                "all_completed": len(wait_result.get("failed", [])) == 0,
            }

    # Return result
    return {
        "success": result.get("success", False),
        "result": result,
        "dry_run": dry_run,
    }


def run_preview_all_for_experiment(
    resolved_name: str,
    positions: list,
    output_dir: Path | None = None,
    fluorescent_only: bool = False,
    filename_prefix: str | None = None,
) -> dict:
    """Run the full --preview-all flow for one experiment.

    Parameters
    ----------
    resolved_name : str
        Fully-resolved experiment name (e.g. 'ops0094_20251217').
    positions : list[str]
        Positions to preview (typically ['A/1/0']).
    output_dir : Path, optional
        Directory for preview canvases. When None uses
        <assembly_dir>/organelle_seg_debug/ (the existing default).
    fluorescent_only : bool
        When True, skip Phase2D / Focus3D / nucleoli_phase2d / nucleoli_focus3d
        configs so only fluorescent + CP + 4i channels are previewed.
    filename_prefix : str, optional
        Prefix for output files (e.g. the experiment name) — useful when multiple
        experiments share one output directory.

    Returns
    -------
    dict
        {'experiment', 'canvas_dir', 'total', 'successful', 'results': [...]}
    """
    from cyclops_utils.data.filesystem import resolve_experiment_name  # noqa: F401
    from organelle_profiler.organelle_seg.organelle_segmentation import (
        segment_single_position_channel,
    )
    from organelle_profiler.organelle_seg.configs import (
        load_experiment_configs,
        load_channel_labels,
    )
    from organelle_profiler.organelle_seg.visualizations import (
        create_combined_canvas,
        save_segmentation_params_yaml,
    )

    exp_channel_configs = load_experiment_configs(resolved_name)
    try:
        exp_channel_labels = load_channel_labels(resolved_name)
    except Exception:
        exp_channel_labels = {}

    def _display_name(ch, st):
        base = f"{ch}_{st}" if st else ch
        lab = exp_channel_labels.get(ch)
        return f"{base}\n{lab}" if lab else base

    preview_configs = []
    if not fluorescent_only:
        preview_configs.extend([
            ("Phase2D", "tubular", None),
            ("Phase2D", "vesicular", None),
            ("Phase2D", "vesicular_dark", None),
            ("Focus3D", "tubular", None),
            ("Focus3D", "vesicular", None),
            ("Focus3D", "vesicular_dark", None),
            ("nucleoli_phase2d", None, None),
            ("nucleoli_focus3d", None, None),
        ])

    for ch_name, channel_config in exp_channel_configs.items():
        if ch_name in ["BF", "Phase", "Phase2D", "Focus3D"]:
            continue
        config_structure_type = channel_config.get("structure_type", "tubular") if isinstance(channel_config, dict) else "tubular"
        config_method = channel_config.get("method") if isinstance(channel_config, dict) else None
        preview_configs.append((ch_name, config_structure_type, config_method))

    print(f"\n{'='*60}")
    print(f"PREVIEW MODE (local) — {resolved_name}")
    print(f"  fluorescent_only={fluorescent_only}")
    print(f"  positions={positions}")
    print(f"  configs ({len(preview_configs)}):")
    for ch, st, method in preview_configs:
        print(f"    - {ch} (structure={st or 'auto'}{', method=' + method if method else ''})")
    print(f"{'='*60}\n")

    all_results = []
    for pos in positions:
        for ch, structure_type, method in preview_configs:
            st_str = f" ({structure_type})" if structure_type else ""
            method_str = f", method={method}" if method else ""
            print(f"\n--- [{resolved_name}] {pos} / {ch}{st_str}{method_str} ---")
            try:
                result = segment_single_position_channel(
                    experiment=resolved_name,
                    position=pos,
                    channel_key=ch,
                    structure_type=structure_type,
                    debug_only=True,
                )
                all_results.append((pos, ch, structure_type, method, result))
                if result.get("success"):
                    print(f"  ✓ {ch}{st_str}: {result.get('num_objects', 0)} objects")
                else:
                    print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append((pos, ch, structure_type, method, {"success": False, "error": str(e)}))

    # Resolve canvas output dir
    if output_dir is None:
        dataset = OpsDataset(resolved_name)
        source_path = dataset.store_paths["pheno_assembled_v3"]
        canvas_dir = source_path.parent / "organelle_seg_debug"
    else:
        canvas_dir = Path(output_dir)
    canvas_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{filename_prefix}_" if filename_prefix else ""

    for pos in positions:
        pos_results = [
            (ch, st, r) for p, ch, st, method, r in all_results
            if p == pos and r.get("success") and r.get("labels") is not None
        ]
        if not pos_results:
            print(f"  No successful results for {pos}; skipping canvas")
            continue

        channel_params = {}
        for ch, st, r in pos_results:
            name = _display_name(ch, st)
            channel_params[name] = {
                'frangi_params': r.get('frangi_params', {}),
                'clahe_params': r.get('clahe_params', {}),
                'structure_type': r.get('structure_type'),
            }

        yaml_path = canvas_dir / f"{prefix}segmentation_params_{pos.replace('/', '_')}.yaml"
        save_segmentation_params_yaml(
            channel_params=channel_params,
            output_path=yaml_path,
            experiment=resolved_name,
            position=pos,
            preview_mode=True,
        )

        canvas_crop_size = 512
        crop_configs = [
            {"name": "region1", "x_offset": -250, "y_offset": 0,   "suffix": ""},
            {"name": "region2", "x_offset": -250, "y_offset": 500, "suffix": "_region2"},
            {"name": "region3", "x_offset":  250, "y_offset": 0,   "suffix": "_region3"},
            {"name": "region4", "x_offset":  250, "y_offset": 500, "suffix": "_region4"},
        ]
        for crop_cfg in crop_configs:
            y_extra_offset = crop_cfg["y_offset"]
            x_offset = crop_cfg["x_offset"]
            suffix = crop_cfg["suffix"]

            images = []
            for ch, st, r in pos_results:
                name = _display_name(ch, st)
                labels = r.get('labels')
                raw = r.get('raw')
                vesselness = r.get('vesselness')
                if labels is not None:
                    h, w = labels.shape[:2]
                    cy = h // 2 + y_extra_offset
                    cx = w // 2 + x_offset
                    half = canvas_crop_size // 2
                    y1, y2 = max(0, cy - half), min(h, cy + half)
                    x1, x2 = max(0, cx - half), min(w, cx + half)
                    labels = labels[y1:y2, x1:x2]
                    if raw is not None:
                        raw = raw[y1:y2, x1:x2]
                    if vesselness is not None:
                        vesselness = vesselness[y1:y2, x1:x2]
                images.append({'name': name, 'labels': labels, 'raw': raw, 'vesselness': vesselness})

            canvas_path = canvas_dir / f"{prefix}preview_{pos.replace('/', '_')}_combined{suffix}.png"
            region_label = f" (Y+{y_extra_offset}px)" if y_extra_offset else ""
            create_combined_canvas(
                images=images,
                labels=[],
                output_path=canvas_path,
                title=f"Preview {resolved_name} {pos}{region_label}",
                channel_params=channel_params,
            )

    successes = sum(1 for _, _, _, _, r in all_results if r.get("success"))
    print(f"\n[{resolved_name}] preview summary: {successes}/{len(all_results)} configs successful; canvases -> {canvas_dir}")

    return {
        "experiment": resolved_name,
        "canvas_dir": str(canvas_dir),
        "total": len(all_results),
        "successful": successes,
        "results": all_results,
    }


def run_compare_preview_for_experiment(
    resolved_name: str,
    new_params_path: str,
    positions: list,
    output_dir: Path | None = None,
    filename_prefix: str | None = None,
) -> dict:
    """Preview OLD (production) vs NEW (overlay) segmentation, side by side.

    Non-destructive: like --preview-all this only renders debug canvases and
    never writes zarr labels. For each channel whose resolved config DIFFERS
    between the production org_seg_params.yaml and ``new_params_path`` (i.e. the
    organelles you are considering swapping), it runs the channel twice — once
    with each params file — and lays the two results adjacent on one canvas so
    you can judge old vs new before promoting anything.

    Channels not present in the overlay (or unchanged by it) resolve identically
    and are skipped — keeping the comparison focused on the candidates.

    Parameters
    ----------
    resolved_name : str
        Fully-resolved experiment name.
    new_params_path : str
        Path to the overlay org_seg_params.yaml (same schema) holding the NEW
        per-organelle options (e.g. method: threshold + threshold block).
    positions : list[str]
        Positions to preview (typically ['A/1/0']).
    output_dir : Path, optional
        Base debug dir. Canvases are written to ``<output_dir>/compare/``. When
        None uses ``<assembly_dir>/organelle_seg_debug/compare/``.
    filename_prefix : str, optional
        Prefix for output files (useful when pooling experiments).

    Returns
    -------
    dict
        {'experiment', 'canvas_dir', 'compared_channels', 'results'}
    """
    from organelle_profiler.organelle_seg.organelle_segmentation import (
        segment_single_position_channel,
    )
    from organelle_profiler.organelle_seg.configs import (
        load_experiment_configs,
        load_channel_labels,
    )
    from organelle_profiler.organelle_seg.visualizations import (
        create_combined_canvas,
    )

    old_cfgs = load_experiment_configs(resolved_name)
    new_cfgs = load_experiment_configs(resolved_name, marker_params_path=new_params_path)
    try:
        labels_map = load_channel_labels(resolved_name)
    except Exception:
        labels_map = {}

    # Channels the overlay actually changes (the swap candidates).
    compare_channels = []
    for ch, new_cfg in new_cfgs.items():
        if ch in ("BF", "Phase", "Phase2D", "Focus3D"):
            continue
        if new_cfg and new_cfg != old_cfgs.get(ch, {}):
            structure_type = (new_cfg.get("structure_type")
                              or old_cfgs.get(ch, {}).get("structure_type")
                              or "tubular")
            compare_channels.append((ch, structure_type))

    if not compare_channels:
        print(f"[{resolved_name}] no channels differ between production params and "
              f"{new_params_path}; nothing to compare.")
        return {"experiment": resolved_name, "canvas_dir": None,
                "compared_channels": [], "results": []}

    print(f"\n{'='*60}")
    print(f"COMPARE PREVIEW (old vs new) — {resolved_name}")
    print(f"  overlay params: {new_params_path}")
    print(f"  positions={positions}")
    print(f"  swap candidates ({len(compare_channels)}):")
    for ch, st in compare_channels:
        old_m = old_cfgs.get(ch, {}).get("method") or "default"
        new_m = new_cfgs.get(ch, {}).get("method") or "default"
        print(f"    - {ch} ({st}): {old_m} -> {new_m}")
    print(f"{'='*60}\n")

    if output_dir is None:
        dataset = OpsDataset(resolved_name)
        source_path = dataset.store_paths["pheno_assembled_v3"]
        canvas_dir = source_path.parent / "organelle_seg_debug" / "compare"
    else:
        canvas_dir = Path(output_dir) / "compare"
    canvas_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    all_results = []
    for pos in positions:
        # Run OLD + NEW for each candidate; keep them paired/adjacent.
        paired = []  # list of (display_name, old_result, new_result)
        for ch, structure_type in compare_channels:
            label = labels_map.get(ch, "")
            print(f"\n--- [{resolved_name}] {pos} / {ch} ({structure_type}) OLD vs NEW ---")
            try:
                old_r = segment_single_position_channel(
                    experiment=resolved_name, position=pos, channel_key=ch,
                    structure_type=structure_type, debug_only=True,
                )
            except Exception as e:
                old_r = {"success": False, "error": str(e)}
            try:
                new_r = segment_single_position_channel(
                    experiment=resolved_name, position=pos, channel_key=ch,
                    structure_type=structure_type, debug_only=True,
                    marker_params_path=new_params_path,
                )
            except Exception as e:
                new_r = {"success": False, "error": str(e)}
            paired.append((f"{ch}\n{label}" if label else ch, old_r, new_r))
            all_results.append((pos, ch, old_r, new_r))

        # Build canvas: OLD panel immediately followed by its NEW panel.
        canvas_crop_size = 512
        crop_configs = [
            {"x_offset": -250, "y_offset": 0,   "suffix": ""},
            {"x_offset": -250, "y_offset": 500, "suffix": "_region2"},
        ]
        for crop_cfg in crop_configs:
            images = []
            for name, old_r, new_r in paired:
                for tag, r in (("OLD", old_r), ("NEW", new_r)):
                    labels = r.get("labels") if r.get("success") else None
                    raw = r.get("raw") if r.get("success") else None
                    vesselness = r.get("vesselness") if r.get("success") else None
                    if labels is not None:
                        h, w = labels.shape[:2]
                        cy = h // 2 + crop_cfg["y_offset"]
                        cx = w // 2 + crop_cfg["x_offset"]
                        half = canvas_crop_size // 2
                        y1, y2 = max(0, cy - half), min(h, cy + half)
                        x1, x2 = max(0, cx - half), min(w, cx + half)
                        labels = labels[y1:y2, x1:x2]
                        if raw is not None:
                            raw = raw[y1:y2, x1:x2]
                        if vesselness is not None:
                            vesselness = vesselness[y1:y2, x1:x2]
                    images.append({"name": f"{name} [{tag}]", "labels": labels,
                                   "raw": raw, "vesselness": vesselness})

            canvas_path = canvas_dir / f"{prefix}compare_{pos.replace('/', '_')}{crop_cfg['suffix']}.png"
            create_combined_canvas(
                images=images, labels=[], output_path=canvas_path,
                title=f"OLD vs NEW {resolved_name} {pos}",
                channel_params={},
            )
            print(f"  canvas -> {canvas_path}")

    print(f"\n[{resolved_name}] compare summary: {len(compare_channels)} candidate channels; "
          f"canvases -> {canvas_dir}")
    return {
        "experiment": resolved_name,
        "canvas_dir": str(canvas_dir),
        "compared_channels": [c for c, _ in compare_channels],
        "results": all_results,
    }


def main():
    """CLI entry point for SLURM batch organelle segmentation submission."""
    parser = argparse.ArgumentParser(
        description="Submit organelle segmentation jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Experiment name (e.g., ops0049_20250626). Required unless --all is used.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all experiments that need segmentation (batch submission)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-run even if outputs exist (use with --all)",
    )

    parser.add_argument(
        "--exclude-experiments",
        nargs="+",
        default=[],
        help="Experiments to exclude from --all (e.g., ops0094 ops0116). "
             "Matches any experiment whose name starts with any of the given tokens.",
    )

    parser.add_argument(
        "--include-experiments",
        nargs="+",
        default=[],
        help="Explicit allowlist for --all (prefix match). When set, ONLY these experiments "
             "are processed. Example: --include-experiments ops0069 ops0070",
    )

    parser.add_argument(
        "--jobs-file",
        type=str,
        default=None,
        help="Path to a CSV with explicit (experiment, position, channel) rows to (re)submit "
             "as a single batch. Bypasses --all auto-discovery. Channel may include the "
             "_tubular/_vesicular/_vesicular_dark suffix; structure_type is inferred. "
             "Example row: ops0069_20250902,A/1/0,GFP",
    )


    parser.add_argument(
        "--positions",
        "-p",
        type=str,
        nargs="+",
        default=None,
        help="Positions to process (e.g., A/1/0 A/2/0). Default: all positions",
    )

    parser.add_argument(
        "--channels",
        "-c",
        type=str,
        nargs="+",
        default=None,
        help="Channels to process (e.g., nuclei_prediction GFP). Default: all channels",
    )

    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="List available positions and channels for the experiment, then exit",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="SLURM timeout in minutes (default: 180 = 3 hours)",
    )

    parser.add_argument(
        "--mem",
        type=str,
        default="256GB",
        help="SLURM memory allocation (default: 256GB with tiled CLAHE)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="SLURM CPUs per task (default: 8 - work is GPU-bound)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="SLURM GPUs per node (default: 1)",
    )

    parser.add_argument(
        "--gpu-constraint",
        type=str,
        default="[a100|h100|h200]",
        help="SLURM GPU constraint (default: [a100|h100|h200] for high-VRAM GPUs). Use 'none' to disable.",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)",
    )

    parser.add_argument(
        "--debug-crop",
        type=float,
        default=None,
        help="Debug mode: process only center crop (e.g., 0.01 for 1%% area)",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: run segmentation on small center crop and save debug images only. "
             "No data is written to zarr. Requires --experiment, --positions, and --channels.",
    )

    parser.add_argument(
        "--preview-all",
        action="store_true",
        help="Preview ALL segmentation types for the given position. Runs: Phase2D tubular/vesicular, "
             "nucleoli, mCherry tubular/vesicular, Focus3D tubular/vesicular. "
             "Requires --experiment and --positions.",
    )

    parser.add_argument(
        "--preview-all-experiments",
        action="store_true",
        help="Run --preview-all across EVERY experiment in ops_channel_maps.yaml and "
             "collect all canvases in a single shared --preview-output-dir. Pairs well "
             "with --fluorescent-only. Default position is A/1/0.",
    )

    parser.add_argument(
        "--fluorescent-only",
        action="store_true",
        help="In --preview-all and --preview-all-experiments, skip label-free channels "
             "(Phase2D/Focus3D/nucleoli_*) and preview only fluorescent + CP + 4i channels.",
    )

    parser.add_argument(
        "--preview-compare",
        action="store_true",
        help="Preview OLD (production org_seg_params.yaml) vs NEW (--new-params overlay) "
             "segmentation side-by-side, for only the channels the overlay changes. "
             "Non-destructive (debug canvases only). Requires --experiment, --positions, --new-params.",
    )

    parser.add_argument(
        "--new-params",
        type=str,
        default=None,
        help="Path to an overlay org_seg_params.yaml (same schema) holding the NEW per-organelle "
             "options to compare against production. Used with --preview-compare.",
    )

    parser.add_argument(
        "--preview-output-dir",
        type=str,
        default=None,
        help="Directory to write preview canvases into. When set, all canvases land in this "
             "single directory (prefixed with the experiment name) instead of being scattered "
             "under each experiment's 3-assembly/organelle_seg_debug/ folder.",
    )

    parser.add_argument(
        "--preview-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes when running --preview-all-experiments "
             "(default: 1 = serial). Each worker handles one experiment at a time. Run on a "
             "GPU salloc with enough resources before setting this above 1.",
    )

    # Sweep mode arguments
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Parameter sweep mode: run segmentation with one parameter varied across a range. "
             "Use with --sweep-var and --sweep-range to specify which parameter to sweep. "
             "Requires --experiment, --positions, and --channels (single channel).",
    )

    parser.add_argument(
        "--sweep-var",
        type=str,
        default="pixel_size_um",
        choices=[
            # Frangi filter parameters
            "pixel_size_um", "alpha", "beta", "threshold",
            "min_radius_um", "max_radius_um", "min_object_size",
            # CLAHE parameter
            "clip_limit",
            # Blob detection parameters (nucleoli, vesicular_dark)
            "overlap",  # Max blob overlap (0-1), lower = fewer merged blobs
            "num_sigma",  # Number of sigma levels for blob_log scale space
            "threshold_mult",  # Multiplier for dynamic thresholding
        ],
        help="Parameter to sweep (default: pixel_size_um). "
             "Frangi: pixel_size_um, alpha, beta, threshold, min_radius_um, max_radius_um, min_object_size. "
             "CLAHE: clip_limit. "
             "Blob: overlap, num_sigma, threshold_mult.",
    )

    parser.add_argument(
        "--sweep-range",
        type=str,
        default="0.1 0.5 5",
        help="Range for sweep as 'min max n_samples' (default: '0.1 0.5 5'). "
             "Values are linearly spaced between min and max. "
             "Example: '0.1 1.0 10' sweeps from 0.1 to 1.0 with 10 samples.",
    )

    parser.add_argument(
        "--sweep-log",
        action="store_true",
        help="Use logarithmic spacing for sweep values instead of linear (useful for pixel_size_um, threshold).",
    )

    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Disable tile stitching in preview/sweep modes. Processes the entire crop as a single tile "
             "(faster but uses more memory). Default uses 2x2 tile grid with overlap correction.",
    )

    parser.add_argument(
        "--preview-size",
        type=int,
        default=2048,
        help="[DEPRECATED] Preview mode now uses fixed 2x2 tile grid (1920x1920 pixels). This argument is ignored.",
    )

    parser.add_argument(
        "--structure-type",
        type=str,
        choices=["tubular", "vesicular"],
        default=None,
        help="Structure type for Frangi segmentation (tubular or vesicular). "
             "If not specified, runs both for dual segmentation.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and submit immediately (use with --all)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce verbosity (suppress job output)",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and return immediately without waiting for completion",
    )

    parser.add_argument(
        "--repair-metadata",
        action="store_true",
        help="Repair/update segmentation_metadata on existing label groups using channel labels "
             "from ops_channel_maps.yaml. Does NOT re-run segmentation — only updates zarr group "
             "attributes. Useful after fixing channel map labels or upgrading metadata schema.",
    )

    args = parser.parse_args()

    # Validation
    if (not args.all and not args.experiment
            and not getattr(args, "preview_all_experiments", False)
            and not getattr(args, "jobs_file", None)):
        parser.error("--experiment is required unless --all, --preview-all-experiments, or --jobs-file is used")

    # Handle --list-channels mode
    if args.list_channels:
        if not args.experiment:
            parser.error("--experiment is required with --list-channels")

        from cyclops_utils.data.filesystem import resolve_experiment_name
        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        try:
            info = get_available_channels(resolved_name)
            print(f"\nExperiment: {resolved_name}")
            print(f"\nPositions ({len(info['positions'])}):")
            for pos in info['positions']:
                print(f"  {pos}")
            print(f"\nAvailable channels ({len(info['channels'])}):")
            for ch in info['channels']:
                method = info['channel_methods'].get(ch, "unknown")
                structure_type = info.get('channel_structure_types', {}).get(ch, None)
                if structure_type:
                    print(f"  {ch} ({method}, {structure_type})")
                else:
                    print(f"  {ch} ({method})")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(0)

    # Handle --repair-metadata mode
    if args.repair_metadata:
        if not args.experiment:
            parser.error("--experiment is required with --repair-metadata")

        from cyclops_utils.data.filesystem import resolve_experiment_name
        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        from organelle_profiler.organelle_seg.metadata import (
            detect_segmentation_status,
            _build_segmentation_metadata,
            get_channel_index,
            _determine_processing_params,
        )
        from organelle_profiler.organelle_seg.naming import get_output_label_name
        from cyclops_utils.io.zarr_labels import _update_labels_metadata
        from organelle_profiler.organelle_seg.channel_processor import (
            build_channel_processing_map,
            resolve_single_channel_info,
        )
        from organelle_profiler.organelle_seg.configs import (
            load_experiment_configs,
            get_channel_type,
            STRUCTURE_TYPES,
        )
        from iohub import open_ome_zarr

        dataset = OpsDataset(resolved_name)
        source_path = dataset.store_paths["pheno_assembled_v3"]

        if not source_path.exists():
            print(f"Error: v3 phenotyping store not found at {source_path}")
            sys.exit(1)

        # Get positions and channel names
        with open_ome_zarr(source_path, mode="r") as store:
            channel_names = store.channel_names
            position_list = [p for p, _ in store.positions()]

        # Filter positions if specified
        if args.positions:
            position_list = [p for p in position_list if p in args.positions]

        # Detect existing labels
        status = detect_segmentation_status(resolved_name)
        labels_with_data = status["labels_with_data"]
        channel_to_label = status["channel_to_label"]

        # Load experiment configs for structure_type
        exp_configs = load_experiment_configs(resolved_name)

        print(f"\n{'='*60}")
        print(f"REPAIR METADATA - {resolved_name}")
        print(f"{'='*60}")
        print(f"Positions: {len(position_list)}")
        print(f"Labels with data: {len(labels_with_data)}")
        print(f"Store: {source_path}\n")

        # For each channel_key -> label_name mapping, rebuild metadata
        updated = 0
        skipped = 0
        for channel_key, label_name in sorted(channel_to_label.items()):
            # Only update labels that actually have data
            if label_name not in labels_with_data:
                continue

            # Resolve channel info (includes channel_label from YAML)
            # Parse structure_type from channel_key
            structure_type = None
            base_channel = channel_key
            for st in STRUCTURE_TYPES:
                if channel_key.endswith(f"_{st}"):
                    structure_type = st
                    base_channel = channel_key[:-len(f"_{st}")]
                    break

            # Nucleoli channels always use vesicular structure_type
            if structure_type is None and base_channel.startswith("nucleoli"):
                structure_type = "vesicular"

            # Also check experiment config for structure_type override
            if structure_type is None and base_channel in exp_configs:
                structure_type = exp_configs[base_channel].get("structure_type")

            ch_info = resolve_single_channel_info(base_channel, channel_names, experiment=resolved_name)
            if ch_info is None:
                print(f"  SKIP {label_name} - could not resolve channel '{base_channel}'")
                skipped += 1
                continue

            organelle_key = ch_info.get("organelle_key", base_channel)
            source_channel = ch_info.get("source_channel", base_channel)
            channel_label = ch_info.get("channel_label", base_channel)

            channel_index = get_channel_index(channel_names, source_channel)

            # Determine processing params for metadata
            detection_params, clahe_metadata, actual_method = _determine_processing_params(
                organelle_key=organelle_key,
                source_channel=source_channel,
                structure_type=structure_type,
                ch_info=ch_info,
            )

            # Build fresh metadata
            seg_metadata = _build_segmentation_metadata(
                label_name=label_name,
                organelle_name=organelle_key,
                channel_name=source_channel,
                channel_label=channel_label,
                channel_index=channel_index,
                segmenter_type=actual_method,
                channel_names=channel_names,
                structure_type=structure_type,
                clahe_params=clahe_metadata,
                detection_params=detection_params,
            )

            bio = seg_metadata.get("biological_annotation", {})
            org = bio.get("organelle", "?")
            marker = bio.get("marker", "?")
            print(f"  {label_name}: organelle={org}, marker={marker}, "
                  f"structure_type={seg_metadata.get('segmentation', {}).get('structure_type')}")

            # Write to all positions
            for pos in position_list:
                _update_labels_metadata(source_path, pos, label_name, metadata=seg_metadata)
            updated += 1

        print(f"\n{'='*60}")
        print(f"REPAIR COMPLETE")
        print(f"{'='*60}")
        print(f"Updated: {updated} labels across {len(position_list)} positions")
        if skipped:
            print(f"Skipped: {skipped} labels (could not resolve channel)")
        print()
        sys.exit(0)

    # Handle --sweep mode (parameter sweep for a single channel)
    if args.sweep:
        if not args.experiment:
            parser.error("--experiment is required with --sweep")
        if not args.positions:
            parser.error("--positions is required with --sweep")
        if not args.channels or len(args.channels) != 1:
            parser.error("--channels with exactly one channel is required with --sweep")

        from cyclops_utils.data.filesystem import resolve_experiment_name
        from organelle_profiler.organelle_seg.debug_utils import run_sweep_mode

        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        # Parse sweep range
        try:
            sweep_parts = args.sweep_range.split()
            if len(sweep_parts) != 3:
                parser.error("--sweep-range must be 'min max n_samples' (3 space-separated values)")
            sweep_min, sweep_max, sweep_n = float(sweep_parts[0]), float(sweep_parts[1]), int(sweep_parts[2])
        except ValueError as e:
            parser.error(f"Invalid --sweep-range format: {e}")

        # Run sweep mode via helper function
        exit_code = run_sweep_mode(
            experiment=resolved_name,
            positions=args.positions,
            channel=args.channels[0],
            sweep_var=args.sweep_var,
            sweep_min=sweep_min,
            sweep_max=sweep_max,
            sweep_n=sweep_n,
            sweep_log=args.sweep_log,
            structure_type=args.structure_type,
            no_stitch=args.no_stitch,
        )
        sys.exit(exit_code)

    # Handle --preview-all-experiments: fan --preview-all out across every experiment
    # in the channel map, collecting canvases in a single shared directory.
    if getattr(args, "preview_all_experiments", False):
        import yaml as _yaml
        from cyclops_utils.data.filesystem import resolve_experiment_name
        from cyclops_utils.data.bad_experiments import is_excluded as _is_bad_exp
        from datetime import datetime as _dt

        dataset = OpsDataset("dummy")
        cm_path = dataset.channel_maps
        if not cm_path.exists():
            print(f"ERROR: channel map not found at {cm_path}")
            sys.exit(1)
        with open(cm_path, "r") as fh:
            channel_maps = _yaml.safe_load(fh) or {}

        # (No library-override filter here — custom-library experiments are valid
        # targets for organelle segmentation preview.)

        if args.preview_output_dir:
            out_dir = Path(args.preview_output_dir)
        else:
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(f"{BASE_PATH}/analysis/organelle_seg_previews/{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        positions = args.positions if args.positions else ["A/1/0"]
        fluorescent_only = True if getattr(args, "fluorescent_only", False) else True

        # Filter experiments: must have fluorescent channels and must not be bad/excluded.
        # (Non-default library experiments are allowed through since segmentation
        # itself doesn't depend on the library / codebook.)
        def _has_fluorescent(entries):
            for e in entries or []:
                if isinstance(e, dict) and "channel_name" in e:
                    if e["channel_name"] not in ("BF", "Phase", "Phase2D", "Focus3D"):
                        return True
            return False

        all_keys = sorted(channel_maps.keys())
        exp_keys = []
        skipped = {"no_fluorescent": [], "bad_or_excluded": []}
        for k in all_keys:
            if not _has_fluorescent(channel_maps.get(k)):
                skipped["no_fluorescent"].append(k)
                continue
            # bad_experiments.is_excluded expects a full name; for prefix-only entries
            # try to resolve first, else fall back to prefix check.
            try:
                resolved_for_check = resolve_experiment_name(k, allow_interactive=False) or k
            except Exception:
                resolved_for_check = k
            if _is_bad_exp(resolved_for_check):
                skipped["bad_or_excluded"].append(k)
                continue
            exp_keys.append(k)

        if skipped["no_fluorescent"]:
            print(f"Skipped {len(skipped['no_fluorescent'])} experiments with no fluorescent channels")
        if skipped["bad_or_excluded"]:
            print(f"Skipped {len(skipped['bad_or_excluded'])} bad/excluded experiments: {skipped['bad_or_excluded']}")

        print(f"\n{'='*60}")
        print(f"PREVIEW ALL EXPERIMENTS")
        print(f"{'='*60}")
        print(f"  experiments to preview: {len(exp_keys)}")
        print(f"  positions:              {positions}")
        print(f"  fluorescent_only:       {fluorescent_only}")
        print(f"  output dir:             {out_dir}")
        print(f"  parallel workers:       {args.preview_jobs}")
        print(f"{'='*60}\n")

        # Resolve every experiment key up-front so the SLURM jobs get full names
        resolved_keys: list[tuple[str, str]] = []  # (channel_map_key, resolved_name)
        for k in exp_keys:
            try:
                r = resolve_experiment_name(k, allow_interactive=False)
            except Exception:
                r = None
            if r is None:
                print(f"  [warn] Could not resolve experiment '{k}'; skipping")
                continue
            resolved_keys.append((k, r))

        if not resolved_keys:
            print("No resolvable experiments to preview. Exiting.")
            sys.exit(1)

        # Build one SLURM job per experiment (each ~3 min, so very short timeout).
        jobs_to_submit = []
        for k, resolved in resolved_keys:
            jobs_to_submit.append({
                "name": f"preview_{resolved}",
                "func": run_preview_all_for_experiment,
                "kwargs": dict(
                    resolved_name=resolved,
                    positions=list(positions),
                    # Each experiment gets its own subdirectory so its 4 region PNGs
                    # + the params YAML stay grouped together.
                    output_dir=out_dir / resolved,
                    fluorescent_only=fluorescent_only,
                    filename_prefix=None,
                ),
                "metadata": {"experiment": resolved, "type": "preview"},
            })

        # SLURM params — preview jobs are short and CPU-only (no GPU needed).
        # 3 min is enough for most 1-2 channel experiments, but CP (8 ch) / 4i (15 ch)
        # / high-object-count experiments need more for canvas rendering. Default to
        # 15 min; user --timeout overrides.
        preview_slurm_params = {
            "timeout_min": args.timeout if args.timeout != 180 else 15,
            "mem": args.mem if args.mem and args.mem != "256GB" else "64GB",
            "cpus_per_task": args.cpus or 8,
            "slurm_partition": args.partition if args.partition != "gpu" else "cpu",
        }

        # submit_parallel_jobs already imported at module level (line 134) — local
        # re-import would make it local throughout main() and break the --all code path.
        result = submit_parallel_jobs(
            jobs_to_submit=jobs_to_submit,
            experiment="preview_all_experiments",
            slurm_params=preview_slurm_params,
            log_dir="organelle_seg_previews",
            manifest_prefix="preview_all",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
        )

        if args.dry_run:
            print(f"\n--dry-run: {len(jobs_to_submit)} preview jobs would have been submitted.")
            sys.exit(0)

        if args.no_wait:
            print(f"\n--no-wait: jobs submitted. Canvases will land in {out_dir} as each finishes.")
            sys.exit(0 if result.get("success", False) else 1)

        # Summary
        print(f"\n{'='*60}")
        print("PREVIEW ALL EXPERIMENTS — FINAL SUMMARY")
        print(f"{'='*60}")
        completed = set(result.get("completed", []))
        failed = set(result.get("failed", []))
        for j in jobs_to_submit:
            status = "OK" if j["name"] in completed else ("FAIL" if j["name"] in failed else "?")
            print(f"  [{status}] {j['metadata']['experiment']}")
        print(f"\nAll canvases -> {out_dir}")
        sys.exit(0 if result.get("all_completed", False) else 1)

    # Handle --preview-compare mode (OLD vs NEW side-by-side for swap candidates)
    if getattr(args, "preview_compare", False):
        if not args.experiment:
            parser.error("--experiment is required with --preview-compare")
        if not args.positions:
            parser.error("--positions is required with --preview-compare")
        if not getattr(args, "new_params", None):
            parser.error("--new-params (overlay org_seg_params.yaml) is required with --preview-compare")
        if not Path(args.new_params).exists():
            parser.error(f"--new-params path does not exist: {args.new_params}")

        from cyclops_utils.data.filesystem import resolve_experiment_name

        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        out_dir = Path(args.preview_output_dir) if getattr(args, "preview_output_dir", None) else None
        if out_dir is not None:
            out_dir = out_dir / resolved_name
        summary = run_compare_preview_for_experiment(
            resolved_name=resolved_name,
            new_params_path=args.new_params,
            positions=args.positions,
            output_dir=out_dir,
            filename_prefix=None,
        )
        sys.exit(0 if summary.get("compared_channels") else 1)

    # Handle --preview-all mode (runs all segmentation types for a position)
    if args.preview_all:
        if not args.experiment:
            parser.error("--experiment is required with --preview-all")
        if not args.positions:
            parser.error("--positions is required with --preview-all")

        from cyclops_utils.data.filesystem import resolve_experiment_name

        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        out_dir = Path(args.preview_output_dir) if getattr(args, "preview_output_dir", None) else None
        # When the user routes output to a shared dir, put each experiment in its
        # own subdirectory so its region PNGs + yaml stay grouped.
        if out_dir is not None:
            out_dir = out_dir / resolved_name
        summary = run_preview_all_for_experiment(
            resolved_name=resolved_name,
            positions=args.positions,
            output_dir=out_dir,
            fluorescent_only=getattr(args, "fluorescent_only", False),
            filename_prefix=None,
        )
        sys.exit(0 if summary["successful"] == summary["total"] else 1)

    if False:
        # Legacy body (dead) kept only because the source of a large block was being
        # trimmed incrementally. It is unreachable. Safe to delete in a later commit.
        exp_channel_configs = load_experiment_configs(resolved_name)

        # Define all preview configurations: (channel, structure_type, method)
        # Label-free channels get all 3 structure types
        # Fluorescent channels use structure_type from config, or default to tubular
        preview_configs = [
            # Label-free channels - all three structure types (method=None means use default)
            ("Phase2D", "tubular", None),
            ("Phase2D", "vesicular", None),
            ("Phase2D", "vesicular_dark", None),
            ("Focus3D", "tubular", None),
            ("Focus3D", "vesicular", None),
            ("Focus3D", "vesicular_dark", None),
            # Nucleoli from Phase2D and Focus3D (uses nuclear mask)
            ("nucleoli_phase2d", None, None),
            ("nucleoli_focus3d", None, None),
        ]

        # Add all configured channels (fluorescent and cell painting)
        for ch_name, channel_config in exp_channel_configs.items():
            # Skip Phase/BF channels (already handled above as label-free)
            if ch_name in ["BF", "Phase", "Phase2D", "Focus3D"]:
                continue

            # Use configured structure_type, or default to 'tubular' for fluorescent channels
            config_structure_type = channel_config.get("structure_type", "tubular")
            config_method = channel_config.get("method")
            preview_configs.append((ch_name, config_structure_type, config_method))

        print(f"\n{'='*60}")
        print(f"PREVIEW ALL MODE - Running locally (no SLURM)")
        print(f"{'='*60}")
        print(f"Experiment: {resolved_name}")
        print(f"Positions: {args.positions}")
        print(f"Configurations to run:")
        for ch, st, method in preview_configs:
            st_str = st if st else "auto"
            method_str = f", method: {method}" if method else ""
            print(f"  - {ch} ({st_str}{method_str})")
        print(f"{'='*60}\n")

        all_results = []
        for pos in args.positions:
            for ch, structure_type, method in preview_configs:
                st_str = f" ({structure_type})" if structure_type else ""
                method_str = f", method: {method}" if method else ""
                print(f"\n--- Processing {pos} / {ch}{st_str}{method_str} ---")
                try:
                    result = segment_single_position_channel(
                        experiment=resolved_name,
                        position=pos,
                        channel_key=ch,
                        structure_type=structure_type,
                        debug_only=True,
                    )
                    all_results.append((pos, ch, structure_type, method, result))

                    if result.get("success"):
                        print(f"✓ {ch}{st_str}: {result.get('num_objects', 0)} objects")
                    else:
                        print(f"✗ Error: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"✗ Exception: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append((pos, ch, structure_type, method, {"success": False, "error": str(e)}))

        # Create combined canvas for each position
        print(f"\n{'='*60}")
        print(f"CREATING COMBINED CANVASES")
        print(f"{'='*60}")

        from organelle_profiler.organelle_seg.visualizations import (
            create_combined_canvas,
            save_segmentation_params_yaml,
        )

        # Get output directory (OpsDataset already imported at module level)
        dataset = OpsDataset(resolved_name)
        source_path = dataset.store_paths["pheno_assembled_v3"]
        assembly_dir = source_path.parent
        canvas_dir = assembly_dir / "organelle_seg_debug"
        canvas_dir.mkdir(parents=True, exist_ok=True)

        for pos in args.positions:
            # Collect all successful results for this position
            # all_results contains 5-tuples: (pos, ch, structure_type, method, result)
            pos_results = [
                (ch, st, r) for p, ch, st, method, r in all_results
                if p == pos and r.get("success") and r.get("labels") is not None
            ]

            if not pos_results:
                print(f"  No successful results for {pos}, skipping canvas")
                continue

            # Build channel_params dict for canvas subtitle and YAML export
            channel_params = {}
            for ch, st, r in pos_results:
                name = f"{ch}_{st}" if st else ch
                channel_params[name] = {
                    'frangi_params': r.get('frangi_params', {}),
                    'clahe_params': r.get('clahe_params', {}),
                    'structure_type': r.get('structure_type'),
                }

            # Save params YAML to debug folder (preview mode)
            yaml_path = canvas_dir / f"segmentation_params_{pos.replace('/', '_')}.yaml"
            save_segmentation_params_yaml(
                channel_params=channel_params,
                output_path=yaml_path,
                experiment=resolved_name,
                position=pos,
                preview_mode=True,
            )

            # Build images list for canvas - crop to 512x512 center region
            # This keeps full 2x2 tile processing but saves only a small region to canvas
            # We create TWO canvases: one at current position, one 500px below (Y offset)
            canvas_crop_size = 512

            # Define two crop regions: current and 500px below
            crop_configs = [
                {"name": "region1", "x_offset": -250, "y_offset": 0, "suffix": ""},
                {"name": "region2", "x_offset": -250, "y_offset": 500, "suffix": "_region2"},
                {"name": "region3", "x_offset": 250, "y_offset": 0, "suffix": "_region3"},
                {"name": "region4", "x_offset": 250, "y_offset": 500, "suffix": "_region4"},
            ]

            for crop_cfg in crop_configs:
                y_extra_offset = crop_cfg["y_offset"]
                x_offset = crop_cfg["x_offset"]
                suffix = crop_cfg["suffix"]

                images = []
                for ch, st, r in pos_results:
                    name = f"{ch}_{st}" if st else ch
                    # Get fresh copies from result dict
                    labels = r.get('labels')
                    raw = r.get('raw')
                    vesselness = r.get('vesselness')

                    # Crop to 512x512 region with offsets
                    if labels is not None:
                        h, w = labels.shape[:2]
                        cy = h // 2 + y_extra_offset  # Add Y offset for second canvas
                        cx = w // 2 + x_offset
                        half = canvas_crop_size // 2
                        y1, y2 = max(0, cy - half), min(h, cy + half)
                        x1, x2 = max(0, cx - half), min(w, cx + half)
                        labels = labels[y1:y2, x1:x2]
                        if raw is not None:
                            raw = raw[y1:y2, x1:x2]
                        if vesselness is not None:
                            vesselness = vesselness[y1:y2, x1:x2]

                    images.append({
                        'name': name,
                        'labels': labels,
                        'raw': raw,
                        'vesselness': vesselness,
                    })

                # Create combined canvas for this region with params in subtitle
                canvas_path = canvas_dir / f"preview_{pos.replace('/', '_')}_combined{suffix}.png"
                region_label = f" (Y+{y_extra_offset}px)" if y_extra_offset else ""
                create_combined_canvas(
                    images=images,
                    labels=[],  # deprecated param
                    output_path=canvas_path,
                    title=f"Preview {pos}{region_label}",
                    channel_params=channel_params,
                )

        # Summary
        print(f"\n{'='*60}")
        print(f"PREVIEW ALL SUMMARY")
        print(f"{'='*60}")
        successes = sum(1 for _, _, _, _, r in all_results if r.get("success"))
        print(f"Processed: {len(all_results)} configurations")
        print(f"Successful: {successes}")

        if successes > 0:
            print(f"\nResults per channel:")
            for pos, ch, st, method, result in all_results:
                if result.get("success"):
                    st_str = f" ({st})" if st else ""
                    tiled_str = " [tiled]" if result.get("tiled") else ""
                    print(f"  {pos}/{ch}{st_str}: {result.get('num_objects', 0)} objects, {result.get('elapsed_time', 0):.1f}s{tiled_str}")
            print(f"\nCombined canvases saved to: {canvas_dir}")
        print(f"{'='*60}\n")

        sys.exit(0 if successes == len(all_results) else 1)

    # Handle --preview mode (runs locally, no SLURM submission)
    if args.preview:
        if not args.experiment:
            parser.error("--experiment is required with --preview")
        if not args.positions or not args.channels:
            parser.error("--positions and --channels are required with --preview")

        from cyclops_utils.data.filesystem import resolve_experiment_name
        # (segment_single_position_channel already imported at module level — local
        # re-import would mark it as a local var and break other code paths in main())

        resolved_name = resolve_experiment_name(args.experiment, allow_interactive=True)
        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"PREVIEW MODE - Running locally (no SLURM)")
        print(f"{'='*60}")
        print(f"Experiment: {resolved_name}")
        print(f"Positions: {args.positions}")
        print(f"Channels: {args.channels}")
        print(f"Structure type: {args.structure_type or 'auto (from config)'}")
        print(f"Tile grid: 2x2 (1024px tiles, 128px overlap) -> 1920x1920px crop")
        print(f"{'='*60}\n")

        all_results = []
        for pos in args.positions:
            for ch in args.channels:
                print(f"\n--- Processing {pos} / {ch} ---")
                result = segment_single_position_channel(
                    experiment=resolved_name,
                    position=pos,
                    channel_key=ch,
                    structure_type=args.structure_type,
                    debug_only=True,
                )
                all_results.append((pos, ch, result))

                if result.get("success"):
                    print(f"✓ Preview saved to: {result.get('debug_dir', 'N/A')}")
                else:
                    print(f"✗ Error: {result.get('error', 'Unknown error')}")

        # Summary
        print(f"\n{'='*60}")
        print(f"PREVIEW SUMMARY")
        print(f"{'='*60}")
        successes = sum(1 for _, _, r in all_results if r.get("success"))
        print(f"Processed: {len(all_results)} position-channel combinations")
        print(f"Successful: {successes}")

        if successes > 0:
            print(f"\nDebug images saved to directories:")
            for pos, ch, result in all_results:
                if result.get("success"):
                    tiled_str = " (with stitching)" if result.get("tiled") else ""
                    print(f"  {pos}/{ch}: {result.get('debug_dir', 'N/A')}")
                    print(f"    Objects: {result.get('num_objects', 0)}, Time: {result.get('elapsed_time', 0):.1f}s{tiled_str}")
        print(f"{'='*60}\n")

        sys.exit(0 if successes == len(all_results) else 1)

    # Build SLURM parameters
    slurm_params = {
        "timeout_min": args.timeout,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "gpus_per_node": args.gpus,
        "slurm_partition": args.partition,
    }
    # Add GPU constraint if specified (and not 'none')
    if args.gpu_constraint and args.gpu_constraint.lower() != "none":
        slurm_params["slurm_constraint"] = args.gpu_constraint

    # Single experiment mode - resolve experiment name with partial matching
    if args.experiment:
        from cyclops_utils.data.filesystem import resolve_experiment_name

        resolved_name = resolve_experiment_name(
            args.experiment,
            allow_interactive=True
        )

        if resolved_name is None:
            print("No experiment selected or found. Exiting.")
            sys.exit(1)

        args.experiment = resolved_name

    # Handle --all mode (also entered for --jobs-file, which bypasses
    # auto-discovery and submits an explicit (exp, pos, channel) list).
    if args.all or args.jobs_file:
        # --jobs-file short-circuits auto-discovery and the
        # detect_experiments_needing_segmentation scan.
        if args.jobs_file:
            import csv as _csv
            from cyclops_utils.data.filesystem import resolve_experiment_name as _resolve_exp
            jobs_csv_rows = []
            with open(args.jobs_file, newline="") as _fh:
                reader = _csv.reader(_fh)
                for row in reader:
                    if not row or row[0].lstrip().startswith("#"):
                        continue
                    if len(row) < 3:
                        raise ValueError(f"--jobs-file row malformed (need 3 cols): {row}")
                    exp_in, pos, ch = (row[0].strip(), row[1].strip(), row[2].strip())
                    if not exp_in or not pos or not ch:
                        continue
                    # Header detection: skip a column-name row if present.
                    if exp_in.lower() == "experiment":
                        continue
                    exp_resolved = _resolve_exp(exp_in, allow_interactive=False) or exp_in
                    jobs_csv_rows.append((exp_resolved, pos, ch))
            if not jobs_csv_rows:
                print(f"--jobs-file {args.jobs_file} contained no valid rows. Exiting.")
                sys.exit(1)
            print(f"--jobs-file: loaded {len(jobs_csv_rows)} jobs from {args.jobs_file}")
            # The submission loop below expects experiments_to_process iterable
            # of (exp, n_done, n_total, _) tuples — synthesize one entry per
            # unique experiment so labels pre-creation still happens.
            _seen_exps = []
            for _exp, _, _ in jobs_csv_rows:
                if _exp not in _seen_exps:
                    _seen_exps.append(_exp)
            experiments_to_process = [(e, 0, 0, None) for e in _seen_exps]
            experiments_completed = []
            # Skip the rest of the auto-discovery block.
            include_tokens, exclude_tokens = [], []
        else:
            jobs_csv_rows = None
        # Detect experiments needing segmentation
        # Push include/exclude into the scanner so we don't open zarrs for filtered-out exps
        if not args.jobs_file:
            include_tokens = [t.strip() for t in (getattr(args, "include_experiments", []) or []) if t.strip()]
            exclude_tokens = [t.strip() for t in (args.exclude_experiments or []) if t.strip()]
            if include_tokens:
                print(f"--include-experiments: scanning only {include_tokens}")
            if exclude_tokens:
                print(f"--exclude-experiments: skipping {exclude_tokens}")

        if not args.jobs_file:
            experiments_to_process, experiments_completed = detect_experiments_needing_segmentation(
                positions=args.positions,
                channels=args.channels,
                force=args.force,
                verbose=not args.quiet,
                include_prefixes=include_tokens,
                exclude_prefixes=exclude_tokens,
            )

        if not experiments_to_process:
            print("\n All experiments are complete! No segmentation jobs needed.\n")
            if not args.quiet and experiments_completed:
                print(f"Completed experiments ({len(experiments_completed)}):")
                for exp, n_done, n_total, _ in experiments_completed:
                    print(f"  {exp}: {n_done}/{n_total}")
            sys.exit(0)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch Segmentation Submission: {len(experiments_to_process)} experiments")
        print(f"{'='*60}\n")

        for exp, n_done, n_total, _ in experiments_to_process:
            status = f"{n_done}/{n_total} complete" if n_total > 0 else "pending"
            print(f"  {exp}: {status}")

        if experiments_completed and not args.quiet:
            print(f"\nAlready completed ({len(experiments_completed)}):")
            for exp, n_done, n_total, _ in experiments_completed[:5]:
                print(f"  {exp}")
            if len(experiments_completed) > 5:
                print(f"  ... and {len(experiments_completed) - 5} more")

        print(f"\n{'='*60}\n")

        # Build job list across all experiments and track positions per experiment
        all_jobs = []
        experiment_positions = {}  # Track positions for each experiment for labels pre-creation
        for experiment, n_done, n_total, _ in experiments_to_process:
            try:
                # With --force we need ALL channels (not just unprocessed) so force-rerun works
                info = get_available_channels(experiment, skip_existing=not args.force)
                if args.jobs_file:
                    # Use only the (pos, ch) tuples explicitly listed for this experiment.
                    pos_ch_pairs = [(p, c) for (e, p, c) in jobs_csv_rows if e == experiment]
                else:
                    pos_list = args.positions if args.positions else info["positions"]
                    ch_list = args.channels if args.channels else info["channels"]

                    # --fluorescent-only: skip label-free (Phase2D/Focus3D/nucleoli_*) channels
                    if getattr(args, "fluorescent_only", False):
                        _labelfree_prefixes = ("Phase2D", "Focus3D", "nucleoli_")
                        ch_list = [c for c in ch_list if not any(c.startswith(p) for p in _labelfree_prefixes)]
                    pos_ch_pairs = [(p, c) for p in pos_list for c in ch_list]

                for pos, ch in pos_ch_pairs:
                    if pos not in info["positions"]:
                        continue
                    if ch not in info["channels"]:
                        continue

                    pos_safe = pos.replace("/", "_")
                    ch_safe = ch.replace("/", "_")
                    job_name = f"{experiment}_{pos_safe}_{ch_safe}"

                    # Parse structure_type from channel key for Frangi channels
                    structure_type = info.get("channel_structure_types", {}).get(ch, None)

                    # Extract base channel name (remove _tubular or _vesicular suffix)
                    if structure_type and ch.endswith(f"_{structure_type}"):
                        base_channel = ch[:-len(f"_{structure_type}")]
                    else:
                        base_channel = ch

                    all_jobs.append({
                        "name": job_name,
                        "func": segment_single_position_channel,
                        "kwargs": {
                            "experiment": experiment,
                            "position": pos,
                            "channel_key": base_channel,
                            "structure_type": structure_type,
                            "debug_crop_fraction": args.debug_crop,
                        },
                        "metadata": {
                            "experiment": experiment,
                            "position": pos,
                            "channel": ch,
                            "base_channel": base_channel,
                            "structure_type": structure_type,
                            "method": info["channel_methods"].get(ch, "unknown"),
                        },
                    })

                    # Track positions per experiment for pre-creation
                    if experiment not in experiment_positions:
                        experiment_positions[experiment] = set()
                    experiment_positions[experiment].add(pos)
            except Exception as e:
                print(f"  Error processing {experiment}: {e}")
                continue

        # Show job plan
        print(f"{'='*60}")
        if args.dry_run:
            print(f"DRY RUN: Job Submission Plan")
        else:
            print(f"Job Submission Plan")
        print(f"{'='*60}\n")
        print(f"Total jobs to submit: {len(all_jobs)}")
        print(f"SLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params['mem']}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  GPUs: {slurm_params['gpus_per_node']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")

        # Group jobs by experiment for cleaner display
        from collections import defaultdict
        jobs_by_exp = defaultdict(list)
        for job in all_jobs:
            exp = job['metadata']['experiment']
            pos = job['metadata']['position']
            ch = job['metadata']['channel']
            jobs_by_exp[exp].append(f"{pos}:{ch}")

        print(f"\nJobs by experiment:")
        for exp, job_list in sorted(jobs_by_exp.items()):
            print(f"  {exp}: {len(job_list)} jobs")

        print(f"\n{'='*60}\n")

        # Exit if dry run
        if args.dry_run:
            print("DRY RUN: No jobs submitted\n")
            sys.exit(0)

        # Prompt user for confirmation
        if not args.yes:
            try:
                response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nCancelled by user. No jobs submitted.\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user. No jobs submitted.\n")
                sys.exit(0)
            print()
        else:
            print("Proceeding with submission (--yes flag provided)...\n")

        # Pre-create labels groups for all experiments to avoid race conditions
        print("Pre-creating labels groups for all experiments...")
        for exp, positions in sorted(experiment_positions.items()):
            try:
                ensure_labels_groups_exist(exp, list(positions), verbose=not args.quiet)
            except Exception as e:
                print(f"  Warning: Failed to pre-create labels for {exp}: {e}")
        print()

        # Submit all jobs
        result = submit_parallel_jobs(
            jobs_to_submit=all_jobs,
            experiment=f"batch_segmentation_{len(experiments_to_process)}_experiments",
            slurm_params=slurm_params,
            log_dir="slurm_segmentation_logs/all/%j",
            manifest_prefix="organelle_seg_batch",
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            post_completion_callback=None,
        )

        # Save experiment-to-job mapping in the all/ directory
        if result.get("success") and not args.dry_run:
            from collections import defaultdict
            import yaml

            base_job_id = result.get("base_job_id")
            jobs_list = result.get("jobs", [])

            # Build experiment -> job IDs mapping
            exp_to_jobs = defaultdict(list)
            for job in jobs_list:
                exp = job.get("experiment", "unknown")
                job_id = job.get("job_id", job.get("array_index", "?"))
                exp_to_jobs[exp].append(job_id)

            # Save mapping to all/ directory
            manifest_dir = Path("slurm_logs/slurm_segmentation_logs/all")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / f"experiment_mapping_{base_job_id}.yaml"

            mapping_data = {
                "slurm_array_id": base_job_id,
                "total_jobs": len(jobs_list),
                "experiments": {exp: {"job_ids": ids, "count": len(ids)} for exp, ids in sorted(exp_to_jobs.items())},
            }

            with open(manifest_file, "w") as f:
                yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)

            print(f"\nExperiment mapping saved: {manifest_file}")

        # Exit based on result
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                sys.exit(0)
        else:
            sys.exit(1)

    # Single experiment mode
    else:
        result = submit_organelle_segmentation_jobs(
            experiment=args.experiment,
            positions=args.positions,
            channels=args.channels,
            slurm_params=slurm_params,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            verbose=not args.quiet,
            debug_crop_fraction=args.debug_crop,
            force=args.force,
        )

        # Exit with appropriate code
        if result.get("dry_run"):
            sys.exit(0)
        elif result.get("success"):
            if result.get("all_completed") is not None:
                sys.exit(0 if result.get("all_completed") else 1)
            else:
                sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
