"""
SPMD (Single Program, Multiple Data) worker for CPU-side feature extraction.

Each SLURM task processes an interleaved slice of tasks and writes partial results
to parquet files on the shared filesystem. A separate merge job
(spmd_cpu_merge.py) combines the partials into the final output.

Task types:
  - network: skan network analysis (branches, per-object features, cell summaries)
  - morph_supplement: 5 morphology properties skipped by GPU (perimeter, solidity, etc.)

Supports both single-batch and multi-batch modes:

Single batch:
    srun python -m organelle_profiler.feature_extraction.spmd_cpu_worker \
        --experiment ops0094_20251217_mark --batch-id A_1_0_9999 \
        --output-dir /path/to/output

Multi-batch (combined SPMD for all batches):
    srun python -m organelle_profiler.feature_extraction.spmd_cpu_worker \
        --experiment ops0094_20251217_mark --batch-ids A_1_0_0000,A_2_0_0000,A_3_0_0000 \
        --output-dir /path/to/output
"""

import argparse
import gc
import os
import time
import traceback
from pathlib import Path

import pandas as pd


def _load_completed_tasks(partials_dir, rank_str):
    """Load already-completed (cell_id, organelle_name, task_type) from existing partials."""
    completed = set()

    def _read_pairs(pattern, task_type):
        for f in sorted(partials_dir.glob(pattern)):
            try:
                df = pd.read_parquet(f, columns=["cell_id", "organelle_name"])
                completed.update((row["cell_id"], row["organelle_name"], task_type)
                                 for _, row in df.iterrows())
            except Exception:
                pass

    # Network: use branches and per_object partials (have explicit organelle_name)
    _read_pairs(f"{rank_str}_branches*.parquet", "network")
    _read_pairs(f"{rank_str}_per_object*.parquet", "network")

    # Morph supplement
    _read_pairs(f"{rank_str}_morph_supplement*.parquet", "morph_supplement")

    return completed


def _load_my_tasks(batch_results_dir, batch_ids, rank, n_ranks):
    """Load only this rank's interleaved slice of tasks from parquet files.

    Instead of loading all 29M rows into memory and slicing, reads each file
    as an Arrow table (columnar, memory-efficient), takes only this rank's
    rows, and frees the full table immediately. Peak memory: ~500MB vs ~9GB.
    """
    import pyarrow.parquet as pq

    # First pass: collect file specs with row counts (metadata only, no data)
    file_specs = []  # [(path, n_rows, batch_id, task_type)]
    for batch_id in batch_ids:
        for task_type, pattern in [("network", "network_tasks")]:
            path = batch_results_dir / f"batch_{batch_id}_{pattern}.parquet"
            if not path.exists():
                if rank == 0:
                    print(f"WARNING: {task_type} tasks not found for {batch_id}: {path}")
                continue
            n_rows = pq.ParquetFile(path).metadata.num_rows
            if n_rows > 0:
                file_specs.append((path, n_rows, batch_id, task_type))

    total_tasks = sum(n for _, n, _, _ in file_specs)
    n_network = sum(n for _, n, _, t in file_specs if t == "network")
    n_morph = sum(n for _, n, _, t in file_specs if t == "morph_supplement")

    # Second pass: read only this rank's rows from each file
    # Global task index space: [0, total_tasks). Rank gets indices rank, rank+n_ranks, ...
    # Each file occupies a contiguous range [offset, offset+n_rows) in global space.
    my_dfs = []
    offset = 0
    for path, n_rows, batch_id, task_type in file_specs:
        # Local indices within this file that map to this rank's global indices
        remainder = (rank - offset) % n_ranks
        local_indices = list(range(remainder, n_rows, n_ranks))

        if local_indices:
            table = pq.read_table(path)
            sliced = table.take(local_indices).to_pandas()
            del table
            sliced["batch_id"] = batch_id
            sliced["task_type"] = task_type
            my_dfs.append(sliced)

        offset += n_rows

    return my_dfs, total_tasks, n_network, n_morph


def main():
    parser = argparse.ArgumentParser(description="SPMD CPU worker (network + morph supplement)")
    parser.add_argument("--experiment", required=True)
    # Support both single-batch (--batch-id) and multi-batch (--batch-ids)
    parser.add_argument("--batch-id", default=None, help="Single batch ID (backward compat)")
    parser.add_argument("--batch-ids", default=None, help="Comma-separated batch IDs for multi-batch SPMD")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--restart", action="store_true",
                        help="Skip ranks that already completed (have _done marker)")
    # Support array job mode: explicit rank/n-ranks args
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank index for array job mode (overrides SLURM_PROCID)")
    parser.add_argument("--n-ranks", type=int, default=None,
                        help="Total ranks for array job mode (overrides SLURM_NPROCS)")
    args = parser.parse_args()

    # Determine batch IDs
    if args.batch_ids:
        batch_ids = [b.strip() for b in args.batch_ids.split(",")]
    elif args.batch_id:
        batch_ids = [args.batch_id]
    else:
        raise ValueError("Must specify --batch-id or --batch-ids")

    multi_batch = len(batch_ids) > 1

    # Rank determination: CLI args (array job mode) > SLURM env vars (srun mode)
    if args.rank is not None and args.n_ranks is not None:
        rank = args.rank
        n_ranks = args.n_ranks
    else:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        n_ranks = int(os.environ.get("SLURM_NPROCS", 1))

    output_dir = Path(args.output_dir)
    batch_results_dir = output_dir / "_batch_results"

    # Partials directory: combined for multi-batch, per-batch for single
    if multi_batch:
        partials_dir = batch_results_dir / "_partials_combined"
    else:
        partials_dir = batch_results_dir / f"_partials_{batch_ids[0]}"

    # All ranks can mkdir (parents=True, exist_ok=True is safe for concurrent calls)
    partials_dir.mkdir(parents=True, exist_ok=True)

    rank_str = f"rank{rank:04d}"
    done_marker = partials_dir / f"{rank_str}_done"

    # --restart: skip ranks that already completed
    if args.restart and done_marker.exists():
        if rank < 4 or rank == n_ranks - 1:
            print(f"  Rank {rank}: already done (marker exists), skipping")
        return

    # Load only this rank's slice of tasks (memory-efficient: ~500MB vs ~9GB)
    my_task_dfs, total_tasks, n_network, n_morph = _load_my_tasks(
        batch_results_dir, batch_ids, rank, n_ranks,
    )

    if not my_task_dfs:
        if rank == 0:
            print("ERROR: No tasks found for any batch")
        return

    my_tasks_df = pd.concat(my_task_dfs, ignore_index=True)
    del my_task_dfs

    if rank == 0:
        # Print SLURM job info header using shared utility
        from cyclops_utils.hpc.slurm_utils import print_slurm_job_header
        print_slurm_job_header(
            title="SPMD CPU Worker",
            extra_info={"Batches": f"{len(batch_ids)} ({', '.join(batch_ids)})"}
        )

        print(f"\nTotal tasks: {total_tasks:,} ({n_network:,} network + {n_morph:,} morph_supplement)")
        min_tasks = total_tasks // n_ranks
        max_tasks = min_tasks + (1 if total_tasks % n_ranks > 0 else 0)
        print(f"  Tasks per rank: {min_tasks}-{max_tasks}")
        # Count already-done ranks at startup
        n_done_at_start = len(list(partials_dir.glob("rank*_done")))
        if n_done_at_start > 0:
            print(f"  Already completed: {n_done_at_start}/{n_ranks} ranks")

    # Build work items from this rank's slice
    work_items = []
    for idx, row in my_tasks_df.iterrows():
        work_items.append({
            'task_idx': idx,
            'task_type': row['task_type'],
            'global_cell_id': row['global_cell_id'],
            'well': row['well'],
            'organelle_name': row['organelle_name'],
            'seg_label_name': row['seg_label_name'],
            'cell_seg_label_name': row.get('cell_seg_label_name'),
            'bbox': (int(row['bbox_y0']), int(row['bbox_x0']),
                     int(row['bbox_y1']), int(row['bbox_x1'])),
            'seg_id': int(row['seg_id']) if pd.notna(row.get('seg_id')) else None,
            'spacing_y': row['spacing_y'],
            'spacing_x': row['spacing_x'],
            'store_path': row['store_path'],
            'result_idx': int(row['result_idx']) if pd.notna(row.get('result_idx')) else 0,
            'batch_id': row['batch_id'],
        })
    del my_tasks_df

    # --restart: filter out already-completed tasks
    n_skipped = 0
    if args.restart:
        completed = _load_completed_tasks(partials_dir, rank_str)
        if completed:
            before = len(work_items)
            work_items = [w for w in work_items
                          if (w['global_cell_id'], w['organelle_name'], w['task_type']) not in completed]
            n_skipped = before - len(work_items)
            if rank < 4:
                print(f"  Rank {rank}: restart — {n_skipped} tasks already done, {len(work_items)} remaining")
        if not work_items:
            if rank < 4 or rank == n_ranks - 1:
                print(f"  Rank {rank}: all tasks already done, writing marker")
            done_marker.touch()
            return

    # Import processing functions
    from organelle_profiler.feature_extraction.fe_workers import (
        _run_network_analysis_from_zarr,
    )

    # Use a restart suffix for output files to avoid overwriting existing partials
    restart_suffix = ""
    if args.restart and n_skipped > 0:
        # Find next available restart round
        existing = list(partials_dir.glob(f"{rank_str}_cell_summaries_r*.parquet"))
        restart_suffix = f"_r{len(existing) + 1}"

    # Per-rank log file for debugging crashes
    import resource
    rank_log_path = partials_dir / f"{rank_str}_log.txt"
    rank_log = open(rank_log_path, "a")  # append mode for restart
    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB

    rank_log.write(f"=== Rank {rank} start: {len(work_items)} tasks, restart_skipped={n_skipped}, rss={rss_mb():.0f}MB ===\n")
    rank_log.flush()

    # Network tasks only (morph supplement disabled for performance — see option 5)
    network_items = [w for w in work_items if w['task_type'] == 'network']

    t_start = time.time()
    cell_summaries = {}  # {(cell_id, batch_id): {col: value}}
    branch_rows = []     # list of dicts for branch DataFrame
    per_object_rows = [] # list of dicts for per-object DataFrame
    network_flush_count = 0  # number of network flush files written
    n_completed = 0
    n_network_done = 0
    n_network_failed = 0
    t_network_total = 0.0

    # ============ Phase 1: Network tasks (in-process, fast) ============
    last_progress_time = time.time()
    progress_interval = 30  # seconds between progress updates

    for i, work_item in enumerate(network_items):
        bbox = work_item.get('bbox', (0,0,0,0))
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else 0

        t_task = time.time()
        try:
            result = _run_network_analysis_from_zarr(work_item)
        except Exception:
            result = None
        dt = time.time() - t_task
        t_network_total += dt

        # Log every task to rank file
        rank_log.write(f"N {i} {work_item.get('organelle_name','')} "
                       f"bbox={bbox_area} dt={dt:.3f}s rss={rss_mb():.0f}MB\n")
        if (i + 1) % 100 == 0:
            rank_log.flush()

        # Rank 0: periodic progress update (count done markers)
        if rank == 0 and (time.time() - last_progress_time) > progress_interval:
            n_done_now = len(list(partials_dir.glob("rank*_done")))
            elapsed = time.time() - t_start
            print(f"  [Progress] {n_done_now}/{n_ranks} ranks done, "
                  f"rank0: {i+1}/{len(network_items)} tasks, {elapsed:.0f}s elapsed", flush=True)
            last_progress_time = time.time()

        if result is None:
            n_network_failed += 1
            n_completed += 1
            continue

        global_cell_id = result['global_cell_id']
        organelle_name = result['organelle_name']
        batch_id = work_item['batch_id']
        network_summary_dict = result.get('network_summary_dict')
        branch_df = result.get('branch_df')
        per_object_network_df = result.get('per_object_network_df')

        if network_summary_dict:
            key = (global_cell_id, batch_id)
            if key not in cell_summaries:
                cell_summaries[key] = {}
            for k, value in network_summary_dict.items():
                cell_summaries[key][f"network_{organelle_name}_{k}"] = value

        if branch_df is not None and not branch_df.empty:
            branch_copy = branch_df.copy()
            branch_copy["cell_id"] = global_cell_id
            branch_copy["organelle_name"] = organelle_name
            branch_copy["batch_id"] = batch_id
            branch_rows.append(branch_copy)

        if per_object_network_df is not None and not per_object_network_df.empty:
            per_obj = per_object_network_df.copy()
            per_obj["cell_id"] = global_cell_id
            per_obj["organelle_name"] = organelle_name
            per_obj["batch_id"] = batch_id
            per_object_rows.append(per_obj)

        n_network_done += 1
        n_completed += 1

    # Flush all network results
    if cell_summaries:
        rows = [{"cell_id": cid, "batch_id": bid, **cols}
                for (cid, bid), cols in cell_summaries.items()]
        pd.DataFrame(rows).to_parquet(
            partials_dir / f"{rank_str}_cell_summaries{restart_suffix}_p{network_flush_count}.parquet",
            index=False)
    if branch_rows:
        pd.concat(branch_rows, ignore_index=True).to_parquet(
            partials_dir / f"{rank_str}_branches{restart_suffix}_p{network_flush_count}.parquet",
            index=False)
    if per_object_rows:
        pd.concat(per_object_rows, ignore_index=True).to_parquet(
            partials_dir / f"{rank_str}_per_object{restart_suffix}_p{network_flush_count}.parquet",
            index=False)
    cell_summaries.clear()
    branch_rows.clear()
    per_object_rows.clear()
    network_flush_count += 1
    gc.collect()

    rank_log.write(f"=== Network done: {n_network_done} ok + {n_network_failed} fail in {t_network_total:.1f}s ===\n")
    rank_log.flush()

    # Morph supplement disabled (option 5: skip for performance)

    t_elapsed = time.time() - t_start

    # Write remaining partial results to parquet
    t_write = time.time()

    # Concat network flush parts into final files
    for name in ["cell_summaries", "branches", "per_object"]:
        part_files = sorted(partials_dir.glob(f"{rank_str}_{name}{restart_suffix}_p*.parquet"))
        if part_files:
            combined = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
            combined.to_parquet(partials_dir / f"{rank_str}_{name}{restart_suffix}.parquet", index=False)
            for p in part_files:
                p.unlink()

    # Mark this rank as done
    done_marker.touch()

    t_write_elapsed = time.time() - t_write

    if rank < 4 or rank == n_ranks - 1:
        net_avg = (t_network_total / n_network_done * 1000) if n_network_done > 0 else 0
        print(f"  Rank {rank}: done — {n_completed}/{len(work_items)} in {t_elapsed:.1f}s (write: {t_write_elapsed:.1f}s)"
              + (f" [restart: {n_skipped} previously done]" if n_skipped else ""))
        print(f"    network:  {n_network_done} ok + {n_network_failed} fail | {t_network_total:.1f}s | {net_avg:.1f}ms/task")

    rank_log.write(f"=== Rank {rank} done: {n_completed} tasks in {t_elapsed:.1f}s, peak_rss={rss_mb():.0f}MB ===\n")
    rank_log.close()

    if rank == 0:
        n_done_final = len(list(partials_dir.glob("rank*_done")))
        print(f"Rank 0 complete. {n_done_final}/{n_ranks} ranks done. Partials in {partials_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SPMD worker FAILED: {e}\n{traceback.format_exc()}")
        raise
