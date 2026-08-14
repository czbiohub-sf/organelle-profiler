"""
Merge job for SPMD CPU analysis (network + morph supplement).

Reads partial parquet files written by spmd_cpu_worker.py ranks, merges with
GPU features, aggregates branch/per-object/morph-supplement features per
organelle, and writes the final _cells.parquet for each batch.

Supports both single-batch and multi-batch modes:

Single batch:
    python -m organelle_profiler.feature_extraction.spmd_cpu_merge \
        --experiment ops0094_20251217_mark --batch-id A_1_0_9999 \
        --output-dir /path/to/output

Multi-batch (combined merge for all batches):
    python -m organelle_profiler.feature_extraction.spmd_cpu_merge \
        --experiment ops0094_20251217_mark --batch-ids A_1_0_0000,A_2_0_0000 \
        --output-dir /path/to/output
"""

import argparse
import gc
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


# ---------------------------------------------------------------------------
# Thread-parallel column-chunked groupby (same pattern as vectorized_groupby_agg)
# ---------------------------------------------------------------------------

def _threaded_groupby_agg(grouped, numeric_cols, agg_funcs, chunk_size=200, n_threads=None):
    """
    Thread-parallel column-chunked groupby aggregation.

    Same pattern as vectorized_groupby_agg (fe_aggregation.py): splits columns
    into chunks and aggregates in parallel using threads.  Pandas releases the
    GIL during numpy operations, so threads give real speedup when there are
    many feature columns.

    For small column counts (< chunk_size) this falls back to a single-pass
    agg with zero overhead.

    Returns DataFrame with columns named ``{col}_{agg}``.
    """
    if not numeric_cols:
        return pd.DataFrame(index=grouped.ngroup().index[:0])

    if n_threads is None:
        n_threads = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))

    chunks = [numeric_cols[i:i + chunk_size]
              for i in range(0, len(numeric_cols), chunk_size)]

    # Single chunk — skip threading overhead
    if len(chunks) <= 1:
        result = grouped[numeric_cols].agg(agg_funcs)
        result.columns = [f"{col}_{agg}" for col, agg in result.columns]
        return result

    def _agg_chunk(chunk_cols):
        r = grouped[chunk_cols].agg(agg_funcs)
        r.columns = [f"{col}_{agg}" for col, agg in r.columns]
        return r

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(_agg_chunk, chunks))

    return pd.concat(results, axis=1)


# ---------------------------------------------------------------------------
# Branch streaming aggregation
# ---------------------------------------------------------------------------

def _partial_agg_one_branch_file(args):
    """
    Read one branch parquet file, filter to batch_id, aggregate per
    (organelle_name, cell_id).  Each (cell_id, organelle) is fully contained
    in one rank file, so per-file aggregation gives exact results including
    median.

    Returns a small DataFrame with columns:
        organelle_name, cell_id, {feature}_{agg}, ..., _branch_count
    or None if no rows match the batch.
    """
    fpath, batch_id, agg_funcs = args
    try:
        df = pd.read_parquet(str(fpath), filters=[("batch_id", "==", batch_id)])
    except Exception:
        df = pd.read_parquet(str(fpath))
        if "batch_id" in df.columns:
            df = df[df["batch_id"] == batch_id]

    if df.empty:
        return None

    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in ("cell_id", "label", "total_index")
    ]
    if not numeric_cols:
        return None

    grouped = df.groupby(["organelle_name", "cell_id"])

    # Thread-parallel column-chunked aggregation (same pattern as vectorized_groupby_agg)
    agg_df = _threaded_groupby_agg(grouped, numeric_cols, agg_funcs)

    # Add branch count
    agg_df["_branch_count"] = grouped.size().values

    return agg_df.reset_index()


def _stream_aggregate_branches(branch_files, batch_id, agg_funcs, max_workers=32):
    """
    Stream-aggregate branch data: read each file independently, aggregate
    per (cell_id, organelle), then combine.  Peak memory is ~50MB per file
    instead of ~50GB for the full batch.

    Returns DataFrame indexed by cell_id with columns like
    network_{org}_{feature}_{agg} matching _aggregate_one_organelle output.
    """
    if not branch_files:
        return pd.DataFrame()

    t0 = time.time()
    work_items = [(f, batch_id, agg_funcs) for f in branch_files]
    n_workers = min(len(work_items), max_workers)

    partials = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_partial_agg_one_branch_file, w) for w in work_items]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                partials.append(result)

    if not partials:
        return pd.DataFrame()

    t_read = time.time() - t0
    print(f"      branch partial agg: {len(partials)} files in {t_read:.1f}s", flush=True)

    # Simple concat — each (org, cell_id) appears in exactly one partial
    combined = pd.concat(partials, ignore_index=True)
    del partials
    gc.collect()

    # Check for duplicate (org, cell_id) — shouldn't happen but handle gracefully
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["organelle_name", "cell_id"])
    n_after = len(combined)
    if n_before != n_after:
        print(f"      WARNING: dropped {n_before - n_after} duplicate (org, cell_id) rows",
              flush=True)

    # Pivot to cell-level columns with network_{org}_ prefix
    agg_cols = [c for c in combined.columns
                if c not in ("cell_id", "organelle_name", "_branch_count")]

    frames = []
    for org_name, org_df in combined.groupby("organelle_name"):
        org_df = org_df.set_index("cell_id")
        result_cols = {}
        for col in agg_cols:
            result_cols[f"network_{org_name}_{col}"] = org_df[col]
        result_cols[f"network_{org_name}_branch_count"] = org_df["_branch_count"]
        frames.append(pd.DataFrame(result_cols))

    del combined
    gc.collect()

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    t_total = time.time() - t0
    print(f"      branch agg total: {len(result)} cells, "
          f"{len(result.columns)} cols in {t_total:.1f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# Morph supplement aggregation (small data — loads fully into memory)
# ---------------------------------------------------------------------------

def _aggregate_one_morph_organelle(args):
    """
    Aggregate morph supplement for a single organelle. Used as ProcessPoolExecutor target.
    args: (org_name, org_df, available_props, agg_funcs)
    Returns DataFrame indexed by cell_id with columns like {org}_{prop}_{agg}.
    """
    org_name, org_df, available_props, agg_funcs = args
    grouped = org_df.groupby('cell_id')
    cell_agg = _threaded_groupby_agg(grouped, available_props, agg_funcs)
    # Prefix with organelle name: {prop}_{agg} -> {org}_{prop}_{agg}
    cell_agg.columns = [f"{org_name}_{c}" for c in cell_agg.columns]
    return cell_agg


def _aggregate_morph_supplement(morph_supp_df, agg_funcs, max_workers=16):
    """
    Aggregate morph supplement per-object features to cell-level columns.
    Uses ProcessPoolExecutor to parallelize across organelles.
    """
    MORPH_PROPS = ['perimeter', 'perimeter_crofton', 'euler_number']
    available_props = [p for p in MORPH_PROPS if p in morph_supp_df.columns]

    if not available_props:
        return pd.DataFrame()

    org_groups = list(morph_supp_df.groupby('organelle_name'))
    if not org_groups:
        return pd.DataFrame()

    work_items = [
        (org_name, org_df, available_props, agg_funcs)
        for org_name, org_df in org_groups
    ]

    n_workers = min(len(work_items), max_workers)
    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_aggregate_one_morph_organelle, w): w[0] for w in work_items}
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                frames.append(result)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Object feature streaming aggregation (from GPU phase chunk parquets)
# ---------------------------------------------------------------------------

def _agg_one_object_organelle(args):
    """
    Aggregate object features for one organelle from multiple chunk parquets.
    Memory-efficient: processes one chunk at a time.

    args: (org_name, chunk_paths, agg_funcs)
    Returns DataFrame indexed by cell_id with columns like {org}_{metric}_{agg}.
    """
    import pyarrow.parquet as pq

    org_name, chunk_paths, agg_funcs = args

    # Accumulate partial aggregates per cell_id
    # For exact aggregation, we need to collect all data per cell
    all_rows = []

    for chunk_path in chunk_paths:
        try:
            df = pd.read_parquet(chunk_path)
            if not df.empty:
                all_rows.append(df)
        except Exception as e:
            print(f"    Warning: failed to read {chunk_path}: {e}")
            continue

    if not all_rows:
        return pd.DataFrame()

    # Concat all chunks for this organelle
    org_df = pd.concat(all_rows, ignore_index=True)
    del all_rows
    gc.collect()

    # Find numeric columns to aggregate (exclude metadata)
    exclude_cols = {'cell_id', 'label', 'total_index', 'organelle_name', 'batch_id'}
    numeric_cols = [c for c in org_df.select_dtypes(include=np.number).columns
                    if c not in exclude_cols]

    if not numeric_cols:
        return pd.DataFrame()

    # Thread-parallel column-chunked aggregation (same pattern as vectorized_groupby_agg)
    grouped = org_df.groupby('cell_id', observed=True)
    cell_agg = _threaded_groupby_agg(grouped, numeric_cols, agg_funcs)

    # Prefix with organelle name: {metric}_{agg} -> {org}_{metric}_{agg}
    cell_agg.columns = [f"{org_name}_{c}" for c in cell_agg.columns]

    # Add object count
    cell_agg[f"{org_name}_count"] = grouped.size()

    return cell_agg


def _stream_aggregate_object_features(obj_chunks_dir: Path, agg_funcs: list, max_workers: int = 16):
    """
    Stream-aggregate object features from GPU phase chunk parquets to cell-level.

    Directory structure expected:
        obj_chunks_dir/
            mito/
                chunk_0000.parquet
                chunk_0001.parquet
                ...
            er/
                chunk_0000.parquet
                ...

    Returns DataFrame indexed by cell_id with aggregated features per organelle.
    """
    if not obj_chunks_dir.exists():
        return pd.DataFrame()

    # Find all organelle subdirectories
    org_dirs = [d for d in obj_chunks_dir.iterdir() if d.is_dir()]
    if not org_dirs:
        return pd.DataFrame()

    # Build work items: (org_name, chunk_paths, agg_funcs)
    work_items = []
    for org_dir in org_dirs:
        org_name = org_dir.name
        chunk_paths = sorted(org_dir.glob("*.parquet"))
        if chunk_paths:
            work_items.append((org_name, chunk_paths, agg_funcs))

    if not work_items:
        return pd.DataFrame()

    print(f"    Object agg: {len(work_items)} organelles...", flush=True)
    t0 = time.time()

    # Process organelles in parallel
    n_workers = min(len(work_items), max_workers)
    frames = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_agg_one_object_organelle, w): w[0] for w in work_items}
        for future in as_completed(futures):
            org_name = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    frames.append(result)
                    print(f"      {org_name}: {len(result)} cells, {len(result.columns)} cols", flush=True)
            except Exception as e:
                print(f"      {org_name}: FAILED - {e}", flush=True)

    if not frames:
        return pd.DataFrame()

    # Combine all organelles (outer join on cell_id index)
    combined = pd.concat(frames, axis=1)
    t_elapsed = time.time() - t0
    print(f"    Object agg total: {len(combined)} cells, {len(combined.columns)} cols in {t_elapsed:.1f}s",
          flush=True)

    return combined


# ---------------------------------------------------------------------------
# Per-batch merge
# ---------------------------------------------------------------------------

def _merge_one_batch(
    batch_id: str,
    batch_results_dir: Path,
    batch_summaries: pd.DataFrame,
    branch_agg: pd.DataFrame,
    batch_per_obj: pd.DataFrame,
    batch_morph_supp: pd.DataFrame,
) -> dict:
    """
    Merge network + morph supplement results for a single batch: join summaries
    and pre-aggregated branch features with GPU features, aggregate per-object
    features, write final _cells.parquet.

    Optimized: sets cell_id index ONCE, collects all feature frames, does a
    single pd.concat + join, then resets index once before writing.  This
    eliminates repeated set_index/join/reset_index cycles.

    Returns dict with status, n_cells, path.
    """
    from organelle_profiler.feature_extraction.fe_anndata import (
        _aggregate_one_organelle, AGG_FUNCS,
    )

    t_start = time.time()

    # Load GPU features for this batch
    gpu_features_path = batch_results_dir / f"batch_{batch_id}_gpu_features.parquet"
    if not gpu_features_path.exists():
        print(f"  [{batch_id}] ERROR: GPU features not found: {gpu_features_path}")
        return {"batch_id": batch_id, "status": "failed", "error": "no GPU features"}

    cell_df = pd.read_parquet(gpu_features_path)
    has_cell_id = 'cell_id' in cell_df.columns

    # Set index ONCE — all subsequent joins use index alignment (no repeated rebuild)
    if has_cell_id:
        cell_df = cell_df.set_index('cell_id')

    # --- Aggregate object features from GPU phase chunk parquets (streaming) ---
    obj_chunks_dir = batch_results_dir / f"batch_{batch_id}_obj_chunks"
    n_obj_cols = 0
    t_obj_start = time.time()
    if obj_chunks_dir.exists():
        obj_agg = _stream_aggregate_object_features(obj_chunks_dir, AGG_FUNCS, max_workers=16)
        if not obj_agg.empty:
            n_obj_cols = len(obj_agg.columns)
            if has_cell_id:
                cell_df = cell_df.join(obj_agg)
            else:
                cell_df = obj_agg
            del obj_agg
    t_obj_elapsed = time.time() - t_obj_start
    if n_obj_cols > 0:
        print(f"  [{batch_id}] Object agg: {n_obj_cols} cols in {t_obj_elapsed:.1f}s")

    # Check if network tasks exist for this batch
    network_tasks_path = batch_results_dir / f"batch_{batch_id}_network_tasks.parquet"
    has_network = network_tasks_path.exists() and len(pd.read_parquet(network_tasks_path)) > 0

    if not has_network and (batch_morph_supp is None or len(batch_morph_supp) == 0):
        print(f"  [{batch_id}] No SPMD tasks, saving GPU features + object agg as final output")
        final_path = batch_results_dir / f"batch_{batch_id}_cells.parquet"
        cell_df.reset_index().to_parquet(final_path, index=False)
        return {"batch_id": batch_id, "status": "success", "n_cells": len(cell_df), "path": str(final_path)}

    n_cols_before_network = len(cell_df.columns)

    # --- Collect all remaining feature frames for a SINGLE concat+join pass ---
    join_frames = []

    # Network cell summaries
    t_net_start = time.time()
    if batch_summaries is not None and len(batch_summaries) > 0:
        summaries = batch_summaries.drop(columns=["batch_id"], errors="ignore")
        merged_summaries = summaries.groupby('cell_id').first()  # already indexed by cell_id
        join_frames.append(merged_summaries)

    # Pre-aggregated branch features (already indexed by cell_id)
    n_branch_cols = 0
    if branch_agg is not None and not branch_agg.empty:
        n_branch_cols = len(branch_agg.columns)
        join_frames.append(branch_agg)

    # --- Per-object network features: DISABLED ---
    # Network analysis binarizes the organelle mask and skeletonizes the whole
    # cell's organelles as one network, producing ~1 connected component per cell.
    # Aggregating per-object features (mean/std/etc.) over 1 row is meaningless —
    # all aggs are identical and std is always NaN. Network summary features and
    # branch-level aggregations already capture all useful information.
    # See fe_anndata.py _aggregate_one_organelle for full explanation.
    n_per_obj_cols = 0

    t_net_elapsed = time.time() - t_net_start

    # Morph supplement
    t_morph_start = time.time()
    n_morph_cols = 0
    n_morph_rows_in = 0
    n_morph_orgs = 0
    if batch_morph_supp is not None and len(batch_morph_supp) > 0:
        morph_supp = batch_morph_supp.drop(columns=["batch_id"], errors="ignore")
        n_morph_rows_in = len(morph_supp)
        n_morph_orgs = morph_supp['organelle_name'].nunique()
        morph_agg = _aggregate_morph_supplement(morph_supp, AGG_FUNCS, max_workers=17)
        if not morph_agg.empty:
            n_morph_cols = len(morph_agg.columns)
            join_frames.append(morph_agg)
    t_morph_elapsed = time.time() - t_morph_start

    # Single join pass — concat all feature frames then join to cell_df
    t_join_start = time.time()
    if join_frames and has_cell_id:
        all_features = pd.concat(join_frames, axis=1)
        cell_df = cell_df.join(all_features)
        del all_features
    del join_frames
    t_join_elapsed = time.time() - t_join_start

    n_net_cols_added = len(cell_df.columns) - n_cols_before_network

    # Reset index ONCE and save
    cell_df = cell_df.reset_index()
    t_write_start = time.time()
    final_path = batch_results_dir / f"batch_{batch_id}_cells.parquet"
    cell_df.to_parquet(final_path, index=False)
    t_write_elapsed = time.time() - t_write_start

    t_elapsed = time.time() - t_start
    print(f"  [{batch_id}] {len(cell_df)} cells, {len(cell_df.columns)} cols, {t_elapsed:.1f}s")
    if n_obj_cols > 0:
        print(f"    object agg:   {n_obj_cols} cols in {t_obj_elapsed:.1f}s")
    print(f"    network: {n_net_cols_added} cols total "
          f"({n_branch_cols} branch + {n_per_obj_cols} per-obj) in {t_net_elapsed:.1f}s")
    print(f"    morph agg:    {n_morph_cols} cols from {n_morph_orgs} orgs "
          f"({n_morph_rows_in} rows) in {t_morph_elapsed:.1f}s")
    print(f"    join+write:   {t_join_elapsed:.1f}s + {t_write_elapsed:.1f}s")

    return {"batch_id": batch_id, "status": "success", "n_cells": len(cell_df), "path": str(final_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Set pyarrow thread pools to use all available CPUs
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
    pa.set_cpu_count(n_cpus)
    pa.set_io_thread_count(n_cpus)
    print(f"PyArrow threads: {n_cpus} CPU, {n_cpus} I/O", flush=True)

    parser = argparse.ArgumentParser(description="SPMD CPU merge job")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-id", default=None, help="Single batch ID (backward compat)")
    parser.add_argument("--batch-ids", default=None, help="Comma-separated batch IDs for multi-batch merge")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Determine batch IDs
    if args.batch_ids:
        batch_ids = [b.strip() for b in args.batch_ids.split(",")]
    elif args.batch_id:
        batch_ids = [args.batch_id]
    else:
        raise ValueError("Must specify --batch-id or --batch-ids")

    from organelle_profiler.feature_extraction.fe_anndata import AGG_FUNCS

    output_dir = Path(args.output_dir)
    batch_results_dir = output_dir / "_batch_results"

    # Partials directory
    multi_batch = len(batch_ids) > 1
    if multi_batch:
        partials_dir = batch_results_dir / "_partials_combined"
    else:
        partials_dir = batch_results_dir / f"_partials_{batch_ids[0]}"

    print(f"\n{'='*60}")
    print(f"SPMD Merge: {len(batch_ids)} batch(es)")
    print(f"Batch IDs: {', '.join(batch_ids)}")
    print(f"Partials dir: {partials_dir}")
    print(f"AGG_FUNCS: {AGG_FUNCS}")
    print(f"{'='*60}\n")

    t_start = time.time()

    # --- Discover partial files ---
    summary_files = sorted(partials_dir.glob("rank*_cell_summaries*.parquet"))
    branch_files = sorted(partials_dir.glob("rank*_branches*.parquet"))
    per_obj_files = sorted(partials_dir.glob("rank*_per_object*.parquet"))
    morph_supp_files = sorted(partials_dir.glob("rank*_morph_supplement*.parquet"))
    print(f"Found {len(summary_files)} summary, {len(branch_files)} branch, "
          f"{len(per_obj_files)} per-object, {len(morph_supp_files)} morph_supplement partials",
          flush=True)

    # --- Pyarrow dataset loader for small file types ---
    def _load_file_type_for_batch(files, batch_id):
        """Load all files of one type, filter to batch_id using pyarrow dataset."""
        if not files:
            return None
        dataset = ds.dataset([str(f) for f in files], format="parquet")
        scanner = dataset.scanner(
            filter=ds.field("batch_id") == batch_id,
            use_threads=True,
            fragment_readahead=64,
            batch_readahead=32,
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return None
        df = table.to_pandas()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")
        return df

    # --- Process each batch separately to limit peak memory ---
    t_merge = time.time()
    print(f"\nMerging {len(batch_ids)} batches...", flush=True)

    total_cells = 0
    for batch_id in batch_ids:
        t_batch_start = time.time()
        print(f"\n  [{batch_id}] Loading partials...", flush=True)

        # Summaries: small data, use pyarrow dataset
        batch_summaries = _load_file_type_for_batch(summary_files, batch_id)
        if batch_summaries is not None:
            print(f"    Summaries: {len(batch_summaries):,} rows", flush=True)

        # Branches: STREAMING aggregation (1.3B+ rows, won't fit in memory)
        print(f"    Aggregating branches (streaming)...", flush=True)
        branch_agg = _stream_aggregate_branches(
            branch_files, batch_id, AGG_FUNCS, max_workers=n_cpus,
        )
        if not branch_agg.empty:
            print(f"    Branch agg: {len(branch_agg):,} cells, "
                  f"{len(branch_agg.columns)} cols", flush=True)

        # Per-object: disabled (see _merge_one_batch comment for explanation)
        batch_per_obj = None

        # Morph supplement: no files in current run
        batch_morph_supp = _load_file_type_for_batch(morph_supp_files, batch_id)

        t_load = time.time() - t_batch_start
        print(f"    Loaded/aggregated in {t_load:.1f}s", flush=True)

        result = _merge_one_batch(
            batch_id=batch_id,
            batch_results_dir=batch_results_dir,
            batch_summaries=batch_summaries,
            branch_agg=branch_agg if not branch_agg.empty else None,
            batch_per_obj=batch_per_obj,
            batch_morph_supp=batch_morph_supp,
        )

        if result.get("status") == "success":
            total_cells += result.get("n_cells", 0)
        else:
            print(f"  [{batch_id}] FAILED: {result.get('error')}")

        # Free memory before processing the next batch
        del batch_summaries, branch_agg, batch_per_obj, batch_morph_supp
        gc.collect()

    t_merge_elapsed = time.time() - t_merge
    t_total = time.time() - t_start
    print(f"\nMerge complete: {total_cells} total cells across {len(batch_ids)} batches")
    print(f"Total: {t_total:.1f}s (merge: {t_merge_elapsed:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SPMD merge FAILED: {e}\n{traceback.format_exc()}")
        raise
