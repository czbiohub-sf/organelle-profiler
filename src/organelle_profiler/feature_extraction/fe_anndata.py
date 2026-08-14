"""
Feature Extraction AnnData Output and Aggregation Functions.

This module handles aggregation of object-level features to cell/guide/gene levels
and saving results as AnnData (.h5ad) files.

Feature metadata (organelle, metric, category, aggregation) is built during
aggregation - NOT parsed from names afterward.

Functions
---------
_generate_summaries
    Aggregate object features to cell level, then to guide and gene levels.
_save_results_as_anndata
    Save aggregated results as three AnnData files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import anndata as ad

from .fe_constants import (
    AGGREGATION_FUNCTIONS,
    CATEGORY_CELL_MORPHOLOGY,
    CATEGORY_MORPHOLOGY,
    CATEGORY_INTENSITY,
    CATEGORY_LOCALIZATION,
    CATEGORY_NETWORK,
    CATEGORY_NETWORK_OBJECT,
    CATEGORY_CONTACT,
    CATEGORY_DISTRIBUTION,
    get_unit_for_metric,
)
from .fe_metadata import create_global_cell_id

# Standard aggregation functions (without count - handled separately)
AGG_FUNCS = [f for f in AGGREGATION_FUNCTIONS if f != "count"]


def _print_timing_summary(timing_df: pd.DataFrame):
    """Print a summary of timing data from feature extraction."""
    timing_cols = [c for c in timing_df.columns if c.startswith("_timing_")]
    if not timing_cols:
        return

    print("\n" + "="*70)
    print("TIMING SUMMARY (milliseconds per cell)")
    print("="*70)

    # Summary stats for main timing categories
    main_cols = ["_timing_total_ms", "_timing_batch_extract_ms", "_timing_cell_morph_ms",
                 "_timing_localization_ms", "_timing_network_ms"]
    main_cols = [c for c in main_cols if c in timing_df.columns]

    print(f"\n{'Metric':<35} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total':>12}")
    print("-" * 70)
    for col in main_cols:
        name = col.replace("_timing_", "").replace("_ms", "")
        mean_val = timing_df[col].mean()
        median_val = timing_df[col].median()
        max_val = timing_df[col].max()
        total_val = timing_df[col].sum()
        print(f"{name:<35} {mean_val:>10.1f} {median_val:>10.1f} {max_val:>10.1f} {total_val:>12.0f}")

    # Per-organelle breakdown for batch_extract (regionprops)
    batch_cols = [c for c in timing_cols if c.startswith("_timing_batch_") and c != "_timing_batch_extract_ms"]
    if batch_cols:
        print(f"\n{'Batch extract by organelle':<35} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total':>12}")
        print("-" * 70)
        batch_totals = [(c, timing_df[c].sum()) for c in batch_cols]
        batch_totals.sort(key=lambda x: -x[1])
        for col, _ in batch_totals[:10]:  # Top 10 slowest
            name = col.replace("_timing_batch_", "").replace("_ms", "")
            mean_val = timing_df[col].mean()
            median_val = timing_df[col].median()
            max_val = timing_df[col].max()
            total_val = timing_df[col].sum()
            print(f"  {name:<33} {mean_val:>10.1f} {median_val:>10.1f} {max_val:>10.1f} {total_val:>12.0f}")

    # Per-organelle breakdown for localization
    loc_cols = [c for c in timing_cols if c.startswith("_timing_loc_")]
    if loc_cols:
        print(f"\n{'Localization by organelle':<35} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total':>12}")
        print("-" * 70)
        loc_totals = [(c, timing_df[c].sum()) for c in loc_cols]
        loc_totals.sort(key=lambda x: -x[1])
        for col, _ in loc_totals[:10]:  # Top 10 slowest
            name = col.replace("_timing_loc_", "").replace("_ms", "")
            mean_val = timing_df[col].mean()
            median_val = timing_df[col].median()
            max_val = timing_df[col].max()
            total_val = timing_df[col].sum()
            print(f"  {name:<33} {mean_val:>10.1f} {median_val:>10.1f} {max_val:>10.1f} {total_val:>12.0f}")

    # Per-organelle breakdown for network
    net_cols = [c for c in timing_cols if c.startswith("_timing_net_")]
    if net_cols:
        print(f"\n{'Network analysis by organelle':<35} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total':>12}")
        print("-" * 70)
        net_totals = [(c, timing_df[c].sum()) for c in net_cols]
        net_totals.sort(key=lambda x: -x[1])
        for col, _ in net_totals[:10]:  # Top 10 slowest
            name = col.replace("_timing_net_", "").replace("_ms", "")
            mean_val = timing_df[col].mean()
            median_val = timing_df[col].median()
            max_val = timing_df[col].max()
            total_val = timing_df[col].sum()
            print(f"  {name:<33} {mean_val:>10.1f} {median_val:>10.1f} {max_val:>10.1f} {total_val:>12.0f}")

    # Print total CPU time
    if "_timing_total_ms" in timing_df.columns:
        total_sec = timing_df["_timing_total_ms"].sum() / 1000
        n_cells = len(timing_df)
        print(f"\n{'='*70}")
        print(f"Total: {n_cells} cells, {total_sec:.1f}s CPU time, {n_cells/total_sec:.1f} cells/s effective rate")
        print(f"{'='*70}\n")


def _save_intermediate_results(self, well: str = None, batch_id: str = None):
    """
    Save intermediate cell-level results as parquet for SLURM aggregation.
    
    Parameters
    ----------
    well : str, optional
        Well identifier (legacy per-well mode)
    batch_id : str, optional
        Batch identifier (new per-batch mode, format: "{well}_{idx}")
    """
    print("Saving intermediate results as parquet...")

    # Use batch_results directory for batch mode, well_results for legacy mode
    if batch_id is not None:
        intermediate_dir = self.save_dir / "_batch_results"
        filename = f"batch_{batch_id}_cells.parquet"
    elif well:
        intermediate_dir = self.save_dir / "_well_results"
        well_safe = well.replace("/", "_")
        filename = f"{well_safe}_cells.parquet"
    else:
        intermediate_dir = self.save_dir / "_well_results"
        filename = "all_cells.parquet"

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    if self.results["cell"] is not None:
        path = intermediate_dir / filename
        print(f"  -> {path}")

        df = self.results["cell"].copy()
        string_cols = ["gene_name", "barcode", "sgRNA", "well", "gene_effect", "NCBI_ID"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} cell features to {path}")

        # Also save timing data if available
        if hasattr(self, 'results') and self.results.get("timing_data") is not None:
            timing_path = intermediate_dir / filename.replace("_cells.parquet", "_timing.parquet")
            self.results["timing_data"].to_parquet(timing_path, index=False)
            print(f"Saved timing data to {timing_path}")
    else:
        print("Warning: No cell results to save.")


def _save_results_as_anndata(self):
    """
    Save results as AnnData objects at cell, guide, and gene levels.

    Uses feature_metadata dict (built during aggregation) to populate var DataFrame.
    """
    print("Saving feature tables as AnnData objects...")
    experiment_name = self.dataset.experiment

    # Get feature metadata built during aggregation
    feature_metadata = self.results.get("feature_metadata", {})

    # String column defaults for sanitization
    string_cols_defaults = {
        "gene_name": "NTC",
        "barcode": "",
        "sgRNA": "",
        "gene_effect": "",
        "NCBI_ID": "",
        "well": "",
        "store_key": "",
        "bbox": "",
        "tile_path": "",
        "dep_map_gene_name": "",
        "subpool": "",
        "cp_bbox": "",
    }

    try:
        # --- Cell-level AnnData ---
        if self.results["cell"] is not None:
            cell_df = self.results["cell"].copy()

            # Feature columns are those with metadata entries
            feature_cols = [c for c in cell_df.columns if c in feature_metadata]

            # Obs columns are everything else
            obs_cols = [c for c in cell_df.columns if c not in feature_cols]

            # Create obs DataFrame
            obs_df = cell_df[obs_cols].copy()
            if "cell_id" in obs_df.columns:
                obs_df.index = obs_df["cell_id"].astype(str)
            else:
                obs_df.index = obs_df.index.astype(str)

            # Standardize coordinate column names
            # Handle multiple naming conventions from different pipelines:
            # - Standard linked_results: x_pheno, y_pheno
            # - Cell painting linked: x_pheno_centroid, y_pheno_centroid
            # - CP1 coordinates: x_cp1, y_cp1 (scaled to full res)
            coord_renames = {
                "x_pheno": "x_global_pheno",
                "y_pheno": "y_global_pheno",
                "x_pheno_centroid": "x_global_pheno",
                "y_pheno_centroid": "y_global_pheno",
            }
            for old_name, new_name in coord_renames.items():
                if old_name in obs_df.columns and new_name not in obs_df.columns:
                    obs_df.rename(columns={old_name: new_name}, inplace=True)

            # Ensure numeric coord columns
            numeric_coord_cols = [
                "x_global_pheno", "y_global_pheno", "x_local_pheno", "y_local_pheno",
                "tile_pheno", "segmentation_id", "cp_cell_seg_id", "og_index",
            ]
            for col in numeric_coord_cols:
                if col in obs_df.columns:
                    obs_df[col] = pd.to_numeric(obs_df[col], errors="coerce")

            # Sanitize string columns
            for col, default in string_cols_defaults.items():
                if col in obs_df.columns:
                    obs_df[col] = obs_df[col].fillna(default).astype(str)

            # Convert bbox columns (may contain tuples) to strings for h5py compatibility
            for bbox_col in ['bbox', 'cp_bbox']:
                if bbox_col in obs_df.columns:
                    def bbox_to_str(x):
                        if pd.isna(x) or x is None:
                            return ""
                        if isinstance(x, (tuple, list, np.ndarray)):
                            return str(tuple(x))
                        return str(x)
                    obs_df[bbox_col] = obs_df[bbox_col].apply(bbox_to_str)

            for col in obs_df.columns:
                if obs_df[col].dtype == object:
                    obs_df[col] = obs_df[col].fillna("").astype(str)

            # Create X matrix
            X = cell_df[feature_cols].values.astype(np.float32)

            # Create var DataFrame from feature_metadata
            var_df = pd.DataFrame(index=feature_cols)
            var_df["organelle"] = [feature_metadata[f]["organelle"] for f in feature_cols]
            var_df["metric"] = [feature_metadata[f]["metric"] for f in feature_cols]
            var_df["category"] = [feature_metadata[f]["category"] for f in feature_cols]
            var_df["aggregation"] = [feature_metadata[f].get("aggregation") for f in feature_cols]
            var_df["unit"] = [get_unit_for_metric(feature_metadata[f]["metric"]) for f in feature_cols]

            # 4i experiments save with 4i-prefixed metadata column names
            # instead of cp_-prefixed ones. The internal pipeline uses
            # cp_* names uniformly (legacy from when only CP was supported)
            # but the saved h5ad should reflect the actual imaging modality.
            # Detection: any var organelle starting with '4i_' marks this
            # as a 4i experiment.
            org_lower = var_df["organelle"].astype(str).str.lower()
            is_4i_exp = bool(org_lower.str.startswith("4i_").any())
            if is_4i_exp:
                obs_renames = {
                    "cp_cell_seg_id": "4i_cell_seg_id",
                    "cp_bbox": "4i_bbox",
                    "cp1_idx": "4i_idx",
                    "cp1_label": "4i_label",
                    "x_cp1": "x_4i",
                    "y_cp1": "y_4i",
                    "cp1_pheno_distance": "4i_pheno_distance",
                    "cp1_iss_distance": "4i_iss_distance",
                    "cp1_cp2_distance": "4i_r1_r2_distance",
                }
                obs_renames = {k: v for k, v in obs_renames.items() if k in obs_df.columns}
                if obs_renames:
                    obs_df = obs_df.rename(columns=obs_renames)
                    print(f"  [4i] renamed obs columns for 4i experiment: "
                          f"{list(obs_renames.keys())} -> {list(obs_renames.values())}")

            cell_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
            cell_adata.uns["creation_date"] = datetime.now().isoformat()
            cell_adata.uns["experiment"] = experiment_name
            cell_adata.uns["level"] = "cell"
            cell_adata.uns["aggregation_functions"] = AGG_FUNCS + ["count"]

            # Store unique organelles list for easy access
            cell_adata.uns["organelles"] = sorted(var_df["organelle"].dropna().unique().tolist())

            path = self.save_dir / f"{experiment_name}_cell_features.h5ad"
            print(f"  -> {path}")
            cell_adata.write_h5ad(path)

        # --- Guide-level AnnData ---
        if self.results["guideRNA"] is not None:
            guide_df = self.results["guideRNA"].copy()

            # Feature columns have aggregation suffixes
            guide_feature_cols = [
                c for c in guide_df.columns
                if pd.api.types.is_numeric_dtype(guide_df[c])
                and any(c.endswith(f"_{agg}") for agg in AGGREGATION_FUNCTIONS)
            ]

            guide_obs_cols = [c for c in guide_df.columns if c not in guide_feature_cols]

            guide_obs_df = guide_df[guide_obs_cols].copy()
            if "barcode" in guide_obs_df.columns:
                guide_obs_df.index = guide_obs_df["barcode"].astype(str)

            # Sanitize string columns (known columns with defaults)
            for col, default in string_cols_defaults.items():
                if col in guide_obs_df.columns:
                    guide_obs_df[col] = guide_obs_df[col].fillna(default).astype(str)
            # Sanitize ALL remaining object columns (like barcode_from_iss)
            for col in guide_obs_df.columns:
                if guide_obs_df[col].dtype == object:
                    guide_obs_df[col] = guide_obs_df[col].fillna("").astype(str)

            if self.results["cell"] is not None:
                cell_counts = self.results["cell"].groupby("barcode").size()
                guide_obs_df["n_cells"] = guide_obs_df.index.map(lambda x: cell_counts.get(x, 0))

            X_guide = guide_df[guide_feature_cols].values.astype(np.float32)

            # Build var from cell-level metadata (guide features are aggregations of cell features)
            var_guide_df = pd.DataFrame(index=guide_feature_cols)
            # For guide level, we need to map back to cell features
            # Guide feature = cell_feature + "_" + guide_agg
            var_guide_df["organelle"] = None
            var_guide_df["metric"] = None
            var_guide_df["category"] = None
            var_guide_df["aggregation"] = None
            var_guide_df["unit"] = None

            for gf in guide_feature_cols:
                # Find the cell-level feature this came from
                for agg in AGGREGATION_FUNCTIONS:
                    if gf.endswith(f"_{agg}"):
                        cell_feat = gf[:-(len(agg)+1)]
                        if cell_feat in feature_metadata:
                            var_guide_df.loc[gf, "organelle"] = feature_metadata[cell_feat]["organelle"]
                            var_guide_df.loc[gf, "metric"] = feature_metadata[cell_feat]["metric"]
                            var_guide_df.loc[gf, "category"] = feature_metadata[cell_feat]["category"]
                            var_guide_df.loc[gf, "aggregation"] = agg
                            var_guide_df.loc[gf, "unit"] = get_unit_for_metric(feature_metadata[cell_feat]["metric"])
                        break

            guide_adata = ad.AnnData(X=X_guide, obs=guide_obs_df, var=var_guide_df)
            guide_adata.uns["creation_date"] = datetime.now().isoformat()
            guide_adata.uns["experiment"] = experiment_name
            guide_adata.uns["level"] = "guide"

            path = self.save_dir / f"{experiment_name}_guide_features.h5ad"
            print(f"  -> {path}")
            guide_adata.write_h5ad(path)

        # --- Gene-level AnnData ---
        if self.results["gene"] is not None:
            gene_df = self.results["gene"].copy()

            gene_feature_cols = [
                c for c in gene_df.columns
                if pd.api.types.is_numeric_dtype(gene_df[c])
                and any(c.endswith(f"_{agg}") for agg in AGGREGATION_FUNCTIONS)
            ]

            gene_obs_cols = [c for c in gene_df.columns if c not in gene_feature_cols]

            gene_obs_df = gene_df[gene_obs_cols].copy()
            # Sanitize string columns (known columns with defaults)
            for col, default in string_cols_defaults.items():
                if col in gene_obs_df.columns:
                    gene_obs_df[col] = gene_obs_df[col].fillna(default).astype(str)
            # Sanitize ALL remaining object columns
            for col in gene_obs_df.columns:
                if gene_obs_df[col].dtype == object:
                    gene_obs_df[col] = gene_obs_df[col].fillna("").astype(str)

            if "gene_name" in gene_obs_df.columns:
                gene_obs_df.index = gene_obs_df["gene_name"].astype(str)

            if self.results["cell"] is not None:
                cell_counts_per_gene = self.results["cell"].groupby("gene_name").size()
                gene_obs_df["n_cells"] = gene_obs_df.index.map(lambda x: cell_counts_per_gene.get(x, 0))
            if self.results["guideRNA"] is not None:
                guide_counts_per_gene = self.results["guideRNA"].groupby("gene_name").size()
                gene_obs_df["n_guides"] = gene_obs_df.index.map(lambda x: guide_counts_per_gene.get(x, 0))

            X_gene = gene_df[gene_feature_cols].values.astype(np.float32)

            var_gene_df = pd.DataFrame(index=gene_feature_cols)
            var_gene_df["organelle"] = None
            var_gene_df["metric"] = None
            var_gene_df["category"] = None
            var_gene_df["aggregation"] = None
            var_gene_df["unit"] = None

            for gf in gene_feature_cols:
                for agg in AGGREGATION_FUNCTIONS:
                    if gf.endswith(f"_{agg}"):
                        cell_feat = gf[:-(len(agg)+1)]
                        if cell_feat in feature_metadata:
                            var_gene_df.loc[gf, "organelle"] = feature_metadata[cell_feat]["organelle"]
                            var_gene_df.loc[gf, "metric"] = feature_metadata[cell_feat]["metric"]
                            var_gene_df.loc[gf, "category"] = feature_metadata[cell_feat]["category"]
                            var_gene_df.loc[gf, "aggregation"] = agg
                            var_gene_df.loc[gf, "unit"] = get_unit_for_metric(feature_metadata[cell_feat]["metric"])
                        break

            gene_adata = ad.AnnData(X=X_gene, obs=gene_obs_df, var=var_gene_df)
            gene_adata.uns["creation_date"] = datetime.now().isoformat()
            gene_adata.uns["experiment"] = experiment_name
            gene_adata.uns["level"] = "gene"

            path = self.save_dir / f"{experiment_name}_gene_features.h5ad"
            print(f"  -> {path}")
            gene_adata.write_h5ad(path)

        print("All AnnData files saved successfully.")
        return True

    except Exception as e:
        import traceback
        print(f"ERROR: Failed to save AnnData files: {e}")
        print(traceback.format_exc())
        return False


def _get_metric_category(metric_name):
    """Determine feature category from metric name."""
    if "intensity" in metric_name:
        return CATEGORY_INTENSITY
    elif "distance_from" in metric_name or "normalized_radial" in metric_name:
        return CATEGORY_LOCALIZATION
    else:
        return CATEGORY_MORPHOLOGY


def _aggregate_one_organelle(work):
    """
    Aggregate features for a single organelle. Module-level for ProcessPoolExecutor pickling.

    Parameters
    ----------
    work : tuple
        (organelle_name, obj_dfs, net_dfs, per_obj_dfs, is_network, cell_area_series, agg_funcs)

    Returns
    -------
    tuple
        (organelle_name, agg_frames, metadata, raw_dfs)
    """
    organelle_name, obj_dfs, net_dfs, per_obj_dfs, is_network, cell_area_series, agg_funcs = work

    agg_frames = []
    metadata = {}
    raw_dfs = {}

    # --- Aggregate standard object features (morphology, intensity) ---
    if obj_dfs:
        organelle_df = pd.concat(obj_dfs, ignore_index=True)
        if not organelle_df.empty:
            raw_dfs["object"] = organelle_df

            exclude_cols = {"cell_id", "label", "total_index"}
            feature_cols = [col for col in organelle_df.columns if col not in exclude_cols]

            object_agg_dict = {col: agg_funcs for col in feature_cols}

            organelle_summary_df = organelle_df.groupby("cell_id").agg(object_agg_dict)
            organelle_summary_df.columns = [
                "_".join(col).strip() for col in organelle_summary_df.columns.values
            ]

            agg_stats = organelle_summary_df
            agg_stats.columns = [f"{organelle_name}_{col}" for col in agg_stats.columns]

            # Register metadata for each aggregated feature
            for col in agg_stats.columns:
                parts = col.split("_")
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i] in agg_funcs:
                        agg = parts[i]
                        metric = "_".join(parts[len(organelle_name.split("_")):i])
                        metadata[col] = {
                            "organelle": organelle_name,
                            "metric": metric,
                            "category": _get_metric_category(metric),
                            "aggregation": agg,
                        }
                        break

            # Add count
            count_col = f"{organelle_name}_count"
            agg_stats[count_col] = organelle_df.groupby("cell_id").size()
            metadata[count_col] = {
                "organelle": organelle_name,
                "metric": "count",
                "category": CATEGORY_MORPHOLOGY,
                "aggregation": None,
            }

            # Add area_fraction if cell_area exists
            if cell_area_series is not None:
                total_area_col = f"{organelle_name}_area_sum"
                if total_area_col in agg_stats.columns:
                    frac_col = f"{organelle_name}_area_fraction"
                    agg_stats[frac_col] = agg_stats[total_area_col].div(cell_area_series)
                    metadata[frac_col] = {
                        "organelle": organelle_name,
                        "metric": "area_fraction",
                        "category": CATEGORY_MORPHOLOGY,
                        "aggregation": None,
                    }

            agg_frames.append(agg_stats)

    # --- Aggregate network features (branch-level) ---
    if is_network and net_dfs:
        full_df = pd.concat(net_dfs, ignore_index=True)
        if not full_df.empty:
            raw_dfs["network"] = full_df

            exclude_cols = {"cell_id", "total_index"}
            feature_cols = [
                col for col in full_df.select_dtypes(include=np.number).columns
                if col not in exclude_cols
            ]
            network_agg_dict = {col: agg_funcs for col in feature_cols}

            if network_agg_dict:
                agg_stats = full_df.groupby("cell_id").agg(network_agg_dict)
                agg_stats.columns = [
                    f"network_{organelle_name}_{col[0]}_{col[1]}"
                    for col in agg_stats.columns
                ]

                # Register metadata
                for col in agg_stats.columns:
                    rest = col[8:]  # Remove "network_"
                    for agg in agg_funcs:
                        if rest.endswith(f"_{agg}"):
                            metric_part = rest[:-(len(agg) + 1)]
                            if metric_part.startswith(organelle_name + "_"):
                                metric = metric_part[len(organelle_name) + 1:]
                            else:
                                metric = metric_part
                            metadata[col] = {
                                "organelle": organelle_name,
                                "metric": metric,
                                "category": CATEGORY_NETWORK,
                                "aggregation": agg,
                            }
                            break

                # Add branch count
                count_col = f"network_{organelle_name}_branch_count"
                agg_stats[count_col] = full_df.groupby("cell_id").size()
                metadata[count_col] = {
                    "organelle": organelle_name,
                    "metric": "branch_count",
                    "category": CATEGORY_NETWORK,
                    "aggregation": None,
                }

                agg_frames.append(agg_stats)

    # --- Per-object network features: DISABLED ---
    # Network analysis binarizes the organelle mask (network_analysis.py:200) and
    # skeletonizes the whole cell's organelles as one network. This means per-object
    # features (per connected component) almost always have exactly 1 row per cell,
    # making all aggregations (mean, median, min, max, sum) identical copies of the
    # same value and std always NaN. The network summary features (num_branches,
    # num_nodes, euler_number, etc.) already capture the whole-network metrics, and
    # branch-level features (branch_length, branch_thickness, tortuosity) provide
    # real distributional information across the many branches within the network.
    # Per-object aggregation would only be meaningful if network analysis was
    # redesigned to skeletonize each segmented instance separately.

    return (organelle_name, agg_frames, metadata, raw_dfs)


def _generate_summaries(
    results: dict,
    cell_features_list: list,
    object_features_dict: dict,
    network_features_dict: dict,
    per_object_network_dict: dict,
    all_organelles: list,
    network_organelles: list,
    metadata_df: pd.DataFrame = None,
    cell_only: bool = False,
):
    """
    Generate cell, gene, and guideRNA summary tables from the collected feature data.

    Builds feature_metadata dict during aggregation for proper AnnData var population.
    """
    if isinstance(cell_features_list, pd.DataFrame):
        if cell_features_list.empty:
            return results
    elif not cell_features_list:
        return results

    agg_funcs = AGG_FUNCS

    # Feature metadata: col_name -> {organelle, metric, category, aggregation}
    feature_metadata = {}

    print("  -> Step 1/3: Creating base cell summary dataframe...")
    if isinstance(cell_features_list, pd.DataFrame):
        cell_summary_df = cell_features_list.set_index("cell_id")
    else:
        cell_summary_df = pd.DataFrame(cell_features_list).set_index("cell_id")

    # Extract and save timing columns before removing them
    internal_cols = [c for c in cell_summary_df.columns if c.startswith("_")]
    if internal_cols:
        timing_df = cell_summary_df[internal_cols].copy()
        timing_df["cell_id"] = cell_summary_df.index
        # Print timing summary
        print(f"     Extracted {len(internal_cols)} internal timing columns.")
        _print_timing_summary(timing_df)
        # Save timing data for profiling analysis
        results["timing_data"] = timing_df
        print(f"     Removing timing columns from main output.")
        cell_summary_df = cell_summary_df.drop(columns=internal_cols)

    # Register cell morphology features (cell_area, cell_perimeter, etc.)
    for col in cell_summary_df.columns:
        if col.startswith("cell_"):
            metric = col[5:]  # Remove "cell_" prefix
            feature_metadata[col] = {
                "organelle": "cell",
                "metric": metric,
                "category": CATEGORY_CELL_MORPHOLOGY,
                "aggregation": None,
            }
        elif col.startswith("cp_cell_"):
            metric = col[8:]  # Remove "cp_cell_" prefix
            feature_metadata[col] = {
                "organelle": "cp_cell",
                "metric": metric,
                "category": CATEGORY_CELL_MORPHOLOGY,
                "aggregation": None,
            }

    print(f"     Base cell dataframe created with shape {cell_summary_df.shape}.")

    if not cell_summary_df.index.is_unique:
        duplicates = cell_summary_df.index[cell_summary_df.index.duplicated()].unique()
        print(f"Warning: Found {len(duplicates)} duplicate cell_ids.")
        cell_summary_df = cell_summary_df[~cell_summary_df.index.duplicated(keep="first")]

    print("  -> Step 2/3: Processing and aggregating all features...")
    all_organelles_to_process = [org for org in all_organelles if org not in ("cell_mask", "cp_cell_mask")]

    import time as _time_mod
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cell_area_series = cell_summary_df["cell_area"] if "cell_area" in cell_summary_df.columns else None

    # Build per-organelle work items: only pass data each organelle needs (avoids pickling everything)
    organelle_work = []
    for organelle_name in all_organelles_to_process:
        obj_dfs = object_features_dict.get(organelle_name)
        is_network = organelle_name in network_organelles
        net_dfs = network_features_dict.get(organelle_name) if is_network else None
        per_obj_dfs = per_object_network_dict.get(organelle_name) if is_network else None
        organelle_work.append((organelle_name, obj_dfs, net_dfs, per_obj_dfs, is_network, cell_area_series, agg_funcs))

    # Process all organelles in parallel using processes (GIL-free)
    t_agg_start = _time_mod.time()
    n_workers = min(len(all_organelles_to_process), 18)
    all_agg_frames = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_aggregate_one_organelle, work): work[0] for work in organelle_work}
        for future in as_completed(futures):
            organelle_name, org_agg_frames, org_metadata, org_raw = future.result()
            # Collect raw DataFrames for results dict
            if "object" in org_raw:
                results["object"][organelle_name] = org_raw["object"]
            if "network" in org_raw:
                results["network"][organelle_name] = org_raw["network"]
            # Collect metadata
            feature_metadata.update(org_metadata)
            # Collect aggregated frames for joining
            all_agg_frames.extend(org_agg_frames)

    # Join all aggregated frames at once (serial, but fast since it's just index alignment)
    if all_agg_frames:
        combined_agg = pd.concat(all_agg_frames, axis=1)
        cell_summary_df = cell_summary_df.join(combined_agg)

    t_agg_elapsed = _time_mod.time() - t_agg_start
    print(f"     Aggregated {len(all_organelles_to_process)} organelles in {t_agg_elapsed:.1f}s ({n_workers} processes)")

    # Also register localization features already in cell_features_list
    contact_metrics = ["overlap_area", "overlap_frac_a", "overlap_frac_b", "n_contacts"]
    for col in cell_summary_df.columns:
        if col not in feature_metadata and not col.startswith("_") and col != "well":
            # Check if it's a localization feature
            loc_metrics = ["distance_from_cell_edge", "distance_from_nucleus",
                          "distance_from_nucleus_centroid", "normalized_radial_position"]
            for loc in loc_metrics:
                if f"_{loc}_" in col:
                    # Parse: {organelle}_{loc_metric}_{agg}
                    for agg in agg_funcs:
                        if col.endswith(f"_{agg}"):
                            idx = col.find(f"_{loc}_")
                            organelle = col[:idx]
                            feature_metadata[col] = {
                                "organelle": organelle,
                                "metric": loc,
                                "category": CATEGORY_LOCALIZATION,
                                "aggregation": agg,
                            }
                            break
                    break

            if col in feature_metadata:
                continue

            # Inter-organelle contact: contact_{A}__{B}_{metric} (cell-level scalar).
            # organelle = the "{A}__{B}" pair string; metric = the trailing suffix.
            if col.startswith("contact_"):
                for cm in contact_metrics:
                    if col.endswith(f"_{cm}"):
                        pair = col[len("contact_"):-(len(cm) + 1)]
                        feature_metadata[col] = {
                            "organelle": pair,
                            "metric": cm,
                            "category": CATEGORY_CONTACT,
                            "aggregation": None,
                        }
                        break
                continue

            # Radial distribution: {organelle}_radial_frac_bin{i} / {organelle}_radial_anisotropy
            for dist_metric in ("_radial_frac_bin", "_radial_anisotropy"):
                idx = col.find(dist_metric)
                if idx != -1:
                    feature_metadata[col] = {
                        "organelle": col[:idx],
                        "metric": col[idx + 1:],
                        "category": CATEGORY_DISTRIBUTION,
                        "aggregation": None,
                    }
                    break

    print("  -> Step 3/3: Generating final summaries and joining metadata...")

    cell_features_df = cell_summary_df.reset_index().fillna(0)

    # Join metadata
    if metadata_df is not None and not metadata_df.empty:
        if "global_cell_id" not in metadata_df.columns:
            metadata_df = metadata_df.copy()
            create_global_cell_id(metadata_df)

        rename_cols = {}
        if "x_pheno" in metadata_df.columns and "x_global_pheno" not in metadata_df.columns:
            rename_cols["x_pheno"] = "x_global_pheno"
        if "y_pheno" in metadata_df.columns and "y_global_pheno" not in metadata_df.columns:
            rename_cols["y_pheno"] = "y_global_pheno"
        if rename_cols:
            metadata_df = metadata_df.rename(columns=rename_cols)

        if "well" in cell_features_df.columns:
            cell_features_df = cell_features_df.drop(columns=["well"])

        meta_renamed = metadata_df.rename(columns={"global_cell_id": "cell_id"})
        # Ensure cell_id types match for merge (features may be int, metadata may be str)
        if cell_features_df["cell_id"].dtype != meta_renamed["cell_id"].dtype:
            # Cast both to string for safe merge
            cell_features_df = cell_features_df.copy()
            cell_features_df["cell_id"] = cell_features_df["cell_id"].astype(str)
            meta_renamed["cell_id"] = meta_renamed["cell_id"].astype(str)
        results["cell"] = pd.merge(
            cell_features_df,
            meta_renamed,
            on="cell_id",
            how="left"
        )
    else:
        results["cell"] = cell_features_df

    # Store feature metadata in results
    results["feature_metadata"] = feature_metadata

    if cell_only:
        print("Summary generation complete (cell_only mode, skipped guide/gene).")
        return results

    # Get feature columns for aggregation
    numeric_cols = results["cell"].select_dtypes(include=np.number).columns.tolist()
    metadata_patterns = {
        "x_global", "y_global", "x_local", "y_local", "x_pheno", "y_pheno",
        "x_bc", "y_bc", "tile", "segmentation_id", "og_index", "total_index",
        "cp_cell_seg_id", "_idx", "_label"
    }
    feature_cols_for_agg = [
        c for c in numeric_cols
        if c != "cell_id" and c in feature_metadata
    ]

    # GuideRNA summary
    # Use sgRNA as the grouping key (not barcode) - sgRNA is the true guide identifier
    # and is consistent across different barcode lengths. Barcode can vary in length
    # depending on effective rounds, causing duplicate guide entries.
    if "sgRNA" in results["cell"].columns and feature_cols_for_agg:
        cells_for_guide = results["cell"]
        # Filter to cells with valid sgRNA
        valid_sgrna_mask = cells_for_guide["sgRNA"].notna() & (cells_for_guide["sgRNA"] != "") & (cells_for_guide["sgRNA"].astype(str) != "None")
        n_filtered = (~valid_sgrna_mask).sum()
        if n_filtered > 0:
            print(f"  Filtering {n_filtered} cells without valid sgRNA from guide aggregation")
        cells_for_guide = cells_for_guide[valid_sgrna_mask]

        guide_summary = cells_for_guide.groupby("sgRNA")[feature_cols_for_agg].agg(agg_funcs)
        guide_summary.columns = ["_".join(col).strip() for col in guide_summary.columns.values]
        guide_summary = guide_summary.reset_index()

        std_cols = [c for c in guide_summary.columns if c.endswith("_std")]
        for col in std_cols:
            guide_summary[col] = guide_summary[col].fillna(0)

        # Include ALL non-feature metadata columns (not just hardcoded subset)
        all_cell_cols = set(cells_for_guide.columns)
        feature_col_set = set(feature_cols_for_agg) | {"cell_id", "sgRNA"}
        guide_metadata_cols = [c for c in all_cell_cols if c not in feature_col_set]
        if guide_metadata_cols:
            guide_metadata = cells_for_guide.groupby("sgRNA")[guide_metadata_cols].first().reset_index()
            results["guideRNA"] = pd.merge(guide_summary, guide_metadata, on="sgRNA", how="left")
        else:
            results["guideRNA"] = guide_summary
    elif "barcode" in results["cell"].columns and feature_cols_for_agg:
        # Fallback to barcode if no sgRNA column (legacy support)
        cells_for_guide = results["cell"]
        guide_summary = cells_for_guide.groupby("barcode")[feature_cols_for_agg].agg(agg_funcs)
        guide_summary.columns = ["_".join(col).strip() for col in guide_summary.columns.values]
        guide_summary = guide_summary.reset_index()

        std_cols = [c for c in guide_summary.columns if c.endswith("_std")]
        for col in std_cols:
            guide_summary[col] = guide_summary[col].fillna(0)

        all_cell_cols = set(cells_for_guide.columns)
        feature_col_set = set(feature_cols_for_agg) | {"cell_id", "barcode"}
        guide_metadata_cols = [c for c in all_cell_cols if c not in feature_col_set]
        if guide_metadata_cols:
            guide_metadata = cells_for_guide.groupby("barcode")[guide_metadata_cols].first().reset_index()
            results["guideRNA"] = pd.merge(guide_summary, guide_metadata, on="barcode", how="left")
        else:
            results["guideRNA"] = guide_summary
    else:
        results["guideRNA"] = None

    # Gene summary
    if "gene_name" in results["cell"].columns and feature_cols_for_agg:
        gene_summary = results["cell"].groupby("gene_name")[feature_cols_for_agg].agg(agg_funcs)
        gene_summary.columns = ["_".join(col).strip() for col in gene_summary.columns.values]
        gene_summary = gene_summary.reset_index()

        std_cols = [c for c in gene_summary.columns if c.endswith("_std")]
        for col in std_cols:
            gene_summary[col] = gene_summary[col].fillna(0)

        # Include ALL non-feature metadata columns (not just hardcoded subset)
        all_cell_cols = set(results["cell"].columns)
        feature_col_set = set(feature_cols_for_agg) | {"cell_id", "gene_name"}
        gene_metadata_cols = [c for c in all_cell_cols if c not in feature_col_set]
        if gene_metadata_cols:
            gene_metadata = results["cell"].groupby("gene_name")[gene_metadata_cols].first().reset_index()
            results["gene"] = pd.merge(gene_summary, gene_metadata, on="gene_name", how="left")
        else:
            results["gene"] = gene_summary
    else:
        results["gene"] = None

    print("Summary generation complete.")
    return results
