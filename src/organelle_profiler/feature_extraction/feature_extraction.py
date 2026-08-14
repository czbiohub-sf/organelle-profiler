"""
Feature Extraction Pipeline for OPS Phenotyping Data.

Extracts morphological, intensity, texture, and network features from segmented
organelles in zarr v3 stores. Outputs AnnData (.h5ad) files at three aggregation
levels: cell, guide, and gene.

The pipeline automatically discovers available segmentation labels from the zarr
store, including cell_seg, nuclear_seg, and any organelle-specific segmentations
(e.g., mitochondria_tomm20_seg, phase_2d_tubular_seg).

Output Files:
-------------
Three AnnData files are generated in the experiment's 4-features directory:
  - {experiment}_cell_features.h5ad:  Per-cell aggregated features
  - {experiment}_guide_features.h5ad: Per-guide aggregated features
  - {experiment}_gene_features.h5ad:  Per-gene aggregated features

Each file contains:
  - X: Feature matrix (n_samples × n_features*7 aggregation functions)
  - obs: Sample metadata (cell/guide/gene identifiers, well, barcode, etc.)
  - var: Feature metadata (organelle, base metric, aggregation function)
  - obsm: Embedding coordinates (X_umap, X_pca added by fe_graphs.py)
  - uns: Processing metadata (creation date, experiment, version)

Usage:
------
# Dry run - preview what segmentations would be discovered and extracted
python -m organelle_profiler.feature_extraction.feature_extraction -e 94 --dry-run

# Run feature extraction for an experiment (shorthand resolves to full name)
python -m organelle_profiler.feature_extraction.feature_extraction -e 94

# Run with full experiment name
python -m organelle_profiler.feature_extraction.feature_extraction -e ops0094_20251217

# Run with full features (includes texture features - slower)
python -m organelle_profiler.feature_extraction.feature_extraction -e 94 --full-features

# Debug with limited tiles
python -m organelle_profiler.feature_extraction.feature_extraction -e 94 --debug_tile_count 10

Discovered Labels:
------------------
The pipeline discovers segmentation labels from the zarr v3 store's labels group:
  - cell_seg: Cell segmentation mask (required for cell-level metrics)
  - nuclear_seg: Nuclear segmentation mask (mapped as "nuclei")
  - Organelle labels: Parsed from naming convention {organelle}_{marker}_seg
    e.g., mitochondria_tomm20_seg, er_sec61b_tubular_seg, phase_2d_vesicular_seg

Feature Types:
--------------
  - Morphological: area, perimeter, eccentricity, solidity, extent, etc.
  - Intensity: mean, max, min, std intensity per channel
  - Texture: GLCM features (contrast, correlation, homogeneity) - with --full-features
  - Network: connectivity metrics for network-forming organelles (ER, mitochondria)

Aggregation:
------------
Object/organelle-level metrics are aggregated to cell level using 7 functions:
  sum, mean, median, std, min, max, count
Cell-level metrics are then aggregated to guide and gene levels.
"""

import time
import warnings

# Suppress FutureWarnings (CUDA deprecation) BEFORE any CUDA imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*regions with <=1 background pixel spacing.*")
warnings.filterwarnings("ignore", message="Found.*regions with <=1 background pixel spacing")
warnings.filterwarnings("ignore", message="Input image is entirely zero")

import numpy as np
import pandas as pd
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.bbox_utils import BaseDataset, normalize_bbox
from iohub import open_ome_zarr
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import random

from cyclops_utils.hpc.resource_manager import get_optimal_workers
from cyclops_utils.profiling.decorators import notify_step, versioned_function
from organelle_profiler.feature_extraction.fe_metadata import create_global_cell_id

# Visualization imports
from .fe_visualization import (
    VALIDATION_FEATURES,
    generate_validation_visualizations,
)

from .fe_anndata import (
    _generate_summaries, 
    _save_results_as_anndata, 
    _save_intermediate_results,
)
from .fe_metadata import (
    _discover_available_labels, 
    is_network_organelle, 
    _load_cells_metadata,
    dry_run_discovery,
)
from .fe_workers import _worker_process_well, process_single_cell, extract_cell_features

# =============================================================================
# Legacy FeatureExtractor class (maintained for backward compatibility)
# =============================================================================

class FeatureExtractor:
    """
    Cell-centric feature extraction pipeline for tile-based Zarr datasets.

    This class processes individual cells cropped from a tile-based morphology
    dataset, extracting morphological features for cells and their organelles.

    The pipeline generates a series of CSV files with features at different
    levels of abstraction:
    - Object-level: Features for each individual organelle instance.
    - Cell-level: Aggregated features for each cell.
    - Gene-level: Summary statistics (mean, median, std) grouped by gene.
    - GuideRNA-level: Summary statistics grouped by guideRNA barcode.

    Parameters
    ----------
    experiment : str
        The name of the experiment to process (e.g., 'ops0049_20250626').
    """

    def __init__(self, experiment: str, output_dir: Path = None):
        self.dataset = OpsDataset(experiment)
        # Use the zarr v3 stitched phenotyping store with cell_seg labels
        self.morphology_path = self.dataset.store_paths["pheno_assembled_v3"]

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.dataset.analysis_path

        self.save_dir = self.output_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "object": {},
            "network": {},
            "cell": None,
            "gene": None,
            "guideRNA": None,
        }

    # Static method wrappers for backward compatibility
    @staticmethod
    def _static_process_single_cell(*args, **kwargs):
        """Backward-compatible wrapper for process_single_cell."""
        return process_single_cell(*args, **kwargs)

    @staticmethod
    def _static_extract_cell_features(cell_mask, spacing):
        """Backward-compatible wrapper for extract_cell_features."""
        return extract_cell_features(cell_mask, spacing)

    

   
    def run(
        self,
        organelle_map: dict = None,
        network_organelles: list = None,
        debug_tile_count: int = None,
        full_features: bool = False,
        well: str = None,
        save_intermediate: bool = False,
        validation_samples: int = 6,
        validation_features: list = None,
        preview_cells: int = None,
        debug: bool = None,
        use_cpu: bool = False,
        max_objects_per_organelle: int = 250,
        cells_df_override: "pd.DataFrame" = None,
        batch_id: str = None,
        sequential: bool = False,
    ):
        """
        Execute the full feature extraction pipeline.

        This method processes 'cell_mask' by default and can be extended with
        a list of other organelles to analyze.

        Parameters
        ----------
        organelle_map : dict, optional
            A dictionary mapping organelle names to their channel names, e.g., {'mitochondria': 'GFP'}.
        network_organelles : list, optional
            A list of organelle names to analyze for network/filament features.
        debug_tile_count : int, optional
            If set, only process cells from the first N tiles for debugging.
        full_features : bool, optional
            If True, calculate a larger but more computationally expensive set of features.
        well : str, optional
            If provided, only process cells from this single well (e.g., 'A/1/0').
            Used for per-well SLURM parallelization.
        save_intermediate : bool, optional
            If True, save intermediate cell-level results as parquet instead of
            final AnnData files. Used for SLURM aggregation workflow.
        validation_samples : int, optional
            Number of cells to sample for feature validation visualizations.
            Set to 0 to disable. Default is 12 (for 4x3 grid).
        validation_features : list, optional
            Features to visualize in validation plots. Default is ["area", "perimeter", "axis_major_length"].
        preview_cells : int, optional
            If set, only process this many cells to test the pipeline end-to-end.
            Useful for quick validation that the pipeline works before full runs.
        debug : bool, optional
            If True, print detailed debug information about organelle loading and feature extraction.
            Automatically enabled when preview_cells is set.
        use_cpu : bool, optional
            If True, force CPU-only mode even if GPU is available. Default False.
        sequential : bool, optional
            If True, process cells sequentially (1 worker) instead of parallel.
            For benchmarking to compare parallel vs sequential performance.
        """
        # CPU-only mode
        print("Using CPU mode for feature extraction")

        if validation_features is None:
            validation_features = VALIDATION_FEATURES
        print(
            "Starting tile-based feature extraction pipeline (In-Memory Parallel Version)."
        )
        print(f"Output will be saved to: {self.save_dir}")

        if not self.morphology_path.exists():
            raise FileNotFoundError(
                f"Zarr v3 phenotyping store not found at {self.morphology_path}. Ensure stitching/conversion is complete."
            )

        # Discover all available segmentation labels from the zarr store
        # Skip if already set (e.g., by extract_features_for_batch_direct)
        if not hasattr(self, 'available_labels') or not self.available_labels:
            print("Discovering available segmentation labels from zarr store...")
            self.available_labels = _discover_available_labels(self.morphology_path)
        else:
            print(f"Using pre-discovered labels ({len(self.available_labels)} organelles)")

        # Build organelles_to_process from discovered labels
        # Always include cell_mask and nuclei if available (core segmentations)
        organelles_to_process = []
        if "cell_mask" in self.available_labels:
            organelles_to_process.append("cell_mask")
        if "nuclei" in self.available_labels:
            organelles_to_process.append("nuclei")

        # Add all other discovered organelle segmentations
        # Exclude core/single-object segmentations (cell masks, nuclei)
        # These are used as reference masks, not as organelles with multiple objects
        core_segmentations = {"cell_mask", "nuclei", "cp_cell_mask"}
        for organelle_name in self.available_labels.keys():
            if organelle_name not in core_segmentations:
                organelles_to_process.append(organelle_name)

        # # Also add any organelles from the channel map that might have intensity data
        # # even if they don't have segmentation masks
        # if organelle_map:
        #     for organelle_name in organelle_map.keys():
        #         if organelle_name not in organelles_to_process:
        #             organelles_to_process.append(organelle_name)

        self.network_organelles = network_organelles or []
        if self.network_organelles:
            print(
                f"Attempting to process network features for: {self.network_organelles}"
            )
            for org in self.network_organelles:
                if org not in organelles_to_process:
                    organelles_to_process.append(org)

        organelles_to_process = sorted(list(set(organelles_to_process)))

        # Skip redundant/problematic organelles
        # - CP2_nuclear: Redundant with CP1_nuclear (same Hoechst channel)
        # - focus3d_vesicular, focus3d_vesicular_dark: Skip vesicular segmentations
        skip_organelles = {"CP2_nuclear"} # , "focus3d_vesicular", "focus3d_vesicular_dark"
        organelles_to_process = [org for org in organelles_to_process if org not in skip_organelles]
        skipped = skip_organelles & set(self.available_labels.keys())
        if skipped:
            print(f"Skipping: {skipped}")

        print(f"Organelles to process ({len(organelles_to_process)}): {organelles_to_process}")
        print(f"Available labels mapping: {self.available_labels}")

        # Load cell metadata - either from override (for chunk processing) or from linked_results
        if cells_df_override is not None:
            print(f"Using provided cells_df with {len(cells_df_override)} cells (chunk processing)")
            cells_df = cells_df_override.copy()
        else:
            print("Loading cell metadata from linked_results...")
            cells_df = _load_cells_metadata(self.dataset, self.morphology_path, debug_tile_count=debug_tile_count)

        if len(cells_df) == 0:
            raise ValueError("No cells found.")

        # Store original metadata column names for later use in AnnData creation
        # These columns come from linked_results CSV and should go to obs, not X
        self.metadata_columns = set(cells_df.columns.tolist())

        # Filter cells: keep those with at least ONE valid bbox system
        # - Standard organelles require: segmentation_id + bbox
        # - CP organelles require: cp_cell_seg_id + cp_bbox
        # This maximizes coverage by including cells that may only have one bbox type
        original_count = len(cells_df)

        # Check which cells have valid standard bbox (for phenotyping organelles)
        has_standard_bbox = (
            cells_df["segmentation_id"].notna() &
            cells_df["bbox"].notna()
        )

        # Check which cells have valid CP bbox (for CellPainting organelles)
        # Initialize as all False, then update if columns exist
        has_cp_bbox = pd.Series([False] * len(cells_df), index=cells_df.index)
        if "cp_cell_seg_id" in cells_df.columns and "cp_bbox" in cells_df.columns:
            has_cp_bbox = (
                cells_df["cp_cell_seg_id"].notna() &
                cells_df["cp_bbox"].notna()
            )

        # Keep cells with at least one valid bbox system
        valid_cells_mask = has_standard_bbox | has_cp_bbox
        cells_df = cells_df[valid_cells_mask].copy()

        # Print detailed coverage statistics
        n_filtered = original_count - len(cells_df)
        n_standard_only = (has_standard_bbox & ~has_cp_bbox).sum()
        n_cp_only = (~has_standard_bbox & has_cp_bbox).sum()
        n_both = (has_standard_bbox & has_cp_bbox).sum()

        if n_filtered > 0:
            print(f"Filtered out {n_filtered} cells without any valid bbox.")
        print(f"Cell bbox coverage:")
        print(f"  - Both bboxes (full dual-pass): {n_both}")
        print(f"  - Standard bbox only (pheno organelles): {n_standard_only}")
        print(f"  - CP bbox only (CP organelles): {n_cp_only}")

        # Create a unique global cell ID for indexing
        create_global_cell_id(cells_df)

        print(
            f"Found {len(cells_df)} cells with valid segmentation IDs to process."
        )

        # Filter to single well if specified (for per-well SLURM parallelization)
        if well:
            pre_filter_count = len(cells_df)
            cells_df = cells_df[cells_df["well"] == well].copy()
            if len(cells_df) == 0:
                print(f"Warning: No cells found for well {well}. Skipping.")
                return
            print(f"Filtered to well {well}: {len(cells_df)} cells (from {pre_filter_count})")

        # Preview mode: limit to N cells to test pipeline end-to-end
        self._preview_mode = preview_cells is not None and preview_cells > 0
        if self._preview_mode:
            if len(cells_df) > preview_cells:
                # Sample cells randomly but ensure we get cells from multiple wells if possible
                cells_df = cells_df.sample(n=preview_cells, random_state=42).copy()
            print(f"\n{'='*60}")
            print(f"PREVIEW MODE: Processing only {len(cells_df)} cells")
            print(f"{'='*60}\n")

        # Sample cells for validation visualization
        validation_indices = set()
        if validation_samples > 0 and not save_intermediate:
            n_to_sample = min(validation_samples, len(cells_df))

            # For CP experiments, prefer cells with valid cp_bbox to ensure CP organelles are visualized
            # Check if cp_bbox column exists and filter to cells with valid cp_bbox
            has_cp_bbox_col = "cp_bbox" in cells_df.columns
            if has_cp_bbox_col:
                # Filter to cells with non-null cp_bbox
                cells_with_cp_bbox = cells_df[cells_df["cp_bbox"].notna() & (cells_df["cp_bbox"] != "None")]
                if len(cells_with_cp_bbox) >= n_to_sample:
                    sample_df = cells_with_cp_bbox
                    print(f"  Sampling from {len(cells_with_cp_bbox)} cells with valid cp_bbox (for CP organelle visualization)")
                else:
                    sample_df = cells_df
                    print(f"  Note: Only {len(cells_with_cp_bbox)} cells have cp_bbox, sampling from all cells")
            else:
                sample_df = cells_df

            # Sample from different wells to get diversity
            wells = sample_df["well"].unique()
            cells_per_well = max(1, n_to_sample // len(wells))
            sampled_indices = []
            for w in wells:
                well_indices = sample_df[sample_df["well"] == w].index.tolist()
                if well_indices:
                    n_from_well = min(cells_per_well, len(well_indices))
                    sampled_indices.extend(random.sample(well_indices, n_from_well))
                if len(sampled_indices) >= n_to_sample:
                    break
            validation_indices = set(sampled_indices[:n_to_sample])
            print(f"Sampled {len(validation_indices)} cells for validation visualization.")

        # Enable debug mode automatically for preview runs
        if debug is None:
            debug = preview_cells is not None and preview_cells > 0

        if debug:
            print("\n" + "="*60)
            print("DEBUG MODE ENABLED - Tracing organelle feature extraction")
            print("="*60 + "\n")

        # Process all cells, collecting results in memory
        cell_features, object_features, network_features, per_object_network, viz_data_list = self._process_cells_parallel(
            cells_df,
            organelles_to_process,
            self.network_organelles,
            organelle_map,
            full_features,
            validation_indices=validation_indices,
            debug=debug,
            max_objects_per_organelle=max_objects_per_organelle,
            sequential=sequential,
        )

        if not cell_features:
            print("Warning: No features were extracted from any cell. Aborting.")
            return

        print("Aggregating features and generating summary tables...")
        self.results = _generate_summaries(
            self.results,
            cell_features,
            object_features,
            network_features,
            per_object_network,
            organelles_to_process,
            self.network_organelles,
            metadata_df=cells_df,  # Pass original metadata for joining at the end
        )

        # Print feature summary AFTER aggregation (preview mode only)
        if self.results["cell"] is not None and getattr(self, '_preview_mode', False):
            self._print_feature_summary_from_df(
                self.results["cell"],
                organelles_processed=organelles_to_process,
                network_organelles=self.network_organelles,
            )

        # Save results based on mode
        if save_intermediate:
            # Save intermediate parquet for SLURM aggregation
            _save_intermediate_results(self, well=well, batch_id=batch_id)
        else:
            # Save final AnnData files
            _save_results_as_anndata(self)

            # Generate validation visualizations if we have viz data
            if viz_data_list and validation_samples > 0:
                print("Generating feature validation visualizations...")

                # Random sampling of 3 organelles and 3 features is now handled
                # inside generate_validation_visualizations() for both CP and regular experiments
                generate_validation_visualizations(
                    viz_data_list,
                    validation_features,
                    self.save_dir,
                    network_organelles=self.network_organelles,
                    visualization_organelles=None,  # Let it randomly select 3
                )

        print("Feature extraction pipeline complete.")

    def _process_cells_parallel(
        self,
        cells_df: pd.DataFrame,
        organelles_to_process: list,
        network_organelles: list,
        organelle_map: dict,
        full_features: bool,
        initial_yx_patch_size: tuple = (300, 300),
        validation_indices: set = None,
        debug: bool = False,
        max_objects_per_organelle: int = None,
        sequential: bool = False,
    ):
        """
        Process all cells in parallel using per-well batching for optimal data locality.

        This method groups cells by well and processes each well as a batch.
        This approach provides:
        - Better data locality (one zarr position per batch)
        - Smaller DataFrames passed to workers (only cells from one well)
        - Better zarr chunk caching efficiency

        The zarr store is opened once and passed to all workers.

        Parameters
        ----------
        validation_indices : set, optional
            Set of DataFrame indices for cells to collect visualization data for.
            If provided, returns viz_data_list as 4th element of return tuple.

        Returns
        -------
        tuple
            (cell_features_list, object_features_dict, network_features_dict, viz_data_list)
            viz_data_list is empty if validation_indices is None or empty.
        """
        if validation_indices is None:
            validation_indices = set()

        # Store path as string for passing to parallel workers (picklable)
        store_path = str(self.morphology_path)

        # Open the store temporarily to get metadata (spacing, channel_names)
        # Each parallel worker will open its own store to avoid pickle issues
        pheno_store = open_ome_zarr(self.morphology_path, mode="r")
        try:
            first_tile_path = next(pheno_store.positions())[0]
            first_pos_obj = pheno_store[first_tile_path]
            source_scale = first_pos_obj.scale
            spacing = (source_scale[-2], source_scale[-1])
            channel_names = pheno_store.channel_names
        except (StopIteration, Exception) as e:
            print(
                f"Warning: Error getting scale from first tile: {e}. Using default (1.0, 1.0)"
            )
            spacing = (1.0, 1.0)
            channel_names = []
        finally:
            # Close the store - workers will open their own
            pheno_store.close()

        # Build organelle -> channel_index mapping for visualization
        # Match segmentation label names to channel names directly from zarr metadata
        organelle_channel_indices = {}
        for org_name in organelles_to_process:
            org_lower = org_name.lower()
            # Try to find matching channel by substring matching
            for idx, channel in enumerate(channel_names):
                ch_lower = channel.lower()
                # Match patterns: "focus_3d_tubular" -> "focus3d", "gfp_vesicular" -> "gfp", etc.
                org_base = org_lower.replace("_", "").replace("-", "")
                ch_base = ch_lower.replace("_", "").replace("-", "")
                if org_base.startswith(ch_base) or ch_base.startswith(org_base):
                    organelle_channel_indices[org_name] = idx
                    break
                # Also try partial match (e.g., "gfp" in "gfp_vesicular")
                if ch_base in org_base or org_base in ch_base:
                    organelle_channel_indices[org_name] = idx
                    break
        print(f"Organelle -> Channel index mapping for visualization: {organelle_channel_indices}")

        # Group cells by well for optimal data locality
        # Each well becomes one batch - much better than random cell batching
        wells = cells_df["well"].unique()
        n_wells = len(wells)
        n_cells = len(cells_df)

        # Create per-well batches: each batch is (well_name, cell_indices, well_cells_df)
        well_batches = []
        for well in wells:
            well_mask = cells_df["well"] == well
            well_indices = cells_df.index[well_mask].tolist()
            well_cells_df = cells_df[well_mask].reset_index(drop=True)
            well_batches.append((well, well_indices, well_cells_df))

        # CPU mode: Use Joblib for CPU parallelism (or 1 worker for sequential benchmarking)
        if sequential:
            num_workers = 1
            print(f"🔬 SEQUENTIAL MODE: processing cells one at a time (for benchmarking)")
        else:
            num_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.1, data_ram_gb=0.2)
            print(f"CPU mode: using {num_workers} Joblib workers")
        print(
            f"Processing {n_cells} cells across {n_wells} wells with {num_workers} workers..."
        )
        print(f"  Average cells per well: {n_cells / n_wells:.0f}")

        # SINGLE-WELL OPTIMIZATION: When processing just one well, use internal parallel mode
        # This opens the zarr store ONCE instead of once per batch, avoiding massive I/O overhead
        if n_wells == 1:
            well = wells[0]
            well_cells_df = cells_df[cells_df["well"] == well].reset_index(drop=True)
            print(f"  Single-well mode: opening store once, using internal parallel processing")
            
            # Use _worker_process_well with show_progress=True for internal parallelism
            all_results = [_worker_process_well(
                well=well,
                well_cells_df=well_cells_df,
                store_path=store_path,
                organelles_to_process=organelles_to_process,
                network_organelles=network_organelles,
                spacing=spacing,
                organelle_map=organelle_map,
                full_features=full_features,
                channel_names=channel_names,
                initial_yx_patch_size=initial_yx_patch_size,
                available_labels=self.available_labels,
                show_progress=True,  # Enable internal progress + parallel processing
                n_jobs=num_workers,
                debug=False,
                max_objects_per_organelle=max_objects_per_organelle,
            )]
        else:
            # MULTI-WELL MODE: Use Joblib batching across wells
            # Process cells in parallel batches using Joblib
            # Use smaller batches for frequent progress updates (every ~30-60 seconds)
            # Sort by well + spatial location for optimal zarr chunk cache locality
            # Cells nearby in the same well will hit the same zarr chunks
            cells_per_batch = max(100, n_cells // (num_workers * 16))
            cell_batches = []
            
            # Sort by well, then by GLOBAL spatial coordinates for zarr chunk locality
            # (y_local_pheno/x_local_pheno are local to cell crop - useless for spatial sorting!)
            sort_cols = ["well"]
            if "y_global_pheno" in cells_df.columns and "x_global_pheno" in cells_df.columns:
                spatial_cols = ["y_global_pheno", "x_global_pheno"]
            elif "y_pheno" in cells_df.columns and "x_pheno" in cells_df.columns:
                spatial_cols = ["y_pheno", "x_pheno"]
            else:
                raise ValueError(
                    f"No global spatial coordinates found for chunk locality sorting. "
                    f"Expected 'y_global_pheno'/'x_global_pheno' or 'y_pheno'/'x_pheno'. "
                    f"Available columns: {list(cells_df.columns)}"
                )
            sort_cols.extend(spatial_cols)
            cells_df_sorted = cells_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
            y_range = f"{cells_df_sorted[spatial_cols[0]].min():.0f}-{cells_df_sorted[spatial_cols[0]].max():.0f}"
            x_range = f"{cells_df_sorted[spatial_cols[1]].min():.0f}-{cells_df_sorted[spatial_cols[1]].max():.0f}"
            print(f"  Sorted by {sort_cols} for zarr chunk locality")
            print(f"  Spatial range: y=[{y_range}], x=[{x_range}]")
            for batch_start in range(0, n_cells, cells_per_batch):
                batch_end = min(batch_start + cells_per_batch, n_cells)
                batch_cells_df = cells_df_sorted.iloc[batch_start:batch_end].copy()
                cell_batches.append(batch_cells_df)

            print(f"  Cell batches: {len(cell_batches)} ({cells_per_batch} cells/batch)")

            _max_objects_for_workers = max_objects_per_organelle
            def _process_cell_batch(batch_cells_df):
                results = []
                for well, well_cells in batch_cells_df.groupby("well"):
                    well_results = _worker_process_well(
                        well=well,
                        well_cells_df=well_cells.reset_index(drop=True),
                        store_path=store_path,
                        organelles_to_process=organelles_to_process,
                        network_organelles=network_organelles,
                        spacing=spacing,
                        organelle_map=organelle_map,
                        full_features=full_features,
                        channel_names=channel_names,
                        initial_yx_patch_size=initial_yx_patch_size,
                        available_labels=self.available_labels,
                        show_progress=False,
                        debug=False,
                        max_objects_per_organelle=_max_objects_for_workers,
                    )
                    if well_results:
                        results.extend(well_results)
                return results

            # Track progress with cells/sec throughput using tqdm + generator
            start_time = time.time()
            cells_per_batch = n_cells // len(cell_batches) if cell_batches else 0

            print(f"Processing {n_cells} cells in {len(cell_batches)} batches across {num_workers} workers...")
            with tqdm(total=n_cells, desc="Extracting features", unit="cells", unit_scale=True) as pbar:
                all_results = []
                # Use loky (multiprocessing) for better isolation with tensorstore
                for result in Parallel(n_jobs=num_workers, return_as="generator")(
                    delayed(_process_cell_batch)(batch_cells_df)
                    for batch_cells_df in cell_batches
                ):
                    all_results.append(result)
                    pbar.update(cells_per_batch)

                    # Update postfix with cells/sec
                    elapsed = time.time() - start_time
                    cells_per_sec = pbar.n / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"cells/sec": f"{cells_per_sec:.1f}"})

        # Flatten results from all wells
        cell_features_list = []
        object_features_dict = {
            org: [] for org in organelles_to_process
        }
        network_features_dict = {org: [] for org in network_organelles}
        per_object_network_dict = {org: [] for org in network_organelles}  # Per-object network features

        for well_result in all_results:
            if well_result is None:
                continue
            for cell_features, object_features, network_features in well_result:
                if cell_features is not None:
                    cell_features_list.append(cell_features)
                if object_features:
                    for org, df in object_features.items():
                        if org in object_features_dict and not df.empty:
                            object_features_dict[org].append(df)
                if network_features:
                    for org, net_data in network_features.items():
                        if org in network_features_dict:
                            # Handle both old format (DataFrame) and new format (dict with branch_df + per_object_df)
                            if isinstance(net_data, dict):
                                branch_df = net_data.get("branch_df", pd.DataFrame())
                                per_object_df = net_data.get("per_object_df", pd.DataFrame())
                            else:
                                branch_df = net_data
                                per_object_df = pd.DataFrame()
                            if not branch_df.empty:
                                network_features_dict[org].append(branch_df)
                            if not per_object_df.empty:
                                # Add cell_id for aggregation
                                per_object_df = per_object_df.copy()
                                per_object_df["cell_id"] = cell_features.get("cell_id") if cell_features else None
                                per_object_network_dict[org].append(per_object_df)

        # Debug: Summary of collected features
        if debug:
            print("\n" + "="*60)
            print("DEBUG SUMMARY: Collected features from all cells")
            print("="*60)
            print(f"  Total cells with features: {len(cell_features_list)}")
            print(f"  Object features collected by organelle:")
            for org, dfs in object_features_dict.items():
                total_objects = sum(len(df) for df in dfs) if dfs else 0
                print(f"    - {org}: {len(dfs)} cells with data, {total_objects} total objects")
            print(f"  Network features collected by organelle:")
            for org, dfs in network_features_dict.items():
                total_branches = sum(len(df) for df in dfs) if dfs else 0
                print(f"    - {org}: {len(dfs)} cells with data, {total_branches} total branches")
            print("="*60 + "\n")

        # Print timing summary if we have timing data (preview mode only)
        if cell_features_list and getattr(self, '_preview_mode', False):
            self._print_timing_summary(cell_features_list)

        # Collect visualization data for validation cells
        # Compute per-object features on-the-fly (lightweight - only 12 cells)
        viz_data_list = []
        if validation_indices:
            print(f"Collecting visualization data for {len(validation_indices)} validation cells...")
            pheno_store = open_ome_zarr(self.morphology_path, mode="r")
            stores = {"pheno_assembled_v3": pheno_store}

            # Create BaseDataset for validation cells only
            validation_cells_df = cells_df.loc[list(validation_indices)].reset_index(drop=True)
            base_dataset = BaseDataset(
                stores=stores,
                labels_df=validation_cells_df,
                initial_yx_patch_size=initial_yx_patch_size,
                final_yx_patch_size=initial_yx_patch_size,
                out_channels="all",
                mask_cell=False,
                use_original_crop_size=False,
            )

            # Debug counters for skipped cells
            skipped_empty_mask = 0
            skipped_no_well = 0
            skipped_no_labels_group = 0
            skipped_no_organelle_labels = 0

            for i in range(len(validation_cells_df)):
                try:
                    batch = base_dataset[i]
                    data = batch["data"].numpy() if hasattr(batch["data"], "numpy") else np.array(batch["data"])
                    mask = batch["mask"].numpy() if hasattr(batch["mask"], "numpy") else np.array(batch["mask"])
                    crop_info = batch["crop_info"]
                    bbox = batch.get("bbox")
                    well = crop_info.get("well")

                    # Normalize bbox from any format (str, tuple, list, numpy array)
                    bbox = normalize_bbox(bbox)
                    cp_bbox = normalize_bbox(crop_info.get("cp_bbox"))

                    cell_specific_mask = mask[0].astype(np.uint8)
                    if not np.any(cell_specific_mask):
                        skipped_empty_mask += 1
                        continue

                    # Skip only if cell has NEITHER standard bbox NOR cp_bbox
                    # Cells with only one bbox type can still get features for their organelles
                    if not well:
                        skipped_no_well += 1
                        continue
                    if bbox is None and cp_bbox is None:
                        skipped_no_well += 1
                        continue

                    # Load organelle labels for this cell (features computed on-demand in visualization)
                    organelle_labels = {}

                    position = pheno_store[well]

                    # Use standard bbox for visualization if available, otherwise use cp_bbox
                    if bbox is not None:
                        y_min, x_min, y_max, x_max = bbox
                    else:
                        # Cell has only cp_bbox - use it for visualization
                        y_min, x_min, y_max, x_max = cp_bbox

                    # Load expanded intensity for visualization (50px padding around cell)
                    # This shows surrounding context while keeping masks at original position
                    viz_padding = 50
                    fov = position["0"]
                    img_h, img_w = fov.shape[-2], fov.shape[-1]

                    # Compute expanded bbox with bounds checking
                    exp_y_min = max(0, y_min - viz_padding)
                    exp_x_min = max(0, x_min - viz_padding)
                    exp_y_max = min(img_h, y_max + viz_padding)
                    exp_x_max = min(img_w, x_max + viz_padding)

                    # Calculate offset of original crop within expanded region
                    viz_offset_y = y_min - exp_y_min
                    viz_offset_x = x_min - exp_x_min

                    # Load expanded intensity region
                    n_channels = fov.shape[1] if fov.ndim >= 2 else 1
                    expanded_intensity = np.array(fov[0, :, 0, exp_y_min:exp_y_max, exp_x_min:exp_x_max])

                    if "labels" not in position.zgroup:
                        skipped_no_labels_group += 1
                        continue

                    labels_group = position.zgroup["labels"]

                    # cp_bbox already parsed above (before skip check)

                    # Get cp_cell_seg_id for CP cell identification (if available)
                    cp_seg_id = crop_info.get("cp_cell_seg_id")
                    if cp_seg_id is not None and pd.notna(cp_seg_id):
                        cp_seg_id = int(cp_seg_id)
                    else:
                        cp_seg_id = None

                    # Load cp_cell_seg if available (for CellPainting organelles)
                    # Use cp_bbox if available for correct crop region
                    cp_cell_mask = None
                    cp_intensity = None  # Intensity crop at cp_bbox for CP organelle visualization
                    cp_viz_offset_y = 0  # Offset for positioning CP masks in expanded intensity
                    cp_viz_offset_x = 0

                    # Debug: Check CP loading conditions
                    has_cp_cell_seg = "cp_cell_seg" in labels_group
                    if i == 0:  # Only print for first cell
                        print(f"    [DEBUG] cp_cell_seg in labels_group: {has_cp_cell_seg}")
                        print(f"    [DEBUG] cp_bbox parsed: {cp_bbox}")
                        print(f"    [DEBUG] labels_group keys: {list(labels_group.keys())[:10]}...")

                    if has_cp_cell_seg and cp_bbox is not None:
                        try:
                            cp_label_array = labels_group["cp_cell_seg"]["0"]
                            cp_y_min, cp_x_min, cp_y_max, cp_x_max = cp_bbox

                            if i == 0:
                                print(f"    [DEBUG] cp_label_array.ndim: {cp_label_array.ndim}, shape: {cp_label_array.shape}")
                                print(f"    [DEBUG] cp_bbox: y={cp_y_min}:{cp_y_max}, x={cp_x_min}:{cp_x_max}")

                            # Handle different dimensionality of label arrays (TCZYX = 5D, CZYX = 4D, etc.)
                            if cp_label_array.ndim == 5:
                                if i == 0:
                                    print(f"    [DEBUG] Using 5D slicing: [0, 0, 0, {cp_y_min}:{cp_y_max}, {cp_x_min}:{cp_x_max}]")
                                cp_cell_mask_raw = np.array(cp_label_array[0, 0, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                            elif cp_label_array.ndim == 4:
                                if i == 0:
                                    print(f"    [DEBUG] Using 4D slicing")
                                cp_cell_mask_raw = np.array(cp_label_array[0, 0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                            elif cp_label_array.ndim == 3:
                                if i == 0:
                                    print(f"    [DEBUG] Using 3D slicing")
                                cp_cell_mask_raw = np.array(cp_label_array[0, cp_y_min:cp_y_max, cp_x_min:cp_x_max])
                            else:
                                if i == 0:
                                    print(f"    [DEBUG] Using 2D slicing")
                                cp_cell_mask_raw = np.array(cp_label_array[cp_y_min:cp_y_max, cp_x_min:cp_x_max])

                            if i == 0:
                                print(f"    [DEBUG] cp_cell_mask_raw shape: {cp_cell_mask_raw.shape}")

                            # Create binary mask for this specific CP cell
                            if cp_seg_id is not None:
                                cp_cell_mask = (cp_cell_mask_raw == cp_seg_id).astype(np.uint8)
                            else:
                                cp_cell_mask = (cp_cell_mask_raw > 0).astype(np.uint8)

                            # Load expanded intensity image at cp_bbox location for CP organelle visualization
                            # This is the cell's location at CellPainting imaging time (different from phenotyping)
                            # Use same viz_padding as pheno bbox for consistency
                            try:
                                # fov already loaded above
                                # Compute expanded CP bbox with bounds checking
                                cp_exp_y_min = max(0, cp_y_min - viz_padding)
                                cp_exp_x_min = max(0, cp_x_min - viz_padding)
                                cp_exp_y_max = min(img_h, cp_y_max + viz_padding)
                                cp_exp_x_max = min(img_w, cp_x_max + viz_padding)

                                # Calculate offset of original CP crop within expanded region
                                cp_viz_offset_y = cp_y_min - cp_exp_y_min
                                cp_viz_offset_x = cp_x_min - cp_exp_x_min

                                # Load expanded CP intensity region
                                cp_intensity = np.array(fov[0, :, 0, cp_exp_y_min:cp_exp_y_max, cp_exp_x_min:cp_exp_x_max])
                            except Exception:
                                cp_intensity = None
                                cp_viz_offset_y = 0
                                cp_viz_offset_x = 0

                        except Exception as e:
                            if i == 0:
                                print(f"    [DEBUG] cp_cell_mask loading failed: {e}")
                            cp_cell_mask = None

                    if i == 0:
                        print(f"    [DEBUG] cp_cell_mask loaded: {cp_cell_mask is not None}")
                        if cp_cell_mask is not None:
                            print(f"    [DEBUG] cp_cell_mask shape: {cp_cell_mask.shape}, sum: {np.sum(cp_cell_mask)}")

                    for internal_name, zarr_label_name in self.available_labels.items():
                        if internal_name == "cell_mask":
                            continue
                        if zarr_label_name not in labels_group:
                            continue

                        try:
                            label_array = labels_group[zarr_label_name]["0"]

                            # For CP organelles, use cp_bbox if available
                            is_cp_organelle = internal_name.lower().startswith("cp")
                            if is_cp_organelle and cp_bbox is not None:
                                crop_y_min, crop_x_min, crop_y_max, crop_x_max = cp_bbox
                            else:
                                crop_y_min, crop_x_min, crop_y_max, crop_x_max = y_min, x_min, y_max, x_max

                            if label_array.ndim == 5:
                                label_crop = label_array[0, 0, 0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            elif label_array.ndim == 4:
                                label_crop = label_array[0, 0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            elif label_array.ndim == 3:
                                label_crop = label_array[0, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                            else:
                                label_crop = label_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                            label_crop = np.array(label_crop)

                            # Determine which cell mask to use for this organelle
                            if is_cp_organelle:
                                # CP organelles MUST have cp_cell_mask - skip if not available
                                if cp_cell_mask is None:
                                    if i == 0:
                                        print(f"    [DEBUG] Skipping CP organelle {internal_name}: cp_cell_mask is None")
                                    continue
                                if i == 0:
                                    print(f"    [DEBUG] Processing CP organelle {internal_name} with cp_cell_mask")
                                mask_to_use = cp_cell_mask
                                # Ensure shapes match
                                if label_crop.shape != mask_to_use.shape:
                                    matched_mask = np.zeros(label_crop.shape, dtype=mask_to_use.dtype)
                                    h = min(mask_to_use.shape[0], matched_mask.shape[0])
                                    w = min(mask_to_use.shape[1], matched_mask.shape[1])
                                    matched_mask[:h, :w] = mask_to_use[:h, :w]
                                    mask_to_use = matched_mask
                            else:
                                # Standard organelles require standard bbox - skip if not available
                                if bbox is None:
                                    if i == 0:
                                        print(f"    [DEBUG] Skipping standard organelle {internal_name}: bbox is None")
                                    continue
                                mask_to_use = cell_specific_mask
                                # Match shape if needed
                                if label_crop.shape != cell_specific_mask.shape:
                                    matched_crop = np.zeros(cell_specific_mask.shape, dtype=label_crop.dtype)
                                    h = min(label_crop.shape[0], matched_crop.shape[0])
                                    w = min(label_crop.shape[1], matched_crop.shape[1])
                                    matched_crop[:h, :w] = label_crop[:h, :w]
                                    label_crop = matched_crop

                            # Skip if empty before masking
                            if np.sum(label_crop > 0) == 0:
                                continue

                            # Mask to cell boundary using appropriate mask
                            label_crop = label_crop * (mask_to_use > 0).astype(label_crop.dtype)

                            if np.sum(label_crop > 0) > 0:
                                organelle_labels[internal_name] = label_crop
                        except Exception:
                            pass

                    if organelle_labels:
                        # Use expanded intensity for visualization (shows surrounding context)
                        # Masks/labels stay at original size - visualization code will position them using offsets
                        viz_data_list.append({
                            "intensity": expanded_intensity,  # Expanded region (150px padding)
                            "cp_intensity": cp_intensity,  # Expanded CP region (if available)
                            "cell_mask": mask,
                            "cp_cell_mask": cp_cell_mask,  # CellPainting cell mask (if available)
                            "organelle_labels": organelle_labels,
                            "per_object_features": {},  # Computed on-demand in visualization
                            "per_branch_features": {},
                            "metadata": crop_info,
                            "channel_names": channel_names,
                            "organelle_channel_indices": organelle_channel_indices,  # Direct lookup: organelle -> channel_idx
                            # Offsets for positioning masks within expanded intensity region
                            "viz_offset": (viz_offset_y, viz_offset_x),
                            "cp_viz_offset": (cp_viz_offset_y, cp_viz_offset_x) if cp_intensity is not None else (0, 0),
                        })
                    else:
                        skipped_no_organelle_labels += 1

                except Exception as e:
                    print(f"  Error processing validation cell {i}: {e}")
                    continue

            pheno_store.close()
            print(f"Collected visualization data for {len(viz_data_list)} cells.")
            if skipped_empty_mask or skipped_no_well or skipped_no_labels_group or skipped_no_organelle_labels:
                print(f"  Skipped cells: empty_mask={skipped_empty_mask}, no_well={skipped_no_well}, "
                      f"no_labels_group={skipped_no_labels_group}, no_organelle_labels={skipped_no_organelle_labels}")

        return cell_features_list, object_features_dict, network_features_dict, per_object_network_dict, viz_data_list

    def _print_timing_summary(self, cell_features_list: list):
        """Print a summary of per-metric timing data aggregated across all cells."""
        if not cell_features_list:
            return

        # Collect timing columns from ALL cells (profiling data only in total_index==0 cell)
        all_timing_cols = set()
        for cell in cell_features_list:
            all_timing_cols.update(k for k in cell.keys() if k.startswith("_timing_"))
        timing_cols = sorted(all_timing_cols)
        if not timing_cols:
            return

        # Build timing DataFrame
        timing_data = {col: [] for col in timing_cols}
        for cell in cell_features_list:
            for col in timing_cols:
                timing_data[col].append(cell.get(col, 0))

        timing_df = pd.DataFrame(timing_data)

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
            # Sort by total time (descending) to show slowest first
            batch_totals = [(c, timing_df[c].sum()) for c in batch_cols]
            batch_totals.sort(key=lambda x: -x[1])
            for col, _ in batch_totals:
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
            # Sort by total time (descending) to show slowest first
            loc_totals = [(c, timing_df[c].sum()) for c in loc_cols]
            loc_totals.sort(key=lambda x: -x[1])
            for col, _ in loc_totals:
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
            for col, _ in net_totals:
                name = col.replace("_timing_net_", "").replace("_ms", "")
                mean_val = timing_df[col].mean()
                median_val = timing_df[col].median()
                max_val = timing_df[col].max()
                total_val = timing_df[col].sum()
                print(f"  {name:<33} {mean_val:>10.1f} {median_val:>10.1f} {max_val:>10.1f} {total_val:>12.0f}")

        # Per-property profiling (from cell with total_index=0 - shows regionprops breakdown)
        prop_cols = [c for c in timing_cols if c.startswith("_timing_property_") and c.endswith("_ms")]
        # DEBUG: Show what property timing columns we found
        print(f"\n[DEBUG] Property timing columns found: {prop_cols}")
        if prop_cols and len(cell_features_list) > 0:
            # Find the cell with total_index == 0 (the one that was profiled)
            profiled_cell = None
            for cell in cell_features_list:
                if cell.get("total_index", -1) == 0:
                    profiled_cell = cell
                    break
            if profiled_cell is None:
                profiled_cell = cell_features_list[0]  # Fallback

            profile_org = profiled_cell.get("_timing_property_profile_organelle", "unknown")
            profile_objs = profiled_cell.get("_timing_property_profile_objects", 0)
            print(f"[DEBUG] Profiled cell total_index={profiled_cell.get('total_index')}, org={profile_org}, objs={profile_objs}")
            if profile_org != "unknown" and profile_objs > 0:
                print(f"\n{'regionprops property breakdown':<35} ({profile_org}, {profile_objs} objects)")
                print("-" * 70)
                for col in sorted(prop_cols):
                    name = col.replace("_timing_property_", "").replace("_ms", "")
                    val = profiled_cell.get(col, 0)
                    print(f"  {name:<33} {val:>10.1f} ms")
                total_prop = sum(profiled_cell.get(c, 0) for c in prop_cols)
                print(f"  {'TOTAL':33} {total_prop:>10.1f} ms")

        # Summary
        n_cells = len(cell_features_list)
        total_time_sec = timing_df["_timing_total_ms"].sum() / 1000 if "_timing_total_ms" in timing_df.columns else 0
        print(f"\n{'Summary':<35}")
        print("-" * 70)
        print(f"  Cells processed: {n_cells}")
        print(f"  Total compute time: {total_time_sec:.1f} seconds")
        print(f"  Average per cell: {total_time_sec/n_cells*1000:.1f} ms" if n_cells > 0 else "")
        print("="*70 + "\n")

    def _print_feature_summary(self, cell_features_list: list):
        """Print a summary of all features measured, grouped by category."""
        if not cell_features_list:
            return

        # Get all feature columns (exclude metadata and timing)
        all_cols = set()
        for cell in cell_features_list:
            all_cols.update(cell.keys())

        # Filter out timing columns and known metadata columns
        metadata_cols = {
            "cell_id", "global_cell_id", "well", "tile", "segmentation_id",
            "barcode", "gene", "guide", "total_index", "x_global_pheno",
            "y_global_pheno", "x_local_pheno", "y_local_pheno", "bbox",
            "cp_bbox", "cp_cell_seg_id", "track_id", "time_point",
        }
        feature_cols = [c for c in all_cols if not c.startswith("_timing_") and c not in metadata_cols]

        # Categorize features
        categories = {
            "Cell Morphology": [],
            "Organelle Morphology": [],
            "Localization": [],
            "Network": [],
            "Intensity": [],
            "Other": [],
        }

        # Base morphology metrics (used for pattern matching)
        morph_metrics = {"area", "perimeter", "perimeter_crofton", "solidity", "extent", "eccentricity",
                        "orientation", "axis_major_length", "axis_minor_length",
                        "aspect_ratio", "circularity", "hu_moment", "moments_weighted_hu",
                        "equivalent_diameter_area", "area_convex", "area_filled", "euler_number",
                        "centroid_y", "centroid_x", "centroid_weighted_y", "centroid_weighted_x",
                        "inertia_eigval"}
        loc_metrics = {"distance_from_cell_edge", "distance_from_nucleus", "distance_from_nucleus_centroid", "normalized_radial_position"}
        intensity_metrics = {"intensity_mean", "intensity_max", "intensity_min", "intensity_std",
                            "intensity_median", "intensity_q25", "intensity_q75", "intensity_iqr",
                            "intensity_mad", "intensity_integrated", "intensity_range", "intensity_cv"}
        agg_funcs = {"sum", "mean", "median", "std", "min", "max", "count"}

        for col in sorted(feature_cols):
            col_lower = col.lower()

            if col.startswith("cell_"):
                categories["Cell Morphology"].append(col)
            elif col.startswith("network_"):
                categories["Network"].append(col)
            elif any(loc in col_lower for loc in loc_metrics):
                categories["Localization"].append(col)
            elif any(intens in col_lower for intens in intensity_metrics):
                categories["Intensity"].append(col)
            elif any(morph in col_lower for morph in morph_metrics):
                categories["Organelle Morphology"].append(col)
            else:
                categories["Other"].append(col)

        print(f"\n{'FEATURES MEASURED BY CATEGORY':<35}")
        print("-" * 70)

        total_features = 0
        for category, features in categories.items():
            if not features:
                continue

            # Group by organelle/prefix for cleaner display
            if category in ("Organelle Morphology", "Localization", "Network", "Intensity"):
                # Extract unique organelle prefixes and base metrics
                organelles = set()
                base_metrics = set()
                for f in features:
                    parts = f.split("_")
                    # Find where the metric starts (after organelle name, before agg func)
                    for i, part in enumerate(parts):
                        if part in agg_funcs and i > 0:
                            organelle = "_".join(parts[:i-1]) if i > 1 else parts[0]
                            metric = parts[i-1]
                            organelles.add(organelle)
                            base_metrics.add(metric)
                            break
                    else:
                        # No agg func found, might be a direct metric
                        if len(parts) >= 2:
                            organelles.add(parts[0])
                            base_metrics.add("_".join(parts[1:]))

                print(f"\n  {category}: {len(features)} features")
                print(f"    Organelles: {len(organelles)}")
                if organelles:
                    org_list = sorted(organelles)[:8]  # Show first 8
                    if len(organelles) > 8:
                        print(f"      {', '.join(org_list)}... (+{len(organelles)-8} more)")
                    else:
                        print(f"      {', '.join(org_list)}")
                print(f"    Base metrics: {len(base_metrics)}")
                if base_metrics:
                    metric_list = sorted(base_metrics)[:10]
                    if len(base_metrics) > 10:
                        print(f"      {', '.join(metric_list)}... (+{len(base_metrics)-10} more)")
                    else:
                        print(f"      {', '.join(metric_list)}")
            else:
                print(f"\n  {category}: {len(features)} features")
                # For cell morphology and other, just list them
                feature_list = sorted(features)[:12]
                if len(features) > 12:
                    print(f"      {', '.join(feature_list)}... (+{len(features)-12} more)")
                else:
                    print(f"      {', '.join(feature_list)}")

            total_features += len(features)

        print(f"\n  {'TOTAL FEATURES:':<33} {total_features}")

    def _print_feature_summary_from_df(self, cell_df: pd.DataFrame, organelles_processed: list = None, network_organelles: list = None):
        """Print a summary of all features measured, grouped by category.

        This version works with the aggregated cell DataFrame (after _generate_summaries).

        Parameters
        ----------
        cell_df : pd.DataFrame
            The aggregated cell features DataFrame.
        organelles_processed : list, optional
            List of organelle names that were processed (for accurate counts).
        network_organelles : list, optional
            List of organelle names with network analysis.
        """
        if cell_df is None or cell_df.empty:
            return

        # Use provided lists or empty defaults
        n_organelles = len(organelles_processed) if organelles_processed else 0
        n_network_organelles = len(network_organelles) if network_organelles else 0

        # Get all columns from the DataFrame
        all_cols = set(cell_df.columns)

        # Filter out timing columns and known metadata columns
        metadata_cols = {
            "cell_id", "global_cell_id", "well", "tile", "segmentation_id",
            "barcode", "gene", "guide", "total_index", "x_global_pheno",
            "y_global_pheno", "x_local_pheno", "y_local_pheno", "bbox",
            "cp_bbox", "cp_cell_seg_id", "track_id", "time_point",
            "gene_name", "sgRNA", "gene_effect", "NCBI_ID", "barcode_from_iss",
            "dep_map_gene_name", "store_key", "tile_path", "subpool",
            "x_pheno", "y_pheno", "tile_pheno", "x_global_bc", "y_global_bc",
            "og_index", "cp1_idx", "cp1_label", "cp2_label", "iss_label",
            "cp1_pheno_distance", "cp1_iss_distance", "cp1_cp2_distance",
        }
        feature_cols = [c for c in all_cols if not c.startswith("_timing_") and c not in metadata_cols]

        # Categorize features
        categories = {
            "Cell Morphology": [],
            "Organelle Morphology": [],
            "Localization": [],
            "Network": [],
            "Intensity": [],
            "Other": [],
        }

        # Known base metrics for each category
        morph_base_metrics = {"area", "perimeter", "perimeter_crofton", "solidity", "extent", "eccentricity",
                             "orientation", "axis_major_length", "axis_minor_length",
                             "aspect_ratio", "circularity", "moments_weighted_hu",
                             "equivalent_diameter_area", "area_convex", "area_filled", "euler_number",
                             "centroid_y", "centroid_x", "centroid_weighted_y", "centroid_weighted_x",
                             "inertia_eigval"}
        loc_base_metrics = {"distance_from_cell_edge", "distance_from_nucleus",
                           "distance_from_nucleus_centroid", "normalized_radial_position"}
        intensity_base_metrics = {"intensity_mean", "intensity_max", "intensity_min", "intensity_std",
                                  "intensity_median", "intensity_q25", "intensity_q75", "intensity_iqr",
                                  "intensity_mad", "intensity_integrated", "intensity_range", "intensity_cv"}
        agg_funcs = {"sum", "mean", "median", "std", "min", "max", "count"}

        for col in sorted(feature_cols):
            col_lower = col.lower()

            if col.startswith("cell_"):
                categories["Cell Morphology"].append(col)
            elif col.startswith("network_"):
                categories["Network"].append(col)
            elif any(loc in col_lower for loc in loc_base_metrics):
                categories["Localization"].append(col)
            elif any(intens in col_lower for intens in intensity_base_metrics):
                categories["Intensity"].append(col)
            elif any(morph in col_lower for morph in morph_base_metrics):
                categories["Organelle Morphology"].append(col)
            else:
                categories["Other"].append(col)

        print(f"\n{'='*70}")
        print("FEATURES MEASURED BY CATEGORY")
        print("="*70)

        total_features = 0
        for category, features in categories.items():
            if not features:
                continue

            if category == "Cell Morphology":
                # Just list cell morphology features
                print(f"\n{category}: {len(features)} features")
                feature_list = sorted(features)
                print(f"  {', '.join(feature_list)}")
            elif category in ("Organelle Morphology", "Localization", "Network", "Intensity"):
                # Determine base metrics and organelle count for this category
                if category == "Organelle Morphology":
                    base_metrics = sorted(morph_base_metrics)
                    org_count = n_organelles
                elif category == "Localization":
                    base_metrics = sorted(loc_base_metrics)
                    org_count = n_organelles
                elif category == "Intensity":
                    base_metrics = sorted(intensity_base_metrics)
                    org_count = n_organelles
                else:  # Network
                    base_metrics = ["branch_count", "skeleton_length", "endpoints", "nodes",
                                   "average_degree", "branching_density", "euler_number"]
                    org_count = n_network_organelles

                print(f"\n{category}: {len(features)} features across {org_count} organelles")
                print(f"  Metrics: {', '.join(base_metrics)}")
                print(f"  Aggregations: {', '.join(sorted(agg_funcs))}")
            else:
                # Other category
                print(f"\n{category}: {len(features)} features")

            total_features += len(features)

        print(f"\n{'='*70}")
        print(f"TOTAL FEATURES: {total_features}")
        print("="*70 + "\n")


def extract_features_for_batch_direct(
    experiment: str,
    well: str,
    batch_idx: int,
    batch_cells_df: "pd.DataFrame",
    output_dir: Path,
    full_features: bool = False,
    max_objects_per_organelle: int = 250,
    sequential: bool = False,
) -> Path:
    """
    Extract features for a batch of cells (pre-loaded DataFrame).

    This is the optimized version called by SLURM workers. Cell metadata
    is pre-loaded from a parquet file to avoid loading 2M+ cells in each worker.

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0') - all cells are from this well
    batch_idx : int
        Batch index within the well (0, 1, 2, ...)
    batch_cells_df : pd.DataFrame
        Pre-loaded DataFrame containing only this batch's cells
    output_dir : Path
        Directory to save batch results
    full_features : bool
        Whether to compute expensive texture features
    max_objects_per_organelle : int
        Maximum objects per organelle per cell
    sequential : bool
        If True, process cells sequentially (1 worker) instead of parallel.
        For benchmarking to compare parallel vs sequential performance.

    Returns
    -------
    Path
        Path to the saved parquet file, or None if no cells processed
    """
    import time
    t_start = time.time()

    from cyclops_utils.data.experiment import OpsDataset
    
    ds = OpsDataset(experiment)
    morphology_path = ds.store_paths["pheno_assembled_v3"]
    print(f"[TIMING] OpsDataset init: {time.time() - t_start:.2f}s")

    # Build organelle->channel map
    def _canonicalize_channel_name(name: str):
        n = name.strip()
        low = n.lower()
        if low in {"brightfield", "bf", "bf_phase", "bright field", "phase"}:
            return "BF"
        if low == "gfp":
            return "GFP"
        if low in {"mcherry", "m-cherry", "cherry"}:
            return "mCherry"
        return n

    ch_to_org = ds.channel_map_data or {}
    organelle_channel_map = {}
    for channel, organelle in ch_to_org.items():
        organelle_channel_map[organelle] = _canonicalize_channel_name(channel)
    print(f"organelle_channel_map: {organelle_channel_map}")

    # Discover available labels (ONCE - pass to extractor to avoid re-discovery)
    t_disc = time.time()
    discovered_labels = _discover_available_labels(morphology_path)
    all_organelles = list(discovered_labels.keys())
    print(f"[TIMING] Label discovery: {time.time() - t_disc:.2f}s")
    print(f"Discovered organelles: {len(all_organelles)}")

    # Determine network organelles
    network_organelles = [org for org in all_organelles if is_network_organelle(org)]
    print(f"Network organelles: {len(network_organelles)}")

    # Create extractor with output directory
    t_ext = time.time()
    extractor = FeatureExtractor(experiment, output_dir=output_dir)
    # Pre-set discovered labels to skip redundant discovery in run()
    extractor.available_labels = discovered_labels
    print(f"[TIMING] FeatureExtractor init: {time.time() - t_ext:.2f}s")
    
    well_safe = well.replace("/", "_")
    batch_id = f"{well_safe}_{batch_idx:04d}"
    print(f"Processing batch {batch_id}: {len(batch_cells_df)} cells from well {well}")
    print(f"[TIMING] Total init: {time.time() - t_start:.2f}s")

    # Run feature extraction on this batch (cells already pre-loaded)
    extractor.run(
        organelle_map=organelle_channel_map,
        network_organelles=network_organelles,
        full_features=full_features,
        cells_df_override=batch_cells_df,  # Pre-loaded cells
        save_intermediate=True,
        batch_id=batch_id,  # Pass batch ID for output naming
        validation_samples=0,  # Skip validation for batch processing
        use_cpu=True,
        max_objects_per_organelle=max_objects_per_organelle,
        sequential=sequential,
    )
    
    # Return path to output file
    output_path = output_dir / "_batch_results" / f"batch_{batch_id}_cells.parquet"
    return output_path if output_path.exists() else None


def extract_features_for_batch(
    experiment: str,
    well: str,
    batch_idx: int,
    cell_indices: list,
    output_dir: Path,
    full_features: bool = False,
    max_objects_per_organelle: int = 250,
) -> Path:
    """
    Extract features for a batch of cells from a single well (legacy version).

    NOTE: This version loads all cells and filters. Use extract_features_for_batch_direct
    with pre-saved batch metadata for better performance in SLURM workers.

    Parameters
    ----------
    experiment : str
        Experiment name
    well : str
        Well identifier (e.g., 'A/1/0') - all cells are from this well
    batch_idx : int
        Batch index within the well (0, 1, 2, ...)
    cell_indices : list
        List of cell indices within the well to process (0-indexed within well)
    output_dir : Path
        Directory to save batch results
    full_features : bool
        Whether to compute expensive texture features
    max_objects_per_organelle : int
        Maximum objects per organelle per cell

    Returns
    -------
    Path
        Path to the saved parquet file, or None if no cells processed
    """
    from cyclops_utils.data.experiment import OpsDataset
    
    ds = OpsDataset(experiment)
    morphology_path = ds.store_paths["pheno_assembled_v3"]

    # Load cell metadata for this well only (slow - loads all cells)
    print("WARNING: Using legacy batch function - loading all cells...")
    cells_df = _load_cells_metadata(ds, morphology_path)
    well_cells_df = cells_df[cells_df["well"] == well].copy()
    
    # Sort by GLOBAL spatial location within the well for zarr chunk locality
    # (y_local_pheno/x_local_pheno are local to cell crop - useless for spatial sorting!)
    if "y_global_pheno" in well_cells_df.columns and "x_global_pheno" in well_cells_df.columns:
        sort_cols = ["y_global_pheno", "x_global_pheno"]
    elif "y_pheno" in well_cells_df.columns and "x_pheno" in well_cells_df.columns:
        sort_cols = ["y_pheno", "x_pheno"]
    else:
        raise ValueError(
            f"No global spatial coordinates found for well {well}. "
            f"Expected 'y_global_pheno'/'x_global_pheno' or 'y_pheno'/'x_pheno'. "
            f"Available columns: {list(well_cells_df.columns)}"
        )
    
    well_cells_df = well_cells_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    print(f"  Sorted {len(well_cells_df)} cells by {sort_cols} for zarr chunk locality")
    
    # Filter to this batch's cell indices within the well
    batch_cells_df = well_cells_df.iloc[cell_indices].copy()
    
    # Use the optimized direct function
    return extract_features_for_batch_direct(
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        batch_cells_df=batch_cells_df,
        output_dir=output_dir,
        full_features=full_features,
        max_objects_per_organelle=max_objects_per_organelle,
    )


@notify_step(
    step_message="Started feature extraction",
    success_message="Finished feature extraction",
)
@versioned_function("v1.0")
def extract_features_for_experiment(
    experiment: str,
    morphology_path: Path,
    output_path: Path = None,
    debug_tile_count: int = None,
    full_features: bool = False,
    well: str = None,
    save_intermediate: bool = False,
    validation_samples: int = 6,
    validation_features: list = None,
    preview_cells: int = None,
    use_cpu: bool = False,
    max_objects_per_organelle: int = 250,
):
    """
    Convenience function to extract features for an experiment from tile-based data.

    This function initializes and runs the FeatureExtractor pipeline on a tile-based
    dataset, collecting all features in memory.

    Parameters
    ----------
    experiment : str
        The name of the experiment (e.g., 'ops0049_20250626').
    output_path : Path, optional
        The directory to save the output files.
    debug_tile_count : int, optional
        Limits processing to the first N tiles for debugging.
    full_features : bool
        Whether to compute the full set of (slower) features.
    well : str, optional
        If provided, only process cells from this well (e.g., 'A/1/0').
        Used for per-well SLURM parallelization.
    save_intermediate : bool
        If True, save intermediate per-well results as parquet files
        instead of final AnnData. Used for SLURM aggregation.
    validation_samples : int
        Number of cells to sample for validation visualizations. Default 12.
    validation_features : list
        Features to visualize. Default ["area", "perimeter", "axis_major_length"].
    preview_cells : int, optional
        If set, only process this many cells to test the pipeline end-to-end.
        Useful for quick validation that the pipeline works before full runs.
    use_cpu : bool, optional
        If True, force CPU-only mode even if GPU is available. Default False.
    """
    # Invert the channel->organelle map to get the organelle->channel map needed for processing
    # Build organelle->channel map from dataset-scoped channel map
    dataset = OpsDataset(experiment)

    # Canonicalize channel names
    def _canonicalize_channel_name(name: str):
        n = name.strip()
        low = n.lower()
        if low in {"brightfield", "bf", "bf_phase", "bright field", "phase"}:
            return "BF"
        if low == "gfp":
            return "GFP"
        if low in {"mcherry", "m-cherry", "cherry"}:
            return "mCherry"
        return n

    ch_to_org = dataset.channel_map_data or {}
    organelle_channel_map = {}
    for channel, organelle in ch_to_org.items():
        organelle_channel_map[organelle] = _canonicalize_channel_name(channel)
    print(f"organelle_channel_map: {organelle_channel_map}")

    # Discover available labels to determine which organelles exist
    # This is needed to build the network_organelles list from discovered segmentations
    discovered_labels = _discover_available_labels(morphology_path)
    all_organelles = list(discovered_labels.keys())
    print(f"Discovered organelles: {len(all_organelles)}")

    # Determine network organelles using the shared module-level function
    # This is the same logic used by dry_run_discovery to display the "Network" column
    network_organelles = [org for org in all_organelles if is_network_organelle(org)]
    print(f"Network organelles: {len(network_organelles)}")

    # Preview mode: use a separate subdirectory to avoid overwriting real data
    if preview_cells is not None and preview_cells > 0:
        if output_path is None:
            output_path = dataset.analysis_path / "_preview"
        else:
            output_path = Path(output_path) / "_preview"
        print(f"\n⚠️  PREVIEW MODE: Output will be saved to {output_path}")
        print(f"   (This will NOT overwrite production data)\n")

    extractor = FeatureExtractor(experiment, output_dir=output_path)
    extractor.run(
        organelle_map=organelle_channel_map,
        network_organelles=network_organelles,
        debug_tile_count=debug_tile_count,
        full_features=full_features,
        well=well,
        save_intermediate=save_intermediate,
        validation_samples=validation_samples,
        validation_features=validation_features,
        preview_cells=preview_cells,
        use_cpu=use_cpu,
        max_objects_per_organelle=max_objects_per_organelle,
    )


if __name__ == "__main__":
    import argparse
    from cyclops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Run feature extraction on a tile-based OME-Zarr dataset."
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        required=True,
        help="Name or shorthand for the experiment (e.g., '94', 'ops94', 'ops0094_20251217').",
    )
    parser.add_argument("--output_dir", type=Path, help="Optional output directory.")
    parser.add_argument(
        "--organelles",
        nargs="+",
        help="Optional mapping of organelle name to channel (e.g., mitochondria:GFP)",
    )
    parser.add_argument(
        "--network_organelles",
        nargs="+",
        help="Optional list of organelle names for network analysis (e.g., mitochondria)",
    )
    parser.add_argument(
        "--debug_tile_count", type=int, help="Number of tiles to process for debugging"
    )
    parser.add_argument(
        "--full_features",
        action="store_true",
        help="Calculate all features, including computationally expensive ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without running extraction.",
    )
    parser.add_argument(
        "--well",
        type=str,
        help="Process only a single well (e.g., 'A/1/0'). Used for per-well SLURM parallelization.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate per-well results as parquet (for SLURM aggregation).",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=6,
        help="Number of cells to visualize for feature validation (0 to disable). Default 6 for 3x2 grid.",
    )
    parser.add_argument(
        "--validation-features",
        type=str,
        nargs="+",
        default=VALIDATION_FEATURES,
        help="Features to visualize in validation plots. Each feature generates one PNG per organelle.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=100,
        default=None,
        help="Preview mode: process only N cells (default 100) to test the pipeline end-to-end.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU-only mode even if GPU is available.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=100,
        help="Max objects per organelle for feature extraction (default: 100). "
             "Limits high object-count organelles (e.g., vesicles with 300+ objects) "
             "to speed up processing while maintaining representative statistics. "
             "Set to 0 for no limit (slower but exhaustive).",
    )

    args = parser.parse_args()

    # Resolve experiment name (e.g., "94" -> "ops0094_20251217")
    # autoselect=True prefers canonical format (ops####_YYYYMMDD) over variants with suffixes
    experiment = resolve_experiment_name(args.experiment, autoselect=True)

    # Always show organelle discovery table (useful for all modes)
    dry_run_discovery(experiment, preview=args.preview is not None)

    if args.dry_run:
        # Dry-run mode: exit after showing discovery table
        pass
    else:
        output_dir = Path(args.output_dir) if args.output_dir else None

        organelle_map = None
        if args.organelles:
            organelle_map = {}
            for item in args.organelles:
                if ":" not in item:
                    print(
                        f"Warning: organelle mapping '{item}' is not in 'name:channel' format. Skipping."
                    )
                    continue
                name, channel = item.split(":", 1)
                organelle_map[name] = channel

        # Convert max_objects=0 to None (meaning no limit)
        max_objs = args.max_objects if args.max_objects > 0 else None

        dataset = OpsDataset(experiment)
        morphology_path = dataset.store_paths["pheno_assembled_v3"]

        extract_features_for_experiment(
            experiment=experiment,
            morphology_path=morphology_path,
            output_path=output_dir,
            debug_tile_count=args.debug_tile_count,
            full_features=args.full_features,
            well=args.well,
            save_intermediate=args.save_intermediate,
            validation_samples=args.validation_samples,
            validation_features=args.validation_features,
            preview_cells=args.preview,
            use_cpu=args.use_cpu,
            max_objects_per_organelle=max_objs,
        )
