"""
Feature Extraction Metadata Loading Functions.

This module handles loading cell metadata and discovering available segmentation
labels from zarr v3 stores for the feature extraction pipeline.

Functions
---------
load_cells_metadata
    Load cell metadata from cell_painting_linked or linked_results CSVs.
discover_available_labels
    Discover segmentation labels from a zarr v3 store.
parse_bbox
    Parse bbox string/tuple to a validated tuple.
create_global_cell_id
    Create unique global cell IDs for a DataFrame.
"""

import pandas as pd
from iohub import open_ome_zarr
from pathlib import Path
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.data.bbox_utils import normalize_bbox

# Alias for backwards compatibility
parse_bbox = normalize_bbox


def create_global_cell_id(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Create unique global cell IDs for a DataFrame.

    Uses segmentation_id as the primary identifier for all workflows.
    For cell painting, CP-only cells (no pheno match) use cp_cell_seg_id.

    The upstream data sources (linked_results, cell_painting_linked) are expected
    to have already deduplicated by segmentation_id, so each segmentation_id
    is unique within a well.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell metadata. Must contain 'well' and 'segmentation_id' columns.
        For cell painting workflows, may also contain 'cp_cell_seg_id'.
    inplace : bool, default True
        If True, modify df in place. If False, return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'global_cell_id' column added.
    """
    if not inplace:
        df = df.copy()

    if "global_cell_id" in df.columns:
        return df

    # Use segmentation_id as the primary identifier
    has_seg_id = df["segmentation_id"].notna() if "segmentation_id" in df.columns else pd.Series(False, index=df.index)
    df["global_cell_id"] = ""
    df.loc[has_seg_id, "global_cell_id"] = (
        df.loc[has_seg_id, "well"].astype(str)
        + "_"
        + df.loc[has_seg_id, "segmentation_id"].astype(int).astype(str)
    )

    # For cell painting: CP-only cells (no pheno match) use cp_cell_seg_id
    if "cp_cell_seg_id" in df.columns:
        has_cp_only = ~has_seg_id & df["cp_cell_seg_id"].notna()
        df.loc[has_cp_only, "global_cell_id"] = (
            df.loc[has_cp_only, "well"].astype(str)
            + "_cp"
            + df.loc[has_cp_only, "cp_cell_seg_id"].astype(int).astype(str)
        )

    return df


def dry_run_discovery(experiment: str, preview: bool = False):
    """
    Discover and display what would be extracted from the zarr store without running extraction.

    Parameters
    ----------
    experiment : str
        The name of the experiment to inspect.
    preview : bool
        If True, indicates this is a preview run (output goes to _preview subdirectory).
    """
    mode_str = "PREVIEW" if preview else "DRY RUN"
    print(f"\n{'='*60}")
    print(f"{mode_str}: Feature Extraction Discovery for {experiment}")
    print(f"{'='*60}\n")

    dataset = OpsDataset(experiment)
    morphology_path = dataset.store_paths["pheno_assembled_v3"]

    if not morphology_path.exists():
        print(f"ERROR: Zarr v3 store not found at {morphology_path}")
        return

    print(f"Zarr store: {morphology_path}\n")

    # Create a temporary extractor to use discovery method
    available_labels = _discover_available_labels(morphology_path)

    # Get additional info from the store
    with open_ome_zarr(morphology_path, mode="r") as store:
        positions = [p for p, _ in store.positions()]
        channel_names = store.channel_names

        # Get first position to check all labels
        first_pos_path = positions[0]
        first_pos = store[first_pos_path]
        if "labels" in first_pos.zgroup:
            all_labels = list(first_pos.zgroup["labels"].group_keys())
        else:
            all_labels = []

    print(f"Positions: {len(positions)}")
    print(f"  First few: {positions[:5]}{'...' if len(positions) > 5 else ''}\n")

    print(f"Channels ({len(channel_names)}): {channel_names}\n")

    print(f"All labels in zarr ({len(all_labels)}):")
    for label in sorted(all_labels):
        print(f"  - {label}")

    print(f"\n{'='*60}")
    print("DISCOVERED SEGMENTATIONS FOR FEATURE EXTRACTION")
    print(f"{'='*60}\n")

    print(f"Organelles to process ({len(available_labels)}):\n")

    # Group by type
    core_segs = {}
    organelle_segs = {}

    for internal_name, zarr_label in available_labels.items():
        if internal_name in ["cell_mask", "nuclei", "cp_cell_mask"]:
            core_segs[internal_name] = zarr_label
        else:
            organelle_segs[internal_name] = zarr_label

    # Check if cp_cell_seg exists (indicates CellPainting experiment)
    has_cp_cell_seg = "cp_cell_seg" in all_labels

    if has_cp_cell_seg:
        print("Experiment Type: CellPainting (dual-pass extraction enabled)")
        print("  - CP organelles (cp*) use cp_bbox + cp_cell_seg")
        print("  - Standard organelles use bbox + cell_seg")
    else:
        print("Experiment Type: Standard (single-pass extraction)")
        print("  - All organelles use bbox + cell_seg")

    if core_segs:
        print("\nCore Segmentations:")
        for name, label in core_segs.items():
            print(f"  {name:30} -> {label}")

    if organelle_segs:
        print("\nOrganelle Segmentations:")
        print(f"  {'Internal Name':<30} {'Zarr Label':<35} {'Bbox Pass':<12} {'Cell Mask':<15} {'Network'}")
        print(f"  {'-'*30} {'-'*35} {'-'*12} {'-'*15} {'-'*8}")
        for name, label in sorted(organelle_segs.items()):
            # CellPainting labels (cp*, CP*) use cp_cell_seg and cp_bbox
            # Standard organelles use cell_seg and bbox
            # Case-insensitive check for CP organelles
            if name.lower().startswith("cp") and has_cp_cell_seg:
                cell_mask = "cp_cell_seg"
                bbox_pass = "cp_bbox"
            else:
                cell_mask = "cell_seg"
                bbox_pass = "bbox"
            # Check if this organelle gets network analysis (uses module-level function)
            network_flag = "Yes" if is_network_organelle(name) else ""
            print(f"  {name:<30} {label:<35} {bbox_pass:<12} {cell_mask:<15} {network_flag}")

    # Show what's NOT being processed
    skipped = [l for l in all_labels if l not in available_labels.values()]
    if skipped:
        print(f"\nSkipped labels (not segmentations):")
        for label in sorted(skipped):
            print(f"  - {label}")

    # Show channel map info
    if dataset.channel_map_data:
        print(f"\n{'='*60}")
        print("CHANNEL MAP (from ops_channel_maps.yaml)")
        print(f"{'='*60}\n")
        for channel, organelle in dataset.channel_map_data.items():
            print(f"  {channel:15} -> {organelle}")

    print(f"\n{'='*60}")
    print("OUTPUT")
    print(f"{'='*60}\n")
    output_path = dataset.analysis_path / "_preview" if preview else dataset.analysis_path
    print(f"Output directory: {output_path}")
    if preview:
        print("  ⚠️  PREVIEW MODE: Output saved to _preview/ subdirectory")
    print(f"Output files:")
    print(f"  - {experiment}_cell_features.h5ad")
    print(f"  - {experiment}_guide_features.h5ad")
    print(f"  - {experiment}_gene_features.h5ad")

    print(f"\n{'='*60}")
    print("To run feature extraction:")
    print(f"  python -m organelle_profiler.feature_extraction.feature_extraction -e {experiment}")
    if preview:
        print(f"  python -m organelle_profiler.feature_extraction.feature_extraction -e {experiment} --preview")
    print(f"{'='*60}\n")

def is_network_organelle(name: str) -> bool:
    """
    Determine if an organelle should have network analysis applied.

    Network analysis applies to:
    - Tubular structures (any name containing "tubular")
    - GFP/mCherry/Cy5 channels (often tag network-forming organelles)
    - CellPainting channels (cp1_*, cp2_*) EXCEPT nuclear/nuclei/nucleoli
    - 4i channels (4i_*) EXCEPT nuclear/nuclei/nucleoli — every IF round labels a
      protein that may form filamentous structures, so we treat them like CP markers

    Excluded (not network structures):
    - Cell segmentations (cell_mask, cp_cell_mask, cell_seg, cp_cell_seg)
    - Nuclear/nuclei/nucleoli segmentations
    - Vesicular structures (discrete puncta, not networks)

    Parameters
    ----------
    name : str
        The organelle/segmentation name to check.

    Returns
    -------
    bool
        True if network analysis should be applied.
    """
    name_lower = name.lower()

    # Exclude cell boundary segmentations - these are not organelles
    if any(cell_name in name_lower for cell_name in ["cell_mask", "cell_seg", "cp_cell"]):
        return False

    # Skip nuclear/nuclei/nucleoli segmentations - these are not network structures
    if "nuclear" in name_lower or "nuclei" in name_lower or "nucleoli" in name_lower:
        return False

    # Skip vesicular structures - discrete puncta, not filamentous networks
    if "vesicular" in name_lower:
        return False

    # Include tubular structures
    if "tubular" in name_lower:
        return True

    # Include GFP/mCherry/Cy5 channels
    if any(ch in name_lower for ch in ["gfp", "mcherry", "cy5"]):
        return True

    # Include CellPainting channels (cp1_*, cp2_*) - these have filamentous structures
    if name_lower.startswith("cp1_") or name_lower.startswith("cp2_"):
        return True

    # Include 4i markers (4i_R1_p53, 4i_r4_b-catenin, etc.). The nuclear/nuclei
    # exclusions above already drop 4i_R*_nuclear_seg and 4i_r*_nuclei_dapi_seg.
    if name_lower.startswith("4i_"):
        return True

    return False


def _load_cells_metadata(dataset: OpsDataset, morphology_path: Path, debug_tile_count: int = None) -> pd.DataFrame:
        """
        Load cell metadata following Option B: use cell_painting_linked as primary source.

        For experiments with CellPainting data:
        - Uses cell_painting_linked CSV as the PRIMARY cell source (not linked_results)
        - This CSV already has the correct 'bbox' computed from cp_cell_seg
        - Has 'cp_cell_seg_id' for cell identification
        - Contains all data from linked_results merged in (barcodes, genes, etc.)

        For experiments WITHOUT CellPainting data:
        - Falls back to linked_results CSV (standard workflow)

        This method adds columns needed for BaseDataset compatibility:
        - well: Position path in format "A/1/0"
        - store_key: Key for the store dict (pheno_assembled_v3)
        - total_index: Unique index for each cell
        - segmentation_id: Mapped from cp_cell_seg_id for CP cells

        Parameters
        ----------
        debug_tile_count : int, optional
            If set, only load cells from the first N wells for debugging.

        Returns
        -------
        pd.DataFrame
            DataFrame with cell metadata ready for feature extraction.

        Raises
        ------
        ValueError
            If bbox is missing or invalid for any cell.
        """
        with open_ome_zarr(morphology_path, mode="r") as pheno_store:
            # Get all available wells
            wells = [f"A/{i}" for i in pheno_store["A"].group_keys()]

        if debug_tile_count is not None:
            wells = wells[:debug_tile_count]
            print(f"DEBUG MODE: Processing cells from {len(wells)} wells")

        # Discover which linker output is present for this experiment.
        #
        # The link_cell_painting pipeline writes CSVs under three patterns
        # depending on mode/version (see MODE_CONFIG in link_cell_painting.py):
        #   - "{well_short}_linked_pheno_iss_cp.csv"   (current cell-painting mode)
        #   - "{well_short}_linked_pheno_iss_4i.csv"   (current 4i mode)
        #   - "cell_painting_linked_{well_safe}.csv"   (legacy cell-painting filename)
        #
        # 4i CSVs use ``4i_segmentation_id`` / ``4i_bbox`` columns; we rename
        # those to ``cp_cell_seg_id`` / ``cp_bbox`` immediately after load so the
        # rest of this loader (and downstream feature extraction) sees the same
        # column schema regardless of source modality. The actual label name on
        # disk (``4i_cell_seg`` vs ``cp_cell_seg``) is independent.
        _CSV_VARIANTS = (
            # (kind, filename template using {well_short}/{well_safe}, seg_col, bbox_col)
            ("cp",        "{well_short}_linked_pheno_iss_cp.csv",  "cp_cell_seg_id",     "cp_bbox"),
            ("4i",        "{well_short}_linked_pheno_iss_4i.csv",  "4i_segmentation_id", "4i_bbox"),
            ("cp_legacy", "cell_painting_linked_{well_safe}.csv",  "cp_cell_seg_id",     "cp_bbox"),
        )

        cp_linked_list = []
        loaded_kind: str | None = None
        for well in wells:
            well_with_site = f"{well}/0"  # "A/1/0"
            well_safe = well_with_site.replace("/", "_")  # "A_1_0"
            well_short = well.replace("/", "")  # "A/1" -> "A1"
            for kind, template, seg_col, bbox_col in _CSV_VARIANTS:
                path = dataset.results_fast / template.format(
                    well_short=well_short, well_safe=well_safe,
                )
                if not path.exists():
                    continue
                if loaded_kind is None:
                    loaded_kind = kind
                elif loaded_kind != kind:
                    print(
                        f"  Warning: well {well_with_site} has a {kind!r} CSV but earlier "
                        f"wells used {loaded_kind!r}. Skipping to keep the schema uniform."
                    )
                    continue
                cp_df = pd.read_csv(path)
                # Normalize 4i columns into the cp_-prefixed schema downstream expects.
                if kind == "4i":
                    rename_map = {}
                    if "4i_segmentation_id" in cp_df.columns:
                        rename_map["4i_segmentation_id"] = "cp_cell_seg_id"
                    if "4i_bbox" in cp_df.columns:
                        rename_map["4i_bbox"] = "cp_bbox"
                    if rename_map:
                        cp_df = cp_df.rename(columns=rename_map)
                cp_df["well"] = well_with_site  # Format for store access: "A/1/0"
                cp_df["store_key"] = "pheno_assembled_v3"
                cp_linked_list.append(cp_df)
                break  # one CSV per well

        if cp_linked_list:
            # Option B: Use linker-output CSV as primary source
            cells_df = pd.concat(cp_linked_list, ignore_index=True)
            _label_for_kind = {
                "cp": "linked_pheno_iss_cp",
                "4i": "linked_pheno_iss_4i (renamed to cp_cell_seg_id/cp_bbox)",
                "cp_legacy": "cell_painting_linked (legacy filename)",
            }
            print(
                f"Using {_label_for_kind.get(loaded_kind, loaded_kind)} CSVs "
                f"as primary source: {len(cells_df)} cells"
            )

            # Custom-perturbation schema bridge: when the library uses a
            # non-standard guide column (config ``gene_name_output_column``) with
            # ``barcode`` instead of ``sgRNA``, project barcode → sgRNA and the
            # custom column → gene_name so the rest of FE (dedup, guide/gene
            # aggregation, NTC normalization) sees the standard schema. The custom
            # column plus any ``gene_target``/``row_type`` cols are preserved.
            guide_col = getattr(dataset, "gene_name_output_column", None)
            if (
                guide_col
                and guide_col != "gene_name"
                and guide_col in cells_df.columns
                and "sgRNA" not in cells_df.columns
                and "barcode" in cells_df.columns
            ):
                cells_df["sgRNA"] = cells_df["barcode"]
                cells_df["gene_name"] = cells_df[guide_col]
                # row_type == "neg_ctrl" marks the negative-control equivalent —
                # labeled "neg_ctrl" (NOT "NTC", which is the CRISPR non-targeting
                # guide term and would mislead anyone reading the obs).
                n_neg = 0
                if "row_type" in cells_df.columns:
                    neg_mask = cells_df["row_type"].astype(str) == "neg_ctrl"
                    n_neg = int(neg_mask.sum())
                    cells_df.loc[neg_mask, "gene_name"] = "neg_ctrl"
                print(f"  Custom-perturbation mode: barcode→sgRNA and "
                      f"{guide_col}→gene_name applied; "
                      f"{n_neg:,} neg_ctrl rows labeled 'neg_ctrl'")

            # Filter to only cells with valid sgRNA (codebook match)
            # Cells without sgRNA have ISS reads that didn't match the codebook - invalid/low-quality reads
            # These can't be assigned to a guide/gene, so skip them.
            if "sgRNA" in cells_df.columns:
                n_before = len(cells_df)
                has_sgrna = cells_df["sgRNA"].notna() & (cells_df["sgRNA"] != "") & (cells_df["sgRNA"] != "None")
                n_valid = has_sgrna.sum()
                n_invalid = n_before - n_valid
                if n_invalid > 0:
                    cells_df = cells_df[has_sgrna].reset_index(drop=True)
                    print(f"  Filtered to cells with valid sgRNA: {n_before:,} -> {n_valid:,} ({n_invalid:,} unmatched ISS reads skipped)")
                else:
                    print(f"  ✓ All {n_valid:,} cells have valid sgRNA")

            # Validate required columns
            if "bbox" not in cells_df.columns:
                raise ValueError(
                    "cell_painting_linked CSV is missing 'bbox' column. "
                    "Re-run link_cell_painting.py to generate bboxes from cp_cell_seg."
                )

            if "cp_cell_seg_id" not in cells_df.columns:
                raise ValueError(
                    "cell_painting_linked CSV is missing 'cp_cell_seg_id' column. "
                    "Re-run link_cell_painting.py to add cell identification."
                )

            # Dual bbox system for CP experiments:
            # - cp_bbox + cp_cell_seg_id: Cell location at CP imaging time (for cp* organelles)
            # - bbox + segmentation_id: Cell location at pheno imaging time (for standard organelles)
            # The cell has moved between imaging sessions, so we need both coordinates.

            # Parse and validate cp_bbox (CellPainting bbox - required for CP experiments)
            if "cp_bbox" in cells_df.columns:
                cells_df["cp_bbox"] = cells_df["cp_bbox"].apply(parse_bbox)
                invalid_cp_bbox = cells_df["cp_bbox"].isna().sum()
                if invalid_cp_bbox > 0:
                    print(f"  Warning: {invalid_cp_bbox} cells missing cp_bbox (CP organelles will be skipped for these)")
                else:
                    print(f"  ✓ All {len(cells_df)} cells have valid cp_bbox from cp_cell_seg")
            else:
                print("  Warning: No cp_bbox column found - CP organelles will use fallback bbox")
                cells_df["cp_bbox"] = None

            # Parse and validate bbox (Phenotyping bbox - for standard organelles)
            if "bbox" in cells_df.columns:
                cells_df["bbox"] = cells_df["bbox"].apply(parse_bbox)
                invalid_bbox = cells_df["bbox"].isna().sum()
                if invalid_bbox > 0:
                    print(f"  Warning: {invalid_bbox} cells missing bbox (standard organelles will be skipped for these)")
                else:
                    print(f"  ✓ All {len(cells_df)} cells have valid bbox from linked_results")
            else:
                print("  Warning: No bbox column found - standard organelles will use cp_bbox as fallback")
                cells_df["bbox"] = cells_df["cp_bbox"]

            # Keep both segmentation IDs for proper cell masking
            # cp_cell_seg_id: for cp_cell_seg (CP organelles)
            # segmentation_id: for cell_seg (standard organelles)
            # Note: segmentation_id should already be in the CSV from linked_results merge
            if "segmentation_id" not in cells_df.columns:
                print("  Warning: No segmentation_id found - using cp_cell_seg_id for all")
                cells_df["segmentation_id"] = cells_df["cp_cell_seg_id"]

            # CRITICAL: Check for duplicate (well, segmentation_id) pairs
            # Duplicates cause Cartesian products during feature extraction merges,
            # leading to massive row inflation (e.g., 2M cells -> 3M+ rows)
            # Note: segmentation_id is only unique WITHIN a well, not globally
            # Note: Cells without segmentation_id (CP-only cells without pheno match) are OK
            cells_with_seg = cells_df[cells_df["segmentation_id"].notna()].copy()
            n_with_seg = len(cells_with_seg)
            # Create (well, segmentation_id) key for uniqueness check
            cells_with_seg["_well_seg_key"] = (
                cells_with_seg["well"].astype(str) + "_" +
                cells_with_seg["segmentation_id"].astype(int).astype(str)
            )
            n_unique_pairs = cells_with_seg["_well_seg_key"].nunique()
            if n_with_seg > n_unique_pairs:
                n_duplicates = n_with_seg - n_unique_pairs
                dup_counts = cells_with_seg["_well_seg_key"].value_counts()
                worst_dups = dup_counts[dup_counts > 1].head(5)
                raise ValueError(
                    f"CRITICAL: Found {n_duplicates} duplicate (well, segmentation_id) pairs in cell_painting_linked CSVs!\n"
                    f"  Cells with segmentation_id: {n_with_seg}, Unique pairs: {n_unique_pairs}\n"
                    f"  Worst offenders (well_seg_id: count):\n"
                    f"    {worst_dups.to_dict()}\n\n"
                    f"This will cause Cartesian products during feature extraction, inflating row counts.\n"
                    f"FIX: Re-run cell painting linking with the latest code:\n"
                    f"  python -m cyclops_process.fixed_cp_4i.link_slurm --experiment <exp>\n\n"
                    f"The updated linking code deduplicates by segmentation_id, keeping the best match."
                )
            else:
                n_without_seg = len(cells_df) - n_with_seg
                print(f"  ✓ No duplicate (well, segmentation_id) pairs ({n_with_seg} with pheno match, {n_without_seg} CP-only)")

            # Calculate detailed bbox coverage statistics
            has_cp_bbox = cells_df["cp_bbox"].notna()
            has_bbox = cells_df["bbox"].notna()
            has_cp_seg_id = cells_df["cp_cell_seg_id"].notna() if "cp_cell_seg_id" in cells_df.columns else pd.Series([False] * len(cells_df))
            has_seg_id = cells_df["segmentation_id"].notna() if "segmentation_id" in cells_df.columns else pd.Series([False] * len(cells_df))

            # Valid bbox systems require both bbox AND segmentation ID
            valid_standard = has_bbox & has_seg_id
            valid_cp = has_cp_bbox & has_cp_seg_id

            n_both = (valid_standard & valid_cp).sum()
            n_standard_only = (valid_standard & ~valid_cp).sum()
            n_cp_only = (~valid_standard & valid_cp).sum()
            n_neither = (~valid_standard & ~valid_cp).sum()

            print(f"\n  Bbox coverage summary:")
            print(f"    - Both bboxes valid (full dual-pass): {n_both} cells")
            print(f"    - Standard bbox only (pheno organelles): {n_standard_only} cells")
            print(f"    - CP bbox only (CP organelles): {n_cp_only} cells")
            print(f"    - Neither bbox valid (will be filtered): {n_neither} cells")

        else:
            # Fallback: Use linked_results (standard workflow for non-CP experiments)
            print("No cell_painting_linked CSVs found - using linked_results (standard workflow)")

            results_list = []
            for well in wells:
                results_path = dataset.append_well("linked_results", well)
                if results_path.exists():
                    df = pd.read_csv(results_path)
                    df["well"] = f"{well}/0"  # Format for store access: "A/1/0"
                    df["store_key"] = "pheno_assembled_v3"
                    results_list.append(df)

            if not results_list:
                raise ValueError(
                    f"No cell metadata found for experiment {dataset.experiment}. "
                    f"Expected either cell_painting_linked CSVs or linked_results CSVs."
                )

            cells_df = pd.concat(results_list, ignore_index=True)
            print(f"Loaded {len(cells_df)} cells from linked_results CSVs")

            # Validate bbox in linked_results
            if "bbox" not in cells_df.columns:
                raise ValueError(
                    "linked_results CSV is missing 'bbox' column. "
                    "Re-run tracking pipeline to generate bboxes."
                )

            cells_df["bbox"] = cells_df["bbox"].apply(parse_bbox)

            invalid_bbox_count = int(cells_df["bbox"].isna().sum())
            if invalid_bbox_count == len(cells_df):
                # 100% missing — likely a tracking-pipeline failure, hard fail.
                raise ValueError(
                    f"All {len(cells_df)} cells have missing/invalid bbox in linked_results. "
                    f"Re-run tracking pipeline to populate bboxes."
                )
            if invalid_bbox_count > 0:
                pct = 100.0 * invalid_bbox_count / len(cells_df)
                print(
                    f"  Warning: dropping {invalid_bbox_count}/{len(cells_df)} cells "
                    f"({pct:.1f}%) with missing/invalid bbox in linked_results."
                )
                cells_df = cells_df[cells_df["bbox"].notna()].copy()

            # For non-CP experiments: don't add cp_bbox or cp_cell_seg_id columns
            # Downstream code checks 'if col in df.columns' so missing columns are handled correctly
            print(f"  ✓ All {len(cells_df)} cells have valid bbox (standard single-pass extraction)")

            # Note: Deduplication by segmentation_id happens at aggregation time, not here.
            # This preserves backwards compatibility with existing feature extraction runs.

        # Add total_index if not present (required by BaseDataset)
        if "total_index" not in cells_df.columns:
            cells_df["total_index"] = cells_df.index

        # Rename coordinate columns to standard names
        # Handle multiple naming conventions from different pipelines:
        # - Standard linked_results: x_pheno, y_pheno
        # - Cell painting linked: x_pheno_centroid, y_pheno_centroid
        rename_cols = {}
        if "x_pheno" in cells_df.columns and "x_global_pheno" not in cells_df.columns:
            rename_cols["x_pheno"] = "x_global_pheno"
        if "y_pheno" in cells_df.columns and "y_global_pheno" not in cells_df.columns:
            rename_cols["y_pheno"] = "y_global_pheno"
        if "x_pheno_centroid" in cells_df.columns and "x_global_pheno" not in cells_df.columns:
            rename_cols["x_pheno_centroid"] = "x_global_pheno"
        if "y_pheno_centroid" in cells_df.columns and "y_global_pheno" not in cells_df.columns:
            rename_cols["y_pheno_centroid"] = "y_global_pheno"
        if rename_cols:
            cells_df = cells_df.rename(columns=rename_cols)

        return cells_df


def _discover_available_labels(morphology_path) -> dict:
        """
        Discover all available segmentation labels from the zarr v3 store.

        Returns a dict mapping internal organelle names to their zarr label names:
        - "cell_mask" -> "cell_seg" (cell segmentation)
        - "nuclei" -> "nuclear_seg" (nuclear segmentation)
        - "mitochondria_tomm20" -> "mitochondria_tomm20_seg" (from organelle_segmentation)
        - "cp1_f_actin_phalloidin" -> "cp1_f_actin_phalloidin_seg" (CellPainting)
        - etc.

        Label naming conventions handled:
        - Standard: {organelle}_{marker}_seg (e.g., mitochondria_tomm20_seg)
        - With structure: {channel}_{structure}_seg (e.g., phase2d_tubular_seg)
        - CellPainting: cp{N}_{organelle}_{marker}_seg (e.g., cp1_nuclei_hoechst_seg)
        - Dark variants: {channel}_{structure}_dark_seg (e.g., focus3d_vesicular_dark_seg)

        Returns
        -------
        dict
            Mapping of organelle name to zarr label name.
        """
        available_labels = {}

        with open_ome_zarr(morphology_path, mode="r") as store:
            # Get first position to check available labels
            first_pos_path = next(store.positions())[0]
            first_pos = store[first_pos_path]

            # Check if labels group exists
            if "labels" not in first_pos.zgroup:
                print("Warning: No labels group found in zarr store")
                return available_labels

            labels_group = first_pos.zgroup["labels"]
            label_names = list(labels_group.group_keys())

        print(f"Found {len(label_names)} labels in zarr store")

        # Labels to skip entirely (not organelle segmentations)
        skip_labels = {
            "cell_seg", "nuclear_seg", "cp_cell_seg",  # Core segmentations mapped separately below
            "grid_edges", "grid_props", "grid_overlay",  # Grid labels
            "iss_points", "iss_points_props", "iss_gene_image", "iss_guide_image",  # ISS labels
            "seg",  # Legacy cell segmentation name
            "cp_cell_seg_unstitched",  # Intermediate CellPainting output
            "cp2_nuclei_hoechst_seg",  # Redundant with cp1_nuclei_hoechst_seg (same channel)
            # Drop redundant per-round nuclear/DAPI segmentations from 4i
            # rounds 2-5 — every IF round restains DAPI / nuclei but the
            # R1 segmentation is already kept. With 5 rounds of redundant
            # nuclear masks loaded into the dask working set, GPU workers
            # OOM (15.83 GiB limit, repeatedly hit 95% on ops0144). Keeping
            # only R1 drops 8 segs (5 rounds × {nuclear_seg, nuclei_dapi_seg}
            # minus R1's two) → ~25% memory reduction in the GPU phase.
            "4i_R2_nuclear_seg", "4i_R3_nuclear_seg",
            "4i_R4_nuclear_seg", "4i_R5_nuclear_seg",
            "4i_r2_nuclei_dapi_seg", "4i_r3_nuclei_dapi_seg",
            "4i_r4_nuclei_dapi_seg", "4i_r5_nuclei_dapi_seg",
        }

        # Map core segmentation labels to internal names
        # These are created by convert_v3.py
        if "cell_seg" in label_names:
            available_labels["cell_mask"] = "cell_seg"
        if "nuclear_seg" in label_names:
            available_labels["nuclei"] = "nuclear_seg"
        # CellPainting cell segmentation - core segmentation for CP organelles
        if "cp_cell_seg" in label_names:
            available_labels["cp_cell_mask"] = "cp_cell_seg"

        # Structure type suffixes to recognize
        structure_types = {"tubular", "vesicular", "dark"}

        # Map all other segmentation labels
        for label_name in label_names:
            if label_name in skip_labels:
                continue

            if not label_name.endswith("_seg"):
                continue

            # Remove "_seg" suffix to get the base name
            base_name = label_name[:-4]

            # Parse structure type suffix (can be compound like "vesicular_dark")
            structure_suffix = None
            parts = base_name.split("_")

            # Check for structure type at end (could be "tubular", "vesicular", "dark", or "vesicular_dark")
            if len(parts) >= 2:
                if parts[-1] == "dark" and len(parts) >= 3 and parts[-2] in {"tubular", "vesicular"}:
                    # Compound structure like "vesicular_dark"
                    structure_suffix = f"{parts[-2]}_{parts[-1]}"
                    parts = parts[:-2]
                elif parts[-1] in structure_types:
                    structure_suffix = parts[-1]
                    parts = parts[:-1]

            # Build internal name from remaining parts
            # Keep the full descriptive name for clarity (e.g., "cp1_nuclei_hoechst", "mitochondria_tomm20")
            if len(parts) >= 1:
                internal_name = "_".join(parts)

                # Add structure suffix if present
                if structure_suffix:
                    internal_name = f"{internal_name}_{structure_suffix}"

                available_labels[internal_name] = label_name

        return available_labels


def validate_segmentation_labels(morphology_path, wells_filter: list = None) -> dict:
    """
    Validate that all positions have segmentation labels with data.

    Checks that the 'labels' group exists for all positions and contains
    the expected segmentation arrays. This catches failed segmentation runs
    before feature extraction starts.

    Parameters
    ----------
    morphology_path : Path
        Path to the zarr v3 store.
    wells_filter : list, optional
        List of wells to check. If None, checks all positions.

    Returns
    -------
    dict
        {"valid": bool, "missing_positions": list, "error": str or None}
    """
    missing_positions = []
    positions_checked = 0

    with open_ome_zarr(morphology_path, mode="r") as store:
        # Get expected labels from first position
        first_pos_path = next(store.positions())[0]
        first_pos = store[first_pos_path]

        if "labels" not in first_pos.zgroup:
            return {
                "valid": False,
                "missing_positions": [],
                "error": "No labels group found in zarr store - segmentation not run"
            }

        expected_labels = set(first_pos.zgroup["labels"].group_keys())

        # Check all positions
        for pos_path, _ in store.positions():
            # Parse well from position path (e.g., "A/1/0" -> "A/1/0")
            well = pos_path

            # Skip if wells_filter provided and this well not in it
            if wells_filter is not None:
                if well not in wells_filter:
                    continue

            positions_checked += 1
            pos = store[pos_path]

            # Check if labels group exists
            if "labels" not in pos.zgroup:
                missing_positions.append({
                    "position": pos_path,
                    "issue": "no labels group"
                })
                continue

            # Check if expected labels exist
            pos_labels = set(pos.zgroup["labels"].group_keys())
            missing_labels = expected_labels - pos_labels

            if missing_labels:
                missing_positions.append({
                    "position": pos_path,
                    "issue": f"missing labels: {sorted(missing_labels)}"
                })

    if missing_positions:
        return {
            "valid": False,
            "missing_positions": missing_positions,
            "positions_checked": positions_checked,
            "error": f"{len(missing_positions)} position(s) have missing segmentation data"
        }

    return {
        "valid": True,
        "missing_positions": [],
        "positions_checked": positions_checked,
        "error": None
    }
