"""Consolidate Alex Lin's top-attention cells (phase + fluorescent v2 CSVs) into
two combined AnnData objects: one with OrganelleProfiler features only and one
with OP + CellProfiler features concatenated.

By default, each gene KO contributes its top-K phase cells plus top-K
cells from EVERY fluor channel Alex's CSV provides for that gene
(~56 channels with the current v3 CSV → ~5700 cells/KO at top-K=100).
Pass `--channel-rank-max N` to clip to the top-N fluor channels per
gene (legacy default was 3).

Outputs (under --output-dir):
    top_attention_cells_op.h5ad        OrganelleProfiler features only
    top_attention_cells_op_cp.h5ad     OrganelleProfiler + CellProfiler features

Matching:
    OP <- CSV : direct join on (experiment, well_canonical, segmentation_id == segmentation)
    CP <- CSV : spatial nearest-neighbor on (x_position, y_position) <-> (x_pheno, y_pheno)
                within (experiment, well). For phase cells the CP source is
                features_processed_Phase.h5ad; for fluorescent cells it is
                features_processed_<chan>.h5ad where <chan> is the suffix of
                viz_channel (e.g. "chaperones_HSPA1B" -> "HSPA1B"; spaces in the
                suffix are replaced with underscores to match the file name).
                Files are looked up in both `cell-profiler/anndata_objects/`
                (phase + 4i) and `cell-profiler-cp/anndata_objects/` (Cell
                Painting), with short-name aliases for CP markers (e.g.
                "Concanavalin_A" -> "ConA", "Wheat_Germ_Agglutinin" -> "WGA").

CLI:
    python -m organelle_profiler.feature_extraction.consolidate_top_attention_cells \\
        --phase-csv /hpc/projects/.../pma_top_phase_cells_v2.csv \\
        --fluor-csv /hpc/projects/.../pma_top_fluorescent_cells_v2.csv \\
        --top-k 100 \\
        --output-dir /hpc/projects/.../alex_lin_attention/consolidated_v2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from organelle_profiler.paths import BASE_PATH

OPS_FAST_ROOT = Path(f"{BASE_PATH}")
CHANNEL_MAPS_YAML = Path(
    f"{BASE_PATH}/configs/ops_channel_maps.yaml"
)
DEFAULT_CHAD_CONFIG = Path(
    f"{BASE_PATH}/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
)


def _load_chad_complexes(yaml_path: Path) -> Dict[str, str]:
    """Load `{gene_name: complex_name}` from a CHAD positive_controls YAML.

    YAML format (per `chad_positive_controls_v*.yml`): top-level integer keys
    map to dicts with `name` (complex/cluster label) and `genes` (list). The
    synthetic "NTCs" cluster is skipped — NTCs in the pipeline are handled
    by the `NTC` gene label, not as a CHAD complex.

    A gene appearing in multiple clusters is assigned to the first one
    encountered (CHAD configs aim for disjoint membership; ambiguity is rare).
    """
    import yaml
    with yaml_path.open() as f:
        clusters = yaml.safe_load(f) or {}
    out: Dict[str, str] = {}
    for entry in clusters.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or name == "NTCs":
            continue
        for g in (entry.get("genes") or []):
            if isinstance(g, str) and g.strip() and g.strip() not in out:
                out[g.strip()] = str(name)
    return out


def _apply_chad_relabel(
    cells: pd.DataFrame,
    chad_map: Dict[str, str],
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    """Replace `cells["gene"]` with the CHAD complex name in-place.

    NTC rows (`gene == "NTC"`) are left untouched. KO cells whose gene has
    no CHAD complex assignment are dropped when `drop_unassigned=True`
    (default — atlas pages won't render meaningful complex panels for them
    anyway), or relabeled to `"(unassigned)"` otherwise.
    """
    if not chad_map:
        raise SystemExit("CHAD complex map is empty — nothing to aggregate")
    gene_col = cells["gene"].astype(str)
    is_ntc = gene_col == "NTC"
    mapped = gene_col[~is_ntc].map(chad_map)
    if drop_unassigned:
        keep_idx = cells.index[is_ntc | mapped.reindex(cells.index).notna().fillna(False)]
        cells = cells.loc[keep_idx].copy()
        gene_col = cells["gene"].astype(str)
        is_ntc = gene_col == "NTC"
        mapped = gene_col[~is_ntc].map(chad_map)
    else:
        mapped = mapped.fillna("(unassigned)")
    new_gene = gene_col.copy()
    new_gene.loc[~is_ntc] = mapped.values
    cells["gene"] = new_gene.values
    return cells


# Biology-synonym aliases for matching CSV viz_channel against yaml labels.
# Alex's CSV uses long/title-case forms ("Endoplasmic Reticulum", "Nucleus");
# the yaml uses short/lowercase forms ("ER", "nuclei"). Normalize both sides
# through this table before lookup. Order matters: longest first.
_VIZ_CHANNEL_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("endoplasmic reticulum", "er"),
    ("wheat germ agglutinin", "wga"),
    ("concanavalin a", "cona"),
    ("nucleus", "nuclei"),
)


def _normalize_viz_channel(s: str) -> str:
    """Canonical form for cross-matching CSV viz_channel vs yaml-derived
    'category_marker' strings. Lowercases and applies _VIZ_CHANNEL_ALIASES.

    Also handles v3 attention CSV labels for ops0144's 4i markers, which
    arrive as `NFkB_NFkB (mouse-488)` / `p21_p21 (rabbit-647)` etc. — i.e.
    'Foo_Foo (antibody-channel)'. Strip the parenthetical and dedupe the
    `Foo_Foo` self-repeat to a single `Foo`. Real biological markers like
    `chromatin_h2bc21` (parts differ) pass through unchanged.
    """
    raw = str(s or "").strip()
    # Strip trailing antibody/channel parenthetical from v3 4i labels.
    paren = raw.find("(")
    if paren > 0:
        raw = raw[:paren].strip()
    # Collapse 'Foo_Foo' -> 'Foo' (v3 4i-style self-repeat).
    parts = raw.split("_")
    if len(parts) == 2 and parts[0] and parts[0].lower() == parts[1].lower():
        raw = parts[0]
    out = raw.lower().strip()
    for long, short in _VIZ_CHANNEL_ALIASES:
        out = out.replace(long, short)
    return out


def _load_channel_maps(yaml_path: Path) -> Dict[str, Dict[str, str]]:
    """Load ops_channel_maps.yaml -> {short_exp: {viz_channel: physical_channel}}.

    `physical_channel` is normalized to one of:
      - 'phase' / 'gfp' / 'mcherry' / 'cy5' for standard reporters
      - the lowercased channel name (e.g. 'cp2_microtubules_tubulin') for
        Cell-Painting markers — these match the corresponding `var.organelle`
        in OP h5ads and become first-class channels in the per-cell mask.
      - the lowercased channel name (e.g. '4i_r1_gh2ax') for 4i markers.
        Same per-cell-mask scheme: a cell tagged with op_channel='4i_r1_gh2ax'
        keeps only that channel's features.
    """
    import yaml
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    def norm_phys(chan: str) -> Optional[str]:
        c = str(chan).strip().lower()
        if c == "bf":
            return "phase"
        if c == "gfp":
            return "gfp"
        if c == "mcherry":
            return "mcherry"
        if c in ("cy5", "farred", "far_red", "far-red"):
            return "cy5"
        if c.startswith(("cp1_", "cp2_")):
            return c  # e.g. 'cp2_microtubules_tubulin' — matches OP organelle
        if c.startswith("4i_"):
            return c  # e.g. '4i_r1_gh2ax' — matches OP organelle for 4i features
        return None

    out: Dict[str, Dict[str, str]] = {}
    for exp_short, entries in data.items():
        m: Dict[str, str] = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            chan = entry.get("channel_name")
            label = entry.get("label")
            if chan is None or label is None:
                continue  # metadata-only entry (four_i / cell_painting flags)
            phys = norm_phys(chan)
            if phys is None:
                continue
            # Yaml label "chaperones, HSPA1B" -> viz_channel-style key
            # "chaperones_hspa1b" after _normalize_viz_channel.
            viz = _normalize_viz_channel(str(label).replace(", ", "_", 1))
            m[viz] = phys
        out[exp_short] = m
    return out


def _organelle_to_op_channel(organelle: object) -> str:
    """OP `var.organelle` -> channel bucket used by the per-cell mask.

    Returns one of: 'phase' / 'gfp' / 'mcherry' / 'cy5' / 'agnostic' /
    'exclude' / a CP-marker string (e.g. 'cp2_microtubules_tubulin') /
    a 4i-marker string (e.g. '4i_r1_gh2ax'). The CP- and 4i-marker buckets
    exactly match the `cp1_*` / `cp2_*` / `4i_*` strings produced by
    `_load_channel_maps`, so a marker cell row keeps only its own features.
    """
    o = str(organelle or "").lower()
    if not o:
        return "agnostic"
    if o.startswith("4i_"):
        return o  # match cell channel format ('4i_r1_gh2ax' etc.)
    if o.startswith(("cp1_", "cp2_")):
        return o  # match cell channel format
    if (o.startswith("phase2d") or o.startswith("focus3d")
            or o == "nucleoli_phase2d" or o == "nucleoli_focus3d"):
        return "phase"
    if o == "gfp":
        return "gfp"
    if o == "mcherry":
        return "mcherry"
    if o in ("cy5", "farred"):
        return "cy5"
    if o in ("nuclei", "cell", "cell_morphology", "cp_cell"):
        return "agnostic"
    return "agnostic"


def _resolve_cell_op_channel(
    modality: str,
    viz_channel: str,
    experiment: str,
    channel_maps: Dict[str, Dict[str, str]],
) -> Optional[str]:
    """One CSV row -> physical OP channel string or None.

    Phase rows are 'phase' by definition. Fluorescent rows resolve via the
    per-experiment yaml map after normalizing the CSV's viz_channel
    (`_normalize_viz_channel` handles case + bio-synonyms like
    "Endoplasmic Reticulum"<->"ER"). Returns None if the pair isn't in the
    yaml — caller should fail loudly.
    """
    if str(modality) == "phase":
        return "phase"
    short = str(experiment).split("_")[0]
    return channel_maps.get(short, {}).get(_normalize_viz_channel(viz_channel))


def _canon_well(w: str) -> str:
    """A1 / A/1 / A/1/0 -> A/1/0 (matches OP and CP obs values)."""
    from cyclops_utils.data.filesystem import canonicalize_well_path
    canon = canonicalize_well_path(str(w))
    return canon if canon.count("/") >= 2 else f"{canon}/0"


def _strip_well(w: str) -> str:
    """A/1/0_ops0125_20260219 -> A/1/0 (CP obs.well includes exp suffix)."""
    parts = str(w).split("_", 1)
    return parts[0]


def _viz_channel_to_file(viz_channel: str) -> str:
    """'chaperones_HSPA1B' -> 'HSPA1B'; 'actin filament_FastAct_SPY555 Live Cell Dye'
    -> 'FastAct_SPY555_Live_Cell_Dye' (matches features_processed_<this>.h5ad)."""
    if "_" not in viz_channel:
        return viz_channel.replace(" ", "_")
    return viz_channel.split("_", 1)[1].replace(" ", "_")


# Aliases used by the Cell Painting CP run when its file names abbreviate
# markers that the attention CSV's viz_channel spells out in full.
# Also handles channel-map typos: `rsp6` in ops_channel_maps.yaml (ops0144
# 4i, viz=rsp6 / phys=4i_r2_rsp6) → CP h5ad lives under the correct
# biological name `RPS6` (ribosomal protein S6).
_CP_CHANNEL_ALIASES: Dict[str, str] = {
    "Concanavalin_A": "ConA",
    "Wheat_Germ_Agglutinin": "WGA",
    "rsp6": "RPS6",
}

# Subdirectories under <exp>/3-assembly/ that may hold features_processed_<chan>.h5ad.
# `cell-profiler` is the original CP run (phase + inline 4i markers in older
# experiments); `cell-profiler-cp` is the separate Cell Painting CP run
# (Hoechst / Phalloidin / Tubulin / TOMM20 / NPM1 / ConA / WGA);
# `cell-profiler-4i` is the separate 4i CP run used for newer experiments
# (e.g. ops0144). The first existing path wins.
_CP_SUBDIRS: Tuple[str, ...] = ("cell-profiler", "cell-profiler-cp", "cell-profiler-4i")

# Wells used for NTC pulls — same as attention_atlas.py.
_LINKED_WELLS: Tuple[str, ...] = ("A1", "A2", "A3")


def _linked_csv_path(exp: str, well: str) -> Path:
    """Per-(experiment, well) standard linked CSV. CP/4i experiments also
    have variant CSVs (`_cp.csv` / `_4i.csv`) — see `_load_linked_for_exp`
    which prefers those when present."""
    return OPS_FAST_ROOT / exp / "3-assembly" / f"{well}_linked_pheno_iss.csv"


# Variant-aware linked CSV loading. For CP/4i experiments the OP feature
# extraction pipeline (`fe_metadata.py:320`) writes BOTH a standard
# `_linked_pheno_iss.csv` AND a modality-specific variant; the variant is
# the source-of-truth used at OP feature extraction time, so cells in the
# resulting `<exp>_cell_features.h5ad` are keyed by `4i_segmentation_id` /
# `cp_cell_seg_id` (NOT by the live-cell `segmentation_id`). Reading the
# wrong CSV → cells reference the wrong segmentation IDs → spatial NN match
# against the OP h5ad fails for most cells (~77% miss rate observed for
# ops0144 4i markers before this fix).
#
# Each variant tuple: (kind, filename_suffix, columns_to_load, column_renames).
# Try in priority order — first existing variant per (exp, well) wins; we
# enforce a single variant per experiment so the schema stays uniform.
# Each variant supplies pheno-frame coords as `x_pheno`/`y_pheno` AND, when
# applicable, CP-imaging-time coords as `x_cp`/`y_cp`. Pheno coords are used
# by `_attach_op_features` (the OP cell h5ad's cells are pheno-keyed). CP
# coords are used by `_attach_cp_features` to match against per-channel
# `features_processed_<chan>.h5ad` files (whose obs carries `x_position`/
# `y_position` at CP imaging time, NOT pheno time — cells move between
# imaging sessions). For live-cell (`std`) experiments the two frames
# coincide, so no `x_cp`/`y_cp` is loaded; the CP attach falls back to
# pheno coords there.
_LINKED_VARIANTS: Tuple[Tuple[str, str, List[str], Dict[str, str]], ...] = (
    ("4i", "_4i",
     ["4i_segmentation_id", "x_pheno_centroid", "y_pheno_centroid",
      "x_4i", "y_4i",
      "sgRNA", "barcode", "gene_name", "tile_pheno"],
     {"4i_segmentation_id": "segmentation_id",
      "x_pheno_centroid": "x_pheno",
      "y_pheno_centroid": "y_pheno",
      "x_4i": "x_cp",
      "y_4i": "y_cp"}),
    ("cp", "_cp",
     ["cp_cell_seg_id", "x_pheno_centroid", "y_pheno_centroid",
      "x_cp1", "y_cp1",
      "sgRNA", "barcode", "gene_name", "tile_pheno"],
     {"cp_cell_seg_id": "segmentation_id",
      "x_pheno_centroid": "x_pheno",
      "y_pheno_centroid": "y_pheno",
      "x_cp1": "x_cp",
      "y_cp1": "y_cp"}),
    ("std", "",
     ["segmentation_id", "x_pheno", "y_pheno",
      "sgRNA", "barcode", "gene_name", "tile_pheno"],
     {}),
)


def _load_linked_for_exp(exp: str) -> pd.DataFrame:
    """Concat A1/A2/A3 linked CSVs for one experiment with the cols we need.

    Variant-aware: prefers `_4i.csv` then `_cp.csv` then the standard
    `_linked_pheno_iss.csv`. After load the variant-specific columns
    (`4i_segmentation_id`, `cp_cell_seg_id`, `x_pheno_centroid`, etc.) are
    renamed to the standard schema so all downstream code (`_attach_op_features`,
    NTC handling, the per-channel consolidator) is variant-agnostic.

    Enforces a single variant per experiment: if A1 loads as `4i` and A2 as
    `cp`, A2 is skipped with a warning. Mirrors `fe_metadata.py:320-355`.
    """
    pieces = []
    loaded_kind: Optional[str] = None
    for w in _LINKED_WELLS:
        for kind, suffix, cols, rename_map in _LINKED_VARIANTS:
            p = OPS_FAST_ROOT / exp / "3-assembly" / f"{w}_linked_pheno_iss{suffix}.csv"
            if not p.exists():
                continue
            if loaded_kind is None:
                loaded_kind = kind
            elif loaded_kind != kind:
                # Don't mix variants within an experiment.
                continue
            try:
                df = pd.read_csv(p, usecols=cols)
            except Exception as e:
                print(f"  [linked] {exp} {w} ({kind}): {e!r}")
                continue
            if rename_map:
                df = df.rename(columns=rename_map)
            df["experiment"] = exp
            # tile_pheno e.g. "A/1/003025" -> well_canonical "A/1/0".
            df["well_canonical"] = (
                df["tile_pheno"].astype(str).str.rsplit("/", n=1).str[0].map(_canon_well)
            )
            pieces.append(df)
            break  # one CSV per well, first variant wins
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _annotate_sgrna(cells: pd.DataFrame, linked_by_exp: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join sgRNA + barcode from linked CSVs onto cells by (exp, segmentation)."""
    cells = cells.copy()
    cells["__row__"] = np.arange(len(cells))
    pieces = []
    for exp, ldf in linked_by_exp.items():
        if ldf is None or ldf.empty:
            continue
        sub = cells[cells["experiment"] == exp]
        if sub.empty:
            continue
        ld = ldf[["segmentation_id", "sgRNA", "barcode"]].rename(
            columns={"segmentation_id": "segmentation"})
        ld["segmentation"] = pd.to_numeric(ld["segmentation"], errors="coerce").astype("Int64")
        m = sub[["__row__", "segmentation"]].merge(ld, on="segmentation", how="left")
        pieces.append(m[["__row__", "sgRNA", "barcode"]])
    cells["sgRNA"] = ""
    cells["barcode"] = ""
    if pieces:
        merged = pd.concat(pieces).drop_duplicates("__row__").set_index("__row__")
        cells.loc[merged.index, "sgRNA"] = merged["sgRNA"].fillna("").astype(str).values
        cells.loc[merged.index, "barcode"] = merged["barcode"].fillna("").astype(str).values
    return cells.drop(columns="__row__")


def _build_ntc_cells(
    linked_by_exp: Dict[str, pd.DataFrame],
    ko_cells: pd.DataFrame,
    ntc_per_channel: int,
    channel_maps: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Sample NTC cells per (experiment, viz_channel) seen in KO cells.

    For each (exp, viz_channel) pair, draw `ntc_per_channel` random NTC rows
    from that exp's linked CSV pool (`gene_name` NaN or 'NTC'). Output rows
    match the KO master-cells schema (experiment, well_canonical, segmentation,
    x_pheno, y_pheno, sgRNA, barcode, gene='NTC', modality, viz_channel,
    channel_rank, rank, op_channel) so they thread through the same
    `_attach_op_features` / `_attach_cp_features` pipeline.
    """
    import zlib
    pieces = []
    pairs = (
        ko_cells[["experiment", "modality", "viz_channel"]]
        .drop_duplicates().reset_index(drop=True)
    )
    for _, row in pairs.iterrows():
        exp = str(row["experiment"])
        modality = str(row["modality"])
        viz = str(row["viz_channel"])
        ldf = linked_by_exp.get(exp)
        if ldf is None or ldf.empty:
            continue
        gn = ldf["gene_name"].astype("string")
        is_ntc = gn.isna() | (gn.str.upper() == "NTC")
        ntcs = ldf[is_ntc].dropna(subset=["x_pheno", "y_pheno", "segmentation_id"])
        if ntcs.empty:
            continue
        n = min(ntc_per_channel, len(ntcs))
        seed = zlib.crc32(f"{exp}|{modality}|{viz}".encode()) & 0xFFFFFFFF
        s = ntcs.sample(n=n, random_state=seed).copy()
        s["modality"] = modality
        s["viz_channel"] = viz
        s["channel_rank"] = 1
        s["rank"] = np.arange(1, len(s) + 1)
        s["pma_attention"] = np.nan
        s["gene"] = "NTC"
        s["model_confidence"] = np.nan
        s["predicted_class"] = ""
        # well column for compat with KO master cells (canonicalized later if needed)
        s["well"] = s["well_canonical"]
        pieces.append(s)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["segmentation"] = pd.to_numeric(out["segmentation_id"], errors="coerce").astype("Int64")
    # Resolve op_channel via the same yaml the KO path uses.
    out["op_channel"] = [
        _resolve_cell_op_channel(m, v, e, channel_maps)
        for m, v, e in zip(out["modality"], out["viz_channel"], out["experiment"])
    ]
    miss = out[out["op_channel"].isna()]
    if not miss.empty:
        unique = miss[["experiment", "viz_channel"]].drop_duplicates()
        print(f"  [ntc] dropping {len(miss):,} NTC rows with unresolved op_channel "
              f"({len(unique)} unique pairs); first few:\n{unique.head(5).to_string(index=False)}")
        out = out[out["op_channel"].notna()].reset_index(drop=True)
    return out


def _build_global_cells(
    linked_by_exp: Dict[str, pd.DataFrame],
    ko_cells: pd.DataFrame,
    global_per_channel: int,
    channel_maps: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Sample RANDOM cells per (experiment, viz_channel) — any gene.

    Mirrors `_build_ntc_cells` but without the NTC filter. Used to
    populate the "global random" negative cohort for SHAP. The point
    is biological neutrality: cells are NOT selected by attention
    rank (neither top nor bottom), NOT filtered by gene. They're a
    representative cross-section of cells imaged in this channel —
    the closest thing to "median attention" without re-scoring
    attention for the median cells. Output schema matches
    `_build_ntc_cells` so the same downstream OP/CP attach pipeline
    works unchanged; cells are labeled `gene="GLOBAL"` so SHAP can
    filter for them.
    """
    import zlib
    pieces = []
    pairs = (
        ko_cells[["experiment", "modality", "viz_channel"]]
        .drop_duplicates().reset_index(drop=True)
    )
    for _, row in pairs.iterrows():
        exp = str(row["experiment"])
        modality = str(row["modality"])
        viz = str(row["viz_channel"])
        ldf = linked_by_exp.get(exp)
        if ldf is None or ldf.empty:
            continue
        eligible = ldf.dropna(subset=["x_pheno", "y_pheno", "segmentation_id"])
        if eligible.empty:
            continue
        n = min(global_per_channel, len(eligible))
        seed = zlib.crc32(f"global|{exp}|{modality}|{viz}".encode()) & 0xFFFFFFFF
        s = eligible.sample(n=n, random_state=seed).copy()
        s["modality"] = modality
        s["viz_channel"] = viz
        s["channel_rank"] = 1
        s["rank"] = np.arange(1, len(s) + 1)
        s["pma_attention"] = np.nan
        s["gene"] = "GLOBAL"
        s["model_confidence"] = np.nan
        s["predicted_class"] = ""
        s["well"] = s["well_canonical"]
        pieces.append(s)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["segmentation"] = pd.to_numeric(out["segmentation_id"], errors="coerce").astype("Int64")
    out["op_channel"] = [
        _resolve_cell_op_channel(m, v, e, channel_maps)
        for m, v, e in zip(out["modality"], out["viz_channel"], out["experiment"])
    ]
    miss = out[out["op_channel"].isna()]
    if not miss.empty:
        unique = miss[["experiment", "viz_channel"]].drop_duplicates()
        print(f"  [global] dropping {len(miss):,} GLOBAL rows with unresolved "
              f"op_channel ({len(unique)} unique pairs); first few:\n"
              f"{unique.head(5).to_string(index=False)}")
        out = out[out["op_channel"].notna()].reset_index(drop=True)
    return out


def _resolve_cp_h5ad(exp: str, chan: str) -> Optional[Path]:
    """Find the features_processed_<chan>.h5ad for one (exp, channel) pair.

    Tries each `_CP_SUBDIRS` location in order, also trying the abbreviated
    `_CP_CHANNEL_ALIASES` form (so `Concanavalin_A` resolves to `ConA.h5ad`
    when the file lives under `cell-profiler-cp/`). Falls back to a
    case-insensitive directory scan when the exact-case lookup fails — needed
    for 4i markers (`gH2AX`, `pRb`, `pS6`, `NFkB`, `c-Myc`, etc.) whose
    channel-map keys are lowercased but whose CP h5ad filenames preserve case.
    Returns the first match or None.
    """
    candidates = [chan]
    alias = _CP_CHANNEL_ALIASES.get(chan)
    if alias and alias != chan:
        candidates.append(alias)
    base = OPS_FAST_ROOT / exp / "3-assembly"
    for subdir in _CP_SUBDIRS:
        anndata_dir = base / subdir / "anndata_objects"
        for c in candidates:
            f = anndata_dir / f"features_processed_{c}.h5ad"
            if f.exists():
                return f
        # Case-insensitive fallback: scan directory for any case variant.
        if anndata_dir.exists():
            for c in candidates:
                target = f"features_processed_{c}.h5ad".lower()
                for path in anndata_dir.iterdir():
                    if path.name.lower() == target:
                        return path
    return None


def _load_master_cells(
    phase_csv: Optional[Path],
    fluor_csv: Optional[Path],
    top_k: int,
    channel_rank_max: Optional[int],
    experiments: Optional[set],
    channel_maps: Optional[Dict[str, Dict[str, str]]] = None,
    ntc_phase_csv: Optional[Path] = None,
    ntc_fluor_csv: Optional[Path] = None,
    ntc_top_k: int = 100,
) -> pd.DataFrame:
    """Concatenate the two attention CSVs into a single per-cell row table.

    Handles both v2 and v3 attention CSV schemas:
      v2: `viz_channel` (per row) + `channel_rank` (per gene's channel order)
      v3: `channel` (renamed) + `rank_type` ∈ {top, bottom}; channel_rank
          derived here from per-(gene, channel) ordering of `rank`.
    """
    def _normalize_v3_schema(df: pd.DataFrame, modality: str) -> pd.DataFrame:
        """Back-rename v3's `channel` -> `viz_channel`; derive channel_rank."""
        if "viz_channel" not in df.columns and "channel" in df.columns:
            df = df.rename(columns={"channel": "viz_channel"})
        if "channel_rank" not in df.columns and modality == "fluorescent":
            # Per-gene rank of each unique viz_channel: the channel whose
            # top-attention cells score lowest `rank` (= most-attended) is
            # channel_rank 1, then 2, etc. Ties on min(rank) are broken by
            # viz_channel name so each unique channel always gets its own
            # channel_rank (= 1 row per (gene, viz_channel) in the rank
            # table).
            #
            # CRITICAL for CHAD aggregation. CHAD complexes pool cells
            # from multiple member genes, each with its own per-experiment
            # rank-1 cell for every imaged channel. The previous
            # `dense_rank(min(rank))` collapsed every channel whose
            # min(rank) tied at 1 to channel_rank=1 — which for CHAD is
            # every channel. SHAP then iterated `(gene, channel_rank)` and
            # got one mega-classifier per complex labeled with whichever
            # viz_channel pandas happened to land on first, silently
            # dropping the other 55. Sequential cumcount on a
            # (min_rank, viz_channel)-sorted unique key fixes that
            # without changing gene-level behavior (3 markers per gene
            # have distinct min_ranks, no ties).
            ch_min = (df.groupby(["gene", "viz_channel"], sort=False)["rank"]
                        .min().reset_index().rename(columns={"rank": "_min_rank"}))
            ch_min = ch_min.sort_values(
                ["gene", "_min_rank", "viz_channel"], kind="stable"
            )
            ch_min["channel_rank"] = (
                ch_min.groupby("gene", sort=False).cumcount() + 1
            )
            df = df.merge(
                ch_min[["gene", "viz_channel", "channel_rank"]],
                on=["gene", "viz_channel"], how="left",
            )
            df["channel_rank"] = df["channel_rank"].astype(int)
        elif "channel_rank" not in df.columns and modality == "phase":
            df["channel_rank"] = 1
        return df

    frames = []
    if phase_csv is not None:
        ph = pd.read_csv(phase_csv)
        ph = _normalize_v3_schema(ph, "phase").reset_index(drop=True)
        ph = ph[ph["rank"] <= top_k].reset_index(drop=True)
        ph["modality"] = "phase"
        # v3 phase CSVs already carry `channel="Phase"` per row; v2 didn't have
        # the column so we set it explicitly. Either way, force "Phase" as the
        # canonical viz_channel for phase rows.
        ph["viz_channel"] = "Phase"
        ph["channel_rank"] = 1
        frames.append(ph)
        print(f"  phase: {len(ph):,} rows after rank<={top_k}")

    if fluor_csv is not None:
        fl = pd.read_csv(fluor_csv)
        fl = _normalize_v3_schema(fl, "fluorescent").reset_index(drop=True)
        if channel_rank_max is not None:
            fl = fl[fl["channel_rank"] <= channel_rank_max].reset_index(drop=True)
        fl = fl[fl["rank"] <= top_k].reset_index(drop=True)
        fl["modality"] = "fluorescent"
        frames.append(fl)
        print(f"  fluor: {len(fl):,} rows after rank<={top_k}"
              + (f", channel_rank<={channel_rank_max}" if channel_rank_max else ""))

    # NTC top-attention cells — read straight from the PMA NTC CSVs and
    # plug into the same OP/CP attach pipeline as KO cells. `gene` is
    # already "NTC" in those CSVs. We restrict to rank_type=top and
    # take the top `ntc_top_k` per (gene, channel) so the per-channel
    # NTC pool matches the KO positives' top-K convention. This
    # eliminates the need to read NTC features from a different
    # extraction pipeline at SHAP time — the resulting consolidated
    # h5ad carries both KO positives and NTC negatives with
    # identical feature-value scales (no per-experiment z-score, same
    # CP match radius, same per-cell channel mask).
    if ntc_phase_csv is not None:
        nph = pd.read_csv(ntc_phase_csv)
        nph = _normalize_v3_schema(nph, "phase").reset_index(drop=True)
        if "rank_type" in nph.columns:
            nph = nph[nph["rank_type"].astype(str) == "top"].reset_index(drop=True)
        nph = nph[nph["rank"] <= ntc_top_k].reset_index(drop=True)
        nph["modality"] = "phase"
        nph["viz_channel"] = "Phase"
        nph["channel_rank"] = 1
        if "gene" not in nph.columns:
            nph["gene"] = "NTC"
        frames.append(nph)
        print(f"  ntc phase: {len(nph):,} rows after rank<={ntc_top_k}")

    if ntc_fluor_csv is not None:
        nfl = pd.read_csv(ntc_fluor_csv)
        nfl = _normalize_v3_schema(nfl, "fluorescent").reset_index(drop=True)
        if "rank_type" in nfl.columns:
            nfl = nfl[nfl["rank_type"].astype(str) == "top"].reset_index(drop=True)
        nfl = nfl[nfl["rank"] <= ntc_top_k].reset_index(drop=True)
        nfl["modality"] = "fluorescent"
        if "gene" not in nfl.columns:
            nfl["gene"] = "NTC"
        frames.append(nfl)
        print(f"  ntc fluor: {len(nfl):,} rows after rank<={ntc_top_k}")

    if not frames:
        raise SystemExit("Neither --phase-csv nor --fluor-csv supplied.")

    # Union of all CSV columns — preserve every metric Alex tags onto each cell
    # (rank_type, sample_channel_allowlist, channel_rank, viz_channel, etc.).
    # Phase rows get NaN for fluor-only columns and vice versa.
    df = pd.concat(frames, ignore_index=True, sort=False)

    if experiments is not None:
        df = df[df["experiment"].isin(experiments)].copy()
        print(f"  filtered to {len(experiments)} experiments: {len(df):,} rows")

    df["well_canonical"] = df["well"].map(_canon_well)
    df["segmentation"] = pd.to_numeric(df["segmentation"], errors="coerce").astype("Int64")

    # Resolve each cell's physical OP imaging channel ('phase' / 'gfp' /
    # 'mcherry' / 'cy5'). Used downstream to mask OP features by channel.
    # Hard-fail on any unresolved (experiment, viz_channel) pair — the channel
    # map yaml is the source of truth and a miss means stale yaml or a typo,
    # not a legitimate "unknown channel" case to silently ignore.
    if channel_maps is not None:
        df["op_channel"] = [
            _resolve_cell_op_channel(m, v, e, channel_maps)
            for m, v, e in zip(df["modality"], df["viz_channel"], df["experiment"])
        ]
        miss = df[df["op_channel"].isna()]
        if not miss.empty:
            unique_pairs = (
                miss[["experiment", "modality", "viz_channel"]]
                .drop_duplicates()
                .sort_values(["experiment", "viz_channel"])
            )
            n_pairs = len(unique_pairs)
            n_cells = len(miss)
            head = unique_pairs.head(20).to_string(index=False)
            raise SystemExit(
                f"\n[FATAL] {n_cells:,} cells across {n_pairs} unique "
                f"(experiment, viz_channel) pairs could not be resolved to a "
                f"physical OP channel via {CHANNEL_MAPS_YAML}.\n"
                f"Add or fix entries in the channel-maps yaml; a miss means "
                f"the yaml is stale relative to Alex's CSVs.\n\n"
                f"Unresolved pairs (showing up to 20 of {n_pairs}):\n{head}"
            )
        print("  per-cell op_channel:", df["op_channel"].value_counts().to_dict())

    return df.reset_index(drop=True)


def _op_pass1_skim(args):
    """Worker: backed read to collect var names + var metadata for one experiment."""
    exp, h5_path = args
    import anndata as ad
    a = ad.read_h5ad(h5_path, backed="r")
    feats = list(a.var_names)
    var_records = a.var.to_dict("index")
    a.file.close()
    return exp, feats, var_records


def _op_pass2_match(args):
    """Worker: per-experiment exact-seg match (with spatial NN fallback) +
    matched-row backed read.

    Match strategy (in order of preference):
      1. Exact (well, segmentation) → OP `cp_cell_seg_id`. The pma CSV's
         `segmentation` is the imaging-time seg ID (= 4i_segmentation_id
         for 4i exps, = cp_cell_seg_id for CP exps). The OP cell h5ad's
         `cp_cell_seg_id` column preserves that same imaging-time seg
         (despite the "cp_" prefix the consolidator's feature-extraction
         step uses unified naming for CP and 4i). This match is
         deterministic — no coordinate ambiguity, no radius tuning.
      2. Exact (well, segmentation) → OP `segmentation_id`. For std
         (live-cell) experiments where `cp_cell_seg_id` is missing or
         pma's `segmentation` is the live seg.
      3. Spatial NN with `match_radius_px` — fallback for cells where
         neither exact key matches. Mostly relevant for very old caches
         that lack the seg-id columns.

    Why this matters: the prior spatial-NN-only path missed >95% of
    4i/CP cells because pma's `x_pheno` is offset ~24-34 px from the
    OP h5ad's `x_global_pheno` (different upstream coord computation),
    pushing most cells outside the 20px radius even though the cells
    ARE in the OP h5ad and could be matched by seg-id directly.

    Returns (exp, target_rows, exp_cols_in_union, X_slice, n_hit, n_total, msg).
    target_rows are indices into the master `cells` dataframe; X_slice is shape
    (len(target_rows), len(exp_cols_in_union)) and parent scatters via np.ix_.
    """
    exp, h5_path, sub_cells, feature_order, match_radius_px = args
    import anndata as ad
    from scipy.spatial import cKDTree

    feat_idx = {f: i for i, f in enumerate(feature_order)}
    t0 = time.time()
    try:
        a = ad.read_h5ad(h5_path, backed="r")
    except (FileNotFoundError, OSError) as e:
        return (exp, None, None, None, 0, len(sub_cells),
                f"READ ERROR: {e.__class__.__name__}")

    obs = a.obs
    well_canon = obs.get("well", pd.Series([pd.NA] * len(obs), index=obs.index)).astype("string")
    cell_ids = obs.get("cell_id", pd.Series(obs.index, index=obs.index)).astype(str)
    cid_well = cell_ids.str.rsplit("_", n=1, expand=True)[0]
    well_canon = well_canon.fillna(cid_well).astype(str)
    well_canon = well_canon.where(well_canon.str.count("/") <= 2,
                                  well_canon.map(_strip_well))

    # Guard against missing coord columns on legacy h5ads — `obs.get`
    # returns None when absent; `pd.to_numeric(None)` returns a scalar
    # nan with no `.to_numpy()`. Fall back to an all-NaN array so the
    # downstream spatial filter cleanly treats every row as invalid.
    def _coord_to_array(colname):
        if colname not in obs.columns:
            return np.full(len(obs), np.nan, dtype=np.float64)
        return pd.to_numeric(obs[colname], errors="coerce").to_numpy(dtype=np.float64)

    x_op = _coord_to_array("x_global_pheno")
    y_op = _coord_to_array("y_global_pheno")
    valid_op = np.isfinite(x_op) & np.isfinite(y_op)

    exp_var_names = list(a.var_names)
    exp_cols_in_union = np.array(
        [feat_idx[f] for f in exp_var_names if f in feat_idx], dtype=np.int64,
    )
    local_cols = np.array(
        [k for k, f in enumerate(exp_var_names) if f in feat_idx], dtype=np.int64,
    )

    well_arr = well_canon.values

    # ------------------------------------------------------------------
    # Build (well, seg) → row-index lookup tables for both seg columns
    # the OP h5ad might carry. New 4i runs (post-2026-05-10) save the
    # imaging-time seg as `4i_cell_seg_id`; CP runs save it as
    # `cp_cell_seg_id`; legacy 4i runs (before the rename fix) also
    # used `cp_cell_seg_id`. Try whichever the obs has — same key
    # semantics either way. Live-cell-only experiments have neither
    # imaging-time-seg column; we fall back to all-NaN so the exact-key
    # match is a no-op there and spatial NN handles everything.
    fluor_seg_col = (
        "4i_cell_seg_id" if "4i_cell_seg_id" in obs.columns
        else "cp_cell_seg_id" if "cp_cell_seg_id" in obs.columns
        else None
    )

    def _seg_series_to_array(colname):
        """Coerce an obs column to a numeric numpy array, or return an
        all-NaN array of the right length if the column is missing.
        `obs.get(missing)` returns None, and `pd.to_numeric(None)` is a
        scalar nan with no `.to_numpy()` — guard against that here."""
        if colname is None or colname not in obs.columns:
            return np.full(len(obs), np.nan, dtype=np.float64)
        return pd.to_numeric(obs[colname], errors="coerce").to_numpy(dtype=np.float64)

    cp_seg_op = _seg_series_to_array(fluor_seg_col)
    live_seg_op = _seg_series_to_array("segmentation_id")
    cp_lookup: Dict[Tuple[str, int], int] = {}
    live_lookup: Dict[Tuple[str, int], int] = {}
    for i in range(len(obs)):
        w = well_arr[i]
        if pd.notna(cp_seg_op[i]):
            cp_lookup[(w, int(cp_seg_op[i]))] = i
        if pd.notna(live_seg_op[i]):
            live_lookup[(w, int(live_seg_op[i]))] = i

    # Match strategy: exact `(well, segmentation)` lookup ONLY.
    # The previous spatial-NN fallback caught 1/3500 cells (0.03%) in
    # empirical testing across live-cell, CP, and 4i experiments — pma
    # CSV segmentation IDs are already authoritative (live seg for
    # live-cell exps, imaging-time seg for CP/4i), so the seg-id maps
    # cover everything that's actually recoverable. Cells that miss
    # both maps simply don't exist in the OP h5ad (incomplete feature
    # extraction); spatial NN won't conjure them up.
    all_targets: list[np.ndarray] = []
    all_srcs: list[np.ndarray] = []
    n_exact_cp = 0
    n_exact_live = 0

    for w, sub in sub_cells.groupby("well_canonical", sort=False):
        sub_segs = pd.to_numeric(sub["segmentation"], errors="coerce").to_numpy()
        sub_idx_arr = sub.index.to_numpy()

        for k in range(len(sub_segs)):
            seg = sub_segs[k]
            if not np.isfinite(seg):
                continue
            key = (w, int(seg))
            row = cp_lookup.get(key)
            if row is None:
                row = live_lookup.get(key)
                if row is not None:
                    n_exact_live += 1
            else:
                n_exact_cp += 1
            if row is not None:
                all_targets.append(np.array([sub_idx_arr[k]], dtype=np.int64))
                all_srcs.append(np.array([row], dtype=np.int64))

    n_total = len(sub_cells)
    if not all_targets:
        a.file.close()
        return (exp, np.array([], dtype=np.int64), exp_cols_in_union,
                np.zeros((0, len(local_cols)), dtype=np.float32), 0, n_total,
                f"matched 0/{n_total:,} in {time.time() - t0:.1f}s")

    target_rows = np.concatenate(all_targets)
    src_rows = np.concatenate(all_srcs)

    # Backed fancy-indexed read of only matched rows. np.unique returns sorted
    # unique values, which is what h5py wants on the first axis.
    unique_src, inverse = np.unique(src_rows, return_inverse=True)
    X_unique = np.asarray(a.X[unique_src, :], dtype=np.float32)
    X_slice = X_unique[:, local_cols][inverse]

    n_hit = len(target_rows)
    msg = (f"matched {n_hit:,}/{n_total:,} ({n_hit / max(n_total, 1):.0%}) "
           f"[exact:cp={n_exact_cp:,} exact:live={n_exact_live:,}]; "
           f"{len(exp_var_names)} feats; {time.time() - t0:.1f}s")
    a.file.close()
    return exp, target_rows, exp_cols_in_union, X_slice, n_hit, n_total, msg


def _cp_pass1_skim(args):
    """Worker: backed read of one (exp, channel) features_processed h5ad var names."""
    exp, chan, h5_path = args
    import anndata as ad
    a = ad.read_h5ad(h5_path, backed="r")
    names = list(a.var_names)
    a.file.close()
    return exp, chan, names


def _cp_pass2_match(args):
    """Worker: per-(exp, channel) spatial NN match + matched-row backed read."""
    exp, chan, h5_path, sub_cells, cp_feature_names, match_radius_px = args
    import anndata as ad
    from scipy.spatial import cKDTree

    t0 = time.time()
    try:
        a = ad.read_h5ad(h5_path, backed="r")
    except (FileNotFoundError, OSError) as e:
        return (exp, chan, None, None, 0, len(sub_cells),
                f"READ ERROR: {e.__class__.__name__}")

    col_idx = np.array([a.var_names.get_loc(f) for f in cp_feature_names], dtype=np.int64)
    well_canon = a.obs["well"].astype(str).map(_strip_well)
    xy = a.obs[["x_position", "y_position"]].to_numpy(dtype=np.float64)
    well_arr = well_canon.values

    # Deterministic seg-id join when the CP h5ad carries pheno-frame seg ids
    # (written by the seg-id-aware CP extraction). x_position vs x_cp frames
    # differ for Cell-Painting experiments, so prefer the seg join; spatial NN
    # is only a fallback for cells it misses (or older seg-less CP h5ads).
    cp_seg = (pd.to_numeric(a.obs["segmentation_id"], errors="coerce").to_numpy()
              if "segmentation_id" in a.obs.columns else None)
    has_sub_seg = "segmentation" in sub_cells.columns

    all_targets, all_srcs = [], []
    for w, sub in sub_cells.groupby("well_canonical", sort=False):
        mask = (well_arr == w)
        if not mask.any():
            continue
        cp_rows = np.flatnonzero(mask)
        matched = np.full(len(sub), -1, dtype=np.int64)

        # 1. deterministic (well, seg) join
        if cp_seg is not None and has_sub_seg:
            seg_to_row = {}
            for r in cp_rows:
                s = cp_seg[r]
                if np.isfinite(s):
                    seg_to_row.setdefault(int(s), r)
            sub_seg = pd.to_numeric(sub["segmentation"], errors="coerce").to_numpy()
            for i in range(len(sub_seg)):
                s = sub_seg[i]
                if np.isfinite(s):
                    r = seg_to_row.get(int(s))
                    if r is not None:
                        matched[i] = r

        # 2. spatial-NN fallback for still-unmatched cells
        need = matched < 0
        if need.any():
            tree = cKDTree(xy[mask])
            qry = sub.loc[need, ["x_pheno", "y_pheno"]].to_numpy(dtype=np.float64)
            dists, ii = tree.query(qry, k=1)
            ok = dists <= match_radius_px
            need_pos = np.flatnonzero(need)
            matched[need_pos[ok]] = cp_rows[ii[ok]]

        got = matched >= 0
        if got.any():
            all_targets.append(sub.index.to_numpy()[got])
            all_srcs.append(matched[got])

    n_total = len(sub_cells)
    if not all_targets:
        a.file.close()
        return (exp, chan, np.array([], dtype=np.int64),
                np.zeros((0, len(cp_feature_names)), dtype=np.float32), 0, n_total,
                f"matched 0/{n_total:,} in {time.time() - t0:.1f}s")

    target_rows = np.concatenate(all_targets)
    src_rows = np.concatenate(all_srcs)
    unique_src, inverse = np.unique(src_rows, return_inverse=True)
    X_unique = np.asarray(a.X[unique_src, :], dtype=np.float32)
    X_slice = X_unique[:, col_idx][inverse]

    n_hit = len(target_rows)
    msg = (f"matched {n_hit:,}/{n_total:,} ({n_hit / max(n_total, 1):.0%}) "
           f"in {time.time() - t0:.1f}s")
    a.file.close()
    return exp, chan, target_rows, X_slice, n_hit, n_total, msg


def _attach_op_features(
    cells: pd.DataFrame,
    match_radius_px: float,
    n_workers: int,
    per_cell_channel_mask: bool = True,
    organelle_filter: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, np.ndarray, Dict]:
    """Return (X_op, op_feature_names, var_op, hit_mask) aligned to cells.index.

    Across the v1-paper set the OP feature schema is heterogeneous (different
    experiments segment different organelles), so we take the union of var
    names. Each experiment fills its own columns; cells x feature pairs not
    present in that experiment's h5ad keep NaN.

    Matching is spatial nearest-neighbor (same as CP): for each experiment
    and well, build a KDTree over OP cells' (x_global_pheno, y_global_pheno)
    and look up the nearest OP cell for each CSV (x_pheno, y_pheno) within
    `match_radius_px`. Sidesteps the seg-id ambiguity across workflows
    (cp_cell_seg_id, 4i_cell_seg_id, segmentation_id).

    Per-experiment work runs on a ProcessPoolExecutor with `n_workers` procs;
    each worker reads its h5ad backed and only loads the matched rows of X.

    `organelle_filter`: optional list of OP-channel buckets to retain. When
    set, the feature union is pre-filtered at pass-1 to features whose
    `_organelle_to_op_channel(var.organelle)` is in the filter — drops the
    dense X_op allocation by 3-6× for single-channel callers (e.g. per-channel
    consolidation: `["phase", "agnostic"]` for Phase, `["gfp", "agnostic"]`
    for GFP-imaged markers). Default `None` preserves the original full-union
    behavior used by the top-attention pipeline.
    """
    n = len(cells)
    hit = np.zeros(n, dtype=bool)
    exps_in_csv = list(cells["experiment"].unique())

    available_paths: Dict[str, Path] = {}
    for exp in exps_in_csv:
        h5_path = OPS_FAST_ROOT / exp / "3-assembly" / "feature_extraction" / f"{exp}_cell_features.h5ad"
        if h5_path.exists():
            available_paths[exp] = h5_path

    if not available_paths:
        raise SystemExit("No OP h5ads were readable; aborting.")

    # ------------------------------------------------------------------
    # Pass 1 — backed-mode skim (parallel) to collect var-name union.
    # joblib's loky backend uses cloudpickle, which handles the case where
    # submitit ships run_consolidation across SLURM nodes (default
    # ProcessPoolExecutor's ForkingPickler can't pickle __main__ helpers).
    # ------------------------------------------------------------------
    pass1_args = [(exp, str(p)) for exp, p in available_paths.items()]
    feature_to_var: Dict[str, dict] = {}
    feature_order: List[str] = []
    pass1_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_op_pass1_skim)(a) for a in pass1_args
    )
    for exp, feats, var_records in pass1_results:
        for f in feats:
            if f not in feature_to_var:
                feature_to_var[f] = var_records[f]
                feature_order.append(f)

    if not feature_order:
        raise SystemExit("OP h5ads had no var names; aborting.")

    # Drop features whose `var.organelle` doesn't map to any channel a cell
    # could reference (i.e. `_organelle_to_op_channel` -> 'exclude'). With the
    # channel-map driving the per-cell mask, any 'exclude' feature would be
    # 100% NaN — no need to allocate the column.
    n_before = len(feature_order)
    kept = [
        f for f in feature_order
        if _organelle_to_op_channel(feature_to_var[f].get("organelle")) != "exclude"
    ]
    n_dropped = n_before - len(kept)
    if n_dropped:
        print(f"  OP auto-drop: removed {n_dropped:,} of {n_before:,} features "
              f"whose organelle has no matching cell channel (e.g. 4i_*); "
              f"kept {len(kept):,}")
    feature_order = kept
    feature_to_var = {f: feature_to_var[f] for f in feature_order}

    # Optional channel-aware pre-filter: when the caller knows the cells DataFrame
    # only references a single OP channel (per-channel consolidation), restrict
    # feature_order to features whose organelle bucket is in `organelle_filter`.
    # This shrinks the dense X_op allocation by 3-6× compared to the full
    # paper_v1 union (e.g. ~9785 → ~1500 for Phase). The per-cell channel mask
    # below is now a near-no-op since we already pre-selected the relevant
    # columns.
    if organelle_filter is not None:
        filter_set = set(organelle_filter)
        n_before_filter = len(feature_order)
        kept = [
            f for f in feature_order
            if _organelle_to_op_channel(feature_to_var[f].get("organelle")) in filter_set
        ]
        n_dropped_filter = n_before_filter - len(kept)
        print(f"  OP organelle filter ({sorted(filter_set)}): kept {len(kept):,}/"
              f"{n_before_filter:,} features ({n_dropped_filter:,} dropped)")
        feature_order = kept
        feature_to_var = {f: feature_to_var[f] for f in feature_order}

    if not feature_order:
        raise SystemExit(
            "OP auto-drop removed every feature; check `var.organelle` values "
            "and `_organelle_to_op_channel`."
        )

    var_op = pd.DataFrame([feature_to_var[f] for f in feature_order], index=feature_order)
    F = len(feature_order)
    X_op = np.full((n, F), np.nan, dtype=np.float32)
    print(f"  OP union schema: {F} features across {len(available_paths)} experiments")

    # ------------------------------------------------------------------
    # Pass 2 — per-experiment spatial NN match in parallel.
    # ------------------------------------------------------------------
    pass2_args = []
    missing_files: List[Tuple[str, int]] = []
    for exp, grp in cells.groupby("experiment", sort=False):
        h5_path = available_paths.get(exp)
        if h5_path is None:
            print(f"  [OP MISS] {exp}: h5ad not found, skipping {len(grp):,} cells")
            missing_files.append((exp, len(grp)))
            continue
        # `segmentation` is the imaging-time seg from pma — `_op_pass2_match`
        # uses it as the primary exact-match key into the OP h5ad's
        # `cp_cell_seg_id` (4i/CP) or `segmentation_id` (std). Spatial NN
        # is the fallback only.
        sub_cells = grp[["well_canonical", "segmentation", "x_pheno", "y_pheno"]].copy()
        sub_cells.index = grp.index
        pass2_args.append((exp, str(h5_path), sub_cells, feature_order, match_radius_px))

    pass2_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_op_pass2_match)(a) for a in pass2_args
    )
    per_exp: List[Tuple[str, int, int, int]] = []
    for result in pass2_results:
        exp, target_rows, exp_cols, X_slice, n_hit, n_total, msg = result
        print(f"  [OP] {exp}: {msg}")
        n_feats = int(len(exp_cols)) if exp_cols is not None else 0
        per_exp.append((exp, int(n_hit), int(n_total), n_feats))
        if target_rows is not None and len(target_rows):
            X_op[np.ix_(target_rows.astype(np.int64), exp_cols)] = X_slice
            hit[target_rows] = True

    # ------------------------------------------------------------------
    # Per-cell channel mask — each row only keeps features whose imaging
    # channel matches the cell's own channel (or features tagged 'agnostic',
    # i.e. cell/nucleus morphology). Implements the policy that a phase row
    # only gets OP phase features, a GFP row only gets GFP features, etc.
    # ------------------------------------------------------------------
    if per_cell_channel_mask and "op_channel" in cells.columns:
        feat_channels = np.array(
            [_organelle_to_op_channel(var_op.iloc[i].get("organelle"))
             for i in range(F)],
            dtype=object,
        )
        if cells["op_channel"].isna().any():
            n_na = int(cells["op_channel"].isna().sum())
            raise SystemExit(
                f"[FATAL] {n_na:,} cells have NaN op_channel reaching the OP "
                "mask step — should have been caught at load time."
            )
        cell_channels = cells["op_channel"].astype(str).to_numpy()
        agnostic_cols = (feat_channels == "agnostic")
        n_masked_pairs = 0
        for chan in np.unique(cell_channels):
            rows = np.flatnonzero(cell_channels == chan)
            if len(rows) == 0:
                continue
            keep = (feat_channels == chan) | agnostic_cols
            drop = np.flatnonzero(~keep)
            if len(drop):
                X_op[np.ix_(rows, drop)] = np.nan
                n_masked_pairs += len(rows) * len(drop)
        chan_counts = pd.Series(cell_channels).value_counts().to_dict()
        feat_chan_counts = pd.Series(feat_channels).value_counts().to_dict()
        print(f"  OP per-cell channel mask: NaN'd {n_masked_pairs:,} "
              f"(cell, feature) pairs across {len(chan_counts)} cell channels")
        print(f"    cells per channel:    {chan_counts}")
        print(f"    features per channel: {feat_chan_counts}")

    op_stats = {"per_exp": per_exp, "missing_files": missing_files}
    return X_op, feature_order, var_op, hit, op_stats


def _attach_cp_features(
    cells: pd.DataFrame,
    match_radius_px: float,
    n_workers: int,
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict]:
    """Return (X_cp, cp_feature_names, hit_mask) aligned to cells.index.

    CP features come from per-channel features_processed_<chan>.h5ad. We use the
    intersection of var names across all touched channels as the shared schema
    (CellProfiler emits the same module set per channel, so this is ~all features
    minus a few channel-specific ones). Per-(exp, channel) work runs in parallel
    on a ProcessPoolExecutor; each worker reads backed and only materializes the
    matched rows of X.
    """
    # Tag every row with its CP channel up front so we can count cells per
    # (exp, channel) — needed for the coverage report's missing-pair sizes.
    cells_with_chan = cells.assign(
        _chan=lambda d: np.where(d["modality"] == "phase", "Phase",
                                 d["viz_channel"].map(_viz_channel_to_file))
    )
    cells_per_pair: Dict[Tuple[str, str], int] = (
        cells_with_chan.groupby(["experiment", "_chan"], sort=False)
        .size().to_dict()
    )
    needed: List[Tuple[str, str]] = sorted(cells_per_pair.keys())

    pass1_args = []
    missing_pairs: List[Tuple[str, str, int]] = []
    resolved_paths: Dict[Tuple[str, str], Path] = {}
    for exp, chan in needed:
        h5_path = _resolve_cp_h5ad(exp, chan)
        if h5_path is None:
            missing_pairs.append((exp, chan, int(cells_per_pair[(exp, chan)])))
            continue
        resolved_paths[(exp, chan)] = h5_path
        pass1_args.append((exp, chan, str(h5_path)))

    if not pass1_args:
        raise SystemExit("No CP h5ads found; check paths or pass --no-cp.")

    # Pass 1: parallel backed skim for var-name intersection.
    var_intersection: Optional[set] = None
    var_first_order: Optional[List[str]] = None
    pass1_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_cp_pass1_skim)(a) for a in pass1_args
    )
    for exp, chan, names in pass1_results:
        if var_intersection is None:
            var_intersection = set(names)
            var_first_order = names
        else:
            var_intersection &= set(names)

    cp_feature_names = [v for v in var_first_order if v in var_intersection]
    print(f"  CP shared var schema: {len(cp_feature_names)} features "
          f"across {len(needed) - len(missing_pairs)}/{len(needed)} (exp, channel) pairs")
    if missing_pairs:
        print(f"  [CP MISS] {len(missing_pairs)} (exp, channel) pairs missing — "
              f"those cells get NaN CP rows. Sample: "
              f"{[(e, c) for e, c, _ in missing_pairs[:3]]}")

    n = len(cells)
    X_cp = np.full((n, len(cp_feature_names)), np.nan, dtype=np.float32)
    hit = np.zeros(n, dtype=bool)

    # Pass 2: parallel per-(exp, channel) match.
    # CP h5ads carry obs.x_position/obs.y_position at CP IMAGING TIME, not
    # pheno time. For CP/4i experiments cells move between sessions, so the
    # spatial NN match has to use CP-time coords too (`x_cp`/`y_cp`, populated
    # by `_load_linked_for_exp`'s variant detection). For live-cell experiments
    # `x_cp`/`y_cp` aren't loaded, so we fall back to pheno coords (which
    # equal CP-time coords there since there's only one imaging session). The
    # per-axis fallback handles partially-populated columns gracefully.
    pass2_args = []
    has_cp_coords = "x_cp" in cells_with_chan.columns and "y_cp" in cells_with_chan.columns
    for (exp, chan), grp in cells_with_chan.groupby(["experiment", "_chan"], sort=False):
        h5_path = resolved_paths.get((exp, chan))
        if h5_path is None:
            continue
        seg_cols = ["segmentation"] if "segmentation" in grp.columns else []
        sub_cells = grp[["well_canonical", *seg_cols, "x_pheno", "y_pheno"]].copy()
        sub_cells.index = grp.index
        if has_cp_coords:
            x_cp = grp["x_cp"]
            y_cp = grp["y_cp"]
            ok = x_cp.notna() & y_cp.notna()
            if ok.any():
                sub_cells.loc[ok.values, "x_pheno"] = x_cp[ok].astype(np.float64).values
                sub_cells.loc[ok.values, "y_pheno"] = y_cp[ok].astype(np.float64).values
        pass2_args.append((exp, chan, str(h5_path), sub_cells, cp_feature_names, match_radius_px))

    pass2_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_cp_pass2_match)(a) for a in pass2_args
    )
    per_pair: List[Tuple[str, str, int, int]] = []
    for result in pass2_results:
        exp, chan, target_rows, X_slice, n_hit, n_total, msg = result
        print(f"  [CP] {exp}/{chan}: {msg}")
        per_pair.append((exp, chan, int(n_hit), int(n_total)))
        if target_rows is not None and len(target_rows):
            X_cp[target_rows, :] = X_slice
            hit[target_rows] = True

    cp_stats = {
        "per_pair": per_pair,
        "missing_pairs": missing_pairs,
        "n_features": len(cp_feature_names),
        "n_attempted": len(needed),
    }
    return X_cp, cp_feature_names, hit, cp_stats


def _audit_output_nans(
    name: str,
    X: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    n_op_cols: int,
) -> List[str]:
    """Build a NaN / column-alignment audit section for one output h5ad.

    Catches misalignment / unexpected NaN spikes:
      - Overall NaN fraction (sanity check)
      - Per-`op_channel` NaN rate over the OP block — surfaces channels that
        somehow lost their values (e.g., an experiment whose OP match rate
        is dragging coverage down for that channel).
      - Per-feature top-K worst NaN columns — surfaces individual columns
        that didn't get filled (likely misaligned metadata).
    """
    L: List[str] = []
    L.append(f"### {name} ({X.shape[0]:,} rows × {X.shape[1]:,} cols, "
             f"{n_op_cols:,} OP cols + {X.shape[1] - n_op_cols:,} CP cols)")
    L.append("")

    overall = float(np.isnan(X).mean())
    L.append(f"- Overall NaN fraction: **{overall:.1%}**")

    # OP block NaN per cell channel
    if "op_channel" in obs.columns and n_op_cols > 0:
        X_op_block = X[:, :n_op_cols]
        L.append("")
        L.append(f"**OP block NaN by `op_channel`** "
                 f"(unexpectedly high → cells from that channel may be losing "
                 f"OP features through misalignment, not just OP-match misses):")
        L.append("")
        L.append("| op_channel | cells | OP NaN frac |")
        L.append("|---|---:|---:|")
        chan_counts = obs["op_channel"].astype(str).value_counts()
        for chan, n in chan_counts.items():
            mask = (obs["op_channel"].astype(str).values == chan)
            sub = X_op_block[mask]
            nan_frac = float(np.isnan(sub).mean()) if sub.size else 0.0
            flag = " ⚠️" if nan_frac > 0.5 else ""
            L.append(f"| `{chan}` | {int(n):,} | {nan_frac:.1%}{flag} |")
        L.append("")

    # Per-feature NaN: surface the top-10 worst columns.
    nan_per_col = np.isnan(X).mean(axis=0)
    high = sorted(
        ((var.index[i], float(nan_per_col[i])) for i in range(len(var))),
        key=lambda r: -r[1],
    )[:10]
    L.append("**Top 10 highest-NaN columns** "
             "(>95% NaN often indicates a column nothing filled):")
    L.append("")
    L.append("| Column | NaN frac |")
    L.append("|---|---:|")
    for col, frac in high:
        flag = " ⚠️" if frac > 0.95 else ""
        L.append(f"| `{col}` | {frac:.1%}{flag} |")
    L.append("")
    n_above_95 = int((nan_per_col > 0.95).sum())
    n_above_99 = int((nan_per_col > 0.99).sum())
    L.append(f"- Columns >95% NaN: **{n_above_95:,}** of {len(var):,}")
    L.append(f"- Columns >99% NaN: **{n_above_99:,}** of {len(var):,}")
    L.append("")
    return L


def _write_coverage_report(
    output_dir: Path,
    cells: pd.DataFrame,
    op_stats: Dict,
    cp_stats: Optional[Dict] = None,
    output_summaries: Optional[Dict[str, Dict]] = None,
    output_files: Optional[List[Path]] = None,
) -> Path:
    """Write coverage_report.md alongside the consolidated h5ads.

    Sections:
      1. NaN / column-alignment audit per output (catches alignment bugs)
      2. Match-coverage summary (OP / CP)
      3. Missing OP / CP h5ads + per-experiment / per-(exp, channel) tables.
    """
    from datetime import datetime

    n_master = len(cells)
    n_exp = int(cells["experiment"].nunique())
    n_genes = int(cells["gene"].nunique()) if "gene" in cells.columns else 0

    L: List[str] = []
    L.append("# Coverage Report — top_attention_cells")
    L.append("")
    L.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    if output_files:
        L.append("Outputs:")
        for f in output_files:
            shape = None
            if output_summaries:
                summ = output_summaries.get(f.stem.replace("top_attention_cells_", ""))
                if summ is not None:
                    shape = summ["X"].shape
            shape_str = f" ({shape[0]:,} × {shape[1]:,})" if shape else ""
            L.append(f"- `{f.name}`{shape_str}")
        L.append("")

    # ---- NaN / alignment audit (top of report) ----
    if output_summaries:
        L.append("## NaN / column-alignment audit")
        L.append("")
        L.append("Quick-look check for unexpected NaN spikes that would point "
                 "at misalignment between obs.op_channel and var.organelle, "
                 "or columns nothing filled. Flagged cells (⚠️) warrant "
                 "investigation.")
        L.append("")
        for name, summ in output_summaries.items():
            L.extend(_audit_output_nans(
                name, summ["X"], summ["obs"], summ["var"], summ["n_op_cols"],
            ))
        L.append("---")
        L.append("")

    # ---- Summary ----
    L.append("## Summary")
    L.append("")
    op_hit = sum(s[1] for s in op_stats["per_exp"])
    op_total = (sum(s[2] for s in op_stats["per_exp"])
                + sum(n for _, n in op_stats["missing_files"]))
    L.append("| Pass | Matched | Total | Missing | Coverage |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| OP | {op_hit:,} | {op_total:,} | {op_total - op_hit:,} | "
             f"{op_hit / max(op_total, 1):.1%} |")
    if cp_stats:
        cp_hit = sum(s[2] for s in cp_stats["per_pair"])
        cp_total = (sum(s[3] for s in cp_stats["per_pair"])
                    + sum(n for _, _, n in cp_stats["missing_pairs"]))
        L.append(f"| CP | {cp_hit:,} | {cp_total:,} | {cp_total - cp_hit:,} | "
                 f"{cp_hit / max(cp_total, 1):.1%} |")
    L.append("")
    L.append(f"- {n_exp} experiments, {n_genes:,} genes, {n_master:,} master cells.")
    L.append(f"- Missing OP h5ad files: **{len(op_stats['missing_files'])}**.")
    if cp_stats:
        L.append(f"- Missing CP h5ad files: **{len(cp_stats['missing_pairs'])}** "
                 f"of {cp_stats['n_attempted']} needed (exp, channel) pairs.")
    L.append("")
    L.append("---")
    L.append("")

    # ---- OP ----
    L.append("## OP — OrganelleProfiler features")
    L.append("")
    L.append("### Missing OP h5ad files")
    L.append("")
    if not op_stats["missing_files"]:
        L.append("**None.** All experiments have an `<exp>_cell_features.h5ad` "
                 "under `3-assembly/feature_extraction/`.")
    else:
        L.append("| Experiment | Affected cells |")
        L.append("|---|---:|")
        for exp, n in sorted(op_stats["missing_files"], key=lambda r: -r[1]):
            L.append(f"| `{exp}` | {n:,} |")
    L.append("")
    L.append("### Per-experiment match coverage (sorted by miss count, descending)")
    L.append("")
    L.append("| Experiment | Matched | Total | Missing | Coverage | OP features |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for exp, hit, total, feats in sorted(op_stats["per_exp"],
                                         key=lambda r: r[2] - r[1], reverse=True):
        miss = total - hit
        pct = hit / total if total else 0
        L.append(f"| `{exp}` | {hit:,} | {total:,} | {miss:,} | {pct:.1%} | {feats:,} |")
    L.append("")

    # ---- CP ----
    if cp_stats:
        L.append("---")
        L.append("")
        L.append("## CP — CellProfiler features")
        L.append("")
        L.append("### Missing CP h5ad files")
        L.append("")
        n_missing = len(cp_stats["missing_pairs"])
        if n_missing == 0:
            L.append("**None.** All needed (exp, channel) pairs have a "
                     "`features_processed_<chan>.h5ad`.")
        else:
            total_missing_cells = sum(n for _, _, n in cp_stats["missing_pairs"])
            L.append(f"{n_missing} of {cp_stats['n_attempted']} needed (exp, channel) "
                     f"pairs have no `features_processed_<chan>.h5ad`. The cells from "
                     f"those pairs get **NaN CP rows** in the combined output.")
            L.append("")
            L.append("| Experiment | Channel | Affected cells |")
            L.append("|---|---|---:|")
            for e, c, n in sorted(cp_stats["missing_pairs"], key=lambda r: -r[2]):
                L.append(f"| `{e}` | `{c}` | {n:,} |")
            L.append("")
            L.append(f"**Total cells missing CP entirely (file absent): "
                     f"{total_missing_cells:,}**")
        L.append("")

        # Per-experiment CP roll-up combining matched + missing-file losses.
        cp_by_exp: Dict[str, List[int]] = {}
        for e, c, hit, total in cp_stats["per_pair"]:
            cp_by_exp.setdefault(e, [0, 0])
            cp_by_exp[e][0] += hit
            cp_by_exp[e][1] += total
        for e, c, n in cp_stats["missing_pairs"]:
            cp_by_exp.setdefault(e, [0, 0])
            cp_by_exp[e][1] += n
        L.append("### Per-experiment CP roll-up (sorted by miss count, descending)")
        L.append("")
        L.append("| Experiment | Matched | Total | Missing | Coverage |")
        L.append("|---|---:|---:|---:|---:|")
        for exp, (hit, total) in sorted(cp_by_exp.items(),
                                        key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
            miss = total - hit
            pct = hit / total if total else 0
            L.append(f"| `{exp}` | {hit:,} | {total:,} | {miss:,} | {pct:.1%} |")
        L.append("")
        L.append("### Per (exp, channel) match coverage (sorted by miss count, descending)")
        L.append("")
        L.append("| Experiment | Channel | Matched | Total | Missing | Coverage |")
        L.append("|---|---|---:|---:|---:|---:|")
        combined = [(e, c, h, t, "") for e, c, h, t in cp_stats["per_pair"]]
        combined += [(e, c, 0, n, " (file missing)")
                     for e, c, n in cp_stats["missing_pairs"]]
        for e, c, hit, total, flag in sorted(combined,
                                             key=lambda r: r[3] - r[2], reverse=True):
            miss = total - hit
            pct = hit / total if total else 0
            L.append(f"| `{e}` | `{c}`{flag} | {hit:,} | {total:,} | {miss:,} | "
                     f"{pct:.1%} |")
        L.append("")

    out_path = output_dir / "coverage_report.md"
    out_path.write_text("\n".join(L))
    return out_path


def _split_op_outputs(
    X_op: np.ndarray,
    var_op: pd.DataFrame,
    obs: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]:
    """Split the post-attach OP matrix into phase + fluor outputs.

    Phase output (`top_attention_cells_phase.h5ad`):
      Rows: every CSV row with `modality == 'phase'`.
      Cols: every var whose channel is 'phase' (phase2d_*, focus3d_*,
      nucleoli_phase2d, nucleoli_focus3d) or 'agnostic' (cell, nuclei,
      cell_morphology, cp_cell). Phase organelle features stay distinct —
      `phase2d_tubular_*` and `phase2d_vesicular_*` measure different
      morphology classes, not different reporter channels.

    Fluor output (`top_attention_cells_fluor.h5ad`):
      Rows: every CSV row with `modality == 'fluorescent'`.
      Cols: a *unified* set of 287 columns built by collapsing every fluor
      channel (gfp, mcherry, cy5, cp1_*, cp2_*) into shared `(metric,
      aggregation)` pairs. Each cell pulls from its own channel's source
      column based on `obs.op_channel`. Plus agnostic columns passed through.
      The `obs.op_channel` column makes the source channel unambiguous.

    Returns: {'phase': (X, obs, var), 'fluor': (X, obs, var)}.
    """
    var_organelle = var_op["organelle"].astype(str).replace({"nan": ""})
    var_metric = var_op["metric"].astype(str).replace({"nan": ""})
    var_aggr = var_op["aggregation"].astype(str).replace({"nan": ""})

    feat_chan = np.array(
        [_organelle_to_op_channel(o) for o in var_organelle], dtype=object
    )

    is_phase = feat_chan == "phase"
    is_agnostic = feat_chan == "agnostic"
    is_live_fluor = np.isin(feat_chan, ["gfp", "mcherry", "cy5"])
    is_cp_marker = np.array(
        [str(c).startswith(("cp1_", "cp2_")) for c in feat_chan], dtype=bool
    )
    is_4i_marker = np.array(
        [str(c).startswith("4i_") for c in feat_chan], dtype=bool
    )
    is_any_fluor = is_live_fluor | is_cp_marker | is_4i_marker

    # ============ Phase split ============
    phase_row_mask = (obs["modality"].values == "phase")
    phase_col_mask = is_phase | is_agnostic
    X_phase = X_op[np.ix_(phase_row_mask, phase_col_mask)]
    obs_phase = obs.loc[phase_row_mask].copy()
    var_phase = var_op.iloc[phase_col_mask].copy()

    # ============ Fluor split with unification ============
    fluor_row_mask = (obs["modality"].values == "fluorescent")
    obs_fluor = obs.loc[fluor_row_mask].copy()
    n_fluor = int(fluor_row_mask.sum())

    # Build the unified (metric, aggregation) column set across all fluor vars.
    fluor_var_idx = np.flatnonzero(is_any_fluor)
    pair_to_unified: Dict[Tuple[str, str], int] = {}
    pair_order: List[Tuple[str, str]] = []
    for i in fluor_var_idx:
        p = (var_metric.iloc[i], var_aggr.iloc[i])
        if p not in pair_to_unified:
            pair_to_unified[p] = len(pair_order)
            pair_order.append(p)
    n_unified = len(pair_order)

    # For each cell, fill the unified columns from its own channel's source
    # var. obs.op_channel matches var.organelle for fluor channels (gfp,
    # mcherry, cy5, cp1_*, cp2_*) by construction.
    X_fluor_unified = np.full((n_fluor, n_unified), np.nan, dtype=np.float32)
    fluor_cell_chan = obs_fluor["op_channel"].astype(str).values
    fluor_global_idx = np.flatnonzero(fluor_row_mask)

    for i in fluor_var_idx:
        organelle = var_organelle.iloc[i]
        unified_col = pair_to_unified[(var_metric.iloc[i], var_aggr.iloc[i])]
        local_rows = np.flatnonzero(fluor_cell_chan == organelle)
        if local_rows.size == 0:
            continue
        global_rows = fluor_global_idx[local_rows]
        X_fluor_unified[local_rows, unified_col] = X_op[global_rows, i]

    unified_names = [
        f"op_{m}_{a}" if a else f"op_{m}" for m, a in pair_order
    ]
    var_fluor_unified = pd.DataFrame({
        "feature_name": unified_names,
        "organelle": "fluor_unified",
        "metric": [m for m, _ in pair_order],
        "category": "fluor_unified",
        "aggregation": [a for _, a in pair_order],
        "unit": "",
        "source": "organelle_profiler",
    }, index=pd.Index(unified_names))

    # Agnostic block — passed through unchanged.
    X_fluor_agnostic = X_op[np.ix_(fluor_row_mask, is_agnostic)]
    var_fluor_agnostic = var_op.iloc[is_agnostic].copy()

    X_fluor = np.concatenate([X_fluor_unified, X_fluor_agnostic], axis=1)
    var_fluor = pd.concat([var_fluor_unified, var_fluor_agnostic])

    return {
        "phase": (X_phase, obs_phase, var_phase),
        "fluor": (X_fluor, obs_fluor, var_fluor),
    }


def _build_obs(cells: pd.DataFrame) -> pd.DataFrame:
    obs = cells.copy()
    obs.index = (obs["experiment"].astype(str) + "_"
                 + obs["well_canonical"].astype(str) + "_"
                 + obs["segmentation"].astype(str) + "_"
                 + obs["modality"].astype(str) + "_"
                 + obs["viz_channel"].astype(str) + "_r"
                 + obs["rank"].astype(str))
    obs.index = obs.index.astype(str)
    # Ensure no object NaN survives into h5ad
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].fillna("").astype(str)
    return obs


def run_consolidation(
    phase_csv: Optional[Path],
    fluor_csv: Optional[Path],
    top_k: int,
    channel_rank_max: Optional[int],
    experiments: Optional[List[str]],
    output_dir: Path,
    cp_match_radius_px: float,
    no_cp: bool,
    per_cell_channel_mask: bool = True,
    channel_maps_yaml: Path = CHANNEL_MAPS_YAML,
    ntc_per_channel: int = 0,
    annotate_sgrna: bool = False,
    aggregation_level: str = "gene",
    ntc_phase_csv: Optional[Path] = None,
    ntc_fluor_csv: Optional[Path] = None,
    ntc_top_k: int = 100,
    global_per_channel: int = 0,
) -> int:
    """The actual consolidation work — module-level so submitit can pickle it."""
    import anndata as ad
    from cyclops_utils.hpc.resource_manager import get_optimal_workers

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nConsolidate top-attention cells\n{'='*72}")

    # CPU-bound, memory-modest workers (each loads only matched rows of a
    # backed h5ad). Model ~ anndata/scipy/numpy import; data ~ matched-row
    # slice + KDTree per experiment.
    n_workers = get_optimal_workers(
        use_gpu=False, model_ram_gb=1.0, data_ram_gb=2.0, verbose=True,
    )
    print(f"Using {n_workers} parallel workers for OP/CP attach passes.\n")

    channel_maps = None
    if per_cell_channel_mask:
        print(f"Loading channel maps from {channel_maps_yaml}")
        channel_maps = _load_channel_maps(channel_maps_yaml)
        print(f"  loaded maps for {len(channel_maps)} experiments")

    exp_set = set(experiments) if experiments else None
    cells = _load_master_cells(
        phase_csv, fluor_csv, top_k, channel_rank_max, exp_set,
        channel_maps=channel_maps,
        ntc_phase_csv=ntc_phase_csv, ntc_fluor_csv=ntc_fluor_csv,
        ntc_top_k=ntc_top_k,
    )
    print(f"\nMaster table: {len(cells):,} cells, "
          f"{cells['experiment'].nunique()} experiments, "
          f"{cells['gene'].nunique()} genes\n")

    # CHAD complex-level: when feeding the CHAD attention CSVs (pma_top_*_chad_v1),
    # each row's `predicted_class` carries the complex name (e.g. "subu ARP2/3");
    # `gene` is the source gene symbol. For complex-level aggregation we relabel
    # `cells["gene"]` from the source gene to `predicted_class` so all downstream
    # code (per-gene cap, sgRNA annotation, SHAP iteration) operates per-complex
    # without changes. NTCs (which lack a predicted_class) keep `gene = "NTC"`.
    if aggregation_level == "complex":
        if "predicted_class" not in cells.columns:
            raise SystemExit(
                "aggregation_level='complex' requires a 'predicted_class' "
                "column (CHAD attention CSVs supply it; gene-level CSVs don't). "
                "Pass the CHAD-level --phase-csv / --fluor-csv (e.g. "
                "pma_top_phase_cells_chad_v1.csv)."
            )
        is_ntc = cells["gene"].astype(str) == "NTC"
        new_gene = cells["gene"].astype(str).copy()
        new_gene.loc[~is_ntc] = cells.loc[~is_ntc, "predicted_class"].astype(str)
        # Drop rows where predicted_class was NaN/empty (no CHAD assignment).
        keep = is_ntc | (new_gene != "") & (new_gene != "nan") & (new_gene != "None")
        n_before = len(cells)
        cells = cells.loc[keep].copy()
        cells["gene"] = new_gene.loc[keep].values
        n_complexes = cells.loc[cells["gene"] != "NTC", "gene"].nunique()
        print(f"  CHAD relabel: gene -> predicted_class (complex name); "
              f"{n_before:,} -> {len(cells):,} cells "
              f"({n_complexes} unique complexes)\n")
    elif aggregation_level != "gene":
        raise SystemExit(
            f"Unknown aggregation_level={aggregation_level!r}; "
            f"choose 'gene' or 'complex'"
        )

    # Optional sgRNA annotation + NTC cell pool + GLOBAL random pool —
    # all three pulled from per-(exp,well) linked-results CSVs
    # (`<well>_linked_pheno_iss.csv`). Loading linked CSVs is expensive
    # so we do it once if any of the three options is active.
    if annotate_sgrna or ntc_per_channel > 0 or global_per_channel > 0:
        print("Loading linked-results CSVs for sgRNA / NTC / GLOBAL pulls...")
        t0 = time.time()
        linked_by_exp: Dict[str, pd.DataFrame] = {}
        for exp in sorted(cells["experiment"].astype(str).unique()):
            ldf = _load_linked_for_exp(exp)
            if not ldf.empty:
                linked_by_exp[exp] = ldf
        n_rows = sum(len(v) for v in linked_by_exp.values())
        print(f"  loaded {len(linked_by_exp)}/{cells['experiment'].nunique()} experiments, "
              f"{n_rows:,} total linked rows ({time.time()-t0:.1f}s)")

        if annotate_sgrna:
            print("Annotating KO cells with sgRNA + barcode...")
            cells = _annotate_sgrna(cells, linked_by_exp)
            n_with_sgrna = (cells["sgRNA"].astype(str) != "").sum()
            print(f"  {n_with_sgrna:,}/{len(cells):,} KO cells got sgRNA")

        if ntc_per_channel > 0:
            print(f"Building NTC pool: {ntc_per_channel} cells per (exp, viz_channel)...")
            ntc = _build_ntc_cells(linked_by_exp, cells, ntc_per_channel,
                                   channel_maps or {})
            print(f"  built {len(ntc):,} NTC cells across "
                  f"{ntc[['experiment','viz_channel']].drop_duplicates().shape[0] if len(ntc) else 0} pairs")
            # Align columns to the existing master table; missing cols -> NaN/"".
            for c in cells.columns:
                if c not in ntc.columns:
                    ntc[c] = "" if cells[c].dtype == object else np.nan
            ntc = ntc[cells.columns]
            cells = pd.concat([cells, ntc], ignore_index=True)
            print(f"  master table now: {len(cells):,} rows "
                  f"({(cells['gene']=='NTC').sum():,} NTC + "
                  f"{(cells['gene']!='NTC').sum():,} KO)")

        if global_per_channel > 0:
            print(f"Building GLOBAL random pool: {global_per_channel} cells "
                  f"per (exp, viz_channel)...")
            glob = _build_global_cells(linked_by_exp, cells, global_per_channel,
                                        channel_maps or {})
            print(f"  built {len(glob):,} GLOBAL cells across "
                  f"{glob[['experiment','viz_channel']].drop_duplicates().shape[0] if len(glob) else 0} pairs")
            for c in cells.columns:
                if c not in glob.columns:
                    glob[c] = "" if cells[c].dtype == object else np.nan
            glob = glob[cells.columns]
            cells = pd.concat([cells, glob], ignore_index=True)
            print(f"  master table now: {len(cells):,} rows "
                  f"({(cells['gene']=='NTC').sum():,} NTC + "
                  f"{(cells['gene']=='GLOBAL').sum():,} GLOBAL + "
                  f"{(~cells['gene'].isin(['NTC','GLOBAL'])).sum():,} KO)")

    print("Loading OrganelleProfiler features...")
    X_op, op_names, var_op, op_hit, op_stats = _attach_op_features(
        cells, cp_match_radius_px, n_workers,
        per_cell_channel_mask=per_cell_channel_mask,
    )
    print(f"OP attach: {op_hit.sum():,}/{len(cells):,} cells matched "
          f"({op_hit.mean():.1%}); shape {X_op.shape}\n")

    obs = _build_obs(cells)
    obs["op_match"] = op_hit

    # Prefix every feature name with op_ / cp_ so source is unambiguous from
    # the var index alone. var["source"] is kept as a redundant signal.
    op_names_prefixed = [f"op_{n}" for n in op_names]
    var_op = var_op.copy()
    var_op.index = op_names_prefixed
    var_op["source"] = "organelle_profiler"
    if "feature_name" in var_op.columns:
        var_op["feature_name"] = op_names_prefixed

    # CP attach (skipped under --no-cp). Done before the modality split so we
    # can append CP columns to both phase and fluor outputs.
    X_cp = None
    var_cp = None
    cp_stats = None
    if not no_cp:
        print("\nLoading CellProfiler features...")
        X_cp, cp_names, cp_hit, cp_stats = _attach_cp_features(
            cells, cp_match_radius_px, n_workers,
        )
        print(f"CP attach: {cp_hit.sum():,}/{len(cells):,} cells matched "
              f"({cp_hit.mean():.1%}); shape {X_cp.shape}\n")
        obs["cp_match"] = cp_hit
        cp_names_prefixed = [f"cp_{c}" for c in cp_names]
        var_cp = pd.DataFrame(
            {"feature_name": cp_names_prefixed, "source": "cellprofiler"},
            index=cp_names_prefixed,
        )

    # Split OP outputs by modality. Phase keeps organelle-distinct columns;
    # fluor unifies live-fluor + CP-marker channels into 287 (metric, aggr)
    # columns with `obs.op_channel` recording each row's source channel.
    splits = _split_op_outputs(X_op, var_op, obs)

    output_files: List[Path] = []
    output_summaries: Dict[str, Dict] = {}
    for name, (X_split, obs_split, var_split) in splits.items():
        if not no_cp:
            row_mask = (obs["modality"].values
                        == ("phase" if name == "phase" else "fluorescent"))
            X_cp_split = X_cp[row_mask, :]
            X_combined = np.concatenate([X_split, X_cp_split], axis=1)
            var_combined = pd.concat([var_split, var_cp])
        else:
            X_combined = X_split
            var_combined = var_split.copy()

        path = output_dir / f"top_attention_cells_{name}.h5ad"
        print(f"Writing {name} AnnData -> {path}  (shape {X_combined.shape})")
        ad.AnnData(
            X=X_combined, obs=obs_split.copy(), var=var_combined.copy(),
        ).write_h5ad(path)
        output_files.append(path)
        output_summaries[name] = {
            "path": path,
            "X": X_combined,
            "obs": obs_split,
            "var": var_combined,
            "n_op_cols": X_split.shape[1],
        }

    report_path = _write_coverage_report(
        output_dir, cells, op_stats,
        cp_stats=cp_stats,
        output_summaries=output_summaries,
        output_files=output_files,
    )
    print(f"Coverage report -> {report_path}")

    print("\nDone.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ATTN_DIR = Path(f"{BASE_PATH}/models/alex_lin_attention")
    _ATTN_V3 = _ATTN_DIR / "v3" / "attention_v3"
    p.add_argument("--phase-csv", type=Path,
                   default=_ATTN_V3 / "pma_top_phase_cells_v3.csv",
                   help="Top-attention phase cells CSV. Default: v3/attention_v3/"
                        "pma_top_phase_cells_v3.csv (v3 renames `viz_channel` -> "
                        "`channel` and adds `rank_type`; `_load_master_cells` "
                        "back-renames at load time).")
    p.add_argument("--fluor-csv", type=Path,
                   default=_ATTN_V3 / "pma_top_fluorescent_cells_v3.csv",
                   help="Top-attention fluorescent cells CSV. Default: "
                        "v3/attention_v3/pma_top_fluorescent_cells_v3.csv.")
    p.add_argument("--top-k", type=int, default=100,
                   help="Top-K cells per gene (per channel for fluor). Default 100.")
    p.add_argument("--channel-rank-max", type=int, default=None,
                   help="For fluor: optional cap on the number of top channels "
                        "per gene to keep. Default None = no cap (all channels "
                        "Alex's CSV provides flow through, ~56 per gene with the "
                        "current v3 CSV). Pass an integer like 3 to clip to the "
                        "top-N per gene (legacy behavior — produced 100*3=300 "
                        "fluor cells, +100 phase = 400 per KO).")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Optional allowlist of experiment names to include.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(f"{BASE_PATH}/models/"
                                "alex_lin_attention/consolidated_v3"),
                   help="Directory where the two .h5ad files will be written. "
                        "Default: alex_lin_attention/consolidated_v3/ (fresh "
                        "dir for the v3 attention CSVs — keeps Alex's "
                        "pre-built consolidated_v2/v3/ untouched). Pass the "
                        "old path explicitly if you want to overwrite those.")
    p.add_argument("--cp-match-radius-px", type=float, default=20.0,
                   help="Max spatial distance (in CP/CSV pixel units) for "
                        "CSV<->CP nearest-neighbor match. Default 20.")
    p.add_argument("--no-cp", action="store_true",
                   help="Skip the OP+CP output; only write the OP-only AnnData.")
    p.add_argument("--no-per-cell-channel-mask", action="store_true",
                   help="Disable the per-cell channel mask. Default ON: each "
                        "cell only keeps OP features for its own imaging "
                        "channel (phase / gfp / mcherry / cy5) plus "
                        "channel-agnostic morphology (cell, nuclei).")
    p.add_argument("--channel-maps-yaml", type=Path, default=CHANNEL_MAPS_YAML,
                   help=f"Path to ops_channel_maps.yaml (default: {CHANNEL_MAPS_YAML}).")
    p.add_argument("--add-ntc", type=int, default=0,
                   help="Sample N random NTC cells per (experiment, viz_channel) "
                        "from linked_pheno_iss CSVs and append to master cells. "
                        "Implies --annotate-sgrna. Default 0 (no NTC).")
    p.add_argument("--annotate-sgrna", action="store_true",
                   help="Annotate cells with sgRNA + barcode from linked_pheno_iss "
                        "CSVs (joined on (experiment, segmentation)). Required for "
                        "guide-level distinctiveness mAP scoring.")
    # ── Unified-negatives ingestion: attention-ranked NTCs (PMA CSV)
    # + random GLOBAL cells (linked_results). When set, the resulting
    # consolidated h5ad carries the negative cohorts for ntc / global
    # SHAP contrasts in the SAME file as KO positives — same feature
    # extraction pipeline, same scale, no all_cells_v2 detour.
    p.add_argument("--ntc-phase-csv", type=Path,
                   default=_ATTN_V3 / "pma_top_phase_cells_ntc_v3.csv",
                   help="PMA top-attention NTC phase CSV. Default: gene-level "
                        "v3 NTC CSV. Auto-swaps to chad_ntc_v3 at "
                        "--aggregation-level complex (when default kept). "
                        "When non-empty, top-K rows (rank_type=top) per "
                        "channel are appended as `gene=NTC` rows passing "
                        "through the same OP/CP feature attach as KO positives. "
                        "Pass empty string ('') to disable.")
    p.add_argument("--ntc-fluor-csv", type=Path,
                   default=_ATTN_V3 / "pma_top_fluorescent_cells_ntc_v3.csv",
                   help="PMA top-attention NTC fluor CSV. Default: gene-level "
                        "v3 NTC CSV. Auto-swaps to chad_ntc_v3 at "
                        "--aggregation-level complex (when default kept). "
                        "Pass empty string ('') to disable.")
    p.add_argument("--ntc-top-k", type=int, default=100,
                   help="Top-K NTC PMA rows per (gene, channel) to include "
                        "(rank_type=top, lowest ranks). Default 100 to match "
                        "KO positives' top-100 convention.")
    p.add_argument("--global-per-channel", type=int, default=100,
                   help="Sample N RANDOM cells (any gene) per (experiment, "
                        "viz_channel) from linked_pheno_iss CSVs and append "
                        "as `gene=GLOBAL`. Used as the negative pool for the "
                        "SHAP `global` contrast. Default 100. Set 0 to skip.")
    p.add_argument(
        "--aggregation-level", choices=("gene", "complex"), default="gene",
        help="Aggregate at gene (default; uses pma_top_*_v3.csv inputs and "
             "obs.gene = source gene) or at CHAD protein-complex level "
             "(uses pma_top_*_chad_v1.csv inputs and relabels obs.gene = "
             "predicted_class = complex name). When 'complex' AND the user "
             "kept the default --phase-csv/--fluor-csv, the defaults swap to "
             "the chad_v1 CSVs and --output-dir auto-suffixes with `_chad`. "
             "These are DIFFERENT cells than the gene-level run — Alex's "
             "attention model is trained separately at each grain.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and per-experiment OP/CP availability, then exit.")

    # SLURM submission controls
    p.add_argument("--local", action="store_true",
                   help="Run inline on the current host instead of submitting to SLURM. "
                        "Default behavior is SLURM submission with progress tracking.")
    p.add_argument("--mem", default="500G",
                   help="SLURM memory request (default 500G — sized for full v3 set).")
    p.add_argument("--cpus", type=int, default=8,
                   help="SLURM cpus_per_task (default 8: feature reads are I/O-bound).")
    p.add_argument("--timeout-min", type=int, default=360,
                   help="SLURM timeout in minutes (default 360 = 6h).")
    p.add_argument("--partition", default="cpu",
                   help="SLURM partition (default cpu).")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit and return immediately; don't block on completion.")
    args = p.parse_args()

    # Auto-swap to CHAD-level CSVs + output dir when --aggregation-level complex
    # AND the user kept defaults. Avoids overwriting the gene-level run.
    _GENE_PHASE_DEFAULT = _ATTN_V3 / "pma_top_phase_cells_v3.csv"
    _GENE_FLUOR_DEFAULT = _ATTN_V3 / "pma_top_fluorescent_cells_v3.csv"
    _CHAD_PHASE = _ATTN_V3 / "pma_top_phase_cells_chad_v1.csv"
    _CHAD_FLUOR = _ATTN_V3 / "pma_top_fluorescent_cells_chad_v1.csv"
    _GENE_NTC_PHASE_DEFAULT = _ATTN_V3 / "pma_top_phase_cells_ntc_v3.csv"
    _GENE_NTC_FLUOR_DEFAULT = _ATTN_V3 / "pma_top_fluorescent_cells_ntc_v3.csv"
    _CHAD_NTC_PHASE = _ATTN_V3 / "pma_top_phase_cells_chad_ntc_v3.csv"
    _CHAD_NTC_FLUOR = _ATTN_V3 / "pma_top_fluorescent_cells_chad_ntc_v3.csv"
    _DEFAULT_OUT = Path(
        f"{BASE_PATH}/models/"
        "alex_lin_attention/consolidated_v3"
    )
    # Empty-string sentinels disable the NTC frames (path parses to '.').
    if str(args.ntc_phase_csv) in ("", "."):
        args.ntc_phase_csv = None
    if str(args.ntc_fluor_csv) in ("", "."):
        args.ntc_fluor_csv = None
    if args.aggregation_level == "complex":
        if args.phase_csv == _GENE_PHASE_DEFAULT:
            args.phase_csv = _CHAD_PHASE
            print(f"[CHAD] swapped --phase-csv default to: {args.phase_csv.name}")
        if args.fluor_csv == _GENE_FLUOR_DEFAULT:
            args.fluor_csv = _CHAD_FLUOR
            print(f"[CHAD] swapped --fluor-csv default to: {args.fluor_csv.name}")
        if args.ntc_phase_csv == _GENE_NTC_PHASE_DEFAULT:
            args.ntc_phase_csv = _CHAD_NTC_PHASE
            print(f"[CHAD] swapped --ntc-phase-csv default to: {args.ntc_phase_csv.name}")
        if args.ntc_fluor_csv == _GENE_NTC_FLUOR_DEFAULT:
            args.ntc_fluor_csv = _CHAD_NTC_FLUOR
            print(f"[CHAD] swapped --ntc-fluor-csv default to: {args.ntc_fluor_csv.name}")
        if args.output_dir == _DEFAULT_OUT:
            args.output_dir = Path(str(_DEFAULT_OUT) + "_chad")
            print(f"[CHAD] swapped --output-dir default to: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --dry-run is fast (just file-existence checks); always run inline.
    if args.dry_run:
        print(f"\n{'='*72}\nConsolidate top-attention cells (--dry-run)\n{'='*72}")
        exp_set = set(args.experiments) if args.experiments else None
        cells = _load_master_cells(args.phase_csv, args.fluor_csv,
                                   args.top_k, args.channel_rank_max, exp_set)
        print(f"\nMaster table: {len(cells):,} cells, "
              f"{cells['experiment'].nunique()} experiments, "
              f"{cells['gene'].nunique()} genes\n")
        print("Per-experiment availability check:")
        for exp in sorted(cells["experiment"].unique()):
            grp = cells[cells["experiment"] == exp]
            n = len(grp)
            op_ok = (OPS_FAST_ROOT / exp / "3-assembly" / "feature_extraction"
                     / f"{exp}_cell_features.h5ad").exists()
            chans_needed = {
                ("Phase" if m == "phase" else _viz_channel_to_file(v))
                for m, v in zip(grp["modality"], grp["viz_channel"])
            }
            cp_resolved = sum(_resolve_cp_h5ad(exp, c) is not None for c in chans_needed)
            cp_total = len(chans_needed)
            print(f"  {exp}: {n:>6,} cells  OP={'Y' if op_ok else 'N'}  "
                  f"CP={cp_resolved}/{cp_total} channels")
        return 0

    annotate_sgrna = args.annotate_sgrna or args.add_ntc > 0 or args.global_per_channel > 0
    if args.local:
        return run_consolidation(
            phase_csv=args.phase_csv,
            fluor_csv=args.fluor_csv,
            top_k=args.top_k,
            channel_rank_max=args.channel_rank_max,
            experiments=args.experiments,
            output_dir=args.output_dir,
            cp_match_radius_px=args.cp_match_radius_px,
            no_cp=args.no_cp,
            per_cell_channel_mask=not args.no_per_cell_channel_mask,
            channel_maps_yaml=args.channel_maps_yaml,
            ntc_per_channel=args.add_ntc,
            annotate_sgrna=annotate_sgrna,
            aggregation_level=args.aggregation_level,
            ntc_phase_csv=args.ntc_phase_csv,
            ntc_fluor_csv=args.ntc_fluor_csv,
            ntc_top_k=args.ntc_top_k,
            global_per_channel=args.global_per_channel,
        )

    # SLURM submission via shared batch utils — inherits progress tracking,
    # manifests, and resource summary.
    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs = [{
        "name": "consolidate_top_attention_cells",
        "func": run_consolidation,
        "kwargs": {
            "phase_csv": args.phase_csv,
            "fluor_csv": args.fluor_csv,
            "top_k": args.top_k,
            "channel_rank_max": args.channel_rank_max,
            "experiments": args.experiments,
            "output_dir": args.output_dir,
            "cp_match_radius_px": args.cp_match_radius_px,
            "no_cp": args.no_cp,
            "per_cell_channel_mask": not args.no_per_cell_channel_mask,
            "channel_maps_yaml": args.channel_maps_yaml,
            "ntc_per_channel": args.add_ntc,
            "annotate_sgrna": annotate_sgrna,
            "aggregation_level": args.aggregation_level,
            "ntc_phase_csv": args.ntc_phase_csv,
            "ntc_fluor_csv": args.ntc_fluor_csv,
            "ntc_top_k": args.ntc_top_k,
            "global_per_channel": args.global_per_channel,
        },
        "metadata": {"experiment": "consolidate_top_attention_cells"},
    }]
    slurm_params = {
        "timeout_min": args.timeout_min,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }
    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="consolidate_top_attention_cells",
        slurm_params=slurm_params,
        log_dir="consolidate_top_attention",
        manifest_prefix="consolidate_top_attention",
        wait_for_completion=not args.no_wait,
    )
    if args.no_wait:
        return 0 if result.get("success") else 1
    return 0 if result.get("all_completed") else 1


if __name__ == "__main__":
    sys.exit(main())
