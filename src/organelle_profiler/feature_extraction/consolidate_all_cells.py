"""Consolidate ALL paper_v1 cells (KO + NTC) × CP+OP features — one h5ad per marker.

Sibling of `consolidate_top_attention_cells.py`, but the cell list is driven by
`<exp>/<well>_linked_pheno_iss.csv` (every cell with a valid segmentation +
gene/sgRNA call) instead of Alex's top-attention CSVs. Same OP+CP attach
pipeline, same per-cell channel mask. Two architectural changes vs the
top-attention consolidator:

  * Output is **one h5ad per viz_channel** (Phase + ~56 fluor markers = ~57
    files), not one per modality. Per-channel files have only that channel's
    cells (~2M max with caps) and only that channel's relevant features
    (channel-specific OP organelle bucket + agnostic + that channel's CP
    h5ad). Per-file size: ~5-15 GB. Total disk: ~290 GB. Peak in-memory
    allocation per channel: ~50-80 GB during the attach pass.
  * Features are z-scored per `experiment` batch BEFORE write so the SHAP
    step downstream can pool cells across experiments without batch effects
    (single viz_channel per file means the batch key is just experiment).

Outputs (under --output-dir):
    op_cp_features_phase.h5ad
    op_cp_features_<sanitized_viz_channel>.h5ad   (one per fluor marker)

paper_v1 source-of-truth = `$OPS_BASE_PATH/configs/good_experiment_list_v1.yml`
(77 experiments). Mirrors `pca_optimization.py:4517`'s `--paper-v1` flag/default.

Stratified caps (off by default — pass to cap on demand):
  * --ko-cap N: at most N KO cells **per sgRNA** (per viz_channel).
  * --ntc-cap N: at most N NTC cells **per NTC sgRNA** (per viz_channel).

When unset, every linked cell with a valid sgRNA becomes a row — uncapped
paper_v1 is ~70M (cell, channel) rows × ~3000 features ≈ 800 GB total on
disk. Sized for downstream consumers that want the full distribution rather
than a SHAP-tuned subsample; see DEFAULT_KO_CAP/NTC_CAP below.

4i markers (`4i_R*_*`) ARE included: the shared helpers
(`_load_channel_maps`, `_organelle_to_op_channel`, `_split_op_outputs`) treat
them as first-class fluor channels (mirror of the cp1_/cp2_ scheme). Each 4i
marker becomes its own (cell, viz_channel) row with the per-cell mask
preserving only that marker's features. ops0144's 4i markers (gH2AX, p53,
c-Myc, RSP6, pRb, Rb, p21, b-catenin, pS6, NFkB) all end up in the fluor
output alongside live-cell and Cell-Painting markers.

CLI:
    # Local dry-run (counts only — no attach)
    python -m organelle_profiler.feature_extraction.consolidate_all_cells \\
        --paper-v1 --dry-run

    # Local full run
    python -m organelle_profiler.feature_extraction.consolidate_all_cells \\
        --paper-v1 --local

    # SLURM (default)
    python -m organelle_profiler.feature_extraction.consolidate_all_cells --paper-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse the heavy helpers from the top-attention consolidator. Anything that
# touches OP/CP h5ads or the per-cell channel mask is shared via this import.
from organelle_profiler.feature_extraction.consolidate_top_attention_cells import (
    OPS_FAST_ROOT,
    CHANNEL_MAPS_YAML,
    _LINKED_WELLS,
    _load_channel_maps,
    _load_linked_for_exp,
    _attach_op_features,
    _attach_cp_features,
    _split_op_outputs,
    _build_obs,
)

# Per-cell multiplet dropper: cells with 2+ distinct sgRNAs are ambiguous-KO
# and removed; cells with no guide are also removed. Matches the rule in
# cyclops_process.utils.cell_count_summary and the portal-parquet dedupe in
# cyclops_utils.data_portal.dedupe_cell_data — one source of truth.
from cyclops_process.utils.dedupe_linked_pheno_iss import dedupe_linked_csv
from organelle_profiler.paths import BASE_PATH

# Path to the marker_volcanoes script directory, used by the lazy import
# in `_apply_op_unit_correction` below. Single source of truth for the
# 0.65→0.325 µm/px CORRECTION_FACTORS map + affected-exp discovery.
_FIX_OP_UNITS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "marker_volcanoes"


DEFAULT_PAPER_V1 = Path(
    f"{BASE_PATH}/configs/good_experiment_list_v1.yml"
)
DEFAULT_OUTPUT_DIR = Path(f"{BASE_PATH}/analysis/op_cp_features")
# Per-sgRNA caps default to None = no cap (truly every linked cell with a
# valid sgRNA becomes a row). Pass --ko-cap / --ntc-cap to cap on demand.
# Uncapped paper_v1 scale: Phase ~54M cells (X_op union ~210 GB), high-cov
# fluor ~5-15M, mid-cov ~2-5M. SLURM defaults below are sized for this.
DEFAULT_KO_CAP: Optional[int] = None
DEFAULT_NTC_CAP: Optional[int] = None
# Per-batch z-score clip — protects the classifier from stretched-tail outliers
# without distorting the bulk distribution.
ZSCORE_CLIP = 10.0
# Floor on cells/batch for z-scoring — below this, leave raw and let the
# classifier handle it (rare in paper_v1).
ZSCORE_MIN_CELLS = 5


def _count_cells_one_exp(exp: str) -> Dict[str, int]:
    """Read `obs.gene_name` from the OP cell h5ad in backed mode and tally
    KO/NTC. Mirrors `pca_optimization`'s strategy of reading h5ad metadata
    only (no X load). NTC = gene_name NaN OR == 'NTC'.
    """
    import anndata as ad
    p = OPS_FAST_ROOT / exp / "3-assembly" / "feature_extraction" / f"{exp}_cell_features.h5ad"
    if not p.exists():
        return {"ko": 0, "ntc": 0, "total": 0}
    try:
        a = ad.read_h5ad(p, backed="r")
        gn = a.obs["gene_name"].astype("string")
        is_ntc = gn.isna() | (gn.str.upper() == "NTC")
        n_total = len(gn)
        n_ko = int((~is_ntc).sum())
        n_ntc = int(is_ntc.sum())
        a.file.close()
        return {"ko": n_ko, "ntc": n_ntc, "total": n_total}
    except Exception as e:
        print(f"    [{exp}] read failed: {e!r}")
        return {"ko": 0, "ntc": 0, "total": 0}


# Inspect cache sits inside the output dir so `--inspect` reuses prior counts
# across runs that share an output location. Invalidated via --refresh-counts.
_INSPECT_CACHE = DEFAULT_OUTPUT_DIR / ".inspect_cell_counts.json"


def _count_cells_per_exp(
    paper_v1: Dict[str, List[str]],
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Count cells per paper_v1 experiment via serial h5py read of obs.gene_name.

    No parallelism — login-node-safe. Reads ~77 h5ads in ~3.5 min total. Cached
    to `cache_path` (JSON) so subsequent calls are instant; pass `refresh=True`
    to recompute. Counts include all wells in the cell h5ad (consolidate later
    restricts to A1/A2/A3 via `_LINKED_WELLS` and applies per-(gene, channel)
    caps; the inspect table is a pre-cap upper-bound view of cohort scale).
    """
    import json

    cache_path = cache_path or _INSPECT_CACHE
    if cache_path.exists() and not refresh:
        try:
            with cache_path.open() as f:
                cached = json.load(f)
            cached_keys = set(cached.keys())
            requested_keys = set(paper_v1.keys())
            if requested_keys.issubset(cached_keys):
                print(f"  Using cached cell counts: {cache_path}")
                return {k: cached[k] for k in paper_v1}
            print(f"  Cache at {cache_path} is missing "
                  f"{len(requested_keys - cached_keys)} experiments; recomputing")
        except Exception as e:
            print(f"  [cache] read failed ({e!r}); recomputing")

    print(f"  Counting cells via OP cell h5ad obs (serial; ~3.5 min for "
          f"{len(paper_v1)} exps)...", flush=True)
    from tqdm import tqdm
    counts: Dict[str, Dict[str, int]] = {}
    sorted_exps = sorted(paper_v1.keys())
    pbar = tqdm(sorted_exps, desc="Counting cells", unit="exp", smoothing=0.1)
    for exp in pbar:
        counts[exp] = _count_cells_one_exp(exp)
        c = counts[exp]
        pbar.set_postfix_str(f"KO={c['ko']:,} NTC={c['ntc']:,}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(counts, f, indent=2)
    print(f"  Saved cell-count cache: {cache_path}")
    return counts


def _print_coverage_table(
    paper_v1: Dict[str, List[str]],
    channel_maps: Dict[str, Dict[str, str]],
    cell_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Print which (viz_channel, physical_channel) pairs are kept vs excluded
    across paper_v1, mirroring `pca_optimization.py:_print_groups` style.

    When `cell_counts` is provided (from `_count_cells_per_exp`), each row
    additionally shows the total KO + NTC cells across all experiments
    imaging that marker — pre-cap counts (no per-(gene, channel) cap applied
    yet, no per-channel cap applied yet). Useful for sizing the cohort.
    """
    # exp_short -> exp_full so we can look up cell counts (which are keyed
    # by full exp name).
    short_to_full: Dict[str, str] = {
        e.split("_")[0]: e for e in paper_v1
    }

    kept_groups: Dict = {}      # (viz, phys) -> [exp_short, ...]
    excluded_groups: Dict = {}  # (viz, phys, reason) -> [exp_short, ...]

    for exp_full in paper_v1.keys():
        short = exp_full.split("_")[0]
        cm = channel_maps.get(short, {})
        for v, p in cm.items():
            if p == "phase":
                kept_groups.setdefault(("Phase", "phase"), []).append(short)
                continue
            plower, vlower = str(p).lower(), str(v).lower()
            if plower.endswith("_nuclei"):
                reason = "DAPI registration"
                excluded_groups.setdefault((v, p, reason), []).append(short)
            elif "no label" in vlower:
                if "autofluor" in vlower:
                    reason = "autofluorescence"
                elif "bleed" in vlower:
                    reason = "bleedthrough"
                elif "empty" in vlower:
                    reason = "empty well"
                else:
                    reason = "no label"
                excluded_groups.setdefault((v, p, reason), []).append(short)
            else:
                kept_groups.setdefault((v, p), []).append(short)

    def _agg_cells(exps_short: List[str]) -> Dict[str, int]:
        """Sum KO + NTC cells across the given short-exp names."""
        if cell_counts is None:
            return {"ko": 0, "ntc": 0, "total": 0}
        agg = {"ko": 0, "ntc": 0, "total": 0}
        for short in set(exps_short):
            full = short_to_full.get(short)
            if full is None:
                continue
            c = cell_counts.get(full, {"ko": 0, "ntc": 0, "total": 0})
            agg["ko"] += c["ko"]
            agg["ntc"] += c["ntc"]
            agg["total"] += c["total"]
        return agg

    def _print_one(title: str, groups: Dict, with_reason: bool) -> None:
        print(f"\n{title}")
        if not groups:
            print("    (none)")
            return
        # Sort by total cells desc when available, else by exp count desc.
        if cell_counts is not None:
            sort_key = lambda kv: (-_agg_cells(kv[1])["total"], kv[0])
        else:
            sort_key = lambda kv: (-len(kv[1]), kv[0])
        for key, exps in sorted(groups.items(), key=sort_key):
            exps_sorted = sorted(set(exps))
            sample = ", ".join(exps_sorted[:3])
            more = (f" ... (+{len(exps_sorted) - 3} more)"
                    if len(exps_sorted) > 3 else "")
            if with_reason:
                viz, phys, reason = key
                tag = f"  [{reason}]"
            else:
                viz, phys = key
                tag = ""
            if cell_counts is not None:
                cells = _agg_cells(exps)
                cells_str = (f" {cells['ko']:>9,} KO + {cells['ntc']:>7,} NTC "
                             f"= {cells['total']:>9,} cells")
            else:
                cells_str = ""
            print(f"    {viz:42s} -> {phys:24s} : "
                  f"{len(exps_sorted):>3} exps{cells_str} -- {sample}{more}{tag}")

    n_kept_pairs = sum(len(v) for v in kept_groups.values())
    n_excluded_pairs = sum(len(v) for v in excluded_groups.values())
    print(f"\n{'='*92}")
    print(f"paper_v1 channel coverage  (experiments: {len(paper_v1)})")
    print(f"{'='*92}")
    print(f"(viz_channel, physical_channel) pairs: "
          f"{n_kept_pairs} kept, {n_excluded_pairs} excluded")
    print(f"Unique markers: {len(kept_groups)} kept, {len(excluded_groups)} excluded")
    if cell_counts is not None:
        total = sum(c["total"] for c in cell_counts.values())
        n_ko = sum(c["ko"] for c in cell_counts.values())
        n_ntc = sum(c["ntc"] for c in cell_counts.values())
        print(f"Total paper_v1 cells (across experiments, pre-expansion): "
              f"{n_ko:,} KO + {n_ntc:,} NTC = {total:,}")
        print("Cell counts below are PRE-CAP totals across all experiments "
              "imaging the marker.")
    _print_one(f"KEPT ({len(kept_groups)} unique markers):",
               kept_groups, with_reason=False)
    _print_one(f"EXCLUDED ({len(excluded_groups)} unique markers):",
               excluded_groups, with_reason=True)


REQUIRED_OBS_COLUMNS = (
    "experiment", "well_canonical", "well",
    "x_pheno", "y_pheno", "segmentation",
    "gene_name", "gene", "sgRNA", "barcode",
    "modality", "viz_channel", "channel_rank", "op_channel",
    "rank", "op_match",
)


def _check_outputs(output_dir: Path, expected_ko_cap: Optional[int] = None) -> int:
    """Validate per-channel h5ads in `output_dir` against the schema we expect.

    Per-file checks:
      * required obs columns present
      * single viz_channel per file (matches the filename suffix)
      * modality matches viz_channel (Phase row -> phase modality; otherwise
        fluorescent)
      * `var.zscored_per_batch` annotation present and == "experiment"
      * z-score sanity: per-experiment per-feature mean ≈ 0, std ≈ 1
        (sampled — checked on up to 50 features for speed)
      * no all-NaN rows; no all-NaN feature columns
      * KO cap respected per gene (when `expected_ko_cap` is given)

    Prints a summary table + per-file issues. Returns 0 if every file passes,
    1 if any file has issues.
    """
    import anndata as ad

    h5ads = sorted(output_dir.glob("op_cp_features_*.h5ad"))
    if not h5ads:
        print(f"\nNo op_cp_features_*.h5ad files in {output_dir}")
        return 1

    print(f"\n{'='*98}")
    print(f"Output check: {len(h5ads)} h5ads in {output_dir}")
    print(f"{'='*98}")

    n_pass = 0
    n_fail = 0
    rng = np.random.default_rng(0)

    summary_rows = []
    for path in h5ads:
        issues: List[str] = []
        try:
            a = ad.read_h5ad(path, backed="r")
        except Exception as e:
            print(f"\n[FAIL] {path.name}: cannot open ({e!r})")
            n_fail += 1
            continue

        n_cells = int(a.n_obs)
        n_features = int(a.n_vars)
        obs = a.obs
        var = a.var

        # 0. obs_names uniqueness — duplicates break per-cell joins in any
        # downstream consumer (SHAP step, atlas, captioning).
        if not a.obs_names.is_unique:
            n_dup = int(a.obs_names.duplicated().sum())
            issues.append(f"{n_dup} duplicate obs_names "
                          f"(call obs_names_make_unique or fix index keys)")

        # 1. Required obs columns
        missing_cols = [c for c in REQUIRED_OBS_COLUMNS if c not in obs.columns]
        if missing_cols:
            issues.append(f"missing obs cols: {missing_cols}")

        # 2. Single viz_channel
        viz_uniques = obs["viz_channel"].astype(str).unique() if "viz_channel" in obs.columns else []
        if len(viz_uniques) != 1:
            issues.append(f"expected 1 viz_channel, got {len(viz_uniques)}: "
                          f"{list(viz_uniques)[:5]}")
        viz = str(viz_uniques[0]) if len(viz_uniques) == 1 else "?"

        # 3. Modality consistency
        if "modality" in obs.columns:
            modality_uniques = obs["modality"].astype(str).unique()
            expected_mod = "phase" if viz == "Phase" else "fluorescent"
            if len(modality_uniques) != 1:
                issues.append(f"mixed modality: {list(modality_uniques)}")
            elif str(modality_uniques[0]) != expected_mod:
                issues.append(f"modality {modality_uniques[0]!r} doesn't match "
                              f"viz_channel {viz!r} (expected {expected_mod!r})")

        # 4. z-score annotation. Both "experiment" (--zscore-per-experiment)
        # and "none" (raw values, the default) are valid outputs; only an
        # unexpected third value is an issue. `is_zscored` gates the mean≈0/
        # std≈1 sanity below — raw output has no such expectation.
        is_zscored = False
        if "zscored_per_batch" not in var.columns:
            issues.append("var.zscored_per_batch annotation missing")
        else:
            zs = var["zscored_per_batch"].astype(str).unique()
            if not (len(zs) == 1 and str(zs[0]) in ("experiment", "none")):
                issues.append(f"var.zscored_per_batch unexpected: {list(zs)[:3]}")
            is_zscored = len(zs) == 1 and str(zs[0]) == "experiment"

        # 5. Cell counts
        gene_col = obs["gene"].astype(str) if "gene" in obs.columns else None
        n_ko = int((gene_col != "NTC").sum()) if gene_col is not None else 0
        n_ntc = int((gene_col == "NTC").sum()) if gene_col is not None else 0
        n_experiments = int(obs["experiment"].nunique()) if "experiment" in obs.columns else 0
        n_genes = int(obs.loc[gene_col != "NTC", "gene"].nunique()) if gene_col is not None else 0

        # 6. KO cap respected (per sgRNA)
        if expected_ko_cap is not None and gene_col is not None and "sgRNA" in obs.columns:
            ko_obs = obs.loc[gene_col != "NTC"]
            per_guide = ko_obs.groupby("sgRNA", observed=True).size()
            over = per_guide[per_guide > expected_ko_cap]
            if len(over):
                issues.append(f"{len(over)} KO sgRNA(s) exceed --ko-cap="
                              f"{expected_ko_cap}: top {over.nlargest(3).to_dict()}")

        # 7. z-score sanity on a sample of features × cells. We sample rows
        # stratified by experiment (up to ~200 cells/exp) so each experiment
        # gets equal weight in the median computation. This keeps peak per-
        # file memory under ~10 MB instead of ~800 MB for an all-rows read.
        n_col_check = min(50, n_features)
        # h5py fancy indexing requires strictly-increasing indices on each
        # axis, so we sort col_idx (`rng.choice` is unsorted by default).
        col_idx = (np.sort(rng.choice(n_features, size=n_col_check, replace=False))
                   if n_features > 0 else np.array([], dtype=int))

        per_exp_rows = 200
        sampled_rows = []
        if "experiment" in obs.columns and n_cells > 0:
            exp_codes = obs["experiment"].astype(str).values
            unique_exps = np.unique(exp_codes)
            for exp in unique_exps:
                exp_idx = np.flatnonzero(exp_codes == exp)
                if len(exp_idx) <= per_exp_rows:
                    sampled_rows.append(exp_idx)
                else:
                    sampled_rows.append(rng.choice(exp_idx, size=per_exp_rows, replace=False))
        row_idx = (np.sort(np.concatenate(sampled_rows)) if sampled_rows
                   else np.arange(min(n_cells, per_exp_rows)))

        X_sample = None
        if len(col_idx) and len(row_idx):
            try:
                # Two-step read: first materialize the row slice into a numpy
                # array (h5py honors row_idx — sorted), then slice columns
                # in-memory (no h5py constraint on col_idx ordering anymore).
                X_rows = np.asarray(a.X[row_idx, :])
                X_sample = X_rows[:, col_idx].astype(np.float64)
            except Exception as e:
                issues.append(f"X read failed: {e!r}")

        zscore_summary = "n/a"
        if X_sample is not None and "experiment" in obs.columns:
            import warnings as _w
            sub_exp = obs["experiment"].astype(str).values[row_idx]
            unique_exps = np.unique(sub_exp)
            mean_abs_means = []
            mean_stds = []
            for exp in unique_exps:
                mask = sub_exp == exp
                if mask.sum() < 5:
                    continue
                sub = X_sample[mask, :]
                # Suppress benign "Mean of empty slice" / "DoF <= 0" warnings
                # that fire when a sampled feature column happens to be
                # all-NaN for one experiment (different experiments measure
                # different OP feature subsets — expected, not an error).
                # We filter out the NaN results via isfinite() right after.
                with _w.catch_warnings(), np.errstate(invalid="ignore"):
                    _w.simplefilter("ignore", category=RuntimeWarning)
                    m = np.nanmean(sub, axis=0)
                    s = np.nanstd(sub, axis=0)
                m = m[np.isfinite(m)]
                s = s[np.isfinite(s) & (s > 0)]
                if len(m):
                    mean_abs_means.append(float(np.median(np.abs(m))))
                if len(s):
                    mean_stds.append(float(np.median(s)))
            if mean_abs_means and mean_stds:
                med_abs_mean = float(np.median(mean_abs_means))
                med_std = float(np.median(mean_stds))
                zscore_summary = f"|mean|≈{med_abs_mean:.2f} std≈{med_std:.2f}"
                # These bounds only apply to z-scored output. Raw output
                # (zscored_per_batch=="none", the default) keeps real-world
                # units, so per-experiment mean/std are arbitrary — skip.
                if is_zscored:
                    # After z-score, per-experiment per-feature mean ~0, std ~1.
                    # Allow 0.5 abs(mean) and 0.3-3.0 std for clipping + small N.
                    if med_abs_mean > 0.5:
                        issues.append(f"z-score mean drift: median |exp_mean|={med_abs_mean:.2f}")
                    if not (0.3 <= med_std <= 3.0):
                        issues.append(f"z-score std off: median exp_std={med_std:.2f}")

        # 8. all-NaN rows / columns (in the sampled window). Thresholds
        # tuned against observed upstream behavior:
        #   * 4i markers (ops0144): ~22% all-NaN rows is the floor — cells
        #     that failed 4i registration in any round aren't in the OP h5ad
        #     nor any CP h5ad. Not a pipeline bug, just upstream coverage.
        #   * CP-painting markers (ops0094): ~18% all-NaN rows from CP h5ad
        #     coverage gaps. Same story.
        #   * Heterogeneous feature schemas: 12-20% all-NaN columns when the
        #     50-feature random sample hits features not measured in every
        #     experiment imaging the marker. Sampling noise.
        # Set thresholds well above these baselines so a true regression
        # (e.g. coordinate frame bug, attach failure) trips clearly.
        ROW_NAN_FRAC_FAIL = 0.35   # > 35% all-NaN rows
        COL_NAN_FRAC_FAIL = 0.25   # > 25% all-NaN columns
        if X_sample is not None and len(X_sample):
            n_nan_rows = int(np.all(np.isnan(X_sample), axis=1).sum())
            row_nan_frac = n_nan_rows / X_sample.shape[0] if X_sample.shape[0] else 0.0
            if row_nan_frac > ROW_NAN_FRAC_FAIL:
                issues.append(
                    f"{n_nan_rows}/{X_sample.shape[0]} all-NaN rows "
                    f"({row_nan_frac:.0%}) — likely upstream OP/CP coverage gap"
                )
            n_nan_cols = int(np.all(np.isnan(X_sample), axis=0).sum())
            col_nan_frac = n_nan_cols / X_sample.shape[1] if X_sample.shape[1] else 0.0
            if col_nan_frac > COL_NAN_FRAC_FAIL:
                issues.append(
                    f"{n_nan_cols}/{X_sample.shape[1]} all-NaN columns "
                    f"({col_nan_frac:.0%}) — sampled features absent across cohort"
                )

        a.file.close()

        # OP / CP match rates
        op_match_rate = float(obs["op_match"].mean()) if "op_match" in obs.columns else float("nan")
        cp_match_rate = float(obs["cp_match"].mean()) if "cp_match" in obs.columns else float("nan")

        status = "OK" if not issues else "FAIL"
        if issues:
            n_fail += 1
        else:
            n_pass += 1

        summary_rows.append({
            "file": path.name,
            "viz_channel": viz,
            "n_cells": n_cells,
            "n_features": n_features,
            "n_KO": n_ko,
            "n_NTC": n_ntc,
            "n_exps": n_experiments,
            "n_genes": n_genes,
            "OP%": op_match_rate,
            "CP%": cp_match_rate,
            "z-score": zscore_summary,
            "status": status,
            "issues": issues,
        })

    # Print summary table.
    print(f"\n{'file':50s}  {'viz_channel':32s}  {'n_cells':>10s}  {'n_feats':>8s}  "
          f"{'n_KO':>9s}  {'n_NTC':>7s}  {'n_exps':>6s}  {'n_genes':>7s}  "
          f"{'OP%':>5s}  {'CP%':>5s}  {'z-score':16s}  status")
    print("-" * 180)
    for r in summary_rows:
        print(f"{r['file']:50s}  {r['viz_channel']:32s}  {r['n_cells']:>10,}  "
              f"{r['n_features']:>8}  {r['n_KO']:>9,}  {r['n_NTC']:>7,}  "
              f"{r['n_exps']:>6}  {r['n_genes']:>7}  "
              f"{r['OP%'] * 100 if not np.isnan(r['OP%']) else 0:>5.1f}  "
              f"{r['CP%'] * 100 if not np.isnan(r['CP%']) else 0:>5.1f}  "
              f"{r['z-score']:16s}  {r['status']}")

    # Print per-file issues.
    bad = [r for r in summary_rows if r["issues"]]
    if bad:
        print(f"\nIssues found in {len(bad)} file(s):")
        for r in bad:
            print(f"  [{r['file']}]")
            for issue in r["issues"]:
                print(f"    - {issue}")

    print(f"\n{'='*98}")
    print(f"Pass: {n_pass}/{len(h5ads)}   Fail: {n_fail}/{len(h5ads)}")
    print(f"{'='*98}")
    return 0 if n_fail == 0 else 1


def _is_excluded_pair(viz: str, phys: str) -> bool:
    """True for (viz_channel, physical_channel) pairs that aren't biological
    readouts and should be dropped before any per-channel processing:
      * 4i DAPI registration (`4i_R*_nuclei` — alignment reference).
        Live-cell `nucleus_Hoechst` is kept (physical channel is gfp-class).
      * Autofluorescence / bleedthrough / empty / no-label markers (all
        normalize to viz_channels containing "no label" in the
        ops_channel_maps.yaml; catches the typo variants "bleedthough" and
        "bleedthrough" in one rule).
    """
    p = str(phys).lower()
    v = str(viz).lower()
    if p.endswith("_nuclei"):
        return True
    if "no label" in v:
        return True
    return False


def _load_paper_v1(path: Path) -> Dict[str, List[str]]:
    """Read `good_experiment_list_v1.yml` -> {full_exp_name: [op_channels...]}.

    Mirrors `pca_optimization.py`'s --paper-v1 semantics: keys are full
    experiment names (`ops0094_20251217`), values are OP physical channel
    names (Phase / GFP / mCherry / Cy5).
    """
    import yaml
    with path.open() as f:
        doc = yaml.safe_load(f) or {}
    ec = doc.get("experiments_channels", {}) or {}
    if not ec:
        raise SystemExit(f"No 'experiments_channels' key in {path}")
    return ec


def _build_all_cells(
    paper_v1: Dict[str, List[str]],
    channel_maps: Dict[str, Dict[str, str]],
    ko_cap: Optional[int],
    ntc_cap: Optional[int],
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Enumerate every (cell, viz_channel) row in paper_v1 with stratified caps.

    Each cell from the linked CSV expands into:
      - 1 phase row (modality=phase, viz_channel="Phase", op_channel="phase")
      - K fluor rows, one per fluor viz_channel imaged in that experiment
        (op_channel = the marker's physical channel).

    Stratified caps:
      KO: <= ko_cap rows per (gene, viz_channel), sampled across exps.
      NTC: <= ntc_cap rows per viz_channel, sampled across paper_v1 exps.

    Returns the master cells DataFrame in the schema `_attach_op_features` /
    `_attach_cp_features` expect (experiment, well_canonical, x_pheno, y_pheno,
    segmentation, modality, viz_channel, channel_rank, op_channel, gene, ...).
    """
    from tqdm import tqdm
    rng = np.random.default_rng(rng_seed)

    # Step 1 — load linked CSVs across paper_v1 + per-exp multiplet dedup.
    pieces = []
    empty_exps = []
    n_raw_total = 0
    n_dedup_total = 0
    sorted_exps = sorted(paper_v1.keys())
    for exp in tqdm(sorted_exps, desc="Loading linked CSVs",
                    unit="exp", smoothing=0.1):
        ldf = _load_linked_for_exp(exp)
        if ldf.empty:
            empty_exps.append(exp)
            continue
        # Multiplet drop: cells with 2+ distinct sgRNAs or no guide are removed.
        # See cyclops_process.utils.dedupe_linked_pheno_iss for the rule.
        ldf = ldf.copy()
        ldf["well"] = ldf["well_canonical"]
        n_before = len(ldf)
        ldf = dedupe_linked_csv(ldf)
        n_raw_total += n_before
        n_dedup_total += len(ldf)
        pieces.append(ldf)
    if empty_exps:
        print(f"  [linked] {len(empty_exps)} experiments empty/missing: "
              f"{empty_exps[:3]}{'...' if len(empty_exps) > 3 else ''}")
    if not pieces:
        raise SystemExit("No linked CSVs loaded for paper_v1.")
    print(f"  Dedup (drop multiplets + no-guide): {n_raw_total:,} → "
          f"{n_dedup_total:,} rows ({n_raw_total - n_dedup_total:,} dropped)")
    cells_raw = pd.concat(pieces, ignore_index=True)

    # Drop rows missing essentials.
    cells_raw = cells_raw.dropna(subset=["x_pheno", "y_pheno", "segmentation_id"]).copy()
    cells_raw["segmentation"] = pd.to_numeric(
        cells_raw["segmentation_id"], errors="coerce",
    ).astype("Int64")
    cells_raw = cells_raw[cells_raw["segmentation"].notna()].copy()

    # KO vs NTC: gene_name NaN OR == "NTC" -> "NTC"; otherwise the gene_name.
    gn = cells_raw["gene_name"].astype("string")
    is_ntc = gn.isna() | (gn.str.upper() == "NTC")
    cells_raw["gene"] = np.where(is_ntc, "NTC", gn.fillna("").astype(str))

    n_ko = int((cells_raw["gene"] != "NTC").sum())
    n_ntc = int((cells_raw["gene"] == "NTC").sum())
    n_genes = cells_raw.loc[cells_raw["gene"] != "NTC", "gene"].nunique()
    print(f"  Raw linked cells: {len(cells_raw):,} ({n_ko:,} KO across {n_genes} genes + {n_ntc:,} NTC)")

    # Step 2 — expand each cell to (cell, viz_channel) rows.
    expanded = []
    exp_groups = list(cells_raw.groupby("experiment", sort=False))
    for exp, exp_grp in tqdm(exp_groups, desc="Expanding (cell, viz_channel) rows",
                             unit="exp", smoothing=0.1):
        short_exp = str(exp).split("_")[0]
        cm = channel_maps.get(short_exp, {})
        # Drop phase entries from the channel map (we synthesize phase rows
        # directly below). Keep only fluor viz_channels. Drop non-biological.
        fluor_pairs = sorted([
            (v, p) for v, p in cm.items()
            if p != "phase" and not _is_excluded_pair(v, p)
        ])
        if not fluor_pairs:
            tqdm.write(f"    [expand] {exp}: no fluor viz_channels; phase rows only")

        # Phase row per cell.
        ph = exp_grp.copy()
        ph["modality"] = "phase"
        ph["viz_channel"] = "Phase"
        ph["channel_rank"] = 1
        ph["op_channel"] = "phase"
        expanded.append(ph)

        # One fluor row per (cell, fluor_viz_channel).
        for ch_rank, (viz, phys) in enumerate(fluor_pairs, start=1):
            fl = exp_grp.copy()
            fl["modality"] = "fluorescent"
            fl["viz_channel"] = viz
            fl["channel_rank"] = ch_rank
            fl["op_channel"] = phys
            expanded.append(fl)

    cells = pd.concat(expanded, ignore_index=True)
    cells["well"] = cells["well_canonical"]
    cells["pma_attention"] = np.nan
    cells["model_confidence"] = np.nan
    cells["predicted_class"] = ""
    cells["rank"] = 1  # placeholder; reassigned post-cap

    pre_cap = len(cells)
    print(f"  Expanded to (cell, viz_channel) rows: {pre_cap:,}")

    # Step 3 — stratified caps. Vectorized via global shuffle + groupby().head()
    # which is O(N log N) for the shuffle and avoids Python-level iteration
    # over ~57k (gene, viz_channel) groups (the row-by-row pattern was the
    # multi-minute hang).
    ko_mask = cells["gene"] != "NTC"
    ko_cells = cells[ko_mask]
    ntc_cells = cells[~ko_mask]
    n_ko_total = len(ko_cells)
    n_ntc_total = len(ntc_cells)

    if ko_cap is None:
        print(f"  KO cap: none — keeping all {n_ko_total:,} KO rows", flush=True)
        ko_capped = ko_cells
    else:
        print(f"  Capping KO cells (cap={ko_cap}/(gene, viz_channel))...", flush=True)
        ko_capped = (
            ko_cells.sample(frac=1.0, random_state=rng_seed)  # randomize within group
            .groupby(["gene", "viz_channel"], sort=False, observed=True)
            .head(ko_cap)
        )
        print(f"    {len(ko_capped):,}/{n_ko_total:,} KO rows kept", flush=True)

    if ntc_cap is None:
        print(f"  NTC cap: none — keeping all {n_ntc_total:,} NTC rows", flush=True)
        ntc_capped = ntc_cells
    else:
        print(f"  Capping NTC cells (cap={ntc_cap}/viz_channel)...", flush=True)
        ntc_capped = (
            ntc_cells.sample(frac=1.0, random_state=rng_seed + 1)
            .groupby("viz_channel", sort=False, observed=True)
            .head(ntc_cap)
        )
        print(f"    {len(ntc_capped):,}/{n_ntc_total:,} NTC rows kept", flush=True)

    capped = pd.concat([ko_capped, ntc_capped], ignore_index=True)
    capped["rank"] = capped.groupby(
        ["gene", "viz_channel"], sort=False, observed=True,
    ).cumcount() + 1

    print(f"  After caps: {len(capped):,} rows "
          f"(KO {len(ko_capped):,}/{n_ko_total:,} kept; "
          f"NTC {len(ntc_capped):,}/{n_ntc_total:,} kept)")
    return capped


def _zscore_per_batch(
    X: np.ndarray,
    obs: pd.DataFrame,
    batch_cols: List[str],
    clip: float = ZSCORE_CLIP,
    min_cells: int = ZSCORE_MIN_CELLS,
) -> int:
    """In-place z-score features per batch defined by `obs[batch_cols]`.

    For each batch, compute per-feature mean+std on non-NaN values and replace
    finite cell entries with (val - mean) / std. NaN-only feature columns in a
    batch stay NaN. Sub-`min_cells` batches are left raw. Result clipped to
    ±`clip` to bound stretched-tail outliers without sign distortion. Returns
    the number of batches that were z-scored.
    """
    if len(batch_cols) == 1:
        batch_keys = obs[batch_cols[0]].astype(str).values
    else:
        batch_keys = obs[batch_cols].astype(str).agg("|".join, axis=1).values

    import warnings as _w
    n_zscored = 0
    for batch in np.unique(batch_keys):
        rows = np.flatnonzero(batch_keys == batch)
        if len(rows) < min_cells:
            continue
        sub = X[rows, :]
        # Suppress benign "Mean of empty slice" / "DoF <= 0" RuntimeWarnings
        # that fire when a feature column is all-NaN within this batch (some
        # experiments measure different OP feature subsets — expected). The
        # `np.where(... finite & sigma>0)` guard below promotes any invalid
        # statistic to NaN, which then propagates as NaN in the z-score
        # output — exactly the right behavior for missing features.
        with _w.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            _w.simplefilter("ignore", category=RuntimeWarning)
            mu = np.nanmean(sub, axis=0)
            sigma = np.nanstd(sub, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, np.nan)
        with np.errstate(invalid="ignore"):
            z = (sub - mu) / sigma
            z = np.clip(z, -clip, clip)
        # Preserve NaNs from input and from sigma==0 / NaN cases.
        X[rows, :] = z
        n_zscored += 1
    return n_zscored


def _sanitize_channel_name(viz_channel: str) -> str:
    """Filename-safe form of a viz_channel string. Lowercases, replaces
    spaces, slashes, commas, and parens with underscores. "Phase" -> "phase";
    "actin filament_FastAct_SPY555 Live Cell Dye" ->
    "actin_filament_fastact_spy555_live_cell_dye".
    """
    s = str(viz_channel).strip().lower()
    for ch in (" ", "/", ",", "(", ")"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _apply_op_unit_correction(
    X_op: np.ndarray,
    var_op: pd.DataFrame,
    cells: pd.DataFrame,
) -> Dict:
    """Apply the missing-2×-bin pixel-size correction to OP feature values.

    OP features extracted from per-experiment h5ads whose assembly zarr had
    the buggy 0.65 µm/px metadata are 2× too large for length-units and 4×
    for area-units (see CORRECTION_FACTORS in fix_op_units.py). Per-experiment
    h5ads that have already been run through fix_op_units.py carry an
    `uns["op_units_correction"]["applied"]` marker — those are skipped here
    (their X rows arrive already-corrected).

    Also writes a clean bool `var_op["op_units_corrected"]` flag (True iff
    the feature's unit has a non-1.0 factor). The pass-1 var union otherwise
    leaves this column as object-dtype mixed bool+NaN (some upstream h5ads
    have it, some don't) — which h5py rejects at vlen-string write time.
    """
    import anndata as ad

    # Lazy import: keeps fix_op_units off the module-level dependency graph
    # so submitit can cloudpickle the worker entrypoint without the worker
    # needing `scripts/marker_volcanoes` on sys.path at unpickle time.
    if str(_FIX_OP_UNITS_DIR) not in sys.path:
        sys.path.insert(0, str(_FIX_OP_UNITS_DIR))
    from fix_op_units import (
        CORRECTION_FACTORS as _OP_UNIT_FACTORS,
        discover_affected_experiments as _discover_affected_experiments,
    )

    if "unit" not in var_op.columns:
        var_op["op_units_corrected"] = False
        return {"rows_corrected": 0, "cols_eligible": 0, "experiments_corrected": []}

    units = var_op["unit"].astype(str).values
    factors = np.array(
        [_OP_UNIT_FACTORS.get(u, 1.0) for u in units], dtype=np.float32,
    )
    col_mask = factors != 1.0
    # Overwrite any inherited mixed bool/NaN column with a clean bool dtype.
    var_op["op_units_corrected"] = col_mask

    col_idx = np.where(col_mask)[0]
    if len(col_idx) == 0:
        return {"rows_corrected": 0, "cols_eligible": 0, "experiments_corrected": []}

    affected = _discover_affected_experiments()
    exp_codes = cells["experiment"].astype(str).values

    # Per-experiment: needs correction iff exp's zarr was corrected (in
    # `affected`) AND the per-exp OP h5ad hasn't been corrected in place.
    needs: Dict[str, bool] = {}
    for exp in np.unique(exp_codes):
        if exp not in affected:
            needs[exp] = False
            continue
        p = (OPS_FAST_ROOT / exp / "3-assembly" / "feature_extraction"
             / f"{exp}_cell_features.h5ad")
        try:
            a = ad.read_h5ad(p, backed="r")
            already = bool(a.uns.get("op_units_correction", {}).get("applied"))
            a.file.close()
        except Exception:
            already = False
        needs[exp] = not already

    row_mask = np.array([needs.get(e, False) for e in exp_codes])
    n_rows = int(row_mask.sum())
    exps_corrected = sorted(e for e, n in needs.items() if n)
    if n_rows == 0:
        return {
            "rows_corrected": 0,
            "cols_eligible": int(len(col_idx)),
            "experiments_corrected": [],
        }

    col_factors = factors[col_idx]
    sub = X_op[row_mask][:, col_idx]
    X_op[np.ix_(row_mask, col_idx)] = (sub * col_factors).astype(X_op.dtype, copy=False)
    return {
        "rows_corrected": n_rows,
        "cols_eligible": int(len(col_idx)),
        "experiments_corrected": exps_corrected,
    }


def _process_channel(
    viz_channel: str,
    cells: pd.DataFrame,
    cp_match_radius_px: float,
    n_workers: int,
    no_cp: bool,
    output_path: Path,
    zscore_per_experiment: bool = False,
) -> Dict:
    """Build one (cells × features) h5ad for a single viz_channel.

    Memory budget: peak ≈ len(cells) × OP_union_features × 4B during attach,
    drops to len(cells) × kept_features × 4B after the all-NaN column filter.
    For a typical 2M-row channel: peak ~75 GB → ~5 GB final.
    """
    import anndata as ad

    n = len(cells)
    if n == 0:
        print(f"  [{viz_channel}] no cells; skipping")
        return {"viz_channel": viz_channel, "n_cells": 0, "skipped": True}

    print(f"\n{'-'*72}\n[{viz_channel}] {n:,} cells\n{'-'*72}", flush=True)

    # OP attach for this channel only. Pass `organelle_filter` so the pass-1
    # union shrinks to features whose organelle bucket matches this channel
    # (`op_channel`) plus channel-agnostic morphology (cell, nuclei, etc.).
    # Cuts the dense X_op alloc 3-6× vs the full paper_v1 union — turns Phase
    # from a ~430 GB peak into ~70 GB at cap=2000/sgRNA.
    op_channels_in_cells = set(cells["op_channel"].astype(str).unique())
    organelle_filter = sorted(op_channels_in_cells | {"agnostic"})
    X_op, op_names, var_op, op_hit, op_stats = _attach_op_features(
        cells, cp_match_radius_px, n_workers, per_cell_channel_mask=True,
        organelle_filter=organelle_filter,
    )
    print(f"  OP attach: {op_hit.sum():,}/{n:,} matched ({op_hit.mean():.1%})  "
          f"shape {X_op.shape}", flush=True)

    # Pixel-size correction (0.65 → 0.325 µm/px). Runs BEFORE the all-NaN
    # column compact and the z-score so corrected values feed both. Skipped
    # per-row for source h5ads already carrying the fix_op_units uns marker.
    unit_corr = _apply_op_unit_correction(X_op, var_op, cells)
    exps_str = ", ".join(unit_corr["experiments_corrected"][:3])
    more = (f" (+{len(unit_corr['experiments_corrected']) - 3} more)"
            if len(unit_corr["experiments_corrected"]) > 3 else "")
    print(f"  OP unit correction: {unit_corr['rows_corrected']:,} rows × "
          f"{unit_corr['cols_eligible']} eligible cols  "
          f"experiments=[{exps_str}{more}]", flush=True)

    # Drop all-NaN OP columns. The per-cell channel mask already NaN'd
    # features for irrelevant channels; this just compacts the matrix.
    keep_op = ~np.all(np.isnan(X_op), axis=0)
    if not keep_op.all():
        n_dropped = int((~keep_op).sum())
        X_op = X_op[:, keep_op]
        op_names = [nm for nm, k in zip(op_names, keep_op) if k]
        var_op = var_op.iloc[keep_op].copy()
        print(f"  OP feature compact: dropped {n_dropped} all-NaN cols, "
              f"kept {len(op_names)}", flush=True)

    obs = _build_obs(cells)
    # `_build_obs` in the shared helper keys obs.index on
    # (experiment, well_canonical, segmentation, modality, viz_channel, rank)
    # which collides when the same segmentation_id appears in two different
    # tiles of the same well (tile-local IDs). Append tile_pheno to make
    # the index globally unique.
    if "tile_pheno" in obs.columns:
        obs.index = (obs.index.astype(str) + "_t"
                     + obs["tile_pheno"].astype(str).str.rsplit("/", n=1).str[-1])
    obs["op_match"] = op_hit

    op_names_prefixed = [f"op_{nm}" for nm in op_names]
    var_op = var_op.copy()
    var_op.index = op_names_prefixed
    var_op["source"] = "organelle_profiler"
    if "feature_name" in var_op.columns:
        var_op["feature_name"] = op_names_prefixed

    if not no_cp:
        X_cp, cp_names, cp_hit, cp_stats = _attach_cp_features(
            cells, cp_match_radius_px, n_workers,
        )
        print(f"  CP attach: {cp_hit.sum():,}/{n:,} matched ({cp_hit.mean():.1%})  "
              f"shape {X_cp.shape}", flush=True)
        keep_cp = ~np.all(np.isnan(X_cp), axis=0)
        if not keep_cp.all():
            n_dropped = int((~keep_cp).sum())
            X_cp = X_cp[:, keep_cp]
            cp_names = [c for c, k in zip(cp_names, keep_cp) if k]
            print(f"  CP feature compact: dropped {n_dropped} all-NaN cols, "
                  f"kept {len(cp_names)}", flush=True)
        obs["cp_match"] = cp_hit
        cp_names_prefixed = [f"cp_{c}" for c in cp_names]
        var_cp = pd.DataFrame(
            {"feature_name": cp_names_prefixed, "source": "cellprofiler"},
            index=cp_names_prefixed,
        )
        X = np.concatenate([X_op, X_cp], axis=1)
        var = pd.concat([var_op, var_cp])
    else:
        X = X_op
        var = var_op.copy()

    # Per-experiment z-score (single viz_channel per file → batch = experiment).
    # OFF by default: the z-scored values land on a totally different
    # scale (~16,000× smaller mean) than the raw values that
    # `consolidate_top_attention_cells.py` writes, so SHAP classifiers
    # comparing the two pipelines saturate at AUROC=1.0 on a trivial
    # threshold — not biology. Pass --zscore-per-experiment to restore
    # the legacy behavior.
    if zscore_per_experiment:
        print(f"  Z-scoring per experiment...", flush=True)
        n_zscored = _zscore_per_batch(X, obs, ["experiment"])
        var = var.copy()
        var["zscored_per_batch"] = "experiment"
    else:
        print(f"  Skipping z-score (--zscore-per-experiment not set; raw values "
              f"kept so downstream pipelines see the same scale as the "
              f"top-attention extraction).", flush=True)
        n_zscored = 0
        var = var.copy()
        var["zscored_per_batch"] = "none"

    # Sanitize object-dtype var columns for h5ad write. Pass-1 union in
    # `_attach_op_features` picks the first exp's value for each feature,
    # and pd.concat(var_op, var_cp) leaves CP rows NaN for OP-only columns
    # (e.g. `op_units_corrected`) — both produce object-dtype mixed types
    # that h5py rejects at vlen-string write time. Coerce bool-like cols to
    # bool (NaN → False), everything else to string (NaN → "").
    for col in var.columns:
        if var[col].dtype != object:
            continue
        ser = var[col]
        non_na = ser.dropna()
        if len(non_na) and all(isinstance(v, (bool, np.bool_)) for v in non_na):
            var[col] = ser.fillna(False).astype(bool)
        else:
            var[col] = ser.fillna("").astype(str)

    print(f"  Writing -> {output_path.name}  (shape {X.shape}, "
          f"{n_zscored} experiments z-scored)", flush=True)
    adata_out = ad.AnnData(
        X=X.astype(np.float32, copy=False),
        obs=obs.copy(),
        var=var.copy(),
    )
    # Defensive: catch any residual obs-name collisions before write so
    # downstream consumers don't hit anndata's "non-unique" warning.
    if not adata_out.obs_names.is_unique:
        n_dup = int(adata_out.obs_names.duplicated().sum())
        print(f"  [{viz_channel}] disambiguating {n_dup} duplicate obs names "
              f"(make_unique appends -1/-2/...)", flush=True)
        adata_out.obs_names_make_unique()
    adata_out.write_h5ad(output_path)

    return {
        "viz_channel": viz_channel,
        "n_cells": n,
        "n_features": int(X.shape[1]),
        "op_match_rate": float(op_hit.mean()),
        "path": str(output_path),
    }


def _build_cells_for_channel(
    viz_channel: str,
    op_channel: str,
    experiments: List[str],
    ko_cap: Optional[int],
    ntc_cap: Optional[int],
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Worker-side cells loader: read linked CSVs for the experiments imaging
    this marker, apply per-channel caps, return cells DataFrame.

    No row expansion needed (single viz_channel per worker), so the cells
    table has at most one row per cell. NTC cap is per viz_channel (single
    pool for this worker); KO cap is per gene (since every gene with cells
    in this channel maps to one (gene, viz_channel) pair).
    """
    from tqdm import tqdm

    rng = np.random.default_rng(rng_seed)

    pieces = []
    desc = f"[{viz_channel}] linked CSVs"
    n_raw_total = 0
    n_dedup_total = 0
    for exp in tqdm(experiments, desc=desc, unit="exp", smoothing=0.1):
        ldf = _load_linked_for_exp(exp)
        if ldf.empty:
            continue
        # Multiplet drop: cells with 2+ distinct sgRNAs or no guide are removed.
        # dedupe_linked_csv keys on (well, tile_pheno, segmentation_id); our
        # loaded df carries well_canonical, so alias it as `well` for the key.
        ldf = ldf.copy()
        ldf["well"] = ldf["well_canonical"]
        n_before = len(ldf)
        ldf = dedupe_linked_csv(ldf)
        n_raw_total += n_before
        n_dedup_total += len(ldf)
        pieces.append(ldf)
    if not pieces:
        return pd.DataFrame()
    print(f"  [{viz_channel}] dedup (drop multiplets + no-guide): "
          f"{n_raw_total:,} → {n_dedup_total:,} rows "
          f"({n_raw_total - n_dedup_total:,} dropped)", flush=True)

    cells = pd.concat(pieces, ignore_index=True)
    cells = cells.dropna(subset=["x_pheno", "y_pheno", "segmentation_id"]).copy()
    cells["segmentation"] = pd.to_numeric(
        cells["segmentation_id"], errors="coerce",
    ).astype("Int64")
    cells = cells[cells["segmentation"].notna()].copy()

    # Belt-and-suspenders cell-identity dedup. dedupe_linked_csv already runs
    # per-well above, but guard the concatenated table against any residual
    # duplicate cell — same physical (experiment, well, tile, segmentation) —
    # so it can never reach the feature attach as two rows. Keeps first.
    id_cols = ["experiment", "well_canonical", "tile_pheno", "segmentation"]
    n_before_dedup = len(cells)
    cells = cells.drop_duplicates(subset=id_cols, keep="first")
    n_dup = n_before_dedup - len(cells)
    if n_dup:
        print(f"  [{viz_channel}] removed {n_dup:,} duplicate cells "
              f"(same {tuple(id_cols)})", flush=True)

    gn = cells["gene_name"].astype("string")
    is_ntc = gn.isna() | (gn.str.upper() == "NTC")
    cells["gene"] = np.where(is_ntc, "NTC", gn.fillna("").astype(str))

    # Tag rows with this channel — single value per worker, no expansion.
    cells["modality"] = "phase" if viz_channel == "Phase" else "fluorescent"
    cells["viz_channel"] = viz_channel
    cells["channel_rank"] = 1
    cells["op_channel"] = op_channel
    cells["well"] = cells["well_canonical"]
    cells["pma_attention"] = np.nan
    cells["model_confidence"] = np.nan
    cells["predicted_class"] = ""

    # Vectorized caps PER sgRNA. Each guide gets equal representation —
    # cells from a single sgRNA are sampled to <= ko_cap (KO) or <= ntc_cap
    # (NTC). With ~3 sgRNAs/gene, KO cells/gene ≈ 3 × ko_cap. With ~hundreds
    # of NTC sgRNAs in the screen, NTC cells/channel ≈ N_ntc_sgRNAs × ntc_cap
    # — much larger than a fixed per-channel cap, and proportionally weighted
    # across all NTC guides instead of dominated by a few high-coverage ones.
    cells_with_guide = cells.dropna(subset=["sgRNA"]).copy()
    cells_with_guide = cells_with_guide[cells_with_guide["sgRNA"].astype(str) != ""]
    n_dropped_no_guide = len(cells) - len(cells_with_guide)
    if n_dropped_no_guide:
        print(f"  [{viz_channel}] dropped {n_dropped_no_guide:,} rows with "
              f"missing sgRNA", flush=True)

    ko_mask = cells_with_guide["gene"] != "NTC"
    ko = cells_with_guide[ko_mask]
    ntc = cells_with_guide[~ko_mask]
    n_ko_total = len(ko)
    n_ntc_total = len(ntc)
    n_ko_guides = ko["sgRNA"].nunique() if n_ko_total else 0
    n_ntc_guides = ntc["sgRNA"].nunique() if n_ntc_total else 0

    print(f"  [{viz_channel}] raw: "
          f"{n_ko_total:,} KO ({n_ko_guides:,} sgRNAs) + "
          f"{n_ntc_total:,} NTC ({n_ntc_guides:,} sgRNAs)", flush=True)

    if ko_cap is None:
        ko_capped = ko
    else:
        ko_capped = (
            ko.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
            .groupby("sgRNA", sort=False, observed=True)
            .head(ko_cap)
        )
    if ntc_cap is None:
        ntc_capped = ntc
    else:
        ntc_capped = (
            ntc.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
            .groupby("sgRNA", sort=False, observed=True)
            .head(ntc_cap)
        )

    capped = pd.concat([ko_capped, ntc_capped], ignore_index=True)
    # Rank within (gene, viz_channel) for unique-index purposes downstream.
    capped["rank"] = capped.groupby(
        ["gene", "sgRNA"], sort=False, observed=True,
    ).cumcount() + 1

    ko_cap_str = "none" if ko_cap is None else f"{ko_cap}/guide"
    ntc_cap_str = "none" if ntc_cap is None else f"{ntc_cap}/guide"
    print(f"  [{viz_channel}] after caps (per-sgRNA: KO {ko_cap_str}, "
          f"NTC {ntc_cap_str}): "
          f"{len(ko_capped):,}/{n_ko_total:,} KO + "
          f"{len(ntc_capped):,}/{n_ntc_total:,} NTC = {len(capped):,}",
          flush=True)
    return capped


def _process_channel_slurm(
    viz_channel: str,
    op_channel: str,
    experiments: List[str],
    output_path: str,
    ko_cap: Optional[int],
    ntc_cap: Optional[int],
    cp_match_radius_px: float,
    no_cp: bool,
    zscore_per_experiment: bool = False,
) -> Dict:
    """SLURM worker entry point — top-level so cloudpickle / submitit can
    serialize it cleanly across nodes.

    Self-contained: loads its own linked CSVs (just for the experiments
    imaging this marker), applies caps, attaches OP+CP features, writes its
    h5ad. No login-node master parquet needed.
    """
    from cyclops_utils.hpc.resource_manager import get_optimal_workers

    print(f"[{viz_channel}] Building cells from {len(experiments)} experiments "
          f"(op_channel={op_channel})...", flush=True)
    cells = _build_cells_for_channel(
        viz_channel=viz_channel,
        op_channel=op_channel,
        experiments=experiments,
        ko_cap=ko_cap,
        ntc_cap=ntc_cap,
    )
    if cells.empty:
        print(f"  [{viz_channel}] no cells; skipping", flush=True)
        return {"viz_channel": viz_channel, "skipped": True}

    n_workers = get_optimal_workers(
        use_gpu=False, model_ram_gb=1.0, data_ram_gb=2.0, verbose=True,
    )

    return _process_channel(
        viz_channel, cells, cp_match_radius_px, n_workers, no_cp,
        Path(output_path), zscore_per_experiment=zscore_per_experiment,
    )


def _plan_channel_jobs(
    paper_v1: Dict[str, List[str]],
    channel_maps: Dict[str, Dict[str, str]],
) -> Dict[str, Dict]:
    """Build the per-channel job plan: viz_channel -> {op_channel, experiments}.

    Iterates paper_v1 experiments, looks up their channel_map entries, and
    groups by viz_channel. Phase rows from each experiment's channel_map
    fold into the canonical "Phase" key with `op_channel='phase'`. Excluded
    channels (autofluor, bleedthrough, empty, no-label, DAPI registration)
    are dropped here so they never become jobs.
    """
    plan: Dict[str, Dict] = {}
    for exp_full in sorted(paper_v1.keys()):
        short = exp_full.split("_")[0]
        cm = channel_maps.get(short, {})
        for v, p in cm.items():
            if p == "phase":
                key, op = "Phase", "phase"
            else:
                if _is_excluded_pair(v, p):
                    continue
                key, op = v, p
            entry = plan.setdefault(key, {"op_channel": op, "experiments": []})
            if entry["op_channel"] != op:
                raise SystemExit(
                    f"viz_channel {key!r} maps to multiple op_channels: "
                    f"{entry['op_channel']!r} vs {op!r} (in {exp_full})"
                )
            entry["experiments"].append(exp_full)
    return plan


def run_consolidation_all_cells(
    paper_v1_path: Path,
    output_dir: Path,
    cp_match_radius_px: float,
    ko_cap: Optional[int],
    ntc_cap: Optional[int],
    no_cp: bool,
    channel_maps_yaml: Path,
    zscore_per_experiment: bool = False,
    skip_existing: bool = True,
    submit_via_slurm: bool = True,
    slurm_params: Optional[Dict] = None,
    wait: bool = True,
) -> int:
    """Login-node orchestrator. Plans + submits one SLURM job per viz_channel
    via `submit_parallel_jobs`. No login-node compute beyond reading two
    yamls; each per-channel worker is fully self-contained (loads its own
    linked CSVs for the experiments imaging its marker, applies its own
    caps, attaches features, writes its h5ad).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nConsolidate ALL paper_v1 cells (KO + NTC) — per-marker fan-out\n{'='*72}")

    paper_v1 = _load_paper_v1(paper_v1_path)
    print(f"paper_v1: {len(paper_v1)} experiments from {paper_v1_path.name}")

    print(f"Loading channel maps from {channel_maps_yaml}")
    channel_maps = _load_channel_maps(channel_maps_yaml)
    print(f"  loaded maps for {len(channel_maps)} experiments")

    ko_cap_str = "none" if ko_cap is None else f"{ko_cap}/sgRNA"
    ntc_cap_str = "none" if ntc_cap is None else f"{ntc_cap}/sgRNA"
    print(f"\nPlanning per-channel jobs (KO cap={ko_cap_str}, "
          f"NTC cap={ntc_cap_str})...")
    plan = _plan_channel_jobs(paper_v1, channel_maps)
    print(f"  {len(plan)} viz_channels × experiments imaging each:")
    for viz, info in sorted(plan.items(),
                            key=lambda kv: (-len(kv[1]["experiments"]), kv[0])):
        n = len(info["experiments"])
        sample = ", ".join(info["experiments"][:3])
        more = f" ... (+{n - 3} more)" if n > 3 else ""
        print(f"    {viz:42s} -> {info['op_channel']:24s} : {n:>3} exps -- {sample}{more}")

    jobs: List[Dict] = []
    skipped: List[str] = []
    for viz, info in plan.items():
        out_name = f"op_cp_features_{_sanitize_channel_name(viz)}.h5ad"
        out_path = output_dir / out_name

        if skip_existing and out_path.exists():
            skipped.append(viz)
            continue

        jobs.append({
            "name": f"channel_{_sanitize_channel_name(viz)}",
            "func": _process_channel_slurm,
            "kwargs": {
                "viz_channel": viz,
                "op_channel": info["op_channel"],
                "experiments": info["experiments"],
                "output_path": str(out_path),
                "ko_cap": ko_cap,
                "ntc_cap": ntc_cap,
                "cp_match_radius_px": cp_match_radius_px,
                "no_cp": no_cp,
                "zscore_per_experiment": zscore_per_experiment,
            },
            "metadata": {
                "viz_channel": viz,
                "n_experiments": len(info["experiments"]),
                "op_channel": info["op_channel"],
            },
        })

    if skipped:
        print(f"\nSkipping {len(skipped)} channels with existing h5ads "
              f"(use --force / -f to overwrite):")
        for viz in skipped:
            print(f"  [skip] {viz}")

    if not jobs:
        print("\nNo jobs to submit. All channels already processed.")
        return 0

    if not submit_via_slurm:
        # Sequential local processing — debug only.
        print(f"\nLocal mode: processing {len(jobs)} channels sequentially.")
        for i, job in enumerate(jobs, start=1):
            print(f"\n[{i}/{len(jobs)}] {job['name']}")
            try:
                job["func"](**job["kwargs"])
            except Exception as e:
                print(f"  FAILED: {e!r}")
                import traceback
                traceback.print_exc()
        return 0

    # SLURM fan-out: phase gets its own array (high-mem, since it loads ALL
    # 77 experiments' OP h5ads → union schema = ~9785 features → 2M cells ×
    # 9785 × 4B ≈ 78 GB just for the dense X_op alloc, plus CP attach + concat
    # + joblib worker overhead → easily 200 GB peak). Fluor markers get the
    # standard 100 GB allocation since each only sees its own experiments'
    # h5ads (typically ~3000 features). Both arrays are tracked together via
    # `wait_for_multiple_job_arrays`.
    from cyclops_utils.hpc.slurm_batch_utils import (
        submit_parallel_jobs, wait_for_multiple_job_arrays,
    )

    base_sp = slurm_params or {
        "timeout_min": 240,
        "slurm_mem": "400G",
        "cpus_per_task": 8,
        "slurm_partition": "cpu",
    }
    fluor_sp = {**base_sp}  # standard allocation for the 56 fluor markers
    # Uncapped Phase loads all 77 experiments' OP h5ads → ~54M cells × ~1500
    # filtered features × 4B ≈ 320 GB X_op alone; +X_cp (~160 GB) + concat
    # (~480 GB transient) → ~600-700 GB data-side peak, +overhead → ~800 GB
    # true peak. 1 TB allocation gives ~25% headroom. Floor at base_sp.slurm_mem
    # so user can override upward via --mem.
    def _max_mem(a: str, b: str) -> str:
        def _gb(s):
            s = str(s).upper().rstrip("B")
            if s.endswith("T"):
                return int(float(s[:-1]) * 1024)
            if s.endswith("G"):
                return int(float(s[:-1]))
            if s.endswith("M"):
                return max(1, int(float(s[:-1]) // 1024))
            return int(float(s)) if s else 0
        return a if _gb(a) >= _gb(b) else b
    phase_sp = {
        **base_sp,
        "slurm_mem": _max_mem(base_sp.get("slurm_mem", "400G"), "1500G"),
        # Halved from 16 → 8 after OOM on a 1160 GB node: each pass-2 loky
        # worker materializes one experiment's matched rows × ~2000 features
        # × 4B ≈ 6 GB transiently, and 16 of those concurrent on top of the
        # ~430 GB X_op buffer overshoots 1 TB. 8 workers keeps peak under
        # ~1.2 TB while still parallelizing pass-2 across the 77 OP h5ads.
        "cpus_per_task": 8,
        "timeout_min": max(int(base_sp.get("timeout_min", 240)), 480),
    }

    phase_jobs = [j for j in jobs if j["metadata"]["viz_channel"] == "Phase"]
    fluor_jobs = [j for j in jobs if j["metadata"]["viz_channel"] != "Phase"]

    if not wait:
        # Fire-and-forget: submit both, return without tracking.
        if phase_jobs:
            submit_parallel_jobs(
                jobs_to_submit=phase_jobs, experiment="consolidate_all_cells_phase",
                slurm_params=phase_sp, log_dir="consolidate_all_cells",
                manifest_prefix="consolidate_all_cells_phase",
                wait_for_completion=False, verbose=True,
            )
        if fluor_jobs:
            submit_parallel_jobs(
                jobs_to_submit=fluor_jobs, experiment="consolidate_all_cells_fluor",
                slurm_params=fluor_sp, log_dir="consolidate_all_cells",
                manifest_prefix="consolidate_all_cells_fluor",
                wait_for_completion=False, verbose=True,
            )
        return 0

    # Submit both arrays without blocking, then jointly track.
    job_arrays: List[Dict] = []
    if phase_jobs:
        print(f"\nSubmitting Phase array ({len(phase_jobs)} job, "
              f"{phase_sp['slurm_mem']} mem, {phase_sp['cpus_per_task']} CPUs, "
              f"{phase_sp['timeout_min']} min)...")
        ph_result = submit_parallel_jobs(
            jobs_to_submit=phase_jobs, experiment="consolidate_all_cells_phase",
            slurm_params=phase_sp, log_dir="consolidate_all_cells",
            manifest_prefix="consolidate_all_cells_phase",
            wait_for_completion=False, verbose=True,
        )
        job_arrays.append({
            "submitted_jobs": ph_result["submitted_jobs"],
            "base_job_id": ph_result["base_job_id"],
            "label": "phase",
            "slurm_params": phase_sp,
        })

    if fluor_jobs:
        print(f"\nSubmitting Fluor array ({len(fluor_jobs)} jobs, "
              f"{fluor_sp['slurm_mem']} mem, {fluor_sp['cpus_per_task']} CPUs, "
              f"{fluor_sp['timeout_min']} min each)...")
        fl_result = submit_parallel_jobs(
            jobs_to_submit=fluor_jobs, experiment="consolidate_all_cells_fluor",
            slurm_params=fluor_sp, log_dir="consolidate_all_cells",
            manifest_prefix="consolidate_all_cells_fluor",
            wait_for_completion=False, verbose=True,
        )
        job_arrays.append({
            "submitted_jobs": fl_result["submitted_jobs"],
            "base_job_id": fl_result["base_job_id"],
            "label": "fluor",
            "slurm_params": fluor_sp,
        })

    if not job_arrays:
        return 0

    print(f"\nTracking {sum(len(a['submitted_jobs']) for a in job_arrays)} jobs "
          f"across {len(job_arrays)} arrays...")
    track = wait_for_multiple_job_arrays(
        job_arrays=job_arrays,
        experiment="consolidate_all_cells",
        verbose=True,
        print_resource_summary=True,
    )
    return 0 if track.get("all_completed") else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--paper-v1", type=str, nargs="?",
        const=str(DEFAULT_PAPER_V1), default=str(DEFAULT_PAPER_V1),
        help=f"paper_v1 YAML manifest (default: {DEFAULT_PAPER_V1})",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--ko-cap", type=int, default=DEFAULT_KO_CAP,
        help="Max KO cells per (sgRNA, viz_channel). Unset by default = "
             "no cap (every linked KO cell becomes a row).",
    )
    p.add_argument(
        "--ntc-cap", type=int, default=DEFAULT_NTC_CAP,
        help="Max NTC cells per (NTC sgRNA, viz_channel). Unset by default = "
             "no cap (every linked NTC cell becomes a row).",
    )
    p.add_argument("--cp-match-radius-px", type=float, default=20.0)
    p.add_argument("--no-cp", action="store_true")
    p.add_argument("--channel-maps-yaml", type=Path, default=CHANNEL_MAPS_YAML)
    p.add_argument(
        "--zscore-per-experiment", action="store_true", default=False,
        help="OFF by default. When set, applies per-experiment z-scoring "
             "to feature values (legacy behavior). Off keeps raw values "
             "on the same scale as consolidate_top_attention_cells.py so "
             "downstream SHAP classifiers comparing the two pipelines "
             "don't trivially saturate on a scale-mismatch threshold.",
    )
    p.add_argument(
        "--inspect", action="store_true",
        help="Print the paper_v1 channel-coverage table (kept vs excluded "
             "viz_channels, num exps and num cells per marker) and exit. "
             "Loads linked CSVs to count cells (~5 min on full paper_v1). "
             "Add --no-counts for the fast channels-only variant.",
    )
    p.add_argument(
        "--no-counts", action="store_true",
        help="With --inspect, skip the cell-counting h5ad reads. "
             "Channels-only summary, runs in seconds.",
    )
    p.add_argument(
        "--refresh-counts", action="store_true",
        help="With --inspect, recompute cell counts even if a cached JSON "
             "exists at all_cells_v1_inspect_cell_counts.json (~3.5 min "
             "serial read of 77 h5ads).",
    )
    p.add_argument(
        "--check-output", action="store_true",
        help="Validate the per-channel h5ads in --output-dir against the "
             "expected schema (required obs columns, single viz_channel per "
             "file, modality consistency, z-score sanity, KO cap respected, "
             "no all-NaN rows/cols). Prints a summary table + per-file "
             "issues. Exit code 0 if all pass, 1 otherwise.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan + per-experiment cell/channel availability, then exit "
             "(no attach, no z-score, no write). Includes --inspect output plus "
             "linked-CSV cell counts (slow: ~5 min on 77 experiments).",
    )

    p.add_argument(
        "--force", "-f", dest="skip_existing", action="store_false",
        default=True,
        help="Force re-processing of every channel, overwriting any existing "
             "h5ads in --output-dir. Default: skip channels whose h5ad "
             "already exists (resumable).",
    )

    # SLURM submission controls. Per-channel jobs fan out via
    # `submit_parallel_jobs` — each gets its own allocation. With the per-channel
    # `organelle_filter` pre-filter on `_attach_op_features`, peak memory drops
    # ~3-6× vs the unfiltered union (Phase: 9785 → ~1500 features kept). Defaults
    # sized for the uncapped paper_v1 scale: fluor 400G, Phase auto-bumps to 1.5TB.
    p.add_argument("--local", action="store_true",
                   help="Process channels sequentially on the login node "
                        "(debug only — slow). Default: SLURM fan-out.")
    p.add_argument("--mem", default="400G",
                   help="Per-job SLURM memory used as the fluor base + the "
                        "Phase floor (default 400G). Phase auto-bumps to "
                        "max(--mem, 1500G) for the uncapped ~54M-cell union.")
    p.add_argument("--cpus", type=int, default=8,
                   help="Per-job CPU count (default 8).")
    p.add_argument("--timeout-min", type=int, default=240,
                   help="Per-job timeout in minutes (default 240; Phase "
                        "auto-bumps to max(--timeout-min, 480)).")
    p.add_argument("--partition", default="cpu")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit and return immediately; don't block on completion.")

    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_v1_path = Path(args.paper_v1)

    if args.inspect:
        paper_v1 = _load_paper_v1(paper_v1_path)
        channel_maps = _load_channel_maps(args.channel_maps_yaml)
        cell_counts = (None if args.no_counts
                       else _count_cells_per_exp(paper_v1, refresh=args.refresh_counts))
        _print_coverage_table(paper_v1, channel_maps, cell_counts=cell_counts)
        return 0

    if args.check_output:
        return _check_outputs(args.output_dir, expected_ko_cap=args.ko_cap)

    if args.dry_run:
        print(f"\n{'='*72}\nConsolidate all_cells (--dry-run)\n{'='*72}")
        paper_v1 = _load_paper_v1(paper_v1_path)
        channel_maps = _load_channel_maps(args.channel_maps_yaml)
        _print_coverage_table(paper_v1, channel_maps)
        print(f"\nLoading linked CSVs to count cells per marker...")
        cells = _build_all_cells(paper_v1, channel_maps, args.ko_cap, args.ntc_cap)
        # Per-marker post-cap cell counts (KO + NTC), sorted by total.
        per_marker = (
            cells.assign(
                _is_ntc=(cells["gene"] == "NTC").astype(int),
            )
            .groupby("viz_channel")
            .agg(
                n_total=("gene", "size"),
                n_ko=("_is_ntc", lambda s: int((s == 0).sum())),
                n_ntc=("_is_ntc", lambda s: int((s == 1).sum())),
                n_exps=("experiment", "nunique"),
            )
            .sort_values("n_total", ascending=False)
        )
        ko_cap_str = "none" if args.ko_cap is None else f"{args.ko_cap}/sgRNA"
        ntc_cap_str = "none" if args.ntc_cap is None else f"{args.ntc_cap}/sgRNA"
        print(f"\nPost-cap cell count per viz_channel "
              f"(KO cap={ko_cap_str}, NTC cap={ntc_cap_str}):")
        print(per_marker.to_string())
        print(f"\nTotal post-cap rows: {len(cells):,}")
        print("\nPer-experiment availability check:")
        for exp in sorted(cells["experiment"].unique()):
            grp = cells[cells["experiment"] == exp]
            n = len(grp)
            op_path = (OPS_FAST_ROOT / exp / "3-assembly" / "feature_extraction"
                       / f"{exp}_cell_features.h5ad")
            chans = sorted(grp["viz_channel"].unique())
            print(f"  {exp}: {n:>7,} rows  OP={'Y' if op_path.exists() else 'N'}  "
                  f"channels={len(chans)}")
        return 0

    # Per-channel SLURM fan-out (or local sequential when --local). The
    # orchestrator runs on the login node — only the linked-CSV reads + the
    # parquet write happen here; everything compute-heavy is fanned out to
    # SLURM nodes via `submit_parallel_jobs`.
    slurm_params = {
        "timeout_min": args.timeout_min,
        "slurm_mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }
    return run_consolidation_all_cells(
        paper_v1_path=paper_v1_path,
        output_dir=args.output_dir,
        cp_match_radius_px=args.cp_match_radius_px,
        ko_cap=args.ko_cap,
        ntc_cap=args.ntc_cap,
        no_cp=args.no_cp,
        channel_maps_yaml=args.channel_maps_yaml,
        zscore_per_experiment=args.zscore_per_experiment,
        skip_existing=args.skip_existing,
        submit_via_slurm=not args.local,
        slurm_params=slurm_params,
        wait=not args.no_wait,
    )


if __name__ == "__main__":
    sys.exit(main())
