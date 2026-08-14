# organelle_profiler

Organelle segmentation, geometric feature extraction, and clustering of Optical Pooled Screening (OPS) data.

This repository accompanies the following preprint

> [A multimodal perturbation atlas defines the phenotypic resolution of cellular morphology](https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1.abstract) — bioRxiv, 2026. doi:10.64898/2026.06.01.728087

## Data availability

The processed image datasets that these pipelines ingest are available for download through the Biohub OPS Explorer portal:

> [OPS Explorer — perturbation atlas collection](https://biohub.ai/ops-explorer?collection=6a3f8b91-1c5e-4d3a-9b4c-f7e0a2d8b6f3)

`organelle_profiler` measures cells: it segments subcellular structures and extracts an interpretable, named feature per structure — area, elongation, branch length, distance from the nucleus — rather than a latent vector.

The pipeline has three stages:

1. **Organelle segmentation** — assembled OPS images → per-organelle label masks written back into the zarr store.
2. **Feature extraction** — label masks → morphology, intensity, texture, network and localization features, aggregated to cell / guide / gene AnnData.
3. **Consolidation** — pool per-experiment features into cross-experiment, per-marker AnnData for downstream analysis.

---

## Installation

`organelle_profiler` is **not a standalone package.** It is one submodule of the
[`czbiohub-sf/cyclops-monorepo`](https://github.com/czbiohub-sf/cyclops-monorepo) uv workspace and is only
supported as a piece of that larger project. Install the monorepo, not this repo on its own:

```bash
git clone --recurse-submodules git@github.com:czbiohub-sf/cyclops-monorepo.git
cd cyclops-monorepo
uv sync
```

All commands are then run from the **monorepo root** (`cyclops-monorepo/`) with `uv run`, as in every
example below — never from inside a subpackage directory, never with a manually activated
conda/venv, and never with bare `python` or `pytest`. See the
[monorepo README](https://github.com/czbiohub-sf/cyclops-monorepo#getting-started) for cluster setup
(`module load uv`, `UV_CACHE_DIR`).

Every storage root in the package derives from `$OPS_BASE_PATH`
([`src/organelle_profiler/paths.py`](src/organelle_profiler/paths.py)), which has **no default**
— as in the sibling packages, importing raises a `RuntimeError` if it is unset, so a
misconfigured run cannot read or write somebody else's storage:

```bash
export OPS_BASE_PATH="/path/to/ops_data"   # required
```

### Dependencies

Declared dependencies and the optional extras (`interactive`, `rapids`, `all`) live in
[`pyproject.toml`](pyproject.toml); Python 3.12 is required. The `rapids` extra (cuCIM / cuML)
enables the GPU morphology and network paths; without it the pipeline still runs, on CPU. Exact
resolved versions for the whole workspace are pinned in the monorepo's `uv.lock`, which is the
authoritative environment specification.

---

## 1. Organelle segmentation

`organelle_seg/` turns each assembled phenotyping image into per-organelle label masks. Every channel goes through the same four steps — CLAHE contrast enhancement, a Frangi vesselness or LoG blob detector, thresholding, then morphological cleanup — with the detector and its parameters chosen per structure type: `tubular` (mitochondria, ER), `vesicular` (lysosomes, puncta), `vesicular_dark` (lipid droplets), or a nuclear-mask-gated nucleoli mode. Frangi channels are submitted twice, once tubular and once vesicular, producing two labels per channel (e.g. `phase_2d_tubular_seg`, `phase_2d_vesicular_seg`). Labels are written back into the experiment's zarr store, so segmentation output *is* the feature-extraction input.

The SLURM entry point submits one job per (position, channel) pair:

```bash
# What channels does this experiment have?
uv run python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0049_20250626 --list-channels

# Submit every position × channel
uv run python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0049_20250626

# Restrict to specific positions / channels
uv run python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0049_20250626 --positions A/1/0 A/2/0 --channels GFP_vesicular
```

Experiments accept shorthand (`-e 94` resolves to the full `ops0094_*` name). `--all` sweeps every experiment that still needs segmentation, `--force` reprocesses existing outputs, and `--dry-run` prints the job list without submitting.

Before committing a parameter set, preview it **locally** on a 2×2 tile crop — this exercises both segmentation and tile stitching, and writes debug panels (raw input, vesselness map, binary mask, labels, overlay):

```bash
uv run python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    -e 33 --positions A/1/0 --channels Phase2D --structure-type tubular --preview

# All segmentation types for one position, side by side
uv run python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    -e 94 --positions A/1/0 --preview-all
```

Per-experiment parameter overrides live in `ops_channel_maps.yaml`, keyed by ops number and channel. Its versioned copy is `cyclops_process/cyclops_process/configs/ops_channel_maps.yaml` (from the monorepo root); at runtime it is read from `$OPS_CONFIGS_DIR`, defaulting to `$OPS_BASE_PATH/configs`. See [`ORGANELLE_SEGMENTATION_GUIDE.md`](src/organelle_profiler/organelle_seg/ORGANELLE_SEGMENTATION_GUIDE.md) for the theory behind each filter, the full parameter tables, per-organelle presets, and a tuning troubleshooter — including why `pixel_size_um`, not `threshold`, is usually the dial you want.

---

## 2. Feature extraction

`feature_extraction/` reads the segmentation labels plus the intensity channels and measures every object in every cell. Four families of features are extracted:

- **Morphology & intensity** — regionprops shape descriptors, Hu moments, intensity distribution statistics.
- **Texture** — Haralick / GLCM and Zernike moments (`--full-features` only; expensive).
- **Network topology** — skeleton-graph branch length, thickness, tortuosity, junction and endpoint counts, connectivity density.
- **Subcellular localization** — KDTree distances from each object to the nuclear boundary, cell edge, and nucleus centroid, plus normalized radial position.

Everything is aggregated (sum / mean / median / std / min / max) from object → cell → guide → gene, producing three AnnData files per experiment in its `feature_extraction/` output directory: `{experiment}_cell_features.h5ad`, `_guide_features.h5ad`, `_gene_features.h5ad`. [`feature_extraction_parameters.md`](src/organelle_profiler/feature_extraction/feature_extraction_parameters.md) is the authoritative feature list and marks which features are gated behind `--full-features`.

### Single experiment — `feature_extraction_slurm.py`

Work is split into per-well batches of ~3500 cells (~1 hr jobs, independently retryable) and run as four phases: **gpu** (morphology + localization on GPU nodes) → **spmd** (network analysis, SPMD across cheap CPU nodes) → **merge** (combine partials) → **aggregate** (write AnnData).

```bash
# Preview: 5 batches from different wells, end to end
uv run python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --preview

# Full experiment
uv run python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e ops0094

# Resume a phase (runs that phase and everything downstream)
uv run python -m organelle_profiler.feature_extraction.feature_extraction_slurm -e 94 --resume-from spmd
```

`--dry-run`, `--wells`, `--checkpoint` (skip completed batches), `--force`, `--check-complete`, `--aggregate-only`, `--cells-per-batch` and `--escalate` (gpu → cpu → preempted partitions) cover the rest of the operational surface. Running the module locally without SLURM — `python -m organelle_profiler.feature_extraction.feature_extraction -e 94` — is supported for debugging, with `--dry-run` to list the segmentations it discovered.

### Many experiments — `run_fe_all_experiments.py`

The wave orchestrator runs the same four phases as *cohorts*: all N experiments' GPU arrays queue together, then all SPMD arrays, and so on, so progress is reported per wave rather than per experiment. Any failure hard-stops the batch and writes `_wave_failures.csv` plus a copy-pasteable rerun command.

```bash
uv run python -m organelle_profiler.feature_extraction.run_fe_all_experiments \
    --paper-v1 \
    --output-base $OPS_BASE_PATH/analysis/fe_paper_v1 \
    --run-name fe_paper_v1_2026apr
```

`--paper-v1` resolves the curated 77-experiment set; `--experiments` takes an explicit list (shorthand allowed) and `--exclude-experiments` subtracts from either. `--modality {phase,fluorescent,all}` and `--organelles` filter what gets measured, and `--resume-from` / `--stop-after` / `--clean` retry a single phase for the failed subset. With `--output-base` unset, each experiment writes to its canonical `<exp>/3-assembly/feature_extraction/` rather than a side-by-side run directory.

---

## 3. Consolidation

Per-experiment AnnData is useful for QC; cross-experiment analysis needs one object per marker. `consolidate_all_cells.py` pools every linked cell (valid segmentation + gene/sgRNA call) across the paper_v1 experiment set, attaches both OrganelleProfiler and CellProfiler features, z-scores per experiment so cells can be pooled without batch effects, and writes **one h5ad per visualization channel** (Phase plus ~56 fluorescent markers, including 4i and Cell Painting):

```bash
# Counts only — no attach
uv run python -m organelle_profiler.feature_extraction.consolidate_all_cells --paper-v1 --dry-run

# SLURM (default; --local runs in-process)
uv run python -m organelle_profiler.feature_extraction.consolidate_all_cells --paper-v1
```

Uncapped paper_v1 is roughly 70M (cell, channel) rows — use `--ko-cap` / `--ntc-cap` to subsample per sgRNA when the full distribution isn't needed.

`consolidate_top_attention_cells.py` is the narrower sibling: it takes top-attention cell CSVs from the attention-model analysis and emits two objects, OrganelleProfiler features alone and OP + CellProfiler concatenated, matching CP cells by spatial nearest neighbor within each well.

---

## Ownership and maintenance

**This repository is the result of work done at [biohub San Francisco](https://github.com/czbiohub-sf).**

This repository is owned by the [Leonetti group](https://biohub.org/leonetti/) at [biohub San Francisco](https://github.com/czbiohub-sf).

Maintainers (see also [`.github/CODEOWNERS`](.github/CODEOWNERS)):

- Gav Sturm ([@gav-sturm](https://github.com/gav-sturm))
- Alexander Hillsley ([@ahillsley](https://github.com/ahillsley))
- Mark Potts

Please open an issue or pull request for questions, bugs, or contributions.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
