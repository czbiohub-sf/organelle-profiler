# Organelle Segmentation Guide

A concise reference for the core methods and tuning parameters in the organelle segmentation pipeline.

---

## Pipeline Overview

```
Raw Image → CLAHE → Frangi/LoG Blob → Threshold → Postprocess → Labels
```

1. **CLAHE** - Enhance local contrast
2. **Detection** - Frangi (tubular/vesicular) or LoG Blob (round objects)
3. **Threshold** - Convert response map to binary mask
4. **Postprocess** - Morphological cleanup + labeling

---

## 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Analogy: Adaptive Photo Editing
Imagine you have a photo where some areas are too dark and others too bright. Instead of adjusting the whole image at once (which might blow out the bright parts), CLAHE works like an editor who divides the image into small tiles and adjusts each tile's brightness separately. The "clip limit" prevents any tile from getting too extreme - like having a volume limiter on a speaker to prevent distortion.

### Theory
CLAHE enhances local contrast by equalizing histograms in small tiles, with a clip limit to prevent noise amplification. Unlike global histogram equalization, it preserves local detail in large images.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_limit` | 0.01 | Limits contrast amplification. Lower = less enhancement, less noise |
| `kernel_size` | (256, 256) | Size of local region for histogram equalization |

### Tuning Tips
- **Low contrast structures**: Increase `clip_limit` (0.02-0.03)
- **Noisy images**: Decrease `clip_limit` (0.005-0.01)
- **Fine structures**: Smaller `kernel_size` (128, 128)
- **Large structures**: Larger `kernel_size` (512, 512)

---

## 2. Frangi Vesselness Filter

### Analogy: Finding Tubes by Feel
Imagine running your fingers across a surface with your eyes closed. A tube (like a blood vessel or mitochondrion) feels different from a blob or flat background:
- **Along the tube**: smooth, no change
- **Across the tube**: steep drop-off on both sides

The Frangi filter does this mathematically - it asks "if I blur slightly and measure how quickly brightness changes in different directions, does it look like a tube?" The **sigma** (blur amount) determines what size tubes you're feeling for - small sigma finds thin tubes, large sigma finds thick ones.

The **beta** parameter is like adjusting your sensitivity: high beta means "only count it if it really feels like a tube" (ignores round blobs), low beta means "round things count too" (for vesicles).

### Theory
The Frangi filter detects tubular/vesicular structures by analyzing the **Hessian matrix eigenvalues** at multiple scales.

**Core idea**: At each pixel, compute second derivatives (Hessian), then examine eigenvalue ratios:
- **Tubular structures**: One small eigenvalue (along tube), one large (across tube)
- **Vesicular/blob structures**: Two similar eigenvalues (isotropic)
- **Background**: Low eigenvalue magnitude overall

The vesselness score combines three terms:
```
V = (1 - exp(-Ra²/2α²)) × exp(-Rb²/2β²) × (1 - exp(-S²/2γ²))
```
- **Ra**: Plate-like suppression (3D only)
- **Rb**: Blob vs tube discrimination (controlled by β)
- **S**: Overall structure magnitude (controlled by γ)

### Multi-Scale Analysis: Detecting Structures of All Sizes

#### Analogy: Trying On Different Glasses
Imagine you're looking for objects of unknown size. Instead of picking one pair of glasses and hoping it works, you try on many pairs (each with a different focal length) and keep the clearest view you got of each object. A small object looks best with one pair; a large object looks best with another.

The Frangi filter does exactly this - it tests multiple sigma values (blur levels) and at each pixel, keeps the **maximum response** across all scales. This is called **"Multi-Scale Filtering with Maximum Response Selection"** or **"Scale-Space Maximum Projection"**.

#### How It Works
```
For each sigma in [sigma_min ... sigma_max]:
    1. Blur the image at this scale
    2. Compute vesselness score at every pixel
    3. If this score > previous best → keep it

Result: Each pixel has the response from its "best-matching" scale
```

#### Why This Matters
- A thin mitochondrion (0.3µm) responds best at small sigma
- A thick mitochondrion (1.2µm) responds best at large sigma
- **Both get detected** because each pixel automatically "chooses" its optimal scale

#### The `num_sigma` Parameter
Controls how many scales are tested between `min_radius_um` and `max_radius_um`:

| num_sigma | Coverage | Speed |
|-----------|----------|-------|
| 3 | Coarse - may miss intermediate sizes | Fast |
| 5 (default) | Good balance | Moderate |
| 10 | Fine - smooth scale coverage | Slower |

Spacing is **logarithmic** (geomspace), which matches how biological structures vary in size.

---

### Adaptive Gamma: Auto-Adjusting Sensitivity

#### Analogy: Auto-Brightness on Your Phone
When you walk from a dark room into sunlight, your phone's auto-brightness adjusts so you can still see the screen. Gamma does something similar for Frangi - it looks at the overall intensity of the blurred image and automatically sets a "what counts as a real structure" threshold.

Without adaptive gamma, the same fixed threshold might work great for a bright image but miss everything in a dim image (or vice versa - detect noise in a dim image that wouldn't register in a bright one).

#### How It Works
For each sigma (blur level), the algorithm:
1. Looks at the Gaussian-smoothed image intensities
2. Computes both Otsu and Triangle thresholds (two automatic thresholding methods)
3. Takes the **minimum** of the two as gamma (more conservative = fewer false positives)
4. Uses this gamma in the `(1 - exp(-S²/2γ²))` term to suppress low-magnitude responses

#### The Effect
- **High gamma** (bright image): Only strong structures pass through
- **Low gamma** (dim image): Weaker structures can still be detected
- **Result**: Consistent detection across images with different overall brightness

#### When Gamma is Fixed vs Adaptive
| Setting | Behavior | Use Case |
|---------|----------|----------|
| `gamma=None` (default) | Adaptive per-sigma | Most cases - handles varying image brightness |
| `gamma=0.3` (fixed) | Same threshold everywhere | When you want exact reproducibility or manual control |

### Key Parameters

| Parameter | Tubular | Vesicular | Description |
|-----------|---------|-----------|-------------|
| `alpha` | 0.5 | 0.5 | Elongation sensitivity (higher = more tubular bias) |
| `beta` | 0.5 | 0.1 | Blob suppression. **High = suppress blobs (tubular), Low = allow blobs (vesicular)** |
| `min_radius_um` | 0.1-0.2 | 0.1-0.2 | Minimum structure size in microns |
| `max_radius_um` | 1.5 | 1.5 | Maximum structure size in microns |
| `threshold` | 0.01 | 0.01 | Fixed threshold on vesselness map |
| `black_ridges` | True | False | Dark structures (True) vs bright structures (False) |

### Structure Types
- **`tubular`**: Mitochondria, ER networks, filaments (β=0.5)
- **`vesicular`**: Bright puncta, lysosomes (β=0.1, black_ridges=False)
- **`vesicular_dark`**: Lipid droplets, vacuoles (β=0.1, black_ridges=True)

### Tuning Tips
- **Missing thin structures**: Decrease `min_radius_um`
- **Too much noise**: Increase `threshold` or `min_radius_um`
- **Fragmented tubes**: Increase `max_radius_um` or decrease `threshold`
- **Detecting blobs instead of tubes**: Increase `beta` (0.5+)
- **Missing round objects**: Decrease `beta` (0.1)

---

## 3. LoG Blob Detection (Laplacian of Gaussian)

### Analogy: Finding Pebbles in Sand
Imagine dragging different-sized ring stamps through sand looking for buried pebbles. When the ring size matches the pebble size, you get a perfect fit - the ring sits right on the pebble's edge. Too small a ring and it sits inside the pebble; too large and it misses entirely.

LoG works the same way: it uses ring-shaped filters of different sizes (controlled by sigma). When the ring matches a blob's size, you get a strong "hit." By trying many ring sizes, you can find blobs of various sizes in one pass.

### Theory
LoG finds blob-like structures by convolving with Gaussian-smoothed Laplacian at multiple scales. A blob produces a **local maximum** in scale-space when σ ≈ radius/√2.

**Core idea**: The Laplacian highlights regions of rapid intensity change. Gaussian smoothing selects the scale. Maximum response occurs when blob size matches filter scale.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_radius_um` | 0.2 | Minimum blob radius in microns |
| `max_radius_um` | 0.4-1.5 | Maximum blob radius (larger for fluorescent) |
| `num_sigma` | 4-8 | Number of scales to test (more = finer detection) |
| `threshold` | 0.03-0.06 | Detection sensitivity (lower = more blobs) |
| `overlap` | 0.3 | Maximum allowed blob overlap (0-1) |

### Use Cases
- **Vesicles**: Small, round (0.2-0.4 µm)
- **Nucleoli**: Larger, round (0.5-3.0 µm)
- **Lysosomes**: Medium puncta (0.2-1.5 µm fluorescent)

### Tuning Tips
- **Missing small objects**: Decrease `min_radius_um`, lower `threshold`
- **False positives (noise)**: Increase `threshold`, increase `min_radius_um`
- **Merged blobs**: Decrease `overlap`
- **Variable sizes**: Increase `num_sigma` and widen radius range

---

## 4. Quick Parameter Reference

### By Channel Type

| Channel | CLAHE clip | Frangi threshold | Notes |
|---------|------------|------------------|-------|
| Phase2D | 0.01 | 0.01 | Label-free, use larger kernel |
| Focus3D | 0.01 | 0.01 | Similar to Phase2D |
| GFP | 0.01 | 0.01 | Good SNR typically |
| mCherry | 0.01 | 0.015 | Higher background, raise threshold |
| Cy5 | 0.01 | 0.008 | Weaker signal, lower threshold |

### By Organelle

| Organelle | Method | Structure Type | Key Params |
|-----------|--------|----------------|------------|
| Mitochondria | Frangi | tubular | β=0.5, 0.1-1.5 µm |
| ER network | Frangi | tubular | β=0.5, 0.1-1.5 µm |
| Lysosomes | Blob | vesicular | 0.2-0.6 µm, thresh=0.03 |
| Lipid droplets | Blob | vesicular_dark | black_ridges=True |
| Nucleoli | Frangi/Blob | vesicular | 0.5-3.0 µm, needs nuclear mask |
| Peroxisomes | Blob | vesicular | 0.2-0.6 µm |

---

## 5. The Power of `pixel_size_um`

### Analogy: Adjusting Your Glasses Prescription
Think of `pixel_size_um` like adjusting the strength of reading glasses:
- **Higher pixel_size_um** = stronger prescription = you only see the sharpest, most well-defined text (crisp structure cores)
- **Lower pixel_size_um** = weaker prescription = you see blurrier text too (faint halos and diffuse edges)

It's not about the actual microscope resolution - it's telling the algorithm "pretend the pixels are this big" which changes how much blur (sigma) gets applied before looking for structures.

### Why It Matters

`pixel_size_um` is one of the most powerful yet underappreciated parameters. It controls **how much of the structure gets detected**, especially fainter, less-defined regions.

### The Mechanism

All detection methods convert physical sizes (µm) to pixel units using `pixel_size_um`:

```
sigma_pixels = radius_um / pixel_size_um
```

**Larger `pixel_size_um`** → **Smaller sigma in pixels** → **Tighter, more selective detection**

**Smaller `pixel_size_um`** → **Larger sigma in pixels** → **Broader, more inclusive detection**

### Sigma Range by pixel_size_um

For a fixed radius range of 0.1-1.5 µm:

| pixel_size_um | sigma range (pixels) | Effect |
|---------------|---------------------|--------|
| 0.1625 | 0.6 - 9.2 px | Large blur → detects broad, diffuse structures |
| 0.325 | 0.3 - 4.6 px | Medium blur → balanced detection |
| 0.65 | 0.15 - 2.3 px | Small blur → only sharp, high-contrast edges |

### Visual Intuition

```
Real structure:      ████████████████████  (bright core + faint halo)
                     ▓▓▓▓████████████▓▓▓▓

Large pixel_size_um: ....████████████....  (only sharp, bright core)
Small pixel_size_um: ▓▓▓▓████████████▓▓▓▓  (includes faint periphery)
```

### Why This Works

1. **Frangi Filter**: Sigma controls the Gaussian smoothing scale before Hessian computation. Smaller sigma = sharper gradients = only high-contrast edges. Larger sigma = smoother gradients = detects broader, lower-contrast structures.

2. **LoG Blob**: Sigma determines which blob sizes produce maximum response. Smaller sigma = only compact, well-defined blobs. Larger sigma = includes diffuse, irregular blobs.

3. **CLAHE**: While not directly affected, the downstream filters see different effective structure sizes after contrast enhancement.

### Practical Examples

| Scenario | Adjust `pixel_size_um` | Effect |
|----------|------------------------|--------|
| Only want bright, sharp mitochondria | Increase (e.g., 0.325 → 0.5) | Excludes faint/diffuse regions |
| Want full ER network including dim tubules | Decrease (e.g., 0.185 → 0.1) | Includes low-intensity structures |
| Nucleoli detection too aggressive | Increase (e.g., 0.65 → 1.0) | Only bright, well-defined nucleoli |
| Missing peripheral vesicles | Decrease (e.g., 0.185 → 0.12) | Captures fainter puncta |

### Default Values by Channel

| Channel Type | Default `pixel_size_um` | Rationale |
|--------------|------------------------|-----------|
| Label-free (Phase2D) | 0.325 | More selective for phase contrast |
| Fluorescent (GFP, mCherry) | 0.185 | Broader to capture full signal |
| Nucleoli | 0.65 | Very selective for bright nucleoli only |

### Key Insight

**`pixel_size_um` acts as a "structure completeness" dial:**
- **Higher values** = conservative, high-confidence detections (just the core)
- **Lower values** = inclusive detections (core + periphery)

This is often more intuitive than adjusting `threshold` because:
- Threshold cuts off by intensity value (absolute)
- `pixel_size_um` controls what scale of features are even considered (relative to structure shape)

### When to Use Instead of Threshold

| Goal | Use `pixel_size_um` | Use `threshold` |
|------|---------------------|-----------------|
| Include faint but well-shaped structures | Decrease pixel_size | - |
| Exclude noise (random bright pixels) | - | Increase threshold |
| Get only the brightest part of structures | Increase pixel_size | Increase threshold |
| Include diffuse halos around structures | Decrease pixel_size | - |

---

## 6. Common Issues & Solutions

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Over-segmentation | Threshold too low | Increase `threshold` |
| Under-segmentation | Threshold too high | Decrease `threshold` |
| Missing fine structures | `min_radius_um` too large | Decrease `min_radius_um` |
| Noisy detections | `clip_limit` too high | Decrease CLAHE `clip_limit` |
| Fragmented objects | Scale range too narrow | Widen `min/max_radius_um` |
| Wrong structure type | Incorrect `beta` | Tubular: β=0.5, Vesicular: β=0.1 |

---

## 7. Usage Examples

### Single Position Segmentation
```python
from organelle_profiler.organelle_seg.organelle_segmentation import (
    segment_single_position_channel
)

# Tubular structures (mitochondria, ER)
segment_single_position_channel(
    experiment='ops0049_20250626',
    position='A/3/0',
    channel_key='GFP',
    structure_type='tubular',
    use_clahe=True,
)

# Vesicular structures (lysosomes)
segment_single_position_channel(
    experiment='ops0049_20250626',
    position='A/3/0',
    channel_key='mCherry',
    structure_type='vesicular',
    use_clahe=True,
)
```

### SLURM Batch Processing
```bash
# All positions, auto-detect structure types
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0049_20250626

# Specific channels and structure types
python -m organelle_profiler.organelle_seg.organelle_segmentation_slurm \
    --experiment ops0049_20250626 \
    --channels GFP mCherry \
    --structure-types tubular vesicular
```

---

## 8. Config Override (ops_channel_maps.yaml)

Override defaults per-experiment in `ops_channel_maps.yaml` — versioned at
`cyclops_process/cyclops_process/configs/ops_channel_maps.yaml` (from the monorepo root), read
at runtime from `$OPS_CONFIGS_DIR` (default `$OPS_BASE_PATH/configs`):

```yaml
ops0049:
  - channel_name: GFP
    label: mitochondria, TOMM20
    segmentation_config:
      structure_type: tubular
      frangi:
        threshold: 0.015
        min_radius_um: 0.15
      clahe:
        clip_limit: 0.02
```
