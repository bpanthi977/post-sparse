# Computing and Plotting the Modality Score Distribution

## Overview

This document describes the procedure to compute and plot the **modality score distribution** of sparse features — a 2D histogram showing each feature's cross-modal balance (modality score) against its activation frequency (activation density). This plot is the primary diagnostic for verifying that learned sparse features are genuinely multimodal rather than unimodal.

---

## Inputs Required

- A run directory (e.g. `logs/250520-143201-abkrxf/`) containing:
  - `best.pt` — the trained model checkpoint with image projection head $P_I$ and text projection head $P_T$
- Pre-cached validation embeddings at `data/coco/embeddings/`:
  - `val_image_embeddings.pt` — shape `[N, 512]`
  - `val_text_embeddings.pt` — shape `[N, 5, 512]`

---

## Step 1: Extract Sparse Activations

Load `best.pt` and restore the `PostSparseModel`. Load the cached val embeddings. Use only the **first caption** per image from `val_text_embeddings.pt` (index 0 along the caption axis), giving a text embedding matrix of shape `[N, 512]`.

Pass the image embeddings through the image projection head and ReLU (without L2 normalization), producing $Z_I \in \mathbb{R}^{N \times k}$. Repeat for the text embeddings using the text projection head and ReLU, producing $Z_T \in \mathbb{R}^{N \times k}$. All values in both matrices are non-negative due to ReLU. Each column $j$ corresponds to one sparse feature; each row corresponds to one sample.

---

## Step 2: Compute Modality Score per Feature

For each feature $j \in \{1, \ldots, k\}$, the modality score is defined as the fraction of activations that come from images versus the total activations across both modalities:

$$\text{ModScore}(j) = \frac{\sum_{i=1}^{N} \mathbf{1}[Z_I^{ij} > \tau]}{\sum_{i=1}^{N} \mathbf{1}[Z_I^{ij} > \tau] + \sum_{i=1}^{N} \mathbf{1}[Z_T^{ij} > \tau]}$$

where $\tau = 0.001$ is a small activation threshold used to determine whether a feature is active for a given sample. The modality score lies in $[0, 1]$:

- Score near **1.0** → feature activates almost exclusively for images (unimodal visual)
- Score near **0.0** → feature activates almost exclusively for text (unimodal textual)
- Score near **0.5** → feature activates equally for both modalities (genuinely multimodal)

**Dead features** — those that never activate above $\tau$ for either modality across all $N$ samples — are excluded from all subsequent steps.

---

## Step 3: Compute Activation Density per Feature

For each feature $j$, the activation density captures both how frequently and how strongly it activates, averaged across both modalities and all samples:

$$\text{Density}(j) = \frac{1}{2N} \sum_{i=1}^{N} \left( Z_I^{ij} + Z_T^{ij} \right)$$

This is a magnitude-weighted frequency: a feature that activates rarely but with high magnitude contributes similarly to one that activates often with low magnitude.

---

## Step 4: Plot the 2D Histogram

Plot a 2D histogram where:

- **X-axis:** Modality score for each alive feature, ranging from 0 to 1
- **Y-axis:** Activation density on a log scale, given the wide dynamic range across features
- **Each bin** is colored by the number of features that fall within it (log-scaled color is recommended)

Use approximately 50 bins along each axis. Add a vertical reference line at modality score = 0.5 to mark the perfectly multimodal point.

Save the figure to `<run_dir>/figs/modality_score.png`.

---

## Step 5: Report Summary Statistics

Save the following summary statistics to `<run_dir>/modality_score.txt`:

| Metric | Description |
|---|---|
| Total features | $k$ — the full projection dimension |
| Dead features | Features with zero activations above $\tau$ for both modalities |
| Active features | Total features minus dead features |
| % Multimodal | Fraction of active features with modality score in $(0.4, 0.6)$ |
| % Unimodal image | Fraction of active features with modality score $> 0.9$ |
| % Unimodal text | Fraction of active features with modality score $< 0.1$ |

---

## What to Look For

The shape of the distribution is the key result:

- **Post-hoc SAEs** produce a **bimodal distribution** — two peaks near 0.0 and 1.0 — indicating most features are unimodal, activating for only one modality
- **post-sparse** should produce a distribution **concentrated around 0.5**, indicating most features are genuinely multimodal
