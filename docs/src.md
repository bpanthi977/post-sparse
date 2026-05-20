## 1. File Structure

```
post-sparse/
  pyproject.toml
  config/
	base.yaml
  scripts/
	cache_embeddings.py
  src/
	model.py
	data.py
	loss.py
	eval.py
	train.py
	utils.py
```

## 2. `config/base.yaml`

Single source of truth for all hyperparameters — saved verbatim to every run folder.

```yaml
model:
  clip_model: "ViT-B-32"
  clip_pretrained: "openai"
  proj_dim: 16384

training:
  batch_size: 512
  lr: 1.0e-3
  weight_decay: 0.01
  warmup_fraction: 0.05
  patience: 3
  grad_accum_steps: 1

data:
  data_dir: "data/coco"
  embeddings_dir: "data/coco/embeddings"
  num_workers: 8

logging:
  log_dir: "logs"
  wandb_project: "post-sparse"
```

---

## 3. `src/model.py`

Two classes: `ProjectionHead` and `PostSparseModel`.

```python
class ProjectionHead(nn.Module):
	# Linear(in_dim, out_dim, bias=False) → ReLU → L2-normalize
	# forward(x) → normalized sparse vector

class PostSparseModel(nn.Module):
	# image_head: ProjectionHead
	# text_head:  ProjectionHead
	# log_tau:    nn.Parameter(log(0.07)), exp().clamp(max=100)
	# forward(img_emb, txt_emb) → (z_i, z_t)
	# property tau → scalar temperature
```

---

## 4. `src/data.py`

Two responsibilities: caching embeddings and serving them during training.

### `CachedPairsDataset`
- Loads `train_image_embeddings.pt` [N, 512] and `train_text_embeddings.pt` [N, 5, 512]
- Flattens to N×5 flat pairs: image embedding repeated 5× to match each caption
- `__getitem__` returns `(img_emb, txt_emb)` — one pair per index
- Length = N × 5 (~591K for COCO train)

### `cache_embeddings(data_dir, save_dir, ...)`
- Loads COCO via `torchvision.datasets.CocoCaptions` for both splits
- Encodes images with `model.encode_image`, encodes all 5 captions per image with `model.encode_text`
- CocoCaptions returns `(image_tensor, [cap1, cap2, cap3, cap4, cap5])` per item; a custom `collate_fn` is needed to transpose the caption lists into 5 per-caption batches for efficient encoding
- Saves:
  - `{split}_image_embeddings.pt` — shape [N, 512]
  - `{split}_text_embeddings.pt` — shape [N, 5, 512]

---

## 5. `src/loss.py`

```python
def infonce_loss(z_i, z_t, tau):
	# sim[i,j] = cosine similarity (z_i, z_t already L2-normalized)
	sim = z_i @ z_t.T / tau          # [B, B]
	labels = torch.arange(B)
	return (CE(sim, labels) + CE(sim.T, labels)) / 2
```

---

## 6. `src/eval.py`

```python
def coco_retrieval_metrics(img_embs, txt_embs, ks=(1, 5, 10)):
	# img_embs: [N, D] (L2-normalized)
	# txt_embs: [N, 5, D] (L2-normalized)
	# Returns dict: i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
```

**Image→Text**: sim matrix [N, N×5]. For image i, correct texts are columns [5i … 5i+4]. A retrieval counts correct if the rank of the first correct text ≤ k.

**Text→Image**: sim matrix [N×5, N]. For text j, correct image is j//5. Rank of correct image ≤ k counts as correct.

Both directions done via `torch.argsort` on the full similarity matrix (5K × 25K = 500MB fp32 — acceptable for val).

Also exports:
```python
def sparsity_metrics(z_i, z_t):
	# Returns mean L0 (fraction of active dims) for image and text heads
	# l0 = (z > 0).float().mean()
```

---

## 7. `src/utils.py`

```python
def create_run_dir(log_dir) -> Path:
	# name = f"{datetime:%y%m%d-%H%M%S}-{6 random lowercase letters}"
	# creates logs/<name>/ and returns Path

def save_hparams(config: dict, run_dir: Path):
	# dumps config to run_dir/hparams.yaml
```

---

## 8. `src/train.py`

Entry point: `python src/train.py config/base.yaml`

### Flow

1. Load config from YAML path (sys.argv[1])
2. `run_dir = create_run_dir(...)` — create timestamped folder
3. `save_hparams(config, run_dir)`
4. `wandb.init(project="post-sparse", name=run_dir.name, config=config)`
5. Check if embeddings are cached; if not, throws an error with message to run `scripts/cache_embeddings.py`
6. Load train embeddings → `CachedPairsDataset` → `DataLoader(shuffle=True)`
7. Load val embeddings onto device (kept resident for fast per-epoch eval)
8. Instantiate `PostSparseModel`, move to device
9. `AdamW(model.parameters(), lr, weight_decay)`
10. `LambdaLR` cosine-with-warmup scheduler (warmup = 5% of total steps; total steps = epochs_max × steps_per_epoch)
11. **Training loop** (max 30 epochs):
	- Forward → `infonce_loss` → backward → optimizer/scheduler step
	- Log `train/loss` and `train/tau` to wandb per step
	- End of epoch: encode val set through projection heads, call `coco_retrieval_metrics` and `sparsity_metrics`
	- Log all val metrics to wandb
	- If `i2t_r1` improved → save `best.pt`, reset patience counter
	- Else → increment patience counter; break if ≥ patience
12. Save `last.pt`
13. `wandb.finish()`

### Gradient accumulation
If `grad_accum_steps > 1`: divide loss by `grad_accum_steps`, call `optimizer.step()` only every N micro-steps.

---

## 9. `scripts/cache_embeddings.py`

Standalone script: `uv run python scripts/cache_embeddings.py config/base.yaml`

Calls `cache_embeddings()` from `src/data.py` for both train and val splits. Prints embedding shapes on completion. Idempotent — skips if files already exist.

---

## 10. `scripts/modality_score_plot.py`

Standalone script: `uv run python scripts/modality_score_plot.py <run_dir> [--base-embedding]`

Computes and plots the modality score distribution of sparse features — the primary diagnostic for whether learned features are genuinely multimodal. Uses the val set (first caption per image only).

### Functions

```python
activations_from_heads(model, val_img, val_txt, device) -> (Z_I, Z_T)
# Passes cached val embeddings through trained projection heads.
# Returns raw ReLU activations [N, k] without L2 normalisation,
# so magnitudes are preserved for thresholding.

activations_from_base_embeddings(val_img, val_txt) -> (Z_I, Z_T)
# Sanity-check mode: applies torch.sign to the raw CLIP embeddings [N, 512].
# A dimension is "active" when sgn = +1 (> TAU = 0.001).
# Expected result: scores concentrated near 0.5, since CLIP embeddings
# are already cross-modally aligned.

compute_modality_scores(Z_I, Z_T) -> (mod_score, n_total, n_dead, n_alive)
# For each feature j: ModScore(j) = img_activations(j) / total_activations(j).
# Dead features (never active above TAU in either modality) are excluded.
# Returns mod_score [n_alive] and counts.

build_stats(mod_score, n_total, n_dead, n_alive) -> list[str]
# Formats summary statistics: dead/active feature counts,
# % multimodal (0.4–0.6), % unimodal image (>0.9), % unimodal text (<0.1).

plot_modality_scores(mod_score, title, fig_path)
# 1D histogram of modality scores (50 bins, range [0, 1]).
# Red dashed line at 0.5 marks the perfectly multimodal point.
```

### Outputs (saved inside `<run_dir>`)

| File | Description |
|------|-------------|
| `figs/modality_score.png` | Modality score histogram (trained heads) |
| `figs/modality_score_base.png` | Modality score histogram (`--base-embedding`) |
| `modality_score.txt` | Summary statistics (trained heads) |
| `modality_score_base.txt` | Summary statistics (`--base-embedding`) |

---
