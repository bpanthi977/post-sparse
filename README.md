# PostSparse

Post-hoc sparse multimodal alignment via high-dimensional contrastive projection on frozen CLIP encoders.

## Documentation

| File | Description |
|------|-------------|
| [PROPOSAL.md](docs/PROPOSAL.md) | Research proposal: motivation, method, and distinction from prior work |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model architecture and loss function design |
| [DATASET.md](docs/DATASET.md) | Dataset choices, download instructions, and embedding caching strategy |
| [TRAINING.md](docs/TRAINING.md) | Optimizer, schedule, batch size, and early stopping criterion |
| [src.md](docs/src.md) | How each file inside src/ is implemented |

## Usage

```bash
# Install dependencies
uv sync

# Download the dataset (one-time)
sh scripts/download_dataset.sh

# Pre-compute CLIP embeddings (one-time, requires COCO images in data/coco/)
uv run python scripts/cache_embeddings.py config/base.yaml

# Train
uv run python src/train.py config/base.yaml
```

## Code

| File | Role |
|------|------|
| `src/model.py` | `ProjectionHead`, `PostSparseModel` |
| `src/data.py` | `CachedPairsDataset`, `cache_embeddings` |
| `src/loss.py` | `infonce_loss` |
| `src/eval.py` | `coco_retrieval_metrics`, `sparsity_metrics` |
| `src/train.py` | Training loop entry point |
| `src/utils.py` | Run folder creation, hparams saving |
| `scripts/cache_embeddings.py` | One-time CLIP embedding precomputation |
| `config/base.yaml` | All hyperparameters |
