# PostSparse Training Setup

## Optimizer & Schedule

- **Optimizer**: AdamW, weight decay 0.01
- **Learning rate**: 1e-3
- **Schedule**: Cosine decay with warmup (~5% of total steps)
- **Batch size**: 512–2048 (larger batches improve InfoNCE quality; use gradient accumulation if needed)
- **Epochs**: 10-30 based on early stopping criterion

## Stopping Criterion

Training uses **early stopping** based on the COCO val R@1 (image→text) score — halt when it hasn't improved for 3 consecutive epochs, and keep the best checkpoint.

### Validation Retrieval Metric (R@1, R@5, R@10)

After each epoch, encode the full COCO val set (5K images, ~25K captions) in the sparse space and measure retrieval quality:

- **Image→Text**: for each image, rank all captions by cosine similarity — what fraction of images find their correct caption in the top 1 / 5 / 10?
- **Text→Image**: reverse direction — for each caption, find the matching image from the 5K pool.

This gives six numbers: R@1/5/10 in both directions. **COCO val image→text R@1** is used as the single early-stopping signal (most sensitive to alignment quality). All six are logged per epoch.

## Run Logging

Before training starts, a run folder is created under `logs/` with the naming pattern:

```
yymmdd-hhmmss-<6 random lowercase letters>
```

Example: `250520-143201-abkrxf`

Each run folder contains:

```
logs/
  250520-143201-abkrxf/
    hparams.yaml      ← all hyperparameters saved before training begins
    train_loss.csv    ← training loss logged per step or epoch
    val_metrics.csv   ← val R@1/5/10 (both directions) logged per epoch
    best.pt           ← model checkpoint with highest COCO val image→text R@1
    last.pt           ← model checkpoint at end of training
```

All losses, metrics, and hyperparameters are also logged to **Weights & Biases** under the project `post-sparse`. The run name matches the folder name (`yymmdd-hhmmss-xxxxxx`).
