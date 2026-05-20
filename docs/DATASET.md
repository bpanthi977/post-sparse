# PostSparse Datasets

## Training Progression

| Phase | Dataset  | Size         | Purpose                                      |
|-------|----------|--------------|----------------------------------------------|
| 1     | MS-COCO  | ~118K images, 5 captions/image | Rapid prototyping, sanity checks, hyperparameter search |

## MS-COCO

- **Source**: `torchvision.datasets.CocoCaptions` (downloads automatically)
- **Train split**: ~118K images × 5 captions = ~591K pairs
- **Val split**: ~5K images (use for retrieval evaluation throughout training)
- **Use**: Fast iteration — the small size allows frequent full-pass evaluation and quick debugging of mechanics

```python
from torchvision.datasets import CocoCaptions
train_dataset = CocoCaptions(
	root="data/coco/train2017",
	annFile="data/coco/annotations/captions_train2017.json",
	transform=preprocess,  # open_clip preprocess
)
val_dataset = CocoCaptions(
	root="data/coco/val2017",
	annFile="data/coco/annotations/captions_val2017.json",
	transform=preprocess,
)
```

Images and annotations are downloaded from [cocodataset.org](https://cocodataset.org/#download): `train2017.zip` (~18GB) and `annotations_trainval2017.zip` (~241MB).

### CLIP Preprocessing
- **Images**: Resize + center-crop to 224×224, normalize with CLIP mean/std
- **Text**: Tokenize with CLIP tokenizer (max 77 tokens)

### Embedding Caching

Since base encoders are frozen, pre-compute and cache CLIP embeddings to disk before training. This avoids re-running the (heavy) CLIP encoder every step:

```
data/
  {dataset}/
	embeddings/
	  train_image_embeddings.pt   # shape [N, 512]
	  train_text_embeddings.pt    # shape [N, 512]
	  val_image_embeddings.pt
	  val_text_embeddings.pt
```

Training then operates entirely on cached embeddings — forward/backward only through the projection heads.

## Evaluation

- **Metric**: Zero-shot retrieval R@1, R@5, R@10 on COCO val set (image→text and text→image)
- **Sparsity**: Mean L0 norm (fraction of active dimensions) of `z_I` and `z_T`
- **Alignment**: Mean cosine similarity of positive pairs in sparse space vs. in CLIP embedding space
