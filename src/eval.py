from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def coco_retrieval_metrics(
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    img_embs: [N, D] L2-normalized image embeddings
    txt_embs: [N, 5, D] L2-normalized text embeddings (5 captions per image)
    """
    N = img_embs.shape[0]
    flat_txt = txt_embs.reshape(N * 5, -1)  # [N*5, D]

    sim_i2t = img_embs @ flat_txt.T   # [N, N*5]
    sim_t2i = flat_txt @ img_embs.T   # [N*5, N]

    # Image→Text: rank of the highest-scoring correct text for each image.
    # Correct texts for image i are at indices [5i, 5i+1, ..., 5i+4].
    correct_txt_idxs = torch.arange(N).unsqueeze(1) * 5 + torch.arange(5)  # [N, 5]
    best_correct_sim = sim_i2t[torch.arange(N).unsqueeze(1).expand(N, 5), correct_txt_idxs].max(dim=1).values
    i2t_rank = (sim_i2t > best_correct_sim.unsqueeze(1)).sum(dim=1)  # [N]

    # Text→Image: rank of the correct image for each caption.
    # Correct image for text j is j // 5.
    correct_imgs = torch.arange(N * 5) // 5  # [N*5]
    correct_sim = sim_t2i[torch.arange(N * 5), correct_imgs]  # [N*5]
    t2i_rank = (sim_t2i > correct_sim.unsqueeze(1)).sum(dim=1)  # [N*5]

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = (i2t_rank < k).float().mean().item() * 100
        metrics[f"t2i_r{k}"] = (t2i_rank < k).float().mean().item() * 100
    return metrics


def sparsity_metrics(z_i: torch.Tensor, z_t: torch.Tensor) -> Dict[str, float]:
    """Mean fraction of active (positive) dimensions per modality."""
    return {
        "l0_image": (z_i > 0).float().mean().item(),
        "l0_text": (z_t > 0).float().mean().item(),
    }


def compute_validation_metrics(
    model,
    val_dataset,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, float]:
    if len(val_dataset) == 0:
        # Fallback for debugging case
        metrics = {
            'l0_image': 0.0,
            'l0_text': 0.0,
        }
        for k in [1, 5, 10]:
            metrics[f"i2t_r{k}"] = 0.0
            metrics[f"t2i_r{k}"] = 0.0

        return metrics

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.eval()
    all_img_z, all_txt_z = [], []
    with torch.no_grad():
        for img_emb, txt_emb in tqdm(val_loader, desc="Validation"):
            all_img_z.append(model.image_head(img_emb.to(device)).cpu())
            all_txt_z.append(model.text_head(txt_emb.to(device)).cpu())

    N = len(val_dataset) // 5
    val_img_z = torch.cat(all_img_z)[::5]              # [N, proj_dim]
    val_txt_z = torch.cat(all_txt_z).reshape(N, 5, -1) # [N, 5, proj_dim]

    metrics = coco_retrieval_metrics(val_img_z, val_txt_z)
    metrics.update(sparsity_metrics(val_img_z, val_txt_z.reshape(-1, val_txt_z.shape[-1])))
    return metrics
