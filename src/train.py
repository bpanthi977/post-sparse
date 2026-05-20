import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import CachedPairsDataset
from eval import coco_retrieval_metrics, sparsity_metrics
from loss import infonce_loss
from model import PostSparseModel
from utils import create_run_dir, save_hparams


def _cosine_with_warmup(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def train(config: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = create_run_dir(config["logging"]["log_dir"])
    save_hparams(config, run_dir)
    wandb.init(project=config["logging"]["wandb_project"], name=run_dir.name, config=config)

    emb_dir = Path(config["data"]["embeddings_dir"])
    if not (emb_dir / "train_image_embeddings.pt").exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_dir}.\n"
            "Run first:  uv run python scripts/cache_embeddings.py config/base.yaml"
        )

    train_img = torch.load(emb_dir / "train_image_embeddings.pt", weights_only=True)
    train_txt = torch.load(emb_dir / "train_text_embeddings.pt", weights_only=True)
    val_img = torch.load(emb_dir / "val_image_embeddings.pt", weights_only=True)
    val_txt = torch.load(emb_dir / "val_text_embeddings.pt", weights_only=True)

    train_dataset = CachedPairsDataset(train_img, train_txt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # val_img: [N, D],  val_txt: [N, 5, D] — TensorDataset yields (img[D], txt[5,D]) per item
    val_loader = DataLoader(
        TensorDataset(val_img, val_txt),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = PostSparseModel(proj_dim=config["model"]["proj_dim"]).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    max_epochs = config["training"]["max_epochs"]
    steps_per_epoch = math.ceil(len(train_dataset) / config["training"]["batch_size"])
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * config["training"]["warmup_fraction"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_with_warmup(warmup_steps, total_steps)
    )

    grad_accum = config["training"]["grad_accum_steps"]
    patience = config["training"]["patience"]
    best_r1 = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for micro_step, (img_emb, txt_emb) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        ):
            img_emb = img_emb.to(device)
            txt_emb = txt_emb.to(device)

            z_i, z_t = model(img_emb, txt_emb)
            raw_loss = infonce_loss(z_i, z_t, model.tau)
            (raw_loss / grad_accum).backward()
            epoch_loss += raw_loss.item()

            if (micro_step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                wandb.log(
                    {"train/loss": raw_loss.item(), "train/tau": model.tau.item()},
                    step=global_step,
                )
                global_step += 1

        # --- validation ---
        model.eval()
        all_img_z, all_txt_z = [], []
        with torch.no_grad():
            for img_emb, txt_emb in tqdm(val_loader, desc='Validation'):         # [B, D], [B, 5, D]
                img_emb = img_emb.to(device)
                txt_emb = txt_emb.to(device)
                B = img_emb.shape[0]
                all_img_z.append(model.image_head(img_emb).cpu())
                all_txt_z.append(
                    model.text_head(txt_emb.reshape(B * 5, -1)).reshape(B, 5, -1).cpu()
                )
        val_img_z = torch.cat(all_img_z, dim=0)        # [N, proj_dim]
        val_txt_z = torch.cat(all_txt_z, dim=0)        # [N, 5, proj_dim]

        val_metrics = coco_retrieval_metrics(val_img_z, val_txt_z)
        spar = sparsity_metrics(val_img_z, val_txt_z.reshape(-1, val_txt_z.shape[-1]))  # flatten [N*5, D]
        avg_loss = epoch_loss / len(train_loader)

        wandb.log(
            {
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"val/{k}": v for k, v in spar.items()},
                "train/epoch_loss": avg_loss,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}  "
            f"i2t_r1={val_metrics['i2t_r1']:.1f}  "
            f"t2i_r1={val_metrics['t2i_r1']:.1f}  "
            f"τ={model.tau.item():.4f}"
        )

        r1 = val_metrics["i2t_r1"]
        if r1 > best_r1:
            best_r1 = r1
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    torch.save(model.state_dict(), run_dir / "last.pt")
    wandb.finish()
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
