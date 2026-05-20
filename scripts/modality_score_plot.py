"""
Compute and plot the modality score distribution for a trained PostSparse run.

Usage:
    uv run python scripts/modality_score_plot.py <run_dir>
    uv run python scripts/modality_score_plot.py <run_dir> --base-embedding

Outputs saved inside <run_dir>:
    figs/modality_score.png        — modality score histogram (trained projection heads)
    figs/modality_score_base.png   — modality score histogram (raw CLIP embeddings, sanity check)
    modality_score.txt / modality_score_base.txt — corresponding summary statistics
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import PostSparseModel

TAU = 0.001
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def activations_from_heads(model: PostSparseModel, val_img: torch.Tensor, val_txt: torch.Tensor, device: str):
    """Pass cached embeddings through trained projection heads (ReLU, no L2 norm)."""
    def _encode(head, embeddings):
        parts = []
        with torch.no_grad():
            for start in range(0, len(embeddings), BATCH_SIZE):
                batch = embeddings[start : start + BATCH_SIZE].to(device)
                parts.append(F.relu(head.proj(batch)).cpu())
        return torch.cat(parts, dim=0)

    print("Extracting image activations...")
    Z_I = _encode(model.image_head, val_img)
    print("Extracting text activations...")
    Z_T = _encode(model.text_head, val_txt)
    return Z_I, Z_T


def activations_from_base_embeddings(val_img: torch.Tensor, val_txt: torch.Tensor):
    """Sanity-check activations: treat each CLIP dimension as a feature via sgn.
    A dimension is active when positive (sgn = +1 > TAU). If CLIP is cross-modally
    aligned, modality scores should cluster near 0.5."""
    return torch.sign(val_img), torch.sign(val_txt)


# ---------------------------------------------------------------------------
# Modality score computation
# ---------------------------------------------------------------------------

def compute_modality_scores(Z_I: torch.Tensor, Z_T: torch.Tensor):
    """Return modality scores and alive/dead counts for all features.

    ModScore(j) = image_activations(j) / total_activations(j), in [0, 1].
    Dead features (never active in either modality) are excluded.
    """
    img_counts = (Z_I > TAU).sum(dim=0).float()  # [k]
    txt_counts = (Z_T > TAU).sum(dim=0).float()  # [k]
    total_counts = img_counts + txt_counts

    alive_mask = total_counts > 0
    mod_score = (img_counts[alive_mask] / total_counts[alive_mask]).numpy()

    n_total = Z_I.shape[1]
    n_dead = int((~alive_mask).sum().item())
    n_alive = int(alive_mask.sum().item())
    return mod_score, n_total, n_dead, n_alive


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_stats(mod_score, n_total: int, n_dead: int, n_alive: int) -> list[str]:
    pct_multimodal = 100.0 * ((mod_score > 0.4) & (mod_score < 0.6)).sum() / n_alive
    pct_uni_image  = 100.0 * (mod_score > 0.9).sum() / n_alive
    pct_uni_text   = 100.0 * (mod_score < 0.1).sum() / n_alive
    return [
        f"Total features:          {n_total}",
        f"Dead features:           {n_dead}  ({100.0 * n_dead / n_total:.2f}%)",
        f"Active features:         {n_alive}  ({100.0 * n_alive / n_total:.2f}%)",
        f"% Multimodal (0.4–0.6):  {pct_multimodal:.2f}%",
        f"% Unimodal image (>0.9): {pct_uni_image:.2f}%",
        f"% Unimodal text  (<0.1): {pct_uni_text:.2f}%",
    ]


def save_stats(stats: list[str], path: Path) -> None:
    for line in stats:
        print(line)
    path.write_text("\n".join(stats) + "\n")
    print(f"Stats → {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_modality_scores(mod_score, title: str, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mod_score, bins=50, range=(0.0, 1.0), color="steelblue", edgecolor="none")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0, label="Perfectly multimodal (0.5)")
    ax.set_xlabel("Modality score  (0 = text only,  1 = image only)")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure → {fig_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(run_dir: Path, base_embedding: bool) -> None:
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"No hparams.yaml found in {run_dir}")
    with open(hparams_path) as f:
        config = yaml.safe_load(f)

    emb_dir = Path(config["data"]["embeddings_dir"])
    val_img = torch.load(emb_dir / "val_image_embeddings.pt", weights_only=True)  # [N, 512]
    val_txt = torch.load(emb_dir / "val_text_embeddings.pt", weights_only=True)   # [N, 5, 512]
    val_txt = val_txt[:, 0, :]  # first caption only → [N, 512]

    if base_embedding:
        Z_I, Z_T = activations_from_base_embeddings(val_img, val_txt)
        suffix, title_tag = "_base", "base CLIP embeddings (sgn)"
    else:
        proj_dim = config["model"]["proj_dim"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint (best.pt / last.pt) found in {run_dir}")
        model = PostSparseModel(proj_dim=proj_dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded {ckpt_path.name} from {run_dir.name}")
        Z_I, Z_T = activations_from_heads(model, val_img, val_txt, device)
        suffix, title_tag = "", "trained projection heads"

    mod_score, n_total, n_dead, n_alive = compute_modality_scores(Z_I, Z_T)
    print(f"Val samples: {val_img.shape[0]}  features: {n_total}")

    stats = build_stats(mod_score, n_total, n_dead, n_alive)
    save_stats(stats, run_dir / f"modality_score{suffix}.txt")

    plot_modality_scores(
        mod_score,
        title=f"Modality Score Distribution — {run_dir.name}\n({title_tag})",
        fig_path=run_dir / "figs" / f"modality_score{suffix}.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot modality score distribution for a PostSparse run.")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (contains hparams.yaml and best.pt)")
    parser.add_argument("--base-embedding", action="store_true",
                        help="Sanity-check mode: use sgn(CLIP embeddings) instead of trained projection heads")
    args = parser.parse_args()
    main(args.run_dir, args.base_embedding)
