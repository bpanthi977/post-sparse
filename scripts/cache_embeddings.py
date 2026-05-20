import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import cache_embeddings

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    emb_dir = Path(config["data"]["embeddings_dir"])
    if (emb_dir / "train_image_embeddings.pt").exists():
        print("Embeddings already cached. Delete the embeddings directory to recompute.")
        sys.exit(0)

    cache_embeddings(
        data_dir=config["data"]["data_dir"],
        save_dir=config["data"]["embeddings_dir"],
        clip_model=config["model"]["clip_model"],
        pretrained=config["model"]["clip_pretrained"],
        num_workers=config["data"]["num_workers"],
    )
