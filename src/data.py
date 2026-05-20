from pathlib import Path
from typing import List, Tuple

import open_clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from tqdm import tqdm


def _caption_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    # Pad/truncate each sample to exactly 5 captions
    captions = [item[1][:5] + [""] * max(0, 5 - len(item[1])) for item in batch]
    return imgs, captions


class CachedPairsDataset(Dataset):
    """Flat (img_emb, txt_emb) pairs from cached embeddings.

    Stores image embeddings as [N, D] and text embeddings as [N*5, D],
    mapping each flat index back to the correct image without materialising
    the repeated image tensor in memory.
    """

    def __init__(self, img_embs: torch.Tensor, txt_embs: torch.Tensor):
        # img_embs: [N, D],  txt_embs: [N, 5, D]
        self._num_caps = txt_embs.shape[1]
        self.img_embs = img_embs                              # [N, D]
        self.txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1])  # [N*5, D]

    def __len__(self) -> int:
        return len(self.txt_embs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.img_embs[idx // self._num_caps], self.txt_embs[idx]


def cache_embeddings(
    data_dir: str,
    save_dir: str,
    clip_model: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
) -> None:
    """Encode COCO train and val splits with a frozen CLIP model and save to disk."""
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)
    model = model.to(device).eval()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        dataset = CocoCaptions(
            root=str(Path(data_dir) / f"{split}2017"),
            annFile=str(Path(data_dir) / "annotations" / f"captions_{split}2017.json"),
            transform=preprocess,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=_caption_collate_fn,
        )

        img_embs: List[torch.Tensor] = []
        txt_embs: List[torch.Tensor] = []

        with torch.no_grad():
            for imgs, captions in tqdm(loader, desc=f"Encoding {split}"):
                img_embs.append(model.encode_image(imgs.to(device)).float().cpu())

                # Encode all 5 captions per image in separate forward passes
                B = len(captions)
                cap_embs = []
                for cap_idx in range(5):
                    cap_batch = [captions[b][cap_idx] for b in range(B)]
                    tokens = tokenizer(cap_batch).to(device)
                    cap_embs.append(model.encode_text(tokens).float().cpu())
                txt_embs.append(torch.stack(cap_embs, dim=1))  # [B, 5, D]

        img_tensor = torch.cat(img_embs, dim=0)   # [N, D]
        txt_tensor = torch.cat(txt_embs, dim=0)   # [N, 5, D]
        torch.save(img_tensor, save_path / f"{split}_image_embeddings.pt")
        torch.save(txt_tensor, save_path / f"{split}_text_embeddings.pt")
        print(f"{split}: images {tuple(img_tensor.shape)}, texts {tuple(txt_tensor.shape)}")
