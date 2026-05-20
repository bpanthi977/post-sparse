import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 16384):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.relu(self.proj(x)), dim=-1)


class PostSparseModel(nn.Module):
    def __init__(self, in_dim: int = 512, proj_dim: int = 16384):
        super().__init__()
        self.image_head = ProjectionHead(in_dim, proj_dim)
        self.text_head = ProjectionHead(in_dim, proj_dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(max=100.0)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        return self.image_head(img_emb), self.text_head(txt_emb)
