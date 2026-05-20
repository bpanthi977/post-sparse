import torch
import torch.nn.functional as F


def infonce_loss(z_i: torch.Tensor, z_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    sim = z_i @ z_t.T / tau
    labels = torch.arange(len(z_i), device=z_i.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
