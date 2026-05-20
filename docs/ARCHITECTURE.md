# PostSparse Architecture & Loss Function

## Base Encoders (Frozen)

- **Model**: OpenAI CLIP `ViT-B/32` loaded via `open_clip_torch`
  - Image encoder output dim: 512
  - Text encoder output dim: 512
- Base encoders are **fully frozen** throughout training — no gradients flow into them.
- Upgrade path: `ViT-L/14` (dim 768) for scale experiments.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
# Weights downloaded to ~/.cache/huggingface/hub/ on first call
```

## Projection Heads

Two independent heads `P_I` and `P_T`, one per modality:

- **Architecture**: Single linear layer (no bias) + ReLU
- **Input dim**: 512 (CLIP embedding dim)
- **Output dim (k)**: 16384 (32× CLIP dim)
- **Post-ReLU normalization**: L2-normalize → cosine similarity in sparse space

```python
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=16384):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return F.normalize(F.relu(self.proj(x)), dim=-1)
```

**Rationale for design choices:**
- *No bias*: prevents constant activation offsets that would reduce effective sparsity.
- *k = 16384*: high over-completeness (32×) maximizes geometric sparsity induced by ReLU in high-dimensional space.

## Loss Function

Symmetric InfoNCE (standard CLIP contrastive loss) applied in the sparse space:

```
L = (1/2) * [ CE(sim / τ, labels) + CE(sim.T / τ, labels) ]
```

where `sim[i,j] = z_I^i · z_T^j` (dot product of L2-normalized sparse vectors, i.e. cosine similarity), and `labels = [0, 1, 2, ..., N-1]` (diagonal).

### Temperature

Learnable log-temperature, initialized at `log(0.07)`, exponentiated and clamped:

```python
log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
tau = log_tau.exp().clamp(max=100)
logits = sim_matrix / tau
```

### Trainable Parameters

Only the projection heads and temperature are trained:
- `P_I.proj.weight` — shape `[16384, 512]`
- `P_T.proj.weight` — shape `[16384, 512]`
- `log_tau` — scalar

Total trainable params: ~16.8M (two projection matrices + 1 scalar).

