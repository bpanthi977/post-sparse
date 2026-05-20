# PostSparse: Post-Hoc Sparse Multimodal Alignment via High-Dimensional Contrastive Projection

## Motivation

Sparse Autoencoders (SAEs) applied post-hoc to CLIP encoders suffer from a fundamental limitation: the learned sparse features are predominantly unimodal. Even when SAEs are trained on CLIP's multimodal representation space, image and text inputs activate *different* sparse features for the same semantic concept — a "car" in text and a "car" in an image occupy distinct dictionary atoms that are highly correlated but not shared. This is because SAEs optimize for reconstruction of each modality's embeddings independently, with no cross-modal alignment signal at the sparse feature level.

Sparse CLIP addresses this by enforcing sparsity *during* contrastive pretraining, yielding genuinely multimodal sparse features. However, this requires training from scratch — an expensive proposition that prevents application to existing pretrained models.

## Proposed Method

We propose a post-hoc method that recovers multimodal sparse features from a frozen, pretrained vision-language model (e.g., CLIP) without retraining the base encoders.

Given a pretrained model with frozen image encoder $f_I$ and text encoder $f_T$, we introduce two independent high-dimensional projection heads $P_I: \mathbb{R}^{m_I} \rightarrow \mathbb{R}^k$ and $P_T: \mathbb{R}^{m_T} \rightarrow \mathbb{R}^k$, where $k \gg m_I, m_T$. For a positive pair $(I, T)$ — an image and its paired caption — we compute:

$$z_I = \text{ReLU}(P_I(f_I(I))), \quad z_T = \text{ReLU}(P_T(f_T(T)))$$

We then train $P_I$ and $P_T$ using the standard contrastive (InfoNCE) loss over a batch of positive pairs, pushing $z_I$ and $z_T$ together while separating them from unpaired samples. The base encoders remain frozen throughout; only the projection heads are trained.

## Key Insight

Unlike SAEs, which optimize a reconstruction objective independently per modality, this method applies a *cross-modal contrastive signal directly in the sparse space*. The ReLU non-negativity combined with high dimensionality induces sparsity (following Wang et al., 2024, who show non-negative contrastive learning is equivalent to NMF), while the contrastive objective forces image and text representations of the same concept to activate the *same sparse dimensions* across the two separate projection heads. Cross-modal feature alignment is thus not assumed structurally — it must be learned entirely through the contrastive loss, making the alignment that emerges a meaningful empirical question.

## Distinction from Prior Work

| Method | Training | Sparse Space | Cross-Modal Alignment in Sparse Space |
|---|---|---|---|
| SAE (post-hoc) | Post-hoc | Yes | No |
| Sparse CLIP | From scratch | Yes | Yes |
| **Proposed** | Post-hoc | Yes | Yes |

## Expected Benefits

- Multimodal sparse features without the cost of pretraining from scratch
- Applicable to any existing pretrained vision-language model
- Enables concept naming, steering, and interpretability analysis on deployed models
