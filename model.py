"""
model.py
--------
Hybrid Vision Transformer + CNN (EfficientNet) model for bone fracture classification.

Architecture:
  - EfficientNet-B3 backbone (ImageNet pretrained, fine-tuned) — local feature extraction
  - ViT patch-embedding head — global attention over spatial features
  - Fusion MLP classifier

Why hybrid?
  X-rays benefit from both fine-grained local texture (CNN) and global structural
  context (attention). The combination consistently outperforms either alone on
  medical imaging benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─── Multi-Head Self-Attention block ────────────────────────────────────────
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # x: (B, N, D)
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ─── Main Model ─────────────────────────────────────────────────────────────
class FractureHybridNet(nn.Module):
    """
    Parameters
    ----------
    num_classes : int   — number of fracture categories
    embed_dim   : int   — transformer embedding dimension
    num_heads   : int   — attention heads
    depth       : int   — number of transformer blocks
    dropout     : float — dropout rate
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── CNN backbone (EfficientNet-B3, ImageNet weights) ──────────────
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Remove final classifier; keep feature extractor
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])   # → (B, 1536, H', W')
        self.pool = nn.AdaptiveAvgPool2d(7)                          # → (B, 1536, 7, 7)  = 49 patches

        cnn_out_channels = 1536
        num_patches = 49

        # ── Patch projection (CNN channels → embed_dim) ────────────────
        self.patch_proj = nn.Sequential(
            nn.Linear(cnn_out_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── CLS token + positional embedding ──────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[SelfAttentionBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classifier head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, H, W)
        B = x.size(0)

        # CNN features
        feat = self.cnn(x)                              # (B, 1536, h, w)
        feat = self.pool(feat)                          # (B, 1536, 7, 7)
        feat = feat.flatten(2).transpose(1, 2)          # (B, 49, 1536)
        feat = self.patch_proj(feat)                    # (B, 49, D)

        # Prepend CLS token + add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)          # (B, 50, D)
        tokens = tokens + self.pos_embed

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Classification from CLS token
        cls_out = tokens[:, 0]                          # (B, D)
        return self.head(cls_out)                       # (B, num_classes)


# ─── GradCAM for explainability ──────────────────────────────────────────────
class GradCAM:
    """
    Generates attention/saliency maps from the last CNN layer.
    Usage:
        cam = GradCAM(model)
        heatmap = cam(image_tensor, target_class)
    """

    def __init__(self, model: FractureHybridNet):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook()

    def _hook(self):
        # Last conv block of EfficientNet-B3 inside self.cnn
        target_layer = list(self.model.cnn.children())[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, img: torch.Tensor, target_class: int):
        self.model.eval()
        img = img.unsqueeze(0).requires_grad_(True)
        logits = self.model(img)
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (h, w)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def build_model(cfg, num_classes: int) -> FractureHybridNet:
    m = cfg["model"]
    return FractureHybridNet(
        num_classes=num_classes,
        embed_dim=m.get("embed_dim", 256),
        num_heads=m.get("num_heads", 8),
        depth=m.get("depth", 4),
        dropout=m.get("dropout", 0.3),
    )
