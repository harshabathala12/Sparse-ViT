import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import VisionTransformer


# ---------------------------
# 🔹 Gumbel Top-K Sparse Gate
# ---------------------------
class GumbelTopKGate(nn.Module):
    def __init__(self, d_model, K=32, temp=1.0):
        super().__init__()
        self.K = K
        self.temp = temp
        # Simple projection to learn relative patch importance
        self.proj = nn.Linear(d_model, 1)

    def forward(self, q, k):
        """
        q, k : [B, heads, N, dim]
        Returns a differentiable binary mask [B, N, N]
        """
        # Compute raw attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / np.sqrt(q.size(-1))  # [B, heads, N, N]

        # Average across heads to get unified patch importance
        logits = attn_scores.mean(dim=1)  # [B, N, N]

        # Gumbel softmax trick for differentiable Top-K selection
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        probs = F.softmax((logits + gumbel) / self.temp, dim=-1)

        # Select top-K patches dynamically
        topk = torch.topk(probs, self.K, dim=-1)[0]
        thresh = topk.min(dim=-1, keepdim=True)[0]
        sparse_mask = (probs >= thresh).float()  # [B, N, N]

        return sparse_mask


# ---------------------------
# 🔹 Sparse Attention Module
# ---------------------------
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=6, K=32):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.K = K

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Trainable attention head importance scores
        self.head_scores = nn.Parameter(torch.ones(num_heads))
        self.gate = GumbelTopKGate(dim // num_heads, K)

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, C = x.shape

        # Standard QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get sparse attention mask via gating
        attn_mask = self.gate(q, k).unsqueeze(1)  # [B, 1, N, N]

        # Compute masked attention
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1) * attn_mask
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Weighted by learned head importance
        return self.proj(out * self.head_scores.mean())


# ---------------------------
# 🔹 Sparse Vision Transformer
# ---------------------------
class SparseViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, num_classes=10,
                 embed_dim=192, depth=6, num_heads=6, K=32, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size,
                         num_classes=num_classes, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, **kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.K = K

        # Replace default attention blocks with SparseAttention
        for blk in self.blocks:
            blk.attn = SparseAttention(self.embed_dim, self.num_heads, K=self.K)
