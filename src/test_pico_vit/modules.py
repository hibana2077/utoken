from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def sigmoid_ab(a: float, b: float, value: torch.Tensor) -> torch.Tensor:
    return a * torch.sigmoid(value) + b


class TinyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        out = self.proj(out)
        return self.proj_drop(out)


class TinyMlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SigmaNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.apply(weight_init)

    def forward(self, seq: torch.Tensor, a: float, b: float) -> torch.Tensor:
        batch_size, length, dim = seq.shape
        seq = seq.reshape(batch_size * length, dim)
        seq = F.relu(self.fc1(seq))
        seq = self.fc2(seq)
        seq = seq.reshape(batch_size, length, dim).mean(dim=2, keepdim=True)
        return sigmoid_ab(a, b, seq)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

