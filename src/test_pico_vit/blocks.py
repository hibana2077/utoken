from typing import Dict, Tuple

import torch
import torch.nn as nn

from .modules import SigmaNet, TinyAttention, TinyMlp

try:
    from src.udtw import uDTW as NativeUDTW
except Exception:
    NativeUDTW = None


class FallbackUDTW(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lightweight fallback when numba-based uDTW is unavailable.
        dist = torch.cdist(x, y, p=2).pow(2).mean(dim=(1, 2))
        sig = torch.cdist(sigma_x, sigma_y, p=1).mean(dim=(1, 2)) * beta
        return dist, sig


class StandardViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads, qkv_bias=True, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x, {"aux_loss": x.new_zeros(())}


class SpecialViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        merge_mode: str,
        use_cuda_dtw: bool,
        udtw_gamma: float,
        udtw_beta: float,
        sigma_hidden_dim: int,
        sigma_a: float,
        sigma_b: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads, qkv_bias=True, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMlp(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.sigmanet_norm1 = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_attn = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_norm2 = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_mlp = SigmaNet(dim, sigma_hidden_dim)
        self.merge_mode = merge_mode
        self.udtw_beta = udtw_beta
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        if NativeUDTW is not None:
            self.udtw = NativeUDTW(use_cuda=use_cuda_dtw, gamma=udtw_gamma, normalize=False)
        else:
            self.udtw = FallbackUDTW()

    def _merge(self, seq_a: torch.Tensor, seq_b: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        if self.merge_mode == "mul":
            return seq_a + seq_b * sigma_b
        if self.merge_mode == "add":
            return seq_a + seq_b + sigma_b
        raise ValueError(f"Unsupported merge mode: {self.merge_mode}")

    def _sigma_stats(self, prefix: str, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}_mean": sigma.mean().detach(),
            f"{prefix}_max": sigma.max().detach(),
            f"{prefix}_min": sigma.min().detach(),
            f"{prefix}_std": sigma.std().detach(),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        original_x = x
        norm1_x = self.norm1(x)
        attn_out = self.attn(norm1_x)
        sigma_x = self.sigmanet_norm1(original_x, self.sigma_a, self.sigma_b)
        sigma_attn = self.sigmanet_attn(attn_out, self.sigma_a, self.sigma_b)
        dtw_attn_d, dtw_attn_s = self.udtw(original_x, attn_out, sigma_x, sigma_attn, beta=self.udtw_beta)
        x = self._merge(x, attn_out, sigma_attn)

        norm2_x = self.norm2(x)
        mlp_out = self.mlp(norm2_x)
        sigma_x = self.sigmanet_norm2(original_x, self.sigma_a, self.sigma_b)
        sigma_mlp = self.sigmanet_mlp(mlp_out, self.sigma_a, self.sigma_b)
        dtw_mlp_d, dtw_mlp_s = self.udtw(original_x, mlp_out, sigma_x, sigma_mlp, beta=self.udtw_beta)
        x = self._merge(x, mlp_out, sigma_mlp)

        aux_loss = (dtw_attn_d.mean() + dtw_attn_s.mean() + dtw_mlp_d.mean() + dtw_mlp_s.mean()) / (x.size(1) ** 2)
        stats = {"aux_loss": aux_loss}
        stats.update(self._sigma_stats("sigma_attn", sigma_attn))
        stats.update(self._sigma_stats("sigma_mlp", sigma_mlp))
        return x, stats
