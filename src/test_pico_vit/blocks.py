from typing import Dict, Tuple

import torch
import torch.nn as nn

from .modules import SigmaNet, TinyAttention, TinyMlp

try:
    from src.udtw import uDTW as NativeUDTW
    print("Using native uDTW implementation.")
except Exception:
    print("Native uDTW not available, using fallback implementation.")
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


class BlockSequenceAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        merge_mode: str,
        use_cuda_dtw: bool,
        udtw_gamma: float,
        udtw_beta: float,
        sigma_hidden_dim: int,
        sigma_a: float,
        sigma_b: float,
    ) -> None:
        super().__init__()
        self.sigmanet_a = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_b = SigmaNet(dim, sigma_hidden_dim)
        self.merge_mode = merge_mode
        self.udtw_beta = udtw_beta
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        if NativeUDTW is not None:
            self.udtw = NativeUDTW(use_cuda=use_cuda_dtw, gamma=udtw_gamma, normalize=False)
        else:
            self.udtw = FallbackUDTW()

    def _merge_into_target(self, seq_b: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        if self.merge_mode == "mul":
            return seq_b * (2.0 - sigma_b)
        raise ValueError(f"Unsupported merge mode: {self.merge_mode}")

    def _sigma_stats(self, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "sigma_mean": sigma.mean().detach(),
            "sigma_max": sigma.max().detach(),
            "sigma_min": sigma.min().detach(),
            "sigma_std": sigma.std().detach(),
        }

    def estimate_seq_a(self, seq_a: torch.Tensor) -> torch.Tensor:
        return self.sigmanet_a(seq_a, self.sigma_a, self.sigma_b)

    def apply_seq_b(
        self,
        seq_a: torch.Tensor,
        sigma_a: torch.Tensor,
        seq_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sigma_b = self.sigmanet_b(seq_b, self.sigma_a, self.sigma_b)
        dtw_d, dtw_s = self.udtw(seq_a, seq_b, sigma_a, sigma_b, beta=self.udtw_beta)
        merged = self._merge_into_target(seq_b, sigma_b)

        seq_len = max(1, seq_b.size(1))
        aux_loss = (dtw_d.mean() + dtw_s.mean()) / (seq_len ** 2)
        stats = {"aux_loss": aux_loss}
        stats.update(self._sigma_stats(sigma_b))
        return merged, stats
