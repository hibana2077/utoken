from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .blocks import SpecialViTBlock, StandardViTBlock
from .config import TrainConfig
from .modules import PatchEmbed


class TestPicoViT(nn.Module):
    def __init__(self, cfg: TrainConfig, use_cuda_dtw: bool) -> None:
        super().__init__()
        if cfg.depth < 4:
            raise ValueError("depth must be >= 4 so two special blocks can stay in the middle.")

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_prefix_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        mid = cfg.depth // 2
        special_indices = {max(1, mid - 1), min(cfg.depth - 2, mid)}
        blocks: List[nn.Module] = []
        for i in range(cfg.depth):
            if i in special_indices:
                block = SpecialViTBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    drop=cfg.drop_rate,
                    merge_mode=cfg.merge_mode,
                    use_cuda_dtw=use_cuda_dtw,
                    udtw_gamma=cfg.udtw_gamma,
                    udtw_beta=cfg.udtw_beta,
                    sigma_hidden_dim=cfg.sigma_hidden_dim,
                    sigma_a=cfg.sigma_a,
                    sigma_b=cfg.sigma_b,
                )
            else:
                block = StandardViTBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    drop=cfg.drop_rate,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # Match timm ViT head flow: norm -> dropout -> linear head.
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head_drop = nn.Dropout(cfg.drop_rate)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        aux_loss = x.new_zeros(())
        last_stats: Dict[str, torch.Tensor] = {}
        for block in self.blocks:
            x, stats = block(x)
            aux_loss = aux_loss + stats["aux_loss"]
            last_stats = stats
        cls_token = self.norm(x[:, 0])
        stats_out = {"aux_loss": aux_loss}
        stats_out.update({k: v for k, v in last_stats.items() if k != "aux_loss"})
        return cls_token, stats_out

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats, stats = self.forward_features(x)
        logits = self.forward_head(feats)
        return logits, stats

