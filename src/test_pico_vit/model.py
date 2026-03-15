from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .blocks import BlockSequenceAdapter, StandardViTBlock
from .config import TrainConfig
from .modules import PatchEmbed


class TestPicoViT(nn.Module):
    SEQUENCE_PAIR_MAP = {
        "em": ("early", "mid"),
        "ml": ("mid", "last"),
        "el": ("early", "last"),
    }

    def __init__(self, cfg: TrainConfig, use_cuda_dtw: bool, enable_sigma_path: bool = True) -> None:
        super().__init__()
        if cfg.depth < 3:
            raise ValueError("depth must be >= 3 so early/mid/last blocks are distinct.")
        if cfg.sequence_pair not in self.SEQUENCE_PAIR_MAP:
            raise ValueError(f"Unsupported sequence_pair: {cfg.sequence_pair}")

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_prefix_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        blocks: List[nn.Module] = []
        for _ in range(cfg.depth):
            block = StandardViTBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.drop_rate,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.enable_sigma_path = enable_sigma_path
        self.sequence_block_indices = self._resolve_sequence_block_indices(cfg.depth)
        self.sequence_pair = cfg.sequence_pair
        self.seq_a_source, self.seq_b_source = self.SEQUENCE_PAIR_MAP[self.sequence_pair]
        self.seq_a_index = self.sequence_block_indices[self.seq_a_source]
        self.seq_b_index = self.sequence_block_indices[self.seq_b_source]
        self.sequence_adapter = BlockSequenceAdapter(
            dim=cfg.embed_dim,
            merge_mode=cfg.merge_mode,
            use_cuda_dtw=use_cuda_dtw,
            udtw_gamma=cfg.udtw_gamma,
            udtw_beta=cfg.udtw_beta,
            sigma_hidden_dim=cfg.sigma_hidden_dim,
            sigma_a=cfg.sigma_a,
            sigma_b=cfg.sigma_b,
        )

        # Match timm ViT head flow: norm -> dropout -> linear head.
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head_drop = nn.Dropout(cfg.drop_rate)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self._init_weights()

    @staticmethod
    def _resolve_sequence_block_indices(depth: int) -> Dict[str, int]:
        return {"early": 0, "mid": depth // 2, "last": depth - 1}

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
        sigma_sums: Dict[str, torch.Tensor] = {}
        sigma_count = 0
        seq_a = None
        sigma_a = None
        for idx, block in enumerate(self.blocks):
            x, stats = block(x)
            aux_loss = aux_loss + stats["aux_loss"]
            if self.enable_sigma_path and idx == self.seq_a_index:
                seq_a = x
                sigma_a = self.sequence_adapter.estimate_seq_a(seq_a)
            if self.enable_sigma_path and idx == self.seq_b_index:
                if seq_a is None or sigma_a is None:
                    raise RuntimeError("seq_a and sigma_a must be prepared before seq_b is processed.")
                x, pair_stats = self.sequence_adapter.apply_seq_b(seq_a, sigma_a, x)
                aux_loss = aux_loss + pair_stats["aux_loss"]
                sigma_count += 1
                for key, value in pair_stats.items():
                    if key == "aux_loss":
                        continue
                    sigma_sums[key] = sigma_sums.get(key, x.new_zeros(())) + value
        cls_token = self.norm(x[:, 0]) # (B, D)
        stats_out = {"aux_loss": aux_loss}
        if sigma_count > 0:
            stats_out.update({k: v / sigma_count for k, v in sigma_sums.items()})
        return cls_token, stats_out

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats, stats = self.forward_features(x) # feats: (B, D)
        logits = self.forward_head(feats) # (B, num_classes)
        return logits, stats
