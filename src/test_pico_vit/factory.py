import torch

from .config import TrainConfig
from .model import TestPicoViT


def create_model(cfg: TrainConfig, device: torch.device) -> TestPicoViT:
    # Keep this as a model factory so we can extend to timm models later.
    if cfg.model_name != "test_pico_vit":
        raise ValueError(f"Unsupported model_name: {cfg.model_name}. Expected 'test_pico_vit'.")
    return TestPicoViT(cfg=cfg, use_cuda_dtw=device.type == "cuda")

