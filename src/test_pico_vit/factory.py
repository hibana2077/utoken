import torch

from .config import TrainConfig
from .model import TestPicoViT


def create_model(cfg: TrainConfig, device: torch.device, variant: str = "special") -> TestPicoViT:
    # Keep this as a model factory so we can extend to timm models later.
    if cfg.model_name != "test_pico_vit":
        raise ValueError(f"Unsupported model_name: {cfg.model_name}. Expected 'test_pico_vit'.")
    if variant not in {"special", "baseline"}:
        raise ValueError(f"Unsupported variant: {variant}. Expected 'special' or 'baseline'.")
    return TestPicoViT(
        cfg=cfg,
        use_cuda_dtw=device.type == "cuda",
        use_special_blocks=variant == "special",
    )
