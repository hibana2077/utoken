from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "test_pico_vit"
    train_mode: str = "both"
    num_classes: int = 10
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 192
    depth: int = 8
    num_heads: int = 3
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    aux_weight: float = 0.1
    merge_mode: str = "mul"
    sequence_pair: str = "el"
    sigma_hidden_dim: int = 64
    sigma_a: float = 1.5
    sigma_b: float = 0.5
    udtw_gamma: float = 0.1
    udtw_beta: float = 0.5
    data_dir: str = "data"
    batch_size: int = 128
    workers: int = 4
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05
    seed: int = 7
