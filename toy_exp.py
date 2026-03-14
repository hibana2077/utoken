import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from udtw import uDTW


UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
UCI_HAR_DIRNAME = "UCI HAR Dataset"
UCI_HAR_SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += param.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


class TinyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.proj(out)


class TinyMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class StandardBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x, {"aux_loss": x.new_zeros(())}


def sigmoid_ab(a: float, b: float, input: torch.Tensor) -> torch.Tensor:
    return a * torch.sigmoid(input) + b


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: float, b: float, input: torch.Tensor) -> torch.Tensor:
        return sigmoid_ab(a, b, input)


class SigmaNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.sigmoid = Sigmoid()

    def forward(self, seq: torch.Tensor, a: float, b: float) -> torch.Tensor:
        batch_size, length, _ = seq.shape
        seq = seq.view(batch_size * length, -1)
        seq = self.fc1(seq)
        seq = F.relu(seq)
        seq = self.fc2(seq)
        seq = seq.view(batch_size, length, -1).mean(2, keepdim=True)
        return self.sigmoid(a, b, seq)


class SpecializedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
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
        self.attn = TinyAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim, mlp_ratio)
        self.sigmanet_norm1 = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_attn = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_norm2 = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_mlp = SigmaNet(dim, sigma_hidden_dim)
        self.sigmanet_norm1.apply(weight_init)
        self.sigmanet_attn.apply(weight_init)
        self.sigmanet_norm2.apply(weight_init)
        self.sigmanet_mlp.apply(weight_init)
        self.merge_mode = merge_mode
        self.udtw_beta = udtw_beta
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.udtw = uDTW(use_cuda=use_cuda_dtw, gamma=udtw_gamma, normalize=False)

    def _merge(
        self,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        sigma_b: torch.Tensor,
    ) -> torch.Tensor:
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
        dtw_attn_d, dtw_attn_s = self.udtw(
            original_x,
            attn_out,
            sigma_x,
            sigma_attn,
            beta=self.udtw_beta,
        )
        x = self._merge(x, attn_out, sigma_attn)

        norm2_x = self.norm2(x)
        mlp_out = self.mlp(norm2_x)
        sigma_x = self.sigmanet_norm2(original_x, self.sigma_a, self.sigma_b)
        sigma_mlp = self.sigmanet_mlp(mlp_out, self.sigma_a, self.sigma_b)
        dtw_mlp_d, dtw_mlp_s = self.udtw(
            original_x,
            mlp_out,
            sigma_x,
            sigma_mlp,
            beta=self.udtw_beta,
        )
        aux_loss = (
            dtw_attn_d.mean()
            + dtw_attn_s.mean()
            + dtw_mlp_d.mean()
            + dtw_mlp_s.mean()
        ) / (x.size(1) * x.size(1))
        x = self._merge(x, mlp_out, sigma_mlp)

        stats = {"aux_loss": aux_loss}
        stats.update(self._sigma_stats("sigma_attn", sigma_attn))
        stats.update(self._sigma_stats("sigma_mlp", sigma_mlp))
        return x, stats


class TinyClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        special_second_block: bool,
        merge_mode: str,
        use_cuda_dtw: bool,
        udtw_gamma: float,
        udtw_beta: float,
        sigma_hidden_dim: int,
        sigma_a: float,
        sigma_b: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, dim) * 0.02)
        self.block1 = StandardBlock(dim, num_heads, mlp_ratio)
        if special_second_block:
            self.block2 = SpecializedBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                merge_mode=merge_mode,
                use_cuda_dtw=use_cuda_dtw,
                udtw_gamma=udtw_gamma,
                udtw_beta=udtw_beta,
                sigma_hidden_dim=sigma_hidden_dim,
                sigma_a=sigma_a,
                sigma_b=sigma_b,
            )
        else:
            self.block2 = StandardBlock(dim, num_heads, mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = self.input_proj(x)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : seq_len + 1]
        x, stats1 = self.block1(x)
        x, stats2 = self.block2(x)
        logits = self.head(self.norm(x[:, 0]))
        aux_loss = stats1["aux_loss"] + stats2["aux_loss"]
        stats = {"aux_loss": aux_loss}
        stats.update({k: v for k, v in stats2.items() if k != "aux_loss"})
        return logits, stats


@dataclass
class ExpConfig:
    dataset: str = "har"
    data_dir: str = "data"
    train_size: int = 256
    val_size: int = 128
    seq_len: int = 8
    input_dim: int = 16
    dim: int = 32
    num_heads: int = 4
    mlp_ratio: float = 2.0
    sigma_hidden_dim: int = 32
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.2
    merge_mode: str = "mul"
    udtw_gamma: float = 0.1
    udtw_beta: float = 0.5
    sigma_a: float = 1.5
    sigma_b: float = 0.5
    teacher_merge_mode: str = "mul"
    seed: int = 7


@dataclass
class DatasetBundle:
    train_ds: TensorDataset
    val_ds: TensorDataset
    seq_len: int
    input_dim: int
    num_classes: int
    description: str


def build_teacher_dataset(
    config: ExpConfig,
    device: torch.device,
) -> DatasetBundle:
    teacher = TinyClassifier(
        input_dim=config.input_dim,
        seq_len=config.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=False,
        merge_mode=config.teacher_merge_mode,
        use_cuda_dtw=device.type == "cuda",
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
        sigma_a=config.sigma_a,
        sigma_b=config.sigma_b,
    ).to(device)
    teacher.eval()

    total = config.train_size + config.val_size
    with torch.no_grad():
        inputs = torch.randn(total * 3, config.seq_len, config.input_dim, device=device)
        logits, _ = teacher(inputs)
        score = logits[:, 1] - logits[:, 0]
        threshold = score.median()
        labels = (score > threshold).long()

        selected_inputs = inputs[:total].cpu()
        selected_labels = labels[:total].cpu()

    train_x = selected_inputs[: config.train_size]
    train_y = selected_labels[: config.train_size]
    val_x = selected_inputs[config.train_size : config.train_size + config.val_size]
    val_y = selected_labels[config.train_size : config.train_size + config.val_size]
    return DatasetBundle(
        train_ds=TensorDataset(train_x, train_y),
        val_ds=TensorDataset(val_x, val_y),
        seq_len=config.seq_len,
        input_dim=config.input_dim,
        num_classes=2,
        description="teacher-generated synthetic dataset: labels come from a frozen 2-layer model with the specialized second block.",
    )


def make_dataloaders(
    dataset: DatasetBundle,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset.train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(dataset.val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_uci_har_downloaded(data_root: Path) -> Path:
    dataset_root = data_root / UCI_HAR_DIRNAME
    if dataset_root.exists():
        return dataset_root

    archive_path = data_root / "uci_har_dataset.zip"
    if not archive_path.exists():
        print(f"downloading UCI HAR dataset to {archive_path}")
        urlretrieve(UCI_HAR_URL, archive_path)

    print(f"extracting UCI HAR dataset into {data_root}")
    with ZipFile(archive_path) as archive:
        archive.extractall(data_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected extracted dataset at {dataset_root}")
    return dataset_root


def read_space_delimited_matrix(path: Path) -> torch.Tensor:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append([float(value) for value in line.strip().split()])
    return torch.tensor(rows, dtype=torch.float32)


def read_label_vector(path: Path) -> torch.Tensor:
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            labels.append(int(line.strip()) - 1)
    return torch.tensor(labels, dtype=torch.long)


def load_uci_har_split(dataset_root: Path, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    signal_dir = dataset_root / split / "Inertial Signals"
    channels = []
    for signal_name in UCI_HAR_SIGNAL_NAMES:
        signal_path = signal_dir / f"{signal_name}_{split}.txt"
        channels.append(read_space_delimited_matrix(signal_path))
    x = torch.stack(channels, dim=-1).contiguous()
    y = read_label_vector(dataset_root / split / f"y_{split}.txt")
    return x, y


def build_har_dataset(config: ExpConfig) -> DatasetBundle:
    dataset_root = ensure_uci_har_downloaded(resolve_data_dir(config.data_dir))
    train_x, train_y = load_uci_har_split(dataset_root, "train")
    val_x, val_y = load_uci_har_split(dataset_root, "test")

    train_mean = train_x.mean(dim=(0, 1), keepdim=True)
    train_std = train_x.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - train_mean) / train_std
    val_x = (val_x - train_mean) / train_std

    return DatasetBundle(
        train_ds=TensorDataset(train_x, train_y),
        val_ds=TensorDataset(val_x, val_y),
        seq_len=train_x.size(1),
        input_dim=train_x.size(2),
        num_classes=int(train_y.max().item() + 1),
        description="UCI HAR inertial-signal benchmark with the official train/test split.",
    )


def build_dataset(config: ExpConfig, device: torch.device) -> DatasetBundle:
    if config.dataset == "synthetic":
        return build_teacher_dataset(config, device)
    if config.dataset == "har":
        return build_har_dataset(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def evaluate(
    model: TinyClassifier,
    loader: DataLoader,
    device: torch.device,
    aux_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, stats = model(x)
            ce_loss = F.cross_entropy(logits, y)
            aux_loss = stats["aux_loss"]
            loss = ce_loss + aux_weight * aux_loss
            total_loss += loss.item() * y.size(0)
            total_ce += ce_loss.item() * y.size(0)
            total_aux += aux_loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_count += y.size(0)
    return {
        "loss": total_loss / total_count,
        "ce": total_ce / total_count,
        "aux": total_aux / total_count,
        "acc": total_correct / total_count,
    }


def train_model(
    name: str,
    model: TinyClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: ExpConfig,
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_val = {"acc": 0.0, "loss": float("inf")}

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        last_stats: Dict[str, float] = {}

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(x)
            ce_loss = F.cross_entropy(logits, y)
            aux_loss = stats["aux_loss"]
            loss = ce_loss + config.aux_weight * aux_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_count += y.size(0)

            last_stats = {
                "ce": ce_loss.item(),
                "aux": aux_loss.item(),
                "attn_grad": grad_norm(model.block2.attn),
                "sigma_attn_max": float(stats.get("sigma_attn_max", 0.0)),
                "sigma_mlp_max": float(stats.get("sigma_mlp_max", 0.0)),
                "sigma_attn_min": float(stats.get("sigma_attn_min", 0.0)),
                "sigma_mlp_min": float(stats.get("sigma_mlp_min", 0.0)),
                "sigma_attn_std": float(stats.get("sigma_attn_std", 0.0)),
                "sigma_mlp_std": float(stats.get("sigma_mlp_std", 0.0)),
            }
            if model.block2.__class__ == SpecializedBlock:
                last_stats["sigma_attn_mean"] = float(stats["sigma_attn_mean"])
                last_stats["sigma_mlp_mean"] = float(stats["sigma_mlp_mean"])

        val_metrics = evaluate(model, val_loader, device, config.aux_weight)
        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        best_val["acc"] = max(best_val["acc"], val_metrics["acc"])
        best_val["loss"] = min(best_val["loss"], val_metrics["loss"])

        parts = [
            f"{name}",
            f"epoch={epoch:02d}",
            f"train_loss={train_loss:.4f}",
            f"train_acc={train_acc:.3f}",
            f"val_loss={val_metrics['loss']:.4f}",
            f"val_acc={val_metrics['acc']:.3f}",
            f"ce={last_stats['ce']:.4f}",
            f"aux={last_stats['aux']:.4f}",
            f"attn_grad={last_stats['attn_grad']:.4e}",
        ]
        if model.block2.__class__ == SpecializedBlock:
            parts.extend(
                [
                    f"sigma_attn_mean={last_stats['sigma_attn_mean']:.3f}",
                    f"sigma_mlp_mean={last_stats['sigma_mlp_mean']:.3f}",
                    f"sigma_attn_max={last_stats['sigma_attn_max']:.3f}",
                    f"sigma_mlp_max={last_stats['sigma_mlp_max']:.3f}",
                    f"sigma_attn_min={last_stats['sigma_attn_min']:.3f}",
                    f"sigma_mlp_min={last_stats['sigma_mlp_min']:.3f}",
                    f"sigma_attn_std={last_stats['sigma_attn_std']:.3f}",
                    f"sigma_mlp_std={last_stats['sigma_mlp_std']:.3f}"
                ]
            )
        print(" | ".join(parts))

    return best_val


def parse_args() -> ExpConfig:
    parser = argparse.ArgumentParser(description="Toy experiment for uDTW-gated residual ViT block.")
    parser.add_argument("--dataset", choices=["har", "synthetic"], default="har")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigma-hidden-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--merge-mode", choices=["mul", "add"], default="mul")
    parser.add_argument(
        "--teacher-merge-mode",
        choices=["mul", "add"],
        default="mul",
        help="Merge mode used only for synthetic-label teacher generation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return ExpConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        dim=args.dim,
        num_heads=args.num_heads,
        aux_weight=args.aux_weight,
        lr=args.lr,
        merge_mode=args.merge_mode,
        teacher_merge_mode=args.teacher_merge_mode,
        seed=args.seed,
        sigma_hidden_dim=args.sigma_hidden_dim,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} | dataset={config.dataset} | merge_mode={config.merge_mode} "
        f"| teacher_merge_mode={config.teacher_merge_mode} | seed={config.seed}"
    )

    dataset = build_dataset(config, device)
    print(
        f"{dataset.description} train={len(dataset.train_ds)} val={len(dataset.val_ds)} "
        f"seq_len={dataset.seq_len} input_dim={dataset.input_dim} num_classes={dataset.num_classes}"
    )

    set_seed(config.seed)
    special_train_loader, special_val_loader = make_dataloaders(dataset, config.batch_size, config.seed)
    special = TinyClassifier(
        input_dim=dataset.input_dim,
        seq_len=dataset.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=True,
        merge_mode=config.merge_mode,
        use_cuda_dtw=device.type == "cuda",
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
        sigma_a=config.sigma_a,
        sigma_b=config.sigma_b,
        num_classes=dataset.num_classes,
    ).to(device)

    special_best = train_model(
        "special ",
        special,
        special_train_loader,
        special_val_loader,
        device,
        config,
    )

    del special

    set_seed(config.seed)
    baseline_train_loader, baseline_val_loader = make_dataloaders(dataset, config.batch_size, config.seed)
    baseline = TinyClassifier(
        input_dim=dataset.input_dim,
        seq_len=dataset.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=False,
        merge_mode=config.merge_mode,
        use_cuda_dtw=False,
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
        sigma_a=config.sigma_a,
        sigma_b=config.sigma_b,
        num_classes=dataset.num_classes,
    ).to(device)

    baseline_best = train_model(
        "baseline",
        baseline,
        baseline_train_loader,
        baseline_val_loader,
        device,
        config,
    )

    del baseline 

    print(
        "summary | "
        f"baseline_best_acc={baseline_best['acc']:.3f} baseline_best_loss={baseline_best['loss']:.4f} | "
        f"special_best_acc={special_best['acc']:.3f} special_best_loss={special_best['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
