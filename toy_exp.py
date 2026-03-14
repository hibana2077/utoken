import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from udtw import uDTW


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


class SigmaNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        raw = self.net(seq)
        return 0.5 + torch.sigmoid(raw)


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
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim, mlp_ratio)
        self.sigma_x = SigmaNet(dim, sigma_hidden_dim)
        self.sigma_attn = SigmaNet(dim, sigma_hidden_dim)
        self.merge_mode = merge_mode
        self.udtw_beta = udtw_beta
        self.udtw = uDTW(use_cuda=use_cuda_dtw, gamma=udtw_gamma, normalize=False)

    def _merge(
        self,
        x_seq: torch.Tensor,
        post_attn_seq: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_attn: torch.Tensor,
    ) -> torch.Tensor:
        if self.merge_mode == "mul":
            return x_seq * sigma_x + post_attn_seq * sigma_attn
        if self.merge_mode == "add":
            return x_seq + sigma_x + post_attn_seq + sigma_attn
        raise ValueError(f"Unsupported merge mode: {self.merge_mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_out = self.attn(self.norm1(x))
        sigma_x = self.sigma_x(x)
        sigma_attn = self.sigma_attn(attn_out)
        dtw_d, dtw_s = self.udtw(x, attn_out, sigma_x, sigma_attn, beta=self.udtw_beta)
        aux_loss = (dtw_d.mean() + dtw_s.mean()) / (x.size(1) * x.size(1))
        x = self._merge(x, attn_out, sigma_x, sigma_attn)
        x = x + self.mlp(self.norm2(x))
        stats = {
            "aux_loss": aux_loss,
            "sigma_x_mean": sigma_x.mean().detach(),
            "sigma_attn_mean": sigma_attn.mean().detach(),
        }
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
    seed: int = 7


def build_teacher_dataset(
    config: ExpConfig,
    device: torch.device,
) -> Tuple[TensorDataset, TensorDataset]:
    teacher = TinyClassifier(
        input_dim=config.input_dim,
        seq_len=config.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=True,
        merge_mode=config.merge_mode,
        use_cuda_dtw=device.type == "cuda",
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
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
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)


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
            }
            if hasattr(model.block2, "sigma_x"):
                last_stats["sigma_x_grad"] = grad_norm(model.block2.sigma_x)
                last_stats["sigma_attn_grad"] = grad_norm(model.block2.sigma_attn)
                last_stats["sigma_x_mean"] = float(stats["sigma_x_mean"])
                last_stats["sigma_attn_mean"] = float(stats["sigma_attn_mean"])

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
        if "sigma_x_grad" in last_stats:
            parts.extend(
                [
                    f"sigma_x_grad={last_stats['sigma_x_grad']:.4e}",
                    f"sigma_attn_grad={last_stats['sigma_attn_grad']:.4e}",
                    f"sigma_x_mean={last_stats['sigma_x_mean']:.3f}",
                    f"sigma_attn_mean={last_stats['sigma_attn_mean']:.3f}",
                ]
            )
        print(" | ".join(parts))

    return best_val


def parse_args() -> ExpConfig:
    parser = argparse.ArgumentParser(description="Toy experiment for uDTW-gated residual ViT block.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--merge-mode", choices=["mul", "add"], default="mul")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return ExpConfig(
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
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | merge_mode={config.merge_mode} | seed={config.seed}")

    train_ds, val_ds = build_teacher_dataset(config, device)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    baseline = TinyClassifier(
        input_dim=config.input_dim,
        seq_len=config.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=False,
        merge_mode=config.merge_mode,
        use_cuda_dtw=False,
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
    ).to(device)

    special = TinyClassifier(
        input_dim=config.input_dim,
        seq_len=config.seq_len,
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        special_second_block=True,
        merge_mode=config.merge_mode,
        use_cuda_dtw=device.type == "cuda",
        udtw_gamma=config.udtw_gamma,
        udtw_beta=config.udtw_beta,
        sigma_hidden_dim=config.sigma_hidden_dim,
    ).to(device)

    print("teacher-generated synthetic dataset: labels come from a frozen 2-layer model with the specialized second block.")
    baseline_best = train_model("baseline", baseline, train_loader, val_loader, device, config)
    special_best = train_model("special ", special, train_loader, val_loader, device, config)

    print(
        "summary | "
        f"baseline_best_acc={baseline_best['acc']:.3f} baseline_best_loss={baseline_best['loss']:.4f} | "
        f"special_best_acc={special_best['acc']:.3f} special_best_loss={special_best['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
