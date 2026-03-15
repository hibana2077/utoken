import argparse
import random
from typing import Dict

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import build_cifar10_loaders
from .factory import create_model

SEQUENCE_PAIRS = ("em", "ml", "el")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    aux_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_brier = 0.0
    total_conf = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, stats = model(images)
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss + aux_weight * stats["aux_loss"]
            probs = F.softmax(logits, dim=-1)
            pred_conf, preds = probs.max(dim=-1)
            correct = preds.eq(labels)
            labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
            brier = (probs - labels_one_hot).pow(2).sum(dim=1).mean()

            total_loss += loss.item() * labels.size(0)
            total_brier += brier.item() * labels.size(0)
            total_correct += correct.sum().item()
            total += labels.size(0)
            total_conf += pred_conf.sum().item()

    avg_conf = total_conf / total
    acc = total_correct / total
    return {
        "loss": total_loss / total,
        "acc": acc,
        "brier": total_brier / total,
        "avg_conf": avg_conf,
        "conf_gap": avg_conf - acc,
    }


def train_one(
    cfg: TrainConfig,
    variant: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    set_seed(cfg.seed)
    model = create_model(cfg, device, variant=variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        last_aux = 0.0
        last_sigma_stats: Dict[str, float] = {}
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(images)
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss + cfg.aux_weight * stats["aux_loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=-1) == labels).sum().item()
            running_total += labels.size(0)
            last_aux = float(stats["aux_loss"].item())
            if variant == "special":
                for key in [
                    "sigma_mean",
                    "sigma_max",
                    "sigma_min",
                    "sigma_std",
                ]:
                    if key in stats:
                        last_sigma_stats[key] = float(stats[key].item())

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate(model, val_loader, device, cfg.aux_weight)
        best_val_acc = max(best_val_acc, val_metrics["acc"])
        line = (
            f"{variant:8s} | epoch={epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.3f} | "
            f"val_brier={val_metrics['brier']:.4f} | val_conf_gap={val_metrics['conf_gap']:.4f} | "
            f"aux={last_aux:.4f}"
        )
        if variant == "special" and last_sigma_stats:
            line += (
                f" | sigma_mean={last_sigma_stats.get('sigma_mean', 0.0):.3f}"
                f" | sigma_max={last_sigma_stats.get('sigma_max', 0.0):.3f}"
                f" | sigma_min={last_sigma_stats.get('sigma_min', 0.0):.3f}"
                f" | sigma_std={last_sigma_stats.get('sigma_std', 0.0):.3f}"
            )
        print(line)
    return best_val_acc


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} | model={cfg.model_name} | dataset=cifar10 | "
        f"train_mode={cfg.train_mode} | sequence_pair={cfg.sequence_pair}"
    )

    train_loader, val_loader = build_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )

    variants = ["special", "baseline"] if cfg.train_mode == "both" else [cfg.train_mode]
    results: Dict[str, float] = {}
    for variant in variants:
        results[variant] = train_one(cfg, variant, train_loader, val_loader, device)

    summary_parts = [f"{name}_best_val_acc={acc:.3f}" for name, acc in results.items()]
    print("summary | " + " | ".join(summary_parts))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("Train test_pico_vit on CIFAR-10")
    parser.add_argument("--model-name", default="test_pico_vit", choices=["test_pico_vit"])
    parser.add_argument("--train-mode", default="both", choices=["special", "baseline", "both"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--sigma-hidden-dim", type=int, default=64)
    parser.add_argument("--sigma-a", type=float, default=1.5)
    parser.add_argument("--sigma-b", type=float, default=0.5)
    parser.add_argument("--merge-mode", type=str, choices=["mul", "add"], default="mul")
    parser.add_argument("--sequence-pair", type=str, choices=SEQUENCE_PAIRS, default="el")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        train_mode=args.train_mode,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        aux_weight=args.aux_weight,
        sigma_hidden_dim=args.sigma_hidden_dim,
        merge_mode=args.merge_mode,
        sequence_pair=args.sequence_pair,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    train(parse_args())
