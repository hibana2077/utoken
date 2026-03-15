import argparse
import random
from typing import Callable, Dict, Iterable

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import build_cifar10_loaders
from .factory import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def compute_ece(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 15,
) -> float:
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = confidences.new_zeros(())
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            bin_acc = correctness[in_bin].float().mean()
            bin_conf = confidences[in_bin].mean()
            bin_weight = in_bin.float().mean()
            ece = ece + (bin_conf - bin_acc).abs() * bin_weight
    return float(ece.item())


def _denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return images * std + mean


def _normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _gaussian_blur_images(images: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.expand(images.size(1), 1, kernel_size, kernel_size)
    padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_2d, groups=images.size(1))


def make_corruption_fn(name: str, severity: int) -> Callable[[torch.Tensor], torch.Tensor]:
    severity = max(1, min(5, severity))
    if name == "gaussian_noise":
        sigma = 0.04 * severity

        def fn(images: torch.Tensor) -> torch.Tensor:
            x = _denormalize_cifar10(images)
            x = x + torch.randn_like(x) * sigma
            x = x.clamp(0.0, 1.0)
            return _normalize_cifar10(x)

        return fn
    if name == "gaussian_blur":
        kernel = 2 * severity + 1
        sigma = 0.4 + 0.3 * severity

        def fn(images: torch.Tensor) -> torch.Tensor:
            x = _denormalize_cifar10(images)
            x = _gaussian_blur_images(x, kernel_size=kernel, sigma=sigma)
            x = x.clamp(0.0, 1.0)
            return _normalize_cifar10(x)

        return fn
    if name == "brightness":
        factor = 1.0 - 0.12 * severity

        def fn(images: torch.Tensor) -> torch.Tensor:
            x = _denormalize_cifar10(images)
            x = (x * factor).clamp(0.0, 1.0)
            return _normalize_cifar10(x)

        return fn
    if name == "contrast":
        factor = 1.0 - 0.15 * severity

        def fn(images: torch.Tensor) -> torch.Tensor:
            x = _denormalize_cifar10(images)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = ((x - mean) * factor + mean).clamp(0.0, 1.0)
            return _normalize_cifar10(x)

        return fn
    raise ValueError(f"Unsupported corruption: {name}")


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    aux_weight: float,
    ece_bins: int,
    corruption_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_brier = 0.0
    total_correct = 0
    total = 0
    all_confidences = []
    all_correctness = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if corruption_fn is not None:
                images = corruption_fn(images)
            logits, stats = model(images)
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss + aux_weight * stats["aux_loss"]
            probs = F.softmax(logits, dim=-1)
            pred_conf, preds = probs.max(dim=-1)
            correct = preds.eq(labels)
            labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
            brier = (probs - labels_one_hot).pow(2).sum(dim=1).mean()

            total_loss += loss.item() * labels.size(0)
            total_nll += ce_loss.item() * labels.size(0)
            total_brier += brier.item() * labels.size(0)
            total_correct += correct.sum().item()
            total += labels.size(0)
            all_confidences.append(pred_conf)
            all_correctness.append(correct)

    confidences = torch.cat(all_confidences, dim=0)
    correctness = torch.cat(all_correctness, dim=0)
    avg_conf = float(confidences.mean().item())
    acc = total_correct / total
    return {
        "loss": total_loss / total,
        "acc": acc,
        "nll": total_nll / total,
        "ece": compute_ece(confidences, correctness, n_bins=ece_bins),
        "brier": total_brier / total,
        "avg_conf": avg_conf,
        "conf_gap": avg_conf - acc,
    }


def evaluate_corruption_robustness(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    aux_weight: float,
    ece_bins: int,
    corruption_names: Iterable[str],
    severity: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    acc_values = []
    for name in corruption_names:
        fn = make_corruption_fn(name, severity)
        m = evaluate(
            model=model,
            loader=loader,
            device=device,
            aux_weight=aux_weight,
            ece_bins=ece_bins,
            corruption_fn=fn,
        )
        metrics[f"acc_{name}"] = m["acc"]
        metrics[f"nll_{name}"] = m["nll"]
        metrics[f"ece_{name}"] = m["ece"]
        acc_values.append(m["acc"])

    metrics["acc_corruption_mean"] = sum(acc_values) / max(len(acc_values), 1)
    return metrics


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
                    "sigma_attn_mean",
                    "sigma_attn_max",
                    "sigma_attn_min",
                    "sigma_attn_std",
                    "sigma_mlp_mean",
                    "sigma_mlp_max",
                    "sigma_mlp_min",
                    "sigma_mlp_std",
                ]:
                    if key in stats:
                        last_sigma_stats[key] = float(stats[key].item())

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate(model, val_loader, device, cfg.aux_weight, cfg.ece_bins)
        corruption_metrics: Dict[str, float] = {}
        if cfg.eval_corruptions:
            corruption_metrics = evaluate_corruption_robustness(
                model=model,
                loader=val_loader,
                device=device,
                aux_weight=cfg.aux_weight,
                ece_bins=cfg.ece_bins,
                corruption_names=("gaussian_noise", "gaussian_blur", "brightness", "contrast"),
                severity=cfg.corruption_severity,
            )
        best_val_acc = max(best_val_acc, val_metrics["acc"])
        line = (
            f"{variant:8s} | epoch={epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.3f} | "
            f"val_nll={val_metrics['nll']:.4f} | val_ece={val_metrics['ece']:.4f} | "
            f"val_brier={val_metrics['brier']:.4f} | val_conf_gap={val_metrics['conf_gap']:.4f} | "
            f"aux={last_aux:.4f}"
        )
        if corruption_metrics:
            corr_drop = val_metrics["acc"] - corruption_metrics["acc_corruption_mean"]
            line += (
                f" | corr_acc_mean={corruption_metrics['acc_corruption_mean']:.3f}"
                f" | corr_drop={corr_drop:.3f}"
                f" | noise_acc={corruption_metrics['acc_gaussian_noise']:.3f}"
                f" | blur_acc={corruption_metrics['acc_gaussian_blur']:.3f}"
                f" | bright_acc={corruption_metrics['acc_brightness']:.3f}"
                f" | contrast_acc={corruption_metrics['acc_contrast']:.3f}"
            )
        if variant == "special" and last_sigma_stats:
            line += (
                f" | sigma_attn_mean={last_sigma_stats.get('sigma_attn_mean', 0.0):.3f}"
                f" | sigma_attn_max={last_sigma_stats.get('sigma_attn_max', 0.0):.3f}"
                f" | sigma_attn_min={last_sigma_stats.get('sigma_attn_min', 0.0):.3f}"
                f" | sigma_attn_std={last_sigma_stats.get('sigma_attn_std', 0.0):.3f}"
                f" | sigma_mlp_mean={last_sigma_stats.get('sigma_mlp_mean', 0.0):.3f}"
                f" | sigma_mlp_max={last_sigma_stats.get('sigma_mlp_max', 0.0):.3f}"
                f" | sigma_mlp_min={last_sigma_stats.get('sigma_mlp_min', 0.0):.3f}"
                f" | sigma_mlp_std={last_sigma_stats.get('sigma_mlp_std', 0.0):.3f}"
            )
        print(line)
    return best_val_acc


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} | model={cfg.model_name} | dataset=cifar10 | "
        f"train_mode={cfg.train_mode}"
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
    parser.add_argument("--merge-mode", type=str, choices=["mul", "add"], default="mul")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--skip-corruption-eval", action="store_true")
    parser.add_argument("--corruption-severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        ece_bins=args.ece_bins,
        eval_corruptions=not args.skip_corruption_eval,
        corruption_severity=args.corruption_severity,
    )


if __name__ == "__main__":
    train(parse_args())
