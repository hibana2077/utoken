from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    workers: int,
) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader

