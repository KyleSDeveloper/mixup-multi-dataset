from typing import Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Use ImageNet normalization (robust across torchvision versions)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _resolve_root(root: str | Path) -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _common_transforms(image_size: int, weights=None, train: bool = True):
    # Ignore `weights` for normalization to avoid API changes in torchvision
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

def get_loaders(
    dataset: str,
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    weights=None,          # kept for signature compatibility; not used
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = dataset.lower()
    root = _resolve_root(data_root)

    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(
            root=root, train=True, download=True,
            transform=_common_transforms(image_size, weights, train=True)
        )
        test_ds = datasets.CIFAR10(
            root=root, train=False, download=True,
            transform=_common_transforms(image_size, weights, train=False)
        )
        num_classes = 10

    elif dataset == "svhn":
        train_ds = datasets.SVHN(
            root=root, split="train", download=True,
            transform=_common_transforms(image_size, weights, train=True)
        )
        test_ds = datasets.SVHN(
            root=root, split="test", download=True,
            transform=_common_transforms(image_size, weights, train=False)
        )
        num_classes = 10

    elif dataset == "pets":
        # Oxford-IIIT Pet (category labels)
        train_ds = datasets.OxfordIIITPet(
            root=root, split="trainval", download=True,
            transform=_common_transforms(image_size, weights, train=True),
            target_types="category"
        )
        test_ds = datasets.OxfordIIITPet(
            root=root, split="test", download=True,
            transform=_common_transforms(image_size, weights, train=False),
            target_types="category"
        )
        num_classes = 37

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Build validation split from train (10%)
    g = torch.Generator().manual_seed(seed)
    val_size = max(1, int(0.1 * len(train_ds)))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes

