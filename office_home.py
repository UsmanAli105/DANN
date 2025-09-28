"""Office-Home dataset helpers for Domain Adaptation.

This file is OPTIONAL and not used by the current MNIST -> MNIST-M DANN
experiment. It provides convenience utilities to adapt the existing
training code to the Office-Home dataset (65 classes, 4 domains:
Art, Clipart, Product, Real World).

Usage example (interactive / new script):

    from office_home import get_office_home_loaders
    import model, train

    src_train, tgt_train, src_test, tgt_test, num_classes = \
        get_office_home_loaders(
            root="D:/Datasets/OfficeHome",  # point to extracted dataset root
            source_domain="Art",
            target_domain="Product",
            batch_size=32,
            image_size=224
        )

    # Rebuild model with correct output dimension (65 for Office-Home)
    encoder = model.Extractor()
    classifier = model.Classifier(num_classes=num_classes)
    discriminator = model.Discriminator()
    # (Adjust Extractor/Classifer architectures for higher-res images; current
    # simple conv net is tailored to 28x28. Prefer a pretrained ResNet.)

NOTE: You must manually download and extract the Office-Home dataset:
    http://hemanthdv.org/OfficeHome-Dataset/ (Approx 2.7GB)

Directory structure after extraction should look like:
    <root>/Art/Alarm_Clock/xxx.jpg
    <root>/Art/Backpack/yyy.jpg
    ...
    <root>/Clipart/...
    <root>/Product/...
    <root>/Real World/...

This helper will create random train/validation splits; you can replace
them with deterministic splits if desired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DOMAINS = ["Art", "Clipart", "Product", "Real World"]


def build_transforms(image_size: int = 224, augment: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats (reasonable default)
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def get_office_home_loaders(
    root: str,
    source_domain: str,
    target_domain: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augment: bool = True,
    val_fraction: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    """Return DANN-style loaders for Office-Home.

    Returns source_train, target_train, source_test, target_test, num_classes
    (Here test loaders are simply validation splits of each domain.)
    """
    if source_domain not in DOMAINS or target_domain not in DOMAINS:
        raise ValueError(f"Domains must be in {DOMAINS}")
    if source_domain == target_domain:
        raise ValueError("Source and target domains must differ.")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Provided root '{root}' does not exist. Point to extracted Office-Home root.")

    train_tf, eval_tf = build_transforms(image_size=image_size, augment=augment)

    # ImageFolder builds class-to-index mapping automatically (shared across domains)
    src_full = datasets.ImageFolder(str(root_path / source_domain), transform=train_tf)
    tgt_full = datasets.ImageFolder(str(root_path / target_domain), transform=train_tf)
    num_classes = len(src_full.classes)
    if num_classes != 65:
        print(f"[office_home] Detected {num_classes} classes (expected 65). Proceeding anyway.")

    def split_dataset(full_ds):
        val_len = max(1, int(len(full_ds) * val_fraction))
        train_len = len(full_ds) - val_len
        return random_split(full_ds, [train_len, val_len])

    src_train_ds, src_val_ds = split_dataset(src_full)
    tgt_train_ds, tgt_val_ds = split_dataset(tgt_full)

    # For evaluation loaders we use eval transforms
    src_val_ds.dataset.transform = eval_tf
    tgt_val_ds.dataset.transform = eval_tf

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    src_train_loader = DataLoader(src_train_ds, **kwargs)
    tgt_train_loader = DataLoader(tgt_train_ds, **kwargs)
    # For test/eval we disable shuffle
    test_kwargs = dict(batch_size=batch_size, num_workers=num_workers, shuffle=False)
    src_test_loader = DataLoader(src_val_ds, **test_kwargs)
    tgt_test_loader = DataLoader(tgt_val_ds, **test_kwargs)

    return src_train_loader, tgt_train_loader, src_test_loader, tgt_test_loader, num_classes


if __name__ == "__main__":  # quick smoke test (won't run without dataset)
    try:
        loaders = get_office_home_loaders(
            root="./OfficeHome", source_domain="Art", target_domain="Product", batch_size=2
        )
        for i, (x, y) in enumerate(loaders[0]):
            print("Source batch shape", x.shape)
            if i > 1:
                break
    except Exception as e:
        print("(Expected if dataset missing)", e)
