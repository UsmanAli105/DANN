"""Entry point for Office-Home domain adaptation with DANN.

Example (PowerShell):
  python main_office_home.py --data-root D:/Datasets/OfficeHome \
      --source Art --target Product --epochs 20 --batch-size 16 --method dann

Methods: source (source-only) | dann (domain adversarial)
"""
from __future__ import annotations

import argparse
import torch

import office_home
from model_office import build_models
import train_office
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def parse_args():
    p = argparse.ArgumentParser(description="Office-Home DANN")
    p.add_argument('--data-root', required=True, help='Path to Office-Home dataset root')
    p.add_argument('--source', default='Art', help='Source domain')
    p.add_argument('--target', default='Product', help='Target domain')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--no-pretrained', action='store_true', help='Disable pretrained ResNet weights')
    p.add_argument('--method', choices=['source', 'dann'], default='dann')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', default='DANN-OfficeHome', help='W&B project name')
    p.add_argument('--wandb-run-name', default=None, help='Optional custom run name')
    p.add_argument('--wandb-group', default=None, help='Optional W&B run group')
    return p.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Source={args.source} Target={args.target} Method={args.method}")

    src_train, tgt_train, src_test, tgt_test, num_classes = office_home.get_office_home_loaders(
        root=args.data_root,
        source_domain=args.source,
        target_domain=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encoder, classifier, discriminator = build_models(num_classes=num_classes, pretrained_backbone=not args.no_pretrained)
    encoder.to(device)
    classifier.to(device)
    discriminator.to(device)

    run = None
    if args.wandb and _WANDB:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config={
                'source_domain': args.source,
                'target_domain': args.target,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'momentum': args.momentum,
                'method': args.method,
                'pretrained': not args.no_pretrained
            }
        )

    if args.method == 'source':
        train_office.source_only(
            encoder, classifier,
            src_train, tgt_train,
            src_test, tgt_test,
            epochs=args.epochs, lr=args.lr, momentum=args.momentum,
            log_interval=args.log_interval, device=device,
            wandb_run=run
        )
    else:
        train_office.dann(
            encoder, classifier, discriminator,
            src_train, tgt_train,
            src_test, tgt_test,
            epochs=args.epochs, lr=args.lr, momentum=args.momentum,
            log_interval=args.log_interval, device=device,
            wandb_run=run
        )

    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
