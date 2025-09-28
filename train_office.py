"""Training routines for Office-Home using DANN.

Mirrors logic from train.py but removes MNIST channel replication and
adds minor conveniences for larger models.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Iterable

from utils import optimizer_scheduler, set_model_mode, save_model


def _iterate_zip(source_loader, target_loader):
    for (s_img, s_lbl), (t_img, t_lbl) in zip(source_loader, target_loader):
        yield s_img, s_lbl, t_img, t_lbl


def source_only(encoder, classifier, source_train_loader, target_train_loader, source_test_loader, target_test_loader, epochs=20, lr=0.01, momentum=0.9, log_interval=50, device='cuda'):
    print("[Office-Home] Training source-only baseline")

    ce = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        set_model_mode('train', [encoder, classifier])
        start_steps = epoch * len(source_train_loader)
        total_steps = epochs * len(target_train_loader)

        for batch_idx, (s_img, s_lbl, t_img, t_lbl) in enumerate(_iterate_zip(source_train_loader, target_train_loader)):
            p = float(batch_idx + start_steps) / total_steps
            s_img = s_img.to(device)
            s_lbl = s_lbl.to(device)

            optimizer = optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()
            feat = encoder(s_img)
            logits = classifier(feat)
            loss = ce(logits, s_lbl)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print(f"  [{batch_idx * len(s_img)}/{len(source_train_loader.dataset)}] Loss={loss.item():.4f}")

        evaluate(encoder, classifier, None, source_test_loader, target_test_loader, mode='Source-only', device=device)

    save_model(encoder, classifier, None, 'OfficeHome-Source')


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, source_test_loader, target_test_loader, epochs=20, lr=0.01, momentum=0.9, log_interval=50, device='cuda'):
    print("[Office-Home] Training with DANN")

    ce = nn.CrossEntropyLoss().to(device)
    ce_domain = nn.CrossEntropyLoss().to(device)
    params = list(encoder.parameters()) + list(classifier.parameters()) + list(discriminator.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=momentum)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        set_model_mode('train', [encoder, classifier, discriminator])
        start_steps = epoch * len(source_train_loader)
        total_steps = epochs * len(target_train_loader)

        for batch_idx, (s_img, s_lbl, t_img, t_lbl) in enumerate(_iterate_zip(source_train_loader, target_train_loader)):
            s_img = s_img.to(device)
            s_lbl = s_lbl.to(device)
            t_img = t_img.to(device)

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer = optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            combined = torch.cat((s_img, t_img), 0)
            combined_features = encoder(combined)
            source_features = combined_features[:s_img.size(0)]

            class_logits = classifier(source_features)
            class_loss = ce(class_logits, s_lbl)

            domain_logits = discriminator(combined_features, alpha)
            domain_labels = torch.cat((torch.zeros(s_img.size(0), dtype=torch.long), torch.ones(t_img.size(0), dtype=torch.long)), 0).to(device)
            domain_loss = ce_domain(domain_logits, domain_labels)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print(f"  [{batch_idx * len(s_img)}/{len(source_train_loader.dataset)}] Total={total_loss.item():.4f} Class={class_loss.item():.4f} Domain={domain_loss.item():.4f}")

        evaluate(encoder, classifier, discriminator, source_test_loader, target_test_loader, mode='DANN', device=device)

    save_model(encoder, classifier, discriminator, 'OfficeHome-DANN')


@torch.no_grad()
def evaluate(encoder, classifier, discriminator, source_loader, target_loader, mode='Source-only', device='cuda'):
    set_model_mode('eval', [encoder, classifier])
    if discriminator is not None:
        set_model_mode('eval', [discriminator])

    s_correct = 0
    t_correct = 0
    d_correct = 0
    s_total = len(source_loader.dataset)
    t_total = len(target_loader.dataset)

    for batch_idx, (s_data, t_data) in enumerate(zip(source_loader, target_loader)):
        s_img, s_lbl = s_data
        t_img, t_lbl = t_data
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        t_img, t_lbl = t_img.to(device), t_lbl.to(device)

        p = float(batch_idx) / len(source_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        s_feat = encoder(s_img)
        t_feat = encoder(t_img)
        s_pred = classifier(s_feat).argmax(1)
        t_pred = classifier(t_feat).argmax(1)
        s_correct += (s_pred == s_lbl).sum().item()
        t_correct += (t_pred == t_lbl).sum().item()

        if discriminator is not None:
            combined_feat = torch.cat((s_feat, t_feat), 0)
            domain_pred = discriminator(combined_feat, alpha).argmax(1)
            domain_labels = torch.cat((torch.zeros(s_img.size(0), dtype=torch.long), torch.ones(t_img.size(0), dtype=torch.long)), 0).to(device)
            d_correct += (domain_pred == domain_labels).sum().item()

    print(f"[{mode}] Source Acc: {s_correct}/{s_total} ({100.*s_correct/s_total:.2f}%)  Target Acc: {t_correct}/{t_total} ({100.*t_correct/t_total:.2f}%)")
    if discriminator is not None:
        print(f"[{mode}] Domain Acc: {d_correct}/{s_total + t_total} ({100.*d_correct/(s_total+t_total):.2f}%)")
