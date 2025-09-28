"""Models for Office-Home domain adaptation.

Uses a ResNet-50 backbone as feature extractor. Exposes:
  OfficeExtractor  - outputs flattened feature vector
  OfficeClassifier - MLP head for class prediction
  OfficeDiscriminator - domain discriminator with GRL

The ReverseLayerF is reused from utils.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from utils import ReverseLayerF


def _load_resnet50(pretrained: bool = True) -> nn.Module:
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    # Remove classification head
    backbone.fc = nn.Identity()
    return backbone


class OfficeExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = _load_resnet50(pretrained=pretrained)
        self.feature_dim = 2048  # ResNet-50 final embedding

    def forward(self, x):
        return self.backbone(x)


class OfficeClassifier(nn.Module):
    def __init__(self, in_dim: int = 2048, num_classes: int = 65, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class OfficeDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, alpha: float):
        x = ReverseLayerF.apply(x, alpha)
        return self.net(x)


def build_models(num_classes: int = 65, pretrained_backbone: bool = True):
    extractor = OfficeExtractor(pretrained=pretrained_backbone)
    classifier = OfficeClassifier(in_dim=extractor.feature_dim, num_classes=num_classes)
    discriminator = OfficeDiscriminator(in_dim=extractor.feature_dim)
    return extractor, classifier, discriminator


if __name__ == "__main__":
    m, c, d = build_models()
    x = torch.randn(2, 3, 224, 224)
    f = m(x)
    print("Feature shape", f.shape)
    print("Class logits", c(f).shape, "Domain logits", d(f, 1.0).shape)
