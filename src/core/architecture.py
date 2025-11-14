"""Arquitectura del modelo SGSNet con backbone ResNet y FPN mejorado."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from .config import Config


class DetectionHead(nn.Module):
    """Head ligero inspirado en YOLO para cada nivel de la FPN."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        hidden = in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.pred = nn.Conv2d(hidden, num_anchors * (num_classes + 5), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.pred(x)


class SGSNet(nn.Module):
    """Detector multiescala para estados de madurez de fresas."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = Config.ANCHORS_PER_SCALE

        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        except AttributeError:  # Compatibilidad con versiones antiguas
            resnet = models.resnet34(pretrained=True)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        self.lateral5 = nn.Conv2d(512, 256, 1)
        self.lateral4 = nn.Conv2d(256, 256, 1)
        self.lateral3 = nn.Conv2d(128, 256, 1)

        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)

        self.head_small = DetectionHead(256, self.num_anchors, num_classes)
        self.head_medium = DetectionHead(256, self.num_anchors, num_classes)
        self.head_large = DetectionHead(256, self.num_anchors, num_classes)

        self._initialize_weights()

    @staticmethod
    def _upsample_add(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(high, size=low.shape[-2:], mode="nearest") + low

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral5(c5)
        p4 = self._upsample_add(p5, self.lateral4(c4))
        p3 = self._upsample_add(p4, self.lateral3(c3))

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        # Refinar pirámide descendente para captar objetos grandes
        pred_small = self.head_small(p3)
        pred_medium = self.head_medium(p4)
        pred_large = self.head_large(p5)

        return pred_small, pred_medium, pred_large
