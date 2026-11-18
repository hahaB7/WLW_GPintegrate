"""DeepLab v3+ (ResNet-101) student of PIPELINE/SEGNET, inference only.

A copy of ``SegNet`` of PIPELINE/SEGNET/utils/Network.py (Chen et al. 2018) with the pretrained-weights
branch removed: the checkpoint holds every weight, so nothing is read from outside ``reference/person_c``.
Input (B, 3, S, S) float 0..1; output (B, K, S / 4, S / 4) logits. The state-dict keys must match the
checkpoint, so the module names are those of the original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RATES = (12, 24, 36)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OUT_STRIDE = 4


def conv(i: int, o: int, k: int = 3, d: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(self, i: int = 2048, o: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList([conv(i, o, 1)] + [conv(i, o, 3, d) for d in RATES])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv(o * (len(RATES) + 2), o, 1), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([b(x) for b in self.branches] + [pooled], 1))


def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class SegNet(nn.Module):
    out_stride = OUT_STRIDE

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        from torchvision.models import resnet101

        r = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.body1 = nn.Sequential(r.maxpool, r.layer1)
        self.body2 = nn.Sequential(r.layer2, r.layer3, r.layer4)
        self.aspp = ASPP()
        self.low4 = conv(256, 48, 1)
        self.dec4 = nn.Sequential(conv(256 + 48, 256), conv(256, 256))
        self.head = nn.Conv2d(256, n_classes, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - self.mean) / self.std
        c4 = self.body1(self.stem(x))
        y = self.aspp(self.body2(c4))
        return self.head(self.dec4(torch.cat([_up(y, c4), self.low4(c4)], 1)))
