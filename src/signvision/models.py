"""Model definitions.

``ASLNetOriginal`` reproduces the architecture from the original SignVision
notebook exactly, including its quirk (see the class docstring), so the upgraded
baseline is genuinely the same network the original result came from.
``ASLNetReLU`` changes exactly one thing about it and is used as a controlled
architecture ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLNetOriginal(nn.Module):
    """The original notebook's ``ASLNet``, preserved verbatim.

    conv(3->27,5) -> ReLU -> maxpool(4) -> conv(27->27,5) -> ReLU -> maxpool(2)
    -> flatten -> Linear(->270) -> Dropout(0.5) -> Linear(->n_classes)

    Note the quirk this class deliberately keeps: there is **no non-linearity
    between the two fully connected layers**. Dropout is not an activation, so
    at inference time ``f2(f1(x))`` collapses to a single affine map and the
    270-unit hidden layer adds no representational capacity. ``ASLNetReLU``
    isolates the effect of fixing this.
    """

    def __init__(self, n_classes: int, image_size: int = 100) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 27, 5)
        self.p1 = nn.MaxPool2d(4, 4)
        self.c2 = nn.Conv2d(27, 27, 5)
        self.p2 = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            d = torch.zeros(1, 3, image_size, image_size)
            d = self.p1(F.relu(self.c1(d)))
            d = self.p2(F.relu(self.c2(d)))
            self.flat_dim = d.view(1, -1).shape[1]
        self.f1 = nn.Linear(self.flat_dim, 270)
        self.drop = nn.Dropout(0.5)
        self.f2 = nn.Linear(270, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.p2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.f1(x))
        return self.f2(x)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last convolutional layer — the standard Grad-CAM target."""
        return self.c2


class ASLNetReLU(ASLNetOriginal):
    """``ASLNetOriginal`` with a ReLU inserted between the two FC layers.

    Single-variable change: identical conv stack, identical widths, identical
    parameter count. Only the missing non-linearity is restored.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.p2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.f1(x)))
        return self.f2(x)



class ASLNetGAP(ASLNetOriginal):
    """``ASLNetOriginal`` with the flatten replaced by global average pooling.

    Single-variable change: identical convolutional stack, identical head widths
    (``270 -> n_classes``) and the same missing FC non-linearity. The only thing
    that changes is *how the spatial map reaches the classifier*.

    ``ASLNetOriginal`` flattens the 27x10x10 map into 2,700 position-addressed
    inputs, so 96% of its parameters sit in ``f1`` and absolute position is
    encoded in that weight matrix. That is the mechanism proposed in
    ``results/*/robustness.json`` for why a 5% translation costs ~33 points while
    a comparable photometric change costs ~1. Averaging each channel over space
    discards position by construction, which turns that explanation into a
    testable prediction: if the mechanism is right, this arm should lose far less
    accuracy under translation and rotation than its flattened twin.

    It also shrinks ``f1`` from 729,270 parameters to 7,560, so the arm doubles
    as a capacity check: 35,723 total parameters against 757,433.
    """

    def __init__(self, n_classes: int, image_size: int = 100) -> None:
        super().__init__(n_classes, image_size)
        self.pooled_dim = self.c2.out_channels
        self.f1 = nn.Linear(self.pooled_dim, 270)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.p2(x)
        x = x.mean(dim=(2, 3))  # global average pool: (B, C, H, W) -> (B, C)
        x = self.drop(self.f1(x))
        return self.f2(x)


def build_model(name: str, n_classes: int, image_size: int = 100) -> nn.Module:
    name = name.lower()
    if name == "aslnet_original":
        return ASLNetOriginal(n_classes, image_size)
    if name == "aslnet_relu":
        return ASLNetReLU(n_classes, image_size)
    if name == "aslnet_gap":
        return ASLNetGAP(n_classes, image_size)
    if name == "resnet18_ft":
        return _resnet18_finetune(n_classes)
    raise ValueError(f"unknown model '{name}'")


def _resnet18_finetune(n_classes: int) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a replaced head, fine-tuned end to end.

    Only used for the optional architecture comparison; evaluated under the same
    split, optimiser, epoch budget and metrics as the CNN arms.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    m.gradcam_layer = m.layer4[-1].conv2  # type: ignore[attr-defined]
    return m
