from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {"resnet50", "resnet50v2", "alexnet", "vgg19"}


def create_model(model_name: str, num_classes: int = 2) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50v2":
        try:
            import timm
        except ImportError as exc:
            raise ValueError("Model 'resnet50v2' requires 'timm'. Install dependencies and retry.") from exc
        model = timm.create_model("resnetv2_50", pretrained=False, num_classes=num_classes, in_chans=3)
        return model
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    if model_name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(SUPPORTED_MODELS)}")
