import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


def get_resnet18(num_classes=10, dropout_rate=0.3):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 inputs).
    Adds a deeper multi-layer classification head with BatchNorm and Dropout
    for improved accuracy and regularization.
    """
    model = resnet18(weights=None)
    # Adapt for 32x32 CIFAR-10 inputs (instead of 224x224 ImageNet)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace the single FC layer with a deeper head for better discrimination
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(256, num_classes),
    )
    return model


def get_resnet50(num_classes=10, dropout_rate=0.4, pretrained_imagenet=False):
    """
    ResNet-50 with a deeper classification head.
    Supports both CIFAR-10 (32x32) and ImageNet-scale (224x224) inputs.
    Use pretrained_imagenet=True when auditing TrojAI-style real-world models.
    """
    weights = "IMAGENET1K_V2" if pretrained_imagenet else None
    model = resnet50(weights=weights)

    if num_classes == 10:
        # Adapt for 32x32 CIFAR-10 inputs
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    # Deep multi-layer head with skip-style residual projection
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )
    return model


def get_model(arch='resnet18', num_classes=10, **kwargs):
    """
    Factory function to instantiate a model by architecture name.
    Supported: 'resnet18', 'resnet50'
    """
    registry = {
        'resnet18': get_resnet18,
        'resnet50': get_resnet50,
    }
    if arch not in registry:
        raise ValueError(f"Unknown architecture: '{arch}'. Supported: {list(registry.keys())}")
    return registry[arch](num_classes=num_classes, **kwargs)
