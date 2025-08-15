from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> Tuple[nn.Module, int]:
    if model_name.lower() == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            net = models.resnet18(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Linear(feat_dim, num_classes)
            return net, weights
        else:
            net = models.resnet18(weights=None)
            feat_dim = net.fc.in_features
            net.fc = nn.Linear(feat_dim, num_classes)
            return net, None
    else:
        raise ValueError(f"Unsupported model: {model_name}")
