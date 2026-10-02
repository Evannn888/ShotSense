"""Frozen EfficientNet-B0 plus standardized physical features and a small head."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_NAME = 'EfficientNet_B0_Weights.IMAGENET1K_V1'


class ShotSenseModel(nn.Module):
    def __init__(self, physical_mean=None, physical_scale=None, pretrained=True):
        super().__init__()
        torch.hub.set_dir(str(ROOT / 'artifacts/torch'))
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.classifier = nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        mean = np.zeros(132,dtype=np.float32) if physical_mean is None else np.asarray(physical_mean,dtype=np.float32)
        scale = np.ones(132,dtype=np.float32) if physical_scale is None else np.asarray(physical_scale,dtype=np.float32)
        if mean.shape != (132,) or scale.shape != (132,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid training physical standardization statistics')
        self.register_buffer('physical_mean',torch.from_numpy(mean.copy()))
        self.register_buffer('physical_scale',torch.from_numpy(scale.copy()))
        self.head=nn.Sequential(nn.Linear(1412,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,6),nn.Tanh())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def fused_features(self, semantic, physical_raw):
        return torch.cat((semantic,(physical_raw-self.physical_mean)/self.physical_scale),dim=1)

    def forward(self, image, physical_raw):
        return self.head(self.fused_features(self.backbone(image),physical_raw))
