"""Experimental source-conditioned display-sRGB curves; no RAW parameter semantics."""
import math

import numpy as np
import torch
from torch import nn


LOG_SPAN = math.log(1 / .15)


def source_features(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite HWC RGB pixels')
    luma = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    histogram = np.histogram(luma, bins=8, range=(0, 1))[0] / luma.size
    return np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
                           np.quantile(luma, [.1, .25, .5, .75, .9]), histogram]).astype(np.float32)


def apply_curve(rgb, gamma, strength=1.):
    rgb = np.asarray(rgb)
    gamma = np.asarray(gamma, dtype=np.float32)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or gamma.shape != (3,) or not np.isfinite(gamma).all()
            or (gamma < .15 - 1e-6).any() or (gamma > 1/.15 + 1e-6).any()
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid RGB, bounded gamma or strength')
    if strength == 0 or np.array_equal(gamma, np.ones(3, dtype=np.float32)):
        return rgb.copy()
    source = rgb.astype(np.float32) / 255
    adjusted = (1 - strength) * source + strength * source ** gamma
    return np.rint(np.clip(adjusted, 0, 1) * 255).astype(np.uint8)


class CurveCandidate(nn.Module):
    def __init__(self, mean, scale, constant=False):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.constant = constant
        if constant:
            self.logits = nn.Parameter(torch.zeros(3))
        else:
            self.head = nn.Sequential(nn.Linear(19, 32), nn.ReLU(), nn.Linear(32, 3))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def gamma(self, features):
        logits = self.logits.expand(len(features), 3) if self.constant else self.head((features-self.mean)/self.scale)
        return torch.exp(torch.tanh(logits) * LOG_SPAN)

    def forward(self, source, features):
        gamma = self.gamma(features)[:, :, None, None]
        # Avoid log(0) exponent gradients while retaining exact black endpoints.
        return torch.where(source > 0, source.clamp_min(1e-8) ** gamma, source)
