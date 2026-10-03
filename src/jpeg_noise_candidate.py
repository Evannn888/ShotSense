"""Experimental filtered/shared brightness curves with bounded color correction."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.jpeg_curve_candidate import CurveCandidate, LOG_SPAN, apply_curve


def prefilter(rgb):
    return cv2.bilateralFilter(rgb, 5, 8, 2)


class NoiseCandidate(CurveCandidate):
    def __init__(self, mean, scale, constant=False):
        super().__init__(mean, scale, constant)
        if constant:
            self.logits = nn.Parameter(torch.zeros(4))
        else:
            self.head[-1] = nn.Linear(32, 4)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def curve_parameters(self, features):
        logits = self.logits.expand(len(features), 4) if self.constant else self.head((features-self.mean)/self.scale)
        return torch.exp(torch.tanh(logits[:, :1])*LOG_SPAN), .15*torch.tanh(logits[:, 1:])

    def forward(self, source, features):
        gamma, color = self.curve_parameters(features)
        peak = source.amax(1, keepdim=True)
        safe = peak.clamp_min(1e-8)
        limited = torch.minimum(safe**gamma[:, :, None, None], 8*peak)
        value = source*(limited/safe)
        return value + color[:, :, None, None]*value*(1-value)


def render_noise_candidate(rgb, gamma, color, strength=1., smooth=True):
    original = apply_curve(rgb, np.ones(3,dtype=np.float32), 0)
    color = np.asarray(color, dtype=np.float32)
    if (not np.isfinite(gamma) or not .15-1e-6 <= gamma <= 1/.15+1e-6
            or color.shape != (3,) or not np.isfinite(color).all() or (np.abs(color)>.150001).any()
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid shared gamma, bounded color or strength')
    if strength == 0: return original
    filtered = prefilter(original) if smooth else original
    source = filtered.astype(np.float32)/255
    peak = source.max(-1, keepdims=True)
    safe = np.maximum(peak, 1e-8)
    value = source*(np.minimum(safe**gamma,8*peak)/safe)
    value = value+color*value*(1-value)
    adjusted = (1-strength)*(original.astype(np.float32)/255)+strength*value
    return np.rint(np.clip(adjusted,0,1)*255).astype(np.uint8)


def quality_loss(output, target, original, color):
    weights = output.new_tensor([.2126,.7152,.0722])[None,:,None,None]
    luma = (output*weights).sum(1,keepdim=True)
    reference = (target*weights).sum(1,keepdim=True)
    gradient = (torch.abs(torch.diff(luma,dim=-1)-torch.diff(reference,dim=-1)).mean()
                +torch.abs(torch.diff(luma,dim=-2)-torch.diff(reference,dim=-2)).mean())/2
    chroma = ((output-luma)-(target-reference)).square().mean()
    flat = (F.max_pool2d(reference,3,1,1)+F.max_pool2d(-reference,3,1,1)<.02)&(original.amax(1,keepdim=True)<.1)
    residual = output-F.avg_pool2d(output,3,1,1,count_include_pad=False)
    variation = (residual.square()*flat).sum()/(flat.sum()*3).clamp_min(1)
    return ((output-target).square().mean()+.1*gradient+.1*chroma+.02*variation+.01*color.square().mean())
