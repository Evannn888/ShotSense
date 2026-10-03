"""Fixed denoising followed by bounded shared illumination; research only."""
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def denoise_base(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 RGB')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 7, 10, 7, 21), cv2.COLOR_BGR2RGB)


class IlluminationCandidate(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.constant = constant
        if constant:
            self.logit = nn.Parameter(torch.zeros(1))
        else:
            self.layers = nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),
                                        nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,1,3,padding=1))
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def gain(self, base):
        if self.constant:
            logits = self.logit[None,:,None,None]
        else:
            size = tuple(max(1,s//4) for s in base.shape[-2:])
            small = F.interpolate(base, size=size, mode='area')
            logits = F.interpolate(self.layers(small),size=base.shape[-2:],mode='bilinear',align_corners=False)
        return torch.exp(math.log(4)*torch.tanh(logits))

    def forward(self, base):
        value = -torch.expm1(self.gain(base)*torch.log1p(-base.clamp(max=1-1e-6)))
        return torch.where(base == 1, torch.ones_like(value), value).clamp(0,1)


def predict_illumination(model, original, denoised, strength=1.):
    if (original.dtype != np.uint8 or denoised.dtype != np.uint8 or original.shape != denoised.shape
            or original.ndim != 3 or original.shape[-1] != 3 or min(original.shape[:2]) < 1
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Expected aligned uint8 RGB and bounded strength')
    if strength == 0:
        return original.copy()
    x = torch.tensor(denoised.transpose(2,0,1)[None].astype(np.float32)/255)
    with torch.no_grad():
        value = model(x)[0].permute(1,2,0).numpy()
    if not np.isfinite(value).all():
        raise ValueError('Non-finite illumination output')
    value = (1-strength)*original.astype(np.float32)/255+strength*value
    return np.rint(np.clip(value,0,1)*255).astype(np.uint8)
