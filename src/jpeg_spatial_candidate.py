"""Small experimental spatial residual over a frozen source-only base pipeline."""
import numpy as np
import torch
from torch import nn


class SpatialCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(6,12,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(12,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,3,3,padding=1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self,source_and_base):
        return (source_and_base[:,3:]+.5*torch.tanh(self.layers(source_and_base))).clamp(0,1)


def predict_spatial(model,rgb,base,strength=1.):
    if (rgb.dtype!=np.uint8 or base.dtype!=np.uint8 or rgb.shape!=base.shape
            or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape[:2])<1
            or not np.isfinite(strength) or not 0<=strength<=1):
        raise ValueError('Expected aligned RGB uint8 source/base and bounded strength')
    if strength==0:return rgb.copy()
    x=torch.tensor(np.concatenate([rgb,base],axis=-1).transpose(2,0,1)[None].astype(np.float32)/255)
    with torch.no_grad():value=model(x)[0].permute(1,2,0).numpy()
    if not np.isfinite(value).all():raise ValueError('Non-finite spatial model output')
    return np.rint(np.clip((1-strength)*(rgb.astype(np.float32)/255)+strength*value,0,1)*255).astype(np.uint8)
