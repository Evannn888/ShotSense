"""Experimental last-MBConv-stage tuning; production model is unchanged."""
from torch import nn

from src.model import ShotSenseModel


class FineTuneCandidate(ShotSenseModel):
    def __init__(self, physical_mean=None, physical_scale=None):
        super().__init__(physical_mean,physical_scale,pretrained=False)
        # Installed torchvision EfficientNet-B0 features.7 is the last MBConv stage.
        self.backbone.features[7].requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.features[7].train(mode)
        for module in self.backbone.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def optimizer_groups(self):
        return [{'params':self.backbone.features[7].parameters(),'lr':1e-5},
                {'params':self.head.parameters(),'lr':1e-4}]

    def forward_tail(self, frozen_prefix, physical_raw):
        features=self.backbone.features[7](frozen_prefix)
        features=self.backbone.features[8](features)
        semantic=self.backbone.avgpool(features).flatten(1)
        return self.head(self.fused_features(semantic,physical_raw))
