from typing import Dict, Optional

import timm
import layer
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        model_name: str,
        in_channels: int = 1,
        pooling_name: str = "GeM",
        args_pooling: Optional[Dict] = None,
    ):
        super(CustomModel, self).__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=in_channels
        )
        final_in_features = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = getattr(layer, pooling_name)(**args_pooling)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
