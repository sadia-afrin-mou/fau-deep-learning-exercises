import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, dropout=0.0):
        super().__init__()
        if backbone == 'resnet18':
            m = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            m = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            m = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_feats = m.fc.in_features
        head = []
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
        head.append(nn.Linear(in_feats, 2))
        self.backbone = m
        self.backbone.fc = nn.Sequential(*head)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)
        return x
