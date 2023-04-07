import torch
import torch.nn as nn
import timm


class BaselineSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(..., 6)

    def forward(self, x):
        return self.fc(x)

class HuggingFaceModel(nn.Module):
    ...


