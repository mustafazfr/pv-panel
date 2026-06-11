import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # weights=None explicitly loads NO pretrained weights (training from scratch)
        self.model = efficientnet_b0(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)
