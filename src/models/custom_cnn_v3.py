import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super().__init__()
        # Branch 1: 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3 Depthwise Separable
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 -> 5x5 Depthwise Separable
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Branch 4: Pool -> 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = self.b4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

class CustomCNN_V3(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1
        self.block1 = MultiScaleBlock(64, out_1x1=32, red_3x3=48, out_3x3=64, red_5x5=16, out_5x5=32, out_pool=32)
        # 32+64+32+32 = 160
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2
        self.block2 = MultiScaleBlock(160, out_1x1=64, red_3x3=64, out_3x3=96, red_5x5=32, out_5x5=64, out_pool=64)
        # 64+96+64+64 = 288
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 3
        self.block3 = MultiScaleBlock(288, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_pool=96)
        # 128+192+96+96 = 512
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
