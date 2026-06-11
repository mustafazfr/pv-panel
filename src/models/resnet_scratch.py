"""
ResNetScratch — ResNet-18 Benzeri Model (Sıfırdan Eğitim)
Giriş: (B, 3, 224, 224)
Çıkış: (B, 6) logits

Mimari:
Stem:   Conv7×7(stride=2) → BN → ReLU → MaxPool → 112×112, ch=64
Layer1: 2×BasicBlock(64→64,   stride=1) → 56×56
Layer2: 2×BasicBlock(64→128,  stride=2) → 28×28
Layer3: 2×BasicBlock(128→256, stride=2) → 14×14
Layer4: 2×BasicBlock(256→512, stride=2) → 7×7, ch=512

Özellikleri:
- Residual shortcuts: x + F(x) → derin ağlar için eğitim kolaylaştırması
- Identity vs Projection shortcut (stride/channel değişimi durumunda)
- BatchNorm + ReLU active (post-activation)
- FC: dropout + single layer sınıflandırma
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock: 2 Conv işlemi + residual bağlantı

    Yapısı:
    Conv3×3(stride) → BN → ReLU
    Conv3×3(stride=1) → BN
    ↓
    + (shortcut)
    ↓
    ReLU

    Shortcut:
    - stride=1 ve in_ch=out_ch: identity
    - Aksi takdirde: 1×1 Conv + BN
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: identity veya projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(identity)
        return self.relu(out)


class ResNetScratch(nn.Module):
    """
    ResNet-18 Benzeri Mimarı
    Residual bağlantılar sayesinde 18 layer derin bile overfitting risk daha düşük
    """

    def __init__(self, num_classes=6):
        super().__init__()

        # Stem (initial conv block)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Pooling + Classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Aynı out_channels ile num_blocks tane BasicBlock
        İlk block'a stride aktarılır (spatial reduction)
        Geri kalanlar stride=1
        """
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        """Kaiming init for conv, Xavier for linear"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
