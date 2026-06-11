"""
VGGScratch — VGG-16 Benzeri Model (Sıfırdan Eğitim)
Giriş: (B, 3, 224, 224)
Çıkış: (B, 6) logits

Mimari (VGG-16'dan uyarlanmış):
Block1: 2×Conv(64)   → MaxPool → 112×112
Block2: 2×Conv(128)  → MaxPool → 56×56
Block3: 3×Conv(256)  → MaxPool → 28×28
Block4: 3×Conv(512)  → MaxPool → 14×14
Block5: 3×Conv(512)  → MaxPool → 7×7

Orijinalden Farklar:
- BatchNorm eklendi (original eksi)
- FC: 4096→1024→1024→6 (küçük dataset için parameter azaltma)
- Dropout(0.5) FC'lerde overfitting'e karşı
"""

import torch.nn as nn


class VGGScratch(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        def make_block(in_channels, out_channels, num_convs):
            """
            Block oluşturan helper: num_convs tane Conv3×3 + BatchNorm + ReLU
            Sonuna MaxPool2d ekler.
            """
            layers = []
            for i in range(num_convs):
                in_ch = in_channels if i == 0 else out_channels
                layers += [
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # Feature extraction blocks
        self.features = nn.Sequential(
            make_block(3, 64, 2),      # Block 1: 224→112
            make_block(64, 128, 2),    # Block 2: 112→56
            make_block(128, 256, 3),   # Block 3: 56→28
            make_block(256, 512, 3),   # Block 4: 28→14
            make_block(512, 512, 3),   # Block 5: 14→7
        )

        # Classifier: 512×7×7 = 25088
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(1024, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
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
