"""
Dataset ve DataLoader Hazırlama
PVDataset class, augmentation pipeline, DataLoader factory
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Config import (opsiyonel, kullanılmıyor ama import et)
# from src.config import DATASET_MEAN, DATASET_STD, AUGMENTATION_PARAMS


def get_augmentation_transforms(mean, std):
    """
    Train ve val/test için augmentation pipeline'ı oluştur

    Args:
        mean: list of 3 floats [r, g, b]
        std: list of 3 floats [r, g, b]

    Returns:
        train_transform, val_test_transform (albumentations compose)
    """

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=10, p=0.25),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=0,
            p=0.2,
        ),
        # Intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2,
                        contrast_limit=0.2, p=0.35),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                    val_shift_limit=10, p=0.2),
        # Noise & blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.1),
        # Dropout
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.08, 0.16),
            hole_width_range=(0.08, 0.16),
            fill=0,
            p=0.1,
        ),
        # Normalization
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_transform, val_test_transform


class PVDataset(Dataset):
    """
    PV Panel Dataset — ImageFolder kullanarak

    albumentations kullanıyoruz, bu yüzden PIL Image'i NumPy'ye çeviriyoruz:
    PIL Image → NumPy (H, W, 3) → albumentations → torch.Tensor (3, H, W)
    """

    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]

        # PIL Image yükle ve RGB'ye çevir
        image = np.array(Image.open(path).convert("RGB"))

        # Albumentations transformation apply (eğer varsa)
        if self.transform:
            # Albumentations {"image": tensor} dict dönüyor
            image = self.transform(image=image)["image"]

        return image, label


def get_dataloaders(train_dir, val_dir, test_dir, mean, std,
                   batch_size=32, num_workers=4, pin_memory=True):
    """
    Train/Val/Test DataLoader'ı oluştur

    Args:
        train_dir, val_dir, test_dir: Dataset klasörleri
        mean, std: Normalizasyon parametreleri
        batch_size: Batch size
        num_workers: DataLoader worker sayısı
        pin_memory: CUDA için memory pinning

    Returns:
        train_loader, val_loader, test_loader
    """

    train_transform, val_test_transform = get_augmentation_transforms(mean, std)

    train_dataset = PVDataset(train_dir, transform=train_transform)
    val_dataset = PVDataset(val_dir, transform=val_test_transform)
    test_dataset = PVDataset(test_dir, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(train_dir, classes):
    """
    Class weight hesapla (CrossEntropyLoss'a verilecek)

    Weight = total_samples / (num_classes * samples_in_class)
    Veya uniform: weight ∝ 1 / sample_count

    Returns:
        torch.Tensor of shape (num_classes,)
    """

    from collections import Counter

    counts = Counter()
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        counts[cls] = len(os.listdir(cls_dir))

    total = sum(counts.values())
    num_classes = len(counts)

    # Weight = total / (num_classes * count)
    weights = {cls: total / (num_classes * cnt)
              for cls, cnt in counts.items()}

    # Sıralanmış sınıf sırasına göre tensor oluştur
    weight_tensor = torch.tensor(
        [weights[c] for c in sorted(weights.keys())],
        dtype=torch.float
    )

    return weight_tensor
