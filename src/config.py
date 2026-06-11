"""
BIM 430 — PV Panel Defect Classification
Merkezi Konfigürasyon Dosyası
"""

import os

# ═══════════════════════════════════════════════════════════════
# VERİ YOLLARI (Dataset directory'de)
# ═══════════════════════════════════════════════════════════════

DATA_ROOT = "dataset"  # SEÇENEK A: dataset/ klasörü olduğu gibi kullan
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# ═══════════════════════════════════════════════════════════════
# ÇIKTI DİZİNLERİ
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_DIR = "outputs/checkpoints"
PLOT_EDA_DIR = "outputs/plots/eda"
PLOT_TRAIN_DIR = "outputs/plots/training"
PLOT_EVAL_DIR = "outputs/plots/evaluation"
RESULT_DIR = "outputs/results"

# Otomatik oluştur
for dir_path in [CHECKPOINT_DIR, PLOT_EDA_DIR, PLOT_TRAIN_DIR, PLOT_EVAL_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SINIF DEFİNİSYONLARI
# ═══════════════════════════════════════════════════════════════

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]
NUM_CLASSES = 6
IMG_SIZE = 224

# ═══════════════════════════════════════════════════════════════
# VERİ YÜKLEME HİPERPARAMETRELERİ
# ═══════════════════════════════════════════════════════════════

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
SEED = 42

# ═══════════════════════════════════════════════════════════════
# NORMALIZASYON DEĞERLERİ (Dataset'ten hesaplanmış)
# ═══════════════════════════════════════════════════════════════

DATASET_MEAN = [0.4220, 0.4486, 0.5041]  # PV Panel Dataset'ten hesaplanan
DATASET_STD = [0.2242, 0.2128, 0.1994]   # PV Panel Dataset'ten hesaplanan

# ═══════════════════════════════════════════════════════════════
# EĞİTİM HİPERPARAMETRELERİ
# ═══════════════════════════════════════════════════════════════

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
LR_STEP_SIZE = 10  # Her 10 epoch'ta LR ×0.5 azal
LR_GAMMA = 0.5
PATIENCE = 10  # Early stopping patience
LABEL_SMOOTHING = 0.05

# Dinamik Öğrenme Oranı (Cosine Annealing WR)
USE_COSINE_ANNEALING = True
COSINE_T0 = 10
COSINE_TMULT = 2

# MixUp & CutMix
USE_MIXUP = True
USE_CUTMIX = True
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# ═══════════════════════════════════════════════════════════════
# AUGMENTATION PARAMETRELERİ (albumentations)
# ═══════════════════════════════════════════════════════════════

AUGMENTATION_PARAMS = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.3,
    "rotate_limit": 15,
    "rotate_p": 0.4,
    "shift_limit": 0.05,
    "scale_limit": 0.1,
    "shift_scale_p": 0.3,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "brightness_contrast_p": 0.5,
    "hue_shift_limit": 10,
    "sat_shift_limit": 20,
    "val_shift_limit": 10,
    "hue_saturation_p": 0.3,
    "gaussian_blur_p": 0.2,
    "gaussian_noise_p": 0.2,
    "coarse_dropout_p": 0.2,
}

# ═══════════════════════════════════════════════════════════════
# MODEL BİLGİLERİ
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "CustomCNN": {
        "blocks": [48, 96, 192, 320, 512],
        "dropout_rates": [0.08, 0.10, 0.12, 0.15, 0.18],
        "fc_dropout": 0.35,
    },
    "VGG-Scratch": {
        "fc_hidden": 1024,
        "fc_dropout": 0.50,
    },
    "ResNet-Scratch": {
        "fc_dropout": 0.40,
    },
    "MobileNetV2-Scratch": {
        "dropout": 0.20,
    },
    "EfficientNet-Scratch": {
        "dropout": 0.30,
    },
}

# ═══════════════════════════════════════════════════════════════
# CİHAZ SEÇİMİ
# ═══════════════════════════════════════════════════════════════

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
