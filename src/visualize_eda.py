"""
Exploratory Data Analysis (EDA) — 8 Görselleştirme

VIZ-1: Sınıf Dağılımı (Train / Val / Test)
VIZ-2: Sınıf Başına Örnek Görüntüler
VIZ-3: Görüntü Boyutu (Width × Height)
VIZ-4: En-Boy Oranı (Aspect Ratio)
VIZ-5: RGB Kanal Histogramları
VIZ-6: Parlaklık ve Kontrast
VIZ-7: Dataset Mean/Std Hesabı
VIZ-8: Augmentation Teknikleri Önizlemesi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
from PIL import Image
import cv2

CLASSES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered",
]


# ═══════════════════════════════════════════════════════════════
# VIZ-1: Sınıf Dağılımı Bar Chart (Train / Val / Test)
# ═══════════════════════════════════════════════════════════════


def plot_class_distribution(data_root, classes, save_dir):
    """
    Her split için sınıf başına görüntü sayısını gruplandırılmış
    bar chart ile gösterir.
    """
    splits = ["train", "val", "test"]
    counts = {s: [] for s in splits}

    for split in splits:
        for cls in classes:
            path = os.path.join(data_root, split, cls)
            counts[split].append(len(os.listdir(path)))

    x = np.arange(len(classes))
    width = 0.25
    colors = ["#3266ad", "#1D9E75", "#D85A30"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (split, color) in enumerate(zip(splits, colors)):
        bars = ax.bar(x + i * width, counts[split], width,
                     label=split.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, counts[split]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 1.5, str(val),
                   ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("-", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("Görüntü Sayısı")
    ax.set_title("Sınıf Dağılımı — Train / Val / Test", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz1_class_distribution.png", dpi=150)
    plt.close()
    print("✓ VIZ-1 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-2: Sınıf Başına Örnek Görüntüler
# ═══════════════════════════════════════════════════════════════


def plot_sample_grid(data_root, classes, split="train", n_per_class=5,
                    save_dir="outputs/plots/eda"):
    """
    Her sınıftan n_per_class örnek görüntü gösterir.
    Satır = sınıf, sütun = örnek.
    """
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2.5, n_classes * 2.5))
    fig.suptitle(f"Sınıf Başına Örnek Görüntüler ({split} seti)",
                fontsize=14, y=1.01)

    for row, cls in enumerate(classes):
        folder = os.path.join(data_root, split, cls)
        files = sorted(os.listdir(folder))[:n_per_class]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = img.resize((128, 128))
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(cls, fontsize=9, rotation=0,
                                         labelpad=60, va="center")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz2_sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ VIZ-2 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-3: Görüntü Boyutu Dağılımı
# ═══════════════════════════════════════════════════════════════


def plot_image_size_distribution(data_root, classes, save_dir="outputs/plots/eda"):
    """
    Tüm görüntülerin genişlik ve yükseklik scatter plot'u
    + marginal histogram ile gösterir.
    """
    widths, heights, labels = [], [], []

    for split in ["train", "val", "test"]:
        for cls in classes:
            folder = os.path.join(data_root, split, cls)
            for fname in os.listdir(folder):
                try:
                    img = Image.open(os.path.join(folder, fname))
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    labels.append(cls)
                except Exception:
                    pass

    widths = np.array(widths)
    heights = np.array(heights)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1],
                          height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    colors_map = {c: plt.cm.tab10(i / len(classes)) for i, c in enumerate(classes)}
    for cls in classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        ax_main.scatter(widths[idx], heights[idx],
                       alpha=0.4, s=15, label=cls, color=colors_map[cls])

    ax_main.axvline(224, color="red", linestyle="--", linewidth=1.2, label="Target: 224px")
    ax_main.axhline(224, color="red", linestyle="--", linewidth=1.2)
    ax_main.set_xlabel("Genişlik (px)")
    ax_main.set_ylabel("Yükseklik (px)")
    ax_main.legend(fontsize=7, markerscale=1.5)
    ax_main.grid(alpha=0.2)

    ax_top.hist(widths, bins=40, color="#3266ad", alpha=0.7, edgecolor="white")
    ax_top.set_ylabel("Sayı")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(heights, bins=40, color="#1D9E75", alpha=0.7,
                 orientation="horizontal", edgecolor="white")
    ax_right.set_xlabel("Sayı")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle("Görüntü Boyutu Dağılımı (Tüm Veri Seti)", fontsize=13, y=0.98)
    plt.savefig(f"{save_dir}/viz3_image_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ VIZ-3 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-4: En-Boy Oranı (Aspect Ratio)
# ═══════════════════════════════════════════════════════════════


def plot_aspect_ratio(data_root, classes, save_dir="outputs/plots/eda"):
    """
    aspect_ratio = width / height
    1.0 → kare, >1.0 → yatay, <1.0 → dikey
    """
    ratios_by_class = defaultdict(list)

    for split in ["train", "val", "test"]:
        for cls in classes:
            folder = os.path.join(data_root, split, cls)
            for fname in os.listdir(folder):
                try:
                    img = Image.open(os.path.join(folder, fname))
                    w, h = img.size
                    ratios_by_class[cls].append(w / h)
                except Exception:
                    pass

    fig, ax = plt.subplots(figsize=(11, 5))
    data_list = [ratios_by_class[c] for c in classes]
    colors_list = ["#3266ad", "#1D9E75", "#D85A30", "#9B59B6", "#E67E22", "#1ABC9C"]
    bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                   medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Kare (ratio=1.0)")
    ax.set_xticklabels([c.replace("-", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("Width / Height")
    ax.set_title("Sınıf Başına En-Boy Oranı (Aspect Ratio) Dağılımı", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz4_aspect_ratio.png", dpi=150)
    plt.close()
    print("✓ VIZ-4 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-5: RGB Kanal Histogramları
# ═══════════════════════════════════════════════════════════════


def plot_rgb_histograms(data_root, classes, split="train", n_samples=30,
                       save_dir="outputs/plots/eda"):
    """
    Her sınıf için R, G, B kanallarının ortalama piksel dağılımı.
    """
    channel_names = ["R (Kırmızı)", "G (Yeşil)", "B (Mavi)"]
    channel_colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    fig, axes = plt.subplots(len(classes), 3, figsize=(14, len(classes) * 2))
    fig.suptitle(
        f"Sınıf Başına RGB Kanal Histogramları ({split} seti, {n_samples} örnek)",
        fontsize=13)

    for row, cls in enumerate(classes):
        folder = os.path.join(data_root, split, cls)
        files = sorted(os.listdir(folder))[:n_samples]
        agg = [np.zeros(256), np.zeros(256), np.zeros(256)]

        for fname in files:
            img = np.array(Image.open(os.path.join(folder, fname)).convert("RGB"))
            for ch in range(3):
                hist, _ = np.histogram(img[:, :, ch], bins=256, range=(0, 256))
                agg[ch] += hist

        for ch in range(3):
            ax = axes[row][ch]
            ax.fill_between(range(256), agg[ch] / len(files),
                           color=channel_colors[ch], alpha=0.6)
            ax.set_xlim(0, 255)
            ax.set_yticks([])
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(channel_names[ch], fontsize=10)
            if ch == 0:
                ax.set_ylabel(cls, fontsize=8, rotation=0,
                             labelpad=75, va="center")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz5_rgb_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ VIZ-5 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-6: Parlaklık ve Kontrast
# ═══════════════════════════════════════════════════════════════


def plot_brightness_contrast(data_root, classes, split="train",
                            save_dir="outputs/plots/eda"):
    """
    Her sınıf için:
    - Ortalama piksel parlaklığı (grayscale mean)
    - Piksel standart sapması (kontrast)
    """
    brightness = defaultdict(list)
    contrast = defaultdict(list)

    for cls in classes:
        folder = os.path.join(data_root, split, cls)
        for fname in os.listdir(folder):
            try:
                img = np.array(Image.open(os.path.join(folder, fname)).convert("L"))
                brightness[cls].append(img.mean())
                contrast[cls].append(img.std())
            except Exception:
                pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors_list = ["#3266ad", "#1D9E75", "#D85A30",
                  "#9B59B6", "#E67E22", "#1ABC9C"]

    for ax, metric_dict, title, ylabel in [
        (ax1, brightness, "Ortalama Parlaklık (Grayscale Mean)", "Piksel Değeri (0–255)"),
        (ax2, contrast, "Kontrast (Piksel Std. Sapma)", "Std. Sapma"),
    ]:
        data_list = [metric_dict[c] for c in classes]
        bp = ax.boxplot(data_list, patch_artist=True,
                       medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels([c.replace("-", "\n") for c in classes], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Sınıf Başına Parlaklık ve Kontrast Analizi", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz6_brightness_contrast.png", dpi=150)
    plt.close()
    print("✓ VIZ-6 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# VIZ-7: Dataset Mean/Std Hesabı
# ═══════════════════════════════════════════════════════════════


def compute_and_plot_mean_std(train_dir, classes, save_dir="outputs/plots/eda"):
    """
    Dataset'in kendi mean/std'si hesaplanır ve görselleştirilir.
    ⚠️ BU DEĞERLERİ config.py'ye YAZMANIZ GEREKLI!
    """
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from PIL import Image

    # Transform: sadece resize ve ToTensor (normalization YOK)
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Manual DataLoader (dataset.py CircularImport'u önlemek için)
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(train_dir, transform=raw_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    mean_acc = torch.zeros(3)
    std_acc = torch.zeros(3)
    n = 0

    print("  Dataset mean/std hesaplanıyor...")
    for imgs, _ in loader:
        b = imgs.size(0)
        imgs_flat = imgs.view(b, 3, -1)
        mean_acc += imgs_flat.mean(2).sum(0)
        std_acc += imgs_flat.std(2).sum(0)
        n += b

    mean_vals = (mean_acc / n).numpy()
    std_vals = (std_acc / n).numpy()

    print(f"\n{'='*60}")
    print("🔴 ÖNEMLI: Aşağıdaki değerleri config.py'ye yazın:")
    print(f"{'='*60}")
    print(f"DATASET_MEAN = {mean_vals.tolist()}")
    print(f"DATASET_STD  = {std_vals.tolist()}")
    print(f"{'='*60}\n")

    channels = ["R", "G", "B"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    x = np.arange(3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title in [(ax1, mean_vals, "Mean"), (ax2, std_vals, "Std")]:
        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.01,
                   f"{val:.4f}", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.set_ylim(0, 1)
        ax.set_title(f"Per-Channel {title} (Train Seti)")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Dataset Normalizasyon Değerleri", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz7_mean_std.png", dpi=150)
    plt.close()
    print("✓ VIZ-7 kaydedildi.")

    return mean_vals.tolist(), std_vals.tolist()


# ═══════════════════════════════════════════════════════════════
# VIZ-8: Augmentation Teknikleri Önizlemesi
# ═══════════════════════════════════════════════════════════════


def plot_augmentation_preview(data_root, classes, save_dir="outputs/plots/eda"):
    """
    Her augmentation tekniğini orijinal görüntü yanında gösterir.
    """
    import albumentations as A

    sample_folder = os.path.join(data_root, "train", "Dusty")
    sample_path = os.path.join(sample_folder,
                              sorted(os.listdir(sample_folder))[5])
    original = np.array(Image.open(sample_path).convert("RGB"))

    augmentations = [
        ("Orijinal", None),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("VerticalFlip", A.VerticalFlip(p=1.0)),
        ("Rotate ±15°", A.Rotate(limit=15, p=1.0)),
        ("Brightness +20%", A.RandomBrightnessContrast(
            brightness_limit=(0.2, 0.2), contrast_limit=0, p=1.0)),
        ("Contrast +20%", A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(0.2, 0.2), p=1.0)),
        ("HueSaturation", A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0)),
        ("GaussianBlur", A.GaussianBlur(blur_limit=(5, 5), p=1.0)),
        ("GaussNoise", A.GaussNoise(var_limit=(50, 50), p=1.0)),
        ("CoarseDropout", A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32, fill_value=0, p=1.0)),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("Augmentation Teknikleri — Önizleme", fontsize=14)

    for i, (name, aug) in enumerate(augmentations):
        if aug is None:
            img_show = cv2.resize(original, (224, 224))
        else:
            pipeline = A.Compose([A.Resize(224, 224), aug])
            img_show = pipeline(image=original)["image"]
        axes[i].imshow(img_show)
        axes[i].set_title(name, fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/viz8_augmentation_preview.png", dpi=150)
    plt.close()
    print("✓ VIZ-8 kaydedildi.")


# ═══════════════════════════════════════════════════════════════
# Ana EDA Fonksiyonu
# ═══════════════════════════════════════════════════════════════


def run_all_eda(data_root="dataset", classes=CLASSES, save_dir="outputs/plots/eda"):
    """
    Tüm EDA görselleştirmelerini art arda çalıştır.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EDA Gorsellestirilmeleri Basliyor")
    print("=" * 60)

    plot_class_distribution(data_root, classes, save_dir)
    plot_sample_grid(data_root, classes, save_dir=save_dir)
    plot_image_size_distribution(data_root, classes, save_dir)
    plot_aspect_ratio(data_root, classes, save_dir)
    plot_rgb_histograms(data_root, classes, save_dir=save_dir)
    plot_brightness_contrast(data_root, classes, save_dir=save_dir)
    mean_v, std_v = compute_and_plot_mean_std(
        os.path.join(data_root, "train"), classes, save_dir)
    plot_augmentation_preview(data_root, classes, save_dir)

    print("\n" + "=" * 60)
    print("  Tum EDA gorselleri tamamlandi!")
    print("=" * 60)
    print("\nONEMLI:")
    print(f"   DATASET_MEAN = {mean_v}")
    print(f"   DATASET_STD  = {std_v}")
    print("\n   Yukardaki degerleri src/config.py'ye yazin!\n")

    return mean_v, std_v
