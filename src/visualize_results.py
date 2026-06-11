"""
Evaluation ve Sonuç Görselleştirmeleri
Confusion matrix, ROC curves, misclassified samples, Grad-CAM, model comparison
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
import cv2

CLASSES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered",
]


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-1: Training History (4 Panel)
# ═══════════════════════════════════════════════════════════════


def plot_training_history(history, model_name, save_dir):
    """
    4-panel visualization:
    - Top-left: Train vs Val Loss
    - Top-right: Train vs Val Accuracy
    - Bottom-left: Overfitting gap
    - Bottom-right: Learning rate schedule
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Egitim Gecmisi - {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0][0].plot(epochs, history["train_loss"], label="Train Loss",
                   linewidth=2, color="#3266ad")
    axes[0][0].plot(epochs, history["val_loss"], label="Val Loss",
                   linewidth=2, linestyle="--", color="#D85A30")
    axes[0][0].set_title("Egitim / Dogrulama Kaybi")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].legend()
    axes[0][0].grid(alpha=0.3)

    # Accuracy
    axes[0][1].plot(epochs, history["train_acc"], label="Train Acc",
                   linewidth=2, color="#3266ad")
    axes[0][1].plot(epochs, history["val_acc"], label="Val Acc",
                   linewidth=2, linestyle="--", color="#D85A30")
    axes[0][1].set_title("Egitim / Dogrulama Dogrulugu")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Accuracy")
    axes[0][1].set_ylim(0, 1)
    axes[0][1].legend()
    axes[0][1].grid(alpha=0.3)

    # Overfitting gap
    gap = [ta - va for ta, va in zip(history["train_acc"], history["val_acc"])]
    fill_color = "#E67E22" if max(gap) > 0.10 else "#1D9E75"
    axes[1][0].fill_between(epochs, gap, alpha=0.4, color=fill_color)
    axes[1][0].axhline(0.10, color="red", linestyle="--", linewidth=1,
                      label="Overfitting esigi (10%)")
    axes[1][0].axhline(0, color="gray", linewidth=0.8)
    axes[1][0].set_title("Overfitting Gap (Train Acc - Val Acc)")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Fark")
    axes[1][0].legend()
    axes[1][0].grid(alpha=0.3)

    # LR schedule
    axes[1][1].plot(epochs, history["lr"], color="#9B59B6", linewidth=2)
    axes[1][1].set_title("Learning Rate Schedule")
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("LR")
    axes[1][1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/training_history_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  > Training history saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-2: Confusion Matrix
# ═══════════════════════════════════════════════════════════════


def plot_confusion_matrix(y_true, y_pred, model_name, classes, save_dir):
    """
    Side-by-side confusion matrices:
    - Left: Raw counts
    - Right: Normalized (recall per class)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix - {model_name}", fontsize=13)

    for ax, data, fmt, title, cmap in [
        (ax1, cm, "d", "Ham Sayilar", "Blues"),
        (ax2, cm_norm, ".2f", "Normalize (Recall)", "YlOrRd"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=[c.replace("-", "\n") for c in classes],
                   yticklabels=[c.replace("-", "\n") for c in classes],
                   ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_ylabel("Gercek Sinif")
        ax.set_xlabel("Tahmin Edilen Sinif")

    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  > Confusion matrix saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-3: Per-Class Metrics (Precision / Recall / F1)
# ═══════════════════════════════════════════════════════════════


def plot_per_class_metrics(y_true, y_pred, model_name, classes, save_dir):
    """
    Per-class Precision, Recall, F1-Score grouped bar chart
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))))

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#3266ad", alpha=0.85)
    ax.bar(x, rec, width, label="Recall", color="#1D9E75", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1-Score", color="#D85A30", alpha=0.85)

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        for offset, val in zip([-width, 0, width], [p, r, f]):
            ax.text(i + offset, val + 0.01, f"{val:.2f}",
                   ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in classes], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Skor")
    ax.set_title(f"Sinif Bazinda Precision / Recall / F1 - {model_name}", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/per_class_metrics_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  > Per-class metrics saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-4: ROC Curves
# ═══════════════════════════════════════════════════════════════


def plot_roc_curves(model, loader, model_name, classes, device, save_dir):
    """
    One-vs-Rest ROC curves with AUC scores
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    y_bin = label_binarize(all_labels, classes=list(range(len(classes))))

    colors = ["#3266ad", "#1D9E75", "#D85A30",
             "#9B59B6", "#E67E22", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f"{cls} (AUC={roc_auc:.3f})")

    # Micro-average
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    auc_m = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, "k--", linewidth=2,
           label=f"Micro-average (AUC={auc_m:.3f})")

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Egrileri (One-vs-Rest) - {model_name}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/roc_curves_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  > ROC curves saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-5: Misclassified Samples
# ═══════════════════════════════════════════════════════════════


def plot_misclassified(model, loader, model_name, classes,
                      device, save_dir, mean, std, n_show=20):
    """
    Grid of misclassified samples
    """
    # Denormalization transform
    unnorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    model.eval()
    wrong_imgs = []
    wrong_true = []
    wrong_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(1).cpu()
            mask = preds != labels
            for img, gt, pr in zip(imgs[mask], labels[mask], preds[mask]):
                if len(wrong_imgs) < n_show:
                    wrong_imgs.append(
                        unnorm(img).permute(1, 2, 0).clamp(0, 1).numpy())
                    wrong_true.append(gt.item())
                    wrong_pred.append(pr.item())

    if len(wrong_imgs) == 0:
        print(f"  > No misclassified samples found for {model_name}")
        return

    cols = 5
    rows = (len(wrong_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten()
    fig.suptitle(f"Yanlis Siniflandirilmis Ornekler - {model_name}", fontsize=13)

    for i, (img, gt, pr) in enumerate(zip(wrong_imgs, wrong_true, wrong_pred)):
        axes[i].imshow(img)
        axes[i].set_title(
            f"Gercek: {classes[gt]}\nTahmin: {classes[pr]}",
            fontsize=7.5, color="red")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = f"{save_dir}/misclassified_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  > Misclassified samples saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# EVAL-VIZ-6: Grad-CAM Heatmaps
# ═══════════════════════════════════════════════════════════════


def visualize_gradcam(model, target_layer, loader, classes,
                     device, model_name, save_dir, mean, std, n_per_class=2):
    """
    Grad-CAM visualization: orijinal + heatmap overlay
    """
    from src.gradcam import GradCAM

    gradcam = GradCAM(model, target_layer)
    unnorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    class_examples = {i: [] for i in range(len(classes))}
    for imgs, labels in loader:
        for img, label in zip(imgs, labels):
            li = label.item()
            if len(class_examples[li]) < n_per_class:
                class_examples[li].append(img)
        if all(len(v) >= n_per_class for v in class_examples.values()):
            break

    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class * 2,
                            figsize=(n_per_class * 5, n_classes * 3))
    fig.suptitle(f"Grad-CAM Isi Haritalari - {model_name}", fontsize=13)

    for cls_idx, cls_name in enumerate(classes):
        for ex_idx, img_t in enumerate(class_examples[cls_idx]):
            inp = img_t.unsqueeze(0).to(device)
            cam, pred_idx = gradcam.generate(inp)

            orig = unnorm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = 0.5 * orig + 0.5 * cv2.cvtColor(
                heatmap, cv2.COLOR_BGR2RGB) / 255.0

            col_o = ex_idx * 2
            col_c = ex_idx * 2 + 1
            axes[cls_idx][col_o].imshow(orig)
            axes[cls_idx][col_o].set_title(f"{cls_name}\nOrijinal", fontsize=8)
            axes[cls_idx][col_o].axis("off")
            axes[cls_idx][col_c].imshow(overlay)
            axes[cls_idx][col_c].set_title(
                f"Tahmin: {classes[pred_idx]}\nGrad-CAM", fontsize=8,
                color="green" if pred_idx == cls_idx else "red")
            axes[cls_idx][col_c].axis("off")

    plt.tight_layout()
    path = f"{save_dir}/gradcam_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  > Grad-CAM saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# Comparison Visualizations
# ═══════════════════════════════════════════════════════════════


def plot_combined_training_curves(histories, model_names, save_dir):
    """
    Compare training curves of all 3 models
    """
    cmap = plt.cm.get_cmap("tab10", max(3, len(model_names)))
    colors = [cmap(i) for i in range(len(model_names))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Tum Modeller - Egitim Karsilastirmasi", fontsize=14)

    for hist, name, color in zip(histories, model_names, colors):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs, hist["train_loss"], color=color,
                linestyle="-", linewidth=1.5, alpha=0.6, label=f"{name} Train")
        ax1.plot(epochs, hist["val_loss"], color=color,
                linestyle="--", linewidth=2.0, label=f"{name} Val")
        ax2.plot(epochs, hist["train_acc"], color=color,
                linestyle="-", linewidth=1.5, alpha=0.6, label=f"{name} Train")
        ax2.plot(epochs, hist["val_acc"], color=color,
                linestyle="--", linewidth=2.0, label=f"{name} Val")

    for ax, ylabel, title in [
        (ax1, "Loss", "Kayip Karsilastirmasi"),
        (ax2, "Accuracy", "Dogruluk Karsilastirmasi"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    path = f"{save_dir}/combined_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  > Combined curves saved: {os.path.basename(path)}")


def plot_model_comparison(results_dict, save_dir):
    """
    Bar chart: model comparison (accuracy, precision, recall, F1)
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    met_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(results_dict.keys())
    x = np.arange(len(metrics))
    width = min(0.8 / max(1, len(model_names)), 0.25)
    cmap = plt.cm.get_cmap("tab10", max(3, len(model_names)))
    colors = [cmap(i) for i in range(len(model_names))]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results_dict[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                     label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.005,
                   f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(met_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor")
    ax.set_title("Model Karsilastirmasi - Test Seti", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/model_comparison_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  > Model comparison saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# Ana Evaluation Fonksiyonu
# ═══════════════════════════════════════════════════════════════


def evaluate_model_full(model, model_name, test_loader, classes, device,
                       save_dir, mean, std, target_layer=None):
    """
    Complete evaluation pipeline for a single model
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Test Sonuclari")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=classes)}")

    plot_confusion_matrix(y_true, y_pred, model_name, classes, save_dir)
    plot_per_class_metrics(y_true, y_pred, model_name, classes, save_dir)
    plot_roc_curves(model, test_loader, model_name, classes, device, save_dir)
    plot_misclassified(model, test_loader, model_name, classes, device, save_dir, mean, std)

    if target_layer is not None:
        try:
            visualize_gradcam(model, target_layer, test_loader, classes,
                             device, model_name, save_dir, mean, std)
        except Exception as e:
            print(f"  > Grad-CAM skipped for {model_name}: {e}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
