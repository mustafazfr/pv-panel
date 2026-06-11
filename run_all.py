"""
BIM 430 — PV Panel Defect Classification
Ana Eğitim Pipeline (run_all.py)

Execution order:
1. EDA (optional - görselleştirmeler)
2. Data loading + Class weights
3. Model definitions
4. Training (Custom CNN, VGG, ResNet)
5. Evaluation (Confusion matrix, ROC, Grad-CAM, etc.)
6. Model comparison
7. Results CSV
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Config
import src.config as cfg

# Dataset
from src.dataset import get_dataloaders, compute_class_weights

# Models
from src.models.custom_cnn import CustomCNN
from src.models.vgg_scratch import VGGScratch
from src.models.resnet_scratch import ResNetScratch
from src.models.mobilenet_v2_scratch import MobileNetV2Scratch
from src.models.efficientnet_scratch import EfficientNetScratch

# Training
from src.train import train_model

# Visualization
from src.visualize_results import (
    plot_training_history,
    evaluate_model_full,
    plot_combined_training_curves,
    plot_model_comparison
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 70)
    print("  BIM 430 - PV Panel Defect Classification")
    print("  Complete Training Pipeline")
    print("=" * 70)
    print(f"\n  Device: {cfg.DEVICE}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ═══════════════════════════════════════════════════════════════
    # SETUP: Seeds + Directories
    # ═══════════════════════════════════════════════════════════════
    set_seed(cfg.SEED)
    print("\n[SETUP] Random seeds set")

    # Create output directories
    for directory in [cfg.CHECKPOINT_DIR, cfg.PLOT_EDA_DIR,
                     cfg.PLOT_TRAIN_DIR, cfg.PLOT_EVAL_DIR, cfg.RESULT_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"[SETUP] Output directories created")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [PHASE 1] Data Loading")
    print("=" * 70)

    print(f"  Loading data from: {cfg.DATA_ROOT}")
    print(f"  - Train: {cfg.TRAIN_DIR}")
    print(f"  - Val: {cfg.VAL_DIR}")
    print(f"  - Test: {cfg.TEST_DIR}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=cfg.TRAIN_DIR,
        val_dir=cfg.VAL_DIR,
        test_dir=cfg.TEST_DIR,
        mean=cfg.DATASET_MEAN,
        std=cfg.DATASET_STD,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Val samples: {len(val_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Batch size: {cfg.BATCH_SIZE}")

    # Compute class weights
    class_weights = compute_class_weights(cfg.TRAIN_DIR, cfg.CLASSES)
    print(f"\n  Class weights computed:")
    for cls, w in zip(cfg.CLASSES, class_weights):
        print(f"    {cls:20s}: {w:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [PHASE 2] Model Training")
    print("=" * 70)

    # Model definitions with Grad-CAM target layers
    models_config = {
        "CustomCNN": (
            CustomCNN(num_classes=cfg.NUM_CLASSES),
            lambda m: m.features[-1].conv2  # Son Conv2d
        ),
        "VGG-Scratch": (
            VGGScratch(num_classes=cfg.NUM_CLASSES),
            lambda m: list(m.features[-1].children())[-4]  # Son Conv2d
        ),
        "ResNet-Scratch": (
            ResNetScratch(num_classes=cfg.NUM_CLASSES),
            lambda m: m.layer4[-1].bn2  # Son BasicBlock'un BN2'si
        ),
        "MobileNetV2-Scratch": (
            MobileNetV2Scratch(num_classes=cfg.NUM_CLASSES, dropout=0.20),
            lambda m: m.features[-1][0]  # Final 1x1 conv
        ),
        "EfficientNet-Scratch": (
            EfficientNetScratch(num_classes=cfg.NUM_CLASSES),
            lambda m: m.model.features[-1][0] # Final conv in features
        ),
    }

    all_histories = {}
    all_results = {}

    for model_name, (model, get_target_layer) in models_config.items():
        print(f"\n  >>> Training {model_name}...")

        # Train
        history, best_ckpt_path = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=cfg.DEVICE,
            cfg=cfg
        )

        all_histories[model_name] = history

        # Save training history visualization
        plot_training_history(history, model_name, cfg.PLOT_TRAIN_DIR)

        # Load best checkpoint
        model.load_state_dict(torch.load(best_ckpt_path, map_location=cfg.DEVICE))
        print(f"  > Best checkpoint loaded: {os.path.basename(best_ckpt_path)}")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: MODEL EVALUATION
        # ═══════════════════════════════════════════════════════════════
        print(f"\n  >>> Evaluating {model_name}...")

        target_layer = get_target_layer(model)

        results = evaluate_model_full(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            classes=cfg.CLASSES,
            device=cfg.DEVICE,
            save_dir=cfg.PLOT_EVAL_DIR,
            mean=cfg.DATASET_MEAN,
            std=cfg.DATASET_STD,
            target_layer=target_layer
        )

        all_results[model_name] = results

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: MODEL COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [PHASE 4] Model Comparison")
    print("=" * 70)

    plot_combined_training_curves(
        list(all_histories.values()),
        list(all_histories.keys()),
        cfg.PLOT_TRAIN_DIR
    )

    plot_model_comparison(all_results, cfg.PLOT_EVAL_DIR)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [PHASE 5] Results Summary")
    print("=" * 70)

    # Create results dataframe
    results_df = pd.DataFrame(all_results).T
    print("\n" + results_df.round(4).to_string())

    # Save CSV
    csv_path = f"{cfg.RESULT_DIR}/final_results.csv"
    results_df.to_csv(csv_path)
    print(f"\n  Results saved to: {csv_path}")

    # Print best model
    print("\n" + "=" * 70)
    print("  BEST MODEL (by Test Accuracy)")
    print("=" * 70)
    best_model_name = results_df["accuracy"].idxmax()
    best_acc = results_df.loc[best_model_name, "accuracy"]
    print(f"  Model: {best_model_name}")
    print(f"  Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n  Output directories:")
    print(f"    - Checkpoints: {cfg.CHECKPOINT_DIR}")
    print(f"    - EDA plots: {cfg.PLOT_EDA_DIR}")
    print(f"    - Training plots: {cfg.PLOT_TRAIN_DIR}")
    print(f"    - Evaluation plots: {cfg.PLOT_EVAL_DIR}")
    print(f"    - Results: {cfg.RESULT_DIR}")


if __name__ == "__main__":
    main()
