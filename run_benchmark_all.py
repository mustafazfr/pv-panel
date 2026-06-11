import os
import torch
import pandas as pd
import numpy as np

# Config and Utilities
import src.config as cfg
from src.dataset import get_dataloaders
from src.dataset import compute_class_weights

# Models
from src.models.custom_cnn import CustomCNN
from src.models.custom_cnn_v2 import CustomCNNV2
from src.models.custom_cnn_v3 import CustomCNN_V3
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

def setup_new_output_dirs(base_dir="outputs_benchmark"):
    """
    Creates a completely separate output directory structure 
    to keep results clean and organized from scratch.
    """
    dirs = {
        'CHECKPOINT_DIR': f"{base_dir}/checkpoints",
        'RESULT_DIR': f"{base_dir}/results",
        'PLOT_DIR': f"{base_dir}/plots",
        'PLOT_TRAIN_DIR': f"{base_dir}/plots/training",
        'PLOT_EVAL_DIR': f"{base_dir}/plots/evaluation",
        'PLOT_EDA_DIR': f"{base_dir}/plots/eda"
    }
    
    for _, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        
    return dirs

def main():
    print("=" * 80)
    print("  🚀 FULL BENCHMARK: PV PANEL DEFECT CLASSIFICATION 🚀")
    print("=" * 80)

    # 1. Override config paths to write into the new clean directory
    new_dirs = setup_new_output_dirs("outputs_benchmark")
    cfg.CHECKPOINT_DIR = new_dirs['CHECKPOINT_DIR']
    cfg.RESULT_DIR = new_dirs['RESULT_DIR']
    cfg.PLOT_DIR = new_dirs['PLOT_DIR']
    cfg.PLOT_TRAIN_DIR = new_dirs['PLOT_TRAIN_DIR']
    cfg.PLOT_EVAL_DIR = new_dirs['PLOT_EVAL_DIR']
    cfg.PLOT_EDA_DIR = new_dirs['PLOT_EDA_DIR']

    print(f"📁 Saving all new results to: outputs_benchmark/")
    
    # 2. Load Dataloaders
    print(f"\n[PHASE 1] Data Loading...")
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
    
    class_weights = compute_class_weights(cfg.TRAIN_DIR, cfg.CLASSES).to(cfg.DEVICE)
    print(f"Class Weights computed: {class_weights.cpu().numpy()}")

    # 3. Model Inventory Setup 
    # Defines model, and its target layer for GradCAM visualization
    models_config = {
        "CustomCNN": (
            CustomCNN(num_classes=cfg.NUM_CLASSES),
            lambda m: m.features[-1].conv2  # Son Conv2d
        ),
        "CustomCNN-V2": (
            CustomCNNV2(num_classes=cfg.NUM_CLASSES),
            lambda m: list(m.stage4[-1].children())[1]  
        ),
        "CustomCNN-V3": (
            CustomCNN_V3(num_classes=cfg.NUM_CLASSES),
            lambda m: m.block3
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
        )
    }

    all_histories = {}
    all_results = {}

    print("\n" + "=" * 80)
    print("  [PHASE 2 & 3] Custom & SOTA Models Training & Evaluation")
    print("=" * 80)

    for model_name, (model, get_target_layer_fn) in models_config.items():
        print(f"\n======================> RUNNING {model_name} <======================")
        
        # --- TRAIN ---
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

        # --- EVALUATE ---
        # Load the best weights we just acquired
        model.load_state_dict(torch.load(best_ckpt_path, map_location=cfg.DEVICE))
        print(f"\n  >>> Evaluating {model_name} on Test Set...")
        
        target_layer = None
        try:
            target_layer = get_target_layer_fn(model)
        except Exception as e:
            print(f"Warning: Target layer for GradCAM failed: {e}")

        try:
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
        except Exception as e:
            import traceback
            print(f"TEST FAILED FOR {model_name}. Skipping graph for this model. Error: {e}\n{traceback.format_exc()}")

    # 4. Final Comparison & Logging
    print("\n" + "=" * 80)
    print("  [PHASE 4] Aggregation & Comprehensive Comparison Plots")
    print("=" * 80)

    # Export aggregated final results dict to CSV
    results_df = pd.DataFrame(all_results).T
    csv_path = f"{cfg.RESULT_DIR}/benchmark_results.csv"
    results_df.to_csv(csv_path)
    print(f"📈 Total Benchmark CSV Logged at: {csv_path}")
    print("\nFinal Results Table:\n", results_df)

    # Plot single chart including ALL models
    import matplotlib
    matplotlib.use('Agg') # Safe plotting without GUI
    
    try:
        plot_model_comparison(results_df.to_dict(orient="index"), cfg.PLOT_EVAL_DIR)
        print(f"📊 Accuracy/Precision/Recall/F1 Bar Comparison Chart -> {cfg.PLOT_EVAL_DIR}/model_comparison_bar.png")
    except Exception as plt_err:
        print(f"Warning: Plot failed: {plt_err}")

    # Plot cumulative training curves
    try:
        plot_combined_training_curves(all_histories, cfg.PLOT_TRAIN_DIR)
        print(f"📉 Train Curves Master Comparison Chart -> {cfg.PLOT_TRAIN_DIR}/combined_training_curves.png")
    except Exception as curve_err:
        print(f"Warning: Curve line-plot failed: {curve_err}")
        
    print("\n🎉 ALL SETUP COMPLETE. You can examine the outputs_benchmark folder! 🎉")

if __name__ == "__main__":
    main()
