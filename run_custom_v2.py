import os
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime

import src.config as cfg
from src.dataset import get_dataloaders, compute_class_weights
from src.models.custom_cnn_v2 import CustomCNNV2
from src.train import train_model
from src.visualize_results import evaluate_model_full, plot_combined_training_curves, plot_model_comparison

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(cfg.SEED)
    
    print("=" * 70)
    print("  [PHASE 1] Data Loading (Only for CustomCNN V2)")
    print("=" * 70)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.TRAIN_DIR, cfg.VAL_DIR, cfg.TEST_DIR, cfg.DATASET_MEAN, cfg.DATASET_STD
    )
    class_weights = compute_class_weights(cfg.TRAIN_DIR, cfg.CLASSES)

    models_config = {
        "CustomCNN-V2": (
            CustomCNNV2(num_classes=cfg.NUM_CLASSES),
            lambda m: list(m.stage4[-1].children())[1] # Last batch norm in CBAM sequence conceptually or conv2
        )
    }

    all_histories = {}
    all_results = {}

    print("=" * 70)
    print("  [PHASE 2] Model Training: CustomCNN V2")
    print("=" * 70)

    for model_name, (model, get_target_layer) in models_config.items():
        best_ckpt = f"{cfg.CHECKPOINT_DIR}/{model_name}_best.pth"
        if not os.path.exists(best_ckpt):
            # Train
            history, best_ckpt = train_model(
                model=model, model_name=model_name, 
                train_loader=train_loader, val_loader=val_loader,
                class_weights=class_weights, device=cfg.DEVICE, cfg=cfg
            )
            all_histories[model_name] = history
        
        # Load best and evaluate
        print(f"Loading {best_ckpt} for evaluation...")
        model.load_state_dict(torch.load(best_ckpt))
        model = model.to(cfg.DEVICE)
        metrics = evaluate_model_full(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            classes=cfg.CLASSES,
            device=cfg.DEVICE,
            save_dir=cfg.PLOT_EVAL_DIR,
            mean=cfg.DATASET_MEAN,
            std=cfg.DATASET_STD,
            target_layer=get_target_layer(model)
        )
        all_results[model_name] = metrics

    print("=" * 70)
    print("  [PHASE 3] Merging Results with Old Runs")
    print("=" * 70)
    
    import csv
    backup_file = "outputs/results/final_results_backup.csv"
    current_file = "outputs/results/final_results.csv"

    # Merge everything to draw the final plot!
    old_df = pd.read_csv(backup_file, index_col=0)
    new_df = pd.DataFrame.from_dict(all_results, orient="index")
    
    # Update the old dataframe with new item
    combined_df = pd.concat([old_df, new_df])
    combined_df.to_csv(current_file)
    print(combined_df)
    
    # Re-plot model comparison with ALL models
    plot_model_comparison(combined_df.to_dict(orient="index"), cfg)
    
    print("\nCustom CNN V2 Training Pipeline Finished! Checkout plots.")

if __name__ == "__main__":
    main()
