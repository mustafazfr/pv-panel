"""
Eğitim Pipeline
train_one_epoch, validate, train_model fonksiyonları
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import time
import numpy as np
from tqdm import tqdm


def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y, y[index], lam


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, use_cutmix=False):
    """
    Bir epoch'ta eğitim

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        criterion: loss function (e.g., CrossEntropyLoss)
        optimizer: optimizer (e.g., SGD)
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: epoch ortalama loss
        accuracy: epoch accuracy (0-1)
    """
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        r = np.random.rand()
        if use_mixup and use_cutmix:
            if r < 0.25:
                imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=0.2, device=device)
            elif r < 0.50:
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels, alpha=1.0, device=device)
            else:
                targets_a, targets_b, lam = labels, labels, 1.0
        elif use_mixup and r < 0.5:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=0.2, device=device)
        elif use_cutmix and r < 0.5:
            imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels, alpha=1.0, device=device)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        
        if lam < 1.0:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Istatistikler
        loss_sum += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        
        if lam < 1.0:
            correct += (lam * (predicted == targets_a).sum().float() + (1 - lam) * (predicted == targets_b).sum().float()).item()
        else:
            correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = loss_sum / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validation epoch

    Args:
        model: PyTorch model
        val_loader: validation DataLoader
        criterion: loss function
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="  Val  ", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(model, model_name, train_loader, val_loader,
               class_weights, device, cfg):
    """
    Tam eğitim döngüsü: train + validation + early stopping + checkpoint save

    Args:
        model: PyTorch model
        model_name: model'in adı (string)
        train_loader, val_loader: DataLoaders
        class_weights: torch.Tensor (num_classes,) class weighting için
        device: 'cpu' or 'cuda'
        cfg: configuration object (from config.py)

    Returns:
        history: dict with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
        best_checkpoint_path: en iyi model'in checkpoint yolu
    """

    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=getattr(cfg, "LABEL_SMOOTHING", 0.0),
    )

    # Optimizer: SGD with Nesterov momentum
    optimizer = optim.SGD(model.parameters(),
                         lr=cfg.LEARNING_RATE,
                         momentum=cfg.MOMENTUM,
                         weight_decay=cfg.WEIGHT_DECAY,
                         nesterov=True)

    # LR Scheduler
    if getattr(cfg, "USE_COSINE_ANNEALING", False):
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=getattr(cfg, "COSINE_T0", 10), 
            T_mult=getattr(cfg, "COSINE_TMULT", 2)
        )
    else:
        scheduler = StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = f"{cfg.CHECKPOINT_DIR}/{model_name}_best.pth"

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    print(f"\n{'=' * 60}")
    print(f"  {model_name} Egitimi")
    print(f"{'=' * 60}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=getattr(cfg, "USE_MIXUP", False),
            use_cutmix=getattr(cfg, "USE_CUTMIX", False)
        )
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # LR scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # History append
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"  [{epoch:03d}/{cfg.NUM_EPOCHS}] "
              f"TLoss:{train_loss:.4f} TAcc:{train_acc:.4f} | "
              f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f} | "
              f"LR:{current_lr:.6f} | {epoch_time:.1f}s")

        # Early stopping: val_loss based
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"    > Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n  Early stopping! No improvement for {cfg.PATIENCE} epochs.")
                break

    print(f"\n{'=' * 60}")
    print(f"  Egitim Tamamlandi")
    print(f"{'=' * 60}\n")

    return history, best_checkpoint_path
