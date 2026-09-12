"""CARVE — Training pipeline cho DeepFeat.

Config chot truoc (Muc 5.2 CARVE v5.1):
- Optimizer: Adam (lr=3e-4, weight_decay=1e-4)
- Scheduler: cosine
- Early stopping: patience=10 theo val ROC-AUC, restore best
- Class weight: pos_weight = N_neg/N_pos (BCEWithLogitsLoss)
- Deterministic (Muc III.6): DeepFeat=True
- max 100 epochs, batch_size=32
"""
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Fix import path khi chay truc tiep: python src/train.py
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==== Config ====
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = 'cosine'
    patience: int = 10
    min_delta: float = 1e-4
    num_workers: int = 0
    device: str = 'auto'
    deterministic: bool = True
    verbose: bool = True


# ==== Seed ====
def set_seed(seed, deterministic=True):
    """Set seed cho numpy/torch/cuda + cuDNN deterministic (Muc III.6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==== Utils ====
def resolve_device(device_str):
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def compute_pos_weight(labels):
    """pos_weight = N_neg / N_pos cho BCEWithLogitsLoss."""
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def build_optimizer_scheduler(model, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0,
            total_iters=config.epochs
        )
    else:
        scheduler = None
    return optimizer, scheduler


# ==== Train / eval epoch ====
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.squeeze(1).cpu().numpy())

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    metrics = {
        'loss': total_loss / max(n, 1),
        'roc_auc': float('nan'),
        'brier': float('nan'),
    }
    if len(np.unique(all_labels)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(all_labels, all_probs))
        metrics['brier'] = float(brier_score_loss(all_labels, all_probs))

    return metrics, all_probs, all_labels


# ==== Full fit ====
def fit(model, train_loader, val_loader, config=None):
    """Train full voi early stopping theo val ROC-AUC."""
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model = model.to(device)

    optimizer, scheduler = build_optimizer_scheduler(model, config)

    train_labels = []
    for _, y in train_loader:
        train_labels.append(y.numpy())
    train_labels = np.concatenate(train_labels)
    pos_weight = torch.tensor([compute_pos_weight(train_labels)],
                              device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if config.verbose:
        print(f"Device: {device}")
        print(f"Train N: {len(train_labels)}, N_pos: {int((train_labels==1).sum())}, "
              f"N_neg: {int((train_labels==0).sum())}, pos_weight: {pos_weight.item():.4f}")

    history = []
    best_metric = -np.inf
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    iterator = range(1, config.epochs + 1)
    if config.verbose:
        iterator = tqdm(iterator, desc='Training')

    for epoch in iterator:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_brier': val_metrics['brier'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(record)

        if config.verbose:
            print(f"  Ep {epoch:3d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_metrics['loss']:.4f} "
                  f"| val_auc={val_metrics['roc_auc']:.4f}")

        metric = val_metrics['roc_auc']
        if np.isnan(metric):
            metric = -np.inf

        if metric > best_metric + config.min_delta:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            if config.verbose:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improve for {config.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'history': history,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'stopped_epoch': history[-1]['epoch'] if history else 0,
    }


@torch.no_grad()
def predict_proba(model, loader, device='auto'):
    """Tra ve (probs, labels) numpy arrays."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


if __name__ == '__main__':
    from torch.utils.data import TensorDataset
    from src.models import DeepFeat

    print("=== Test train pipeline voi dummy data ===")
    set_seed(42)

    torch.manual_seed(0)
    X_train = torch.randn(64, 1, 128, 500)
    y_train = torch.tensor([1.0] * 16 + [0.0] * 48)
    X_val = torch.randn(16, 1, 128, 500)
    y_val = torch.tensor([1.0] * 4 + [0.0] * 12)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = DeepFeat()

    config = TrainConfig(epochs=3, batch_size=8, patience=10,
                         device='cpu', verbose=True)
    result = fit(model, train_loader, val_loader, config)

    print(f"\n=== Ket qua ===")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val ROC-AUC: {result['best_metric']:.4f}")
    print(f"Stopped epoch: {result['stopped_epoch']}")
    print(f"History length: {len(result['history'])}")

    probs, labels = predict_proba(result['model'], val_loader, device='cpu')
    print(f"\nPredict shape: {probs.shape}, labels shape: {labels.shape}")
    print(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("\nOK")