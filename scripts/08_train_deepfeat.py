#!/usr/bin/env python3
"""CARVE — Train DeepFeat tren mel cache.

Load mel cache (X, y, meta) → speaker-aware split → train DeepFeat.

Usage:
    # L1 pilot: 1 fold x 1 seed, 20 epoch
    python scripts/08_train_deepfeat.py \\
        --cache-dir /kaggle/working/mel_cache \\
        --output-dir /kaggle/working/results_deepfeat \\
        --fold 0 --seed 0 --epochs 20

    # L1 pilot: 1 fold x 3 seeds
    python scripts/08_train_deepfeat.py --seeds 0 1 2 --epochs 20 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DeepFeat, count_parameters
from src.train import TrainConfig, fit, predict_proba, set_seed


class MelCacheDataset(Dataset):
    """Dataset load tu numpy array (pre-computed mel)."""

    def __init__(self, X, y, indices=None, spec_augment=None,
                 mel_mean=None, mel_std=None):
        self.X = X
        self.y = y
        self.indices = indices if indices is not None else np.arange(len(X))
        self.spec_augment = spec_augment
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        mel = torch.from_numpy(self.X[i].astype(np.float32)).unsqueeze(0)
        label = float(self.y[i])

        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean) / (self.mel_std + 1e-8)

        if self.spec_augment is not None:
            mel = self.spec_augment(mel)

        return mel, torch.tensor(label, dtype=torch.float32)


def speaker_aware_split(meta, n_folds=5, fold=0, seed=0):
    """Chia speakers thanh n_folds, tra ve (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    speakers = meta['speaker_id'].unique()
    rng.shuffle(speakers)
    folds = np.array_split(speakers, n_folds)
    val_speakers = set(folds[fold])
    train_speakers = set()
    for k, f in enumerate(folds):
        if k != fold:
            train_speakers.update(f)

    train_idx = meta.index[meta['speaker_id'].isin(train_speakers)].values
    val_idx = meta.index[meta['speaker_id'].isin(val_speakers)].values
    return train_idx, val_idx


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    x_path = cache_dir / "mel_cache_X.npy"
    y_path = cache_dir / "mel_cache_y.npy"
    meta_path = cache_dir / "mel_cache_meta.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Chua merge. Thieu: {x_path}")

    print(f"Load X: {x_path}")
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path)
    meta = pd.read_csv(meta_path)
    print(f"  X shape: {X.shape} ({X.dtype})")
    print(f"  y shape: {y.shape}, N_pos={int((y==1).sum())}, N_neg={int((y==0).sum())}")
    print(f"  meta: {len(meta)} rows, {meta['speaker_id'].nunique()} speakers")
    return X, y, meta


def train_one_run(X, y, meta, args, seed):
    """1 run = 1 fold x 1 seed."""
    from src.data import SpecAugment

    print(f"\n{'=' * 72}")
    print(f"Run: fold={args.fold}, seed={seed}")
    print(f"{'=' * 72}")

    set_seed(seed, deterministic=True)

    train_idx, val_idx = speaker_aware_split(
        meta, n_folds=args.n_folds, fold=args.fold, seed=seed
    )
    print(f"Train: {len(train_idx)} samples ({meta.iloc[train_idx]['speaker_id'].nunique()} spk)")
    print(f"Val:   {len(val_idx)} samples ({meta.iloc[val_idx]['speaker_id'].nunique()} spk)")

    # Label distribution
    y_train = y[train_idx]
    y_val = y[val_idx]
    print(f"Train labels: N_pos={int((y_train==1).sum())}, N_neg={int((y_train==0).sum())}")
    print(f"Val labels:   N_pos={int((y_val==1).sum())}, N_neg={int((y_val==0).sum())}")

    # Compute mel stats tren train (max 200 samples)
    n_stats = min(200, len(train_idx))
    stats_idx = np.random.RandomState(seed).choice(train_idx, size=n_stats, replace=False)
    vals = []
    for i in stats_idx:
        vals.append(torch.from_numpy(X[i].astype(np.float32)).flatten())
    vals = torch.cat(vals)
    mel_mean = float(vals.mean())
    mel_std = float(vals.std())
    print(f"mel_mean={mel_mean:.4f}, mel_std={mel_std:.4f}")

    spec_aug = SpecAugment()
    train_ds = MelCacheDataset(X, y, train_idx, spec_augment=spec_aug,
                               mel_mean=mel_mean, mel_std=mel_std)
    val_ds = MelCacheDataset(X, y, val_idx, spec_augment=None,
                             mel_mean=mel_mean, mel_std=mel_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = DeepFeat()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device='auto',
        verbose=True,
    )

    result = fit(model, train_loader, val_loader, config)

    # Predict on val (out-of-fold)
    probs, labels = predict_proba(result['model'], val_loader, device='auto')

    return {
        'fold': args.fold,
        'seed': seed,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'best_epoch': result['best_epoch'],
        'best_val_roc_auc': result['best_metric'],
        'stopped_epoch': result['stopped_epoch'],
        'mel_mean': mel_mean,
        'mel_std': mel_std,
        'val_probs': probs.tolist(),
        'val_labels': labels.tolist(),
        'val_speaker_ids': meta.iloc[val_idx]['speaker_id'].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='/kaggle/working/mel_cache')
    parser.add_argument('--output-dir', default='/kaggle/working/results_deepfeat')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Neu set, chay nhieu seed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    print("=" * 72)
    print("CARVE — Train DeepFeat tren mel cache")
    print("=" * 72)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_cache(args.cache_dir)

    seeds = args.seeds if args.seeds else [args.seed]
    all_results = []

    for seed in seeds:
        result = train_one_run(X, y, meta, args, seed)
        all_results.append(result)

        # Luu tung run
        run_path = output_dir / f"run_f{args.fold}_s{seed}.json"
        with open(run_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nLuu: {run_path}")

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'seed':>5} {'best_ep':>8} {'val_auc':>10} {'n_train':>8} {'n_val':>6}")
    for r in all_results:
        print(f"{r['seed']:>5} {r['best_epoch']:>8} {r['best_val_roc_auc']:>10.4f} "
              f"{r['n_train']:>8} {r['n_val']:>6}")

    aucs = [r['best_val_roc_auc'] for r in all_results]
    print(f"\nMean val ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Luu summary
    summary_path = output_dir / f"summary_f{args.fold}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'fold': args.fold,
            'seeds': seeds,
            'aucs': aucs,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
        }, f, indent=2)
    print(f"\nLuu: {summary_path}")


if __name__ == '__main__':
    main()