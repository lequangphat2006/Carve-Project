#!/usr/bin/env python3
"""CARVE — L2 HPO cho DeepFeat (random search 20 trial).

Search space (Muc 5.4 Phu luc A CARVE v5.1):
- lr: log_uniform(1e-5, 1e-3)
- weight_decay: log_uniform(1e-6, 1e-3)
- batch_size: categorical [16, 32, 64]
- dropout: uniform(0.3, 0.7)

Setup: 1 fold (fold 0), 1 seed (seed 0), full 100 epoch
Selection: best val ROC-AUC
Output: best_config.json + trials.json

Usage:
    python scripts/09_hpo_deepfeat.py \\
        --cache-dir /kaggle/input/datasets/lequangphat260206/carve-mel-cache \\
        --output-dir /kaggle/working/results_hpo \\
        --n-trials 20 \\
        --epochs 100
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SpecAugment
from src.models import DeepFeat, count_parameters
from src.train import TrainConfig, fit, predict_proba, set_seed


# ==== Search space ====
SEARCH_SPACE = {
    'lr': ('log_uniform', 1e-5, 1e-3),
    'weight_decay': ('log_uniform', 1e-6, 1e-3),
    'batch_size': ('categorical', [16, 32, 64]),
    'dropout': ('uniform', 0.3, 0.7),
}


def sample_config(rng):
    """Sample 1 config tu search space."""
    config = {}
    for name, spec in SEARCH_SPACE.items():
        if spec[0] == 'log_uniform':
            lo, hi = np.log10(spec[1]), np.log10(spec[2])
            config[name] = float(10 ** rng.uniform(lo, hi))
        elif spec[0] == 'uniform':
            config[name] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == 'categorical':
            config[name] = int(rng.choice(spec[1]))
    return config


class MelCacheDataset(Dataset):
    """Copy tu 08_train_deepfeat.py."""

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
    rng = np.random.RandomState(seed)
    speakers = meta['speaker_id'].unique()
    rng.shuffle(speakers)
    folds = np.array_split(speakers, n_folds)
    val_spk = set(folds[fold])
    train_spk = set()
    for k, f in enumerate(folds):
        if k != fold:
            train_spk.update(f)
    train_idx = meta.index[meta['speaker_id'].isin(train_spk)].values
    val_idx = meta.index[meta['speaker_id'].isin(val_spk)].values
    return train_idx, val_idx


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    X = np.load(cache_dir / "mel_cache_X.npy", mmap_mode='r')
    y = np.load(cache_dir / "mel_cache_y.npy")
    meta = pd.read_csv(cache_dir / "mel_cache_meta.csv")
    print(f"X: {X.shape}, y: {y.shape}, meta: {len(meta)}")
    return X, y, meta


def run_trial(X, y, meta, config, args, trial_id):
    """Chay 1 trial voi 1 config."""
    print(f"\n{'=' * 72}")
    print(f"Trial {trial_id}/{args.n_trials}")
    print(f"{'=' * 72}")
    print(f"Config: {json.dumps(config, indent=2)}")

    set_seed(args.seed, deterministic=True)

    train_idx, val_idx = speaker_aware_split(
        meta, n_folds=5, fold=args.fold, seed=args.seed
    )

    # Compute mel stats
    n_stats = min(200, len(train_idx))
    stats_idx = np.random.RandomState(args.seed).choice(
        train_idx, size=n_stats, replace=False
    )
    vals = [torch.from_numpy(X[i].astype(np.float32)).flatten() for i in stats_idx]
    vals = torch.cat(vals)
    mel_mean = float(vals.mean())
    mel_std = float(vals.std())

    spec_aug = SpecAugment()
    train_ds = MelCacheDataset(X, y, train_idx, spec_augment=spec_aug,
                               mel_mean=mel_mean, mel_std=mel_std)
    val_ds = MelCacheDataset(X, y, val_idx, spec_augment=None,
                             mel_mean=mel_mean, mel_std=mel_std)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    # Model voi dropout tu config
    model = DeepFeat(dropout=config['dropout'])
    print(f"Params: {count_parameters(model):,}")

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=config['batch_size'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        patience=args.patience,
        device='auto',
        verbose=False,  # Tat verbose de gon log
    )

    t0 = time.time()
    result = fit(model, train_loader, val_loader, train_config)
    elapsed = time.time() - t0

    print(f"Best epoch: {result['best_epoch']}, "
          f"Best val AUC: {result['best_metric']:.4f}, "
          f"Time: {elapsed/60:.1f} min")

    return {
        'trial_id': trial_id,
        'config': config,
        'best_epoch': result['best_epoch'],
        'best_val_roc_auc': result['best_metric'],
        'stopped_epoch': result['stopped_epoch'],
        'elapsed_min': elapsed / 60,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',
                        default='/kaggle/input/datasets/lequangphat260206/carve-mel-cache')
    parser.add_argument('--output-dir', default='/kaggle/working/results_hpo')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hpo-seed', type=int, default=42,
                        help='Seed cho random search sampling')
    args = parser.parse_args()

    print("=" * 72)
    print("CARVE — L2 HPO DeepFeat (random search)")
    print("=" * 72)
    print(f"n_trials: {args.n_trials}")
    print(f"epochs/trial: {args.epochs}")
    print(f"fold: {args.fold}, seed: {args.seed}")
    print(f"HPO sampling seed: {args.hpo_seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_cache(args.cache_dir)

    # Sample 20 configs
    rng = np.random.RandomState(args.hpo_seed)
    configs = [sample_config(rng) for _ in range(args.n_trials)]

    # Save configs sampled
    with open(output_dir / 'sampled_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"\nSampled {len(configs)} configs. Luu tai {output_dir}/sampled_configs.json")

    # Run trials
    trials = []
    t_start = time.time()
    for i, cfg in enumerate(configs, 1):
        try:
            result = run_trial(X, y, meta, cfg, args, i)
            trials.append(result)

            # Save incremental
            with open(output_dir / 'trials.json', 'w') as f:
                json.dump(trials, f, indent=2)

            # Print best so far
            best_so_far = max(trials, key=lambda t: t['best_val_roc_auc'])
            print(f"  Best so far: trial {best_so_far['trial_id']}, "
                  f"AUC={best_so_far['best_val_roc_auc']:.4f}")
        except Exception as e:
            print(f"  Trial {i} FAIL: {e}")
            trials.append({
                'trial_id': i, 'config': cfg,
                'error': str(e),
                'best_val_roc_auc': float('nan'),
            })

    total_min = (time.time() - t_start) / 60

    # Find best
    valid_trials = [t for t in trials if not np.isnan(t.get('best_val_roc_auc', np.nan))]
    if not valid_trials:
        print("LOI: Khong co trial nao thanh cong")
        return

    best = max(valid_trials, key=lambda t: t['best_val_roc_auc'])

    # Save best config
    best_config = {
        'trial_id': best['trial_id'],
        'config': best['config'],
        'best_val_roc_auc': best['best_val_roc_auc'],
        'best_epoch': best['best_epoch'],
        'fold': args.fold,
        'seed': args.seed,
        'epochs': args.epochs,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    # Summary
    print(f"\n{'=' * 72}")
    print("HPO SUMMARY")
    print(f"{'=' * 72}")
    print(f"Total time: {total_min:.1f} min ({total_min/60:.1f}h)")
    print(f"\nTop 5 trials:")
    top5 = sorted(valid_trials, key=lambda t: -t['best_val_roc_auc'])[:5]
    for t in top5:
        cfg = t['config']
        print(f"  Trial {t['trial_id']:2d}: AUC={t['best_val_roc_auc']:.4f} | "
              f"lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.2e} "
              f"bs={cfg['batch_size']} do={cfg['dropout']:.2f}")

    print(f"\nBest config (trial {best['trial_id']}):")
    print(f"  {json.dumps(best['config'], indent=4)}")
    print(f"  Val AUC: {best['best_val_roc_auc']:.4f}")

    print(f"\nLuu: {output_dir}/best_config.json")
    print(f"Luu: {output_dir}/trials.json")


if __name__ == '__main__':
    main()