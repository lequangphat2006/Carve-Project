#!/usr/bin/env python3
"""CARVE — L0 Smoke Test cho DeepFeat pipeline.

Chay 1 fold x 1 seed x vai epoch tren audio that de verify:
- Load wav -> mel-spectrogram [1, 128, 500]
- Forward + backward
- Training loop khong NaN
- ROC-AUC tinh duoc

Luu y: label la GIA (random 0/1) — chi de verify pipeline chay,
khong phai danh gia performance. Moi speaker gan 1 label co dinh.

Usage:
    python scripts/05_smoke_test.py --audio-dir /path/to/extracted
    python scripts/05_smoke_test.py --audio-dir ... --epochs 2 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import VowelEDataset, MelTransform, SpecAugment, compute_mel_stats
from src.models import DeepFeat, count_parameters, gradcam_from_features
from src.train import TrainConfig, fit, predict_proba, set_seed


def find_wavs(audio_dir):
    audio_dir = Path(audio_dir)
    wavs = sorted(audio_dir.rglob("*vowel-e*.wav"))
    # Loai macOS metadata
    wavs = [w for w in wavs if not w.name.startswith("._")]
    return wavs


def build_df(wavs, seed=0):
    """Build DataFrame voi speaker_id, wav_path, label (fake)."""
    rng = np.random.RandomState(seed)
    speaker_ids = [w.parent.name for w in wavs]
    unique_speakers = sorted(set(speaker_ids))

    # Gan label co dinh cho moi speaker (fake)
    speaker_label = {sid: int(rng.randint(0, 2)) for sid in unique_speakers}

    df = pd.DataFrame({
        'speaker_id': speaker_ids,
        'wav_path': [str(w) for w in wavs],
        'label': [speaker_label[s] for s in speaker_ids],
    })
    return df


def split_by_speaker(df, train_ratio=0.8, seed=0):
    """Split train/val theo speaker_id (khong leak speaker)."""
    rng = np.random.RandomState(seed)
    speakers = df['speaker_id'].unique()
    rng.shuffle(speakers)
    n_train = int(len(speakers) * train_ratio)
    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train:])
    train_df = df[df['speaker_id'].isin(train_spk)].reset_index(drop=True)
    val_df = df[df['speaker_id'].isin(val_spk)].reset_index(drop=True)
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', required=True,
                        help='Path toi folder chua cac speaker vowel-e wav')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-files', type=int, default=None,
                        help='Gioi han so file de chay nhanh (default: all)')
    args = parser.parse_args()

    print("=" * 72)
    print("CARVE — L0 Smoke Test (DeepFeat)")
    print("=" * 72)

    # 1. Seed
    set_seed(args.seed, deterministic=True)
    print(f"\nSeed: {args.seed}, deterministic=True")

    # 2. Tim audio
    print(f"\nTim vowel-e wav trong: {args.audio_dir}")
    wavs = find_wavs(args.audio_dir)
    print(f"  Found: {len(wavs)} files")
    if len(wavs) == 0:
        print("LOI: Khong tim thay file nao. Kiem tra --audio-dir.")
        sys.exit(1)

    if args.max_files:
        wavs = wavs[:args.max_files]
        print(f"  Gioi han: {len(wavs)} files")

    # 3. Build df + split
    df = build_df(wavs, seed=args.seed)
    print(f"\nSpeakers: {df['speaker_id'].nunique()}")
    print(f"Labels (fake): {df['label'].value_counts().to_dict()}")

    train_df, val_df = split_by_speaker(df, train_ratio=0.8, seed=args.seed)
    print(f"\nTrain: {len(train_df)} samples ({train_df['speaker_id'].nunique()} spk)")
    print(f"Val:   {len(val_df)} samples ({val_df['speaker_id'].nunique()} spk)")

    # Check ca 2 split co ca 2 label
    for name, d in [('train', train_df), ('val', val_df)]:
        counts = d['label'].value_counts().to_dict()
        if len(counts) < 2:
            print(f"CANH BAO: {name} chi co 1 label: {counts} — ROC-AUC se NaN")

    # 4. Build dataset
    print("\nBuild dataset...")
    mel_tf = MelTransform()
    spec_aug = SpecAugment()  # standard preset

    # Compute mel stats tren train (max 20 file de nhanh)
    print("  Tinh mel mean/std tren train (20 file)...")
    mel_mean, mel_std = compute_mel_stats(train_df, mel_tf, max_samples=20)
    print(f"  mel_mean={mel_mean:.4f}, mel_std={mel_std:.4f}")

    train_ds = VowelEDataset(train_df, mel_tf, spec_augment=spec_aug,
                             mel_mean=mel_mean, mel_std=mel_std)
    val_ds = VowelEDataset(val_df, mel_tf, spec_augment=None,
                           mel_mean=mel_mean, mel_std=mel_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # 5. Verify 1 batch
    print("\nVerify 1 batch:")
    xb, yb = next(iter(train_loader))
    print(f"  Batch shape: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    assert xb.shape[1:] == (1, 128, 500), f"Shape sai: {xb.shape}"
    assert not torch.isnan(xb).any(), "Mel co NaN"

    # 6. Model
    print("\nBuild model:")
    model = DeepFeat()
    print(f"  DeepFeat params: {count_parameters(model):,}")

    # 7. Train
    print(f"\nTrain {args.epochs} epochs...")
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=10,
        device='auto',
        verbose=True,
    )
    result = fit(model, train_loader, val_loader, config)

    print(f"\n=== Ket qua L0 ===")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val ROC-AUC: {result['best_metric']:.4f}")
    print(f"Stopped epoch: {result['stopped_epoch']}")
    print(f"History length: {len(result['history'])}")

    # 8. Predict + Grad-CAM sanity
    print("\nPredict on val:")
    probs, labels = predict_proba(result['model'], val_loader, device='auto')
    print(f"  probs shape: {probs.shape}")
    print(f"  probs range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Grad-CAM sanity: 1 sample tu val
    print("\nGrad-CAM sanity check:")
    model = result['model']
    model.eval()
    device = next(model.parameters()).device
    x_one = val_ds[0][0].unsqueeze(0).to(device)
    x_one.requires_grad = True
    logit = model(x_one)
    logit.backward()
    cam = gradcam_from_features(model._feature_map)
    print(f"  CAM shape: {tuple(cam.shape)}")
    print(f"  CAM range: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
    cam_std = cam.std().item()
    print(f"  CAM std: {cam_std:.4f}")
    if cam_std < 1e-6:
        print("  CANH BAO: CAM uniform — model chua hoc duoc feature")
    else:
        print("  CAM khong uniform — OK")

    print("\n" + "=" * 72)
    print("L0 PASS — Pipeline chay end-to-end, khong NaN.")
    print("=" * 72)


if __name__ == '__main__':
    main()