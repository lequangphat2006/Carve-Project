#!/usr/bin/env python3
"""CARVE — Pre-compute mel-spectrogram cache cho Coswara subset.

Muc dich: chuyen audio (12 GB) thanh mel cache (~300 MB) de train nhanh.

Subset: ACUTE_BROAD | NEGATIVE (loai recovered/exposed/excluded)
  - ACUTE_BROAD = {positive_mild, positive_moderate, positive_asymp}
  - NEGATIVE = {healthy}
  - Total ~2114 speakers

Output: mel cache dang npz per-day, sau do merge thanh 1 file lon.

Usage:
    # Xu ly 1 ngay
    python scripts/06_precompute_mel.py --audio-dir /path/to/ext_DATE \\
        --meta-csv data/raw/combined_data.csv --date 20200413 \\
        --output-dir /kaggle/working/mel_cache

    # Merge tat ca thanh 1 file
    python scripts/06_precompute_mel.py --merge \\
        --output-dir /kaggle/working/mel_cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MelTransform, load_audio, N_MELS, T_FRAMES


# ==== Subset definition (dong bo B.1) ====
ACUTE_BROAD = {'positive_mild', 'positive_moderate', 'positive_asymp'}
NEGATIVE = {'healthy'}
KEEP = ACUTE_BROAD | NEGATIVE


def process_day(audio_dir, meta_csv, date, output_dir):
    """Xu ly 1 ngay: extract wav -> mel -> luu npz."""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"mel_{date}.npz"
    if out_path.exists():
        print(f"  Da co: {out_path} — skip")
        return

    # Load metadata
    meta = pd.read_csv(meta_csv)
    meta = meta[['id', 'covid_status']].rename(columns={'id': 'speaker_id'})

    # Filter subset
    meta_subset = meta[meta['covid_status'].isin(KEEP)].copy()
    subset_speakers = set(meta_subset['speaker_id'])
    print(f"  Subset speakers (ACUTE_BROAD | NEGATIVE): {len(subset_speakers)}")

    # Tim wav
    wavs = sorted(audio_dir.rglob("*vowel-e*.wav"))
    wavs = [w for w in wavs if not w.name.startswith("._")]
    print(f"  Wav files trong folder: {len(wavs)}")

    # Filter wav theo subset
    mel_tf = MelTransform()
    mels, labels, speaker_ids, wav_names = [], [], [], []
    n_skip_corrupt = 0
    n_skip_not_subset = 0

    for i, w in enumerate(wavs):
        sid = w.parent.name
        if sid not in subset_speakers:
            n_skip_not_subset += 1
            continue

        try:
            waveform = load_audio(w)
            mel = mel_tf(waveform)  # [1, 128, 500]
            mels.append(mel.squeeze(0).numpy().astype(np.float16))

            status = meta_subset.loc[
                meta_subset['speaker_id'] == sid, 'covid_status'
            ].iloc[0]
            label = 1 if status in ACUTE_BROAD else 0
            labels.append(label)
            speaker_ids.append(sid)
            wav_names.append(w.name)
        except Exception as e:
            n_skip_corrupt += 1
            continue

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(wavs)}...")

    print(f"  Kept: {len(mels)} samples")
    print(f"  Skipped (not subset): {n_skip_not_subset}")
    print(f"  Skipped (corrupt): {n_skip_corrupt}")

    if len(mels) == 0:
        print(f"  CANH BAO: 0 samples cho ngay {date}")
        return

    X = np.stack(mels)  # [N, 128, 500] float16
    y = np.array(labels, dtype=np.int8)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        speaker_ids=np.array(speaker_ids),
        wav_names=np.array(wav_names),
        date=date,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Luu: {out_path} ({size_mb:.1f} MB)")


def merge_all(output_dir):
    """Merge tat ca mel_*.npz thanh 1 file lon."""
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("mel_*.npz"))
    print(f"Tim thay {len(files)} file mel_*.npz")

    if not files:
        print("LOI: Khong co file nao de merge")
        return

    all_X, all_y, all_sid, all_wname, all_date = [], [], [], [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        all_X.append(data['X'])
        all_y.append(data['y'])
        all_sid.extend(data['speaker_ids'].tolist())
        all_wname.extend(data['wav_names'].tolist())
        all_date.extend([str(data['date'])] * len(data['y']))
        print(f"  {f.name}: {data['X'].shape}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.DataFrame({
        'speaker_id': all_sid,
        'wav_name': all_wname,
        'date': all_date,
        'label': y,
    })

    print(f"\nTong X: {X.shape} (dtype={X.dtype})")
    print(f"Tong y: {y.shape}, N_pos={int((y==1).sum())}, N_neg={int((y==0).sum())}")
    print(f"Speakers unique: {meta['speaker_id'].nunique()}")
    print(f"Days: {meta['date'].nunique()}")

    x_path = output_dir / "mel_cache_X.npy"
    y_path = output_dir / "mel_cache_y.npy"
    meta_path = output_dir / "mel_cache_meta.csv"

    np.save(x_path, X)
    np.save(y_path, y)
    meta.to_csv(meta_path, index=False)

    size_x = x_path.stat().st_size / 1024 / 1024
    size_y = y_path.stat().st_size / 1024 / 1024
    print(f"\nLuu:")
    print(f"  {x_path} ({size_x:.1f} MB)")
    print(f"  {y_path} ({size_y:.1f} MB)")
    print(f"  {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', type=str, default=None)
    parser.add_argument('--meta-csv', type=str,
                        default='data/raw/combined_data.csv')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    print("=" * 72)
    print("CARVE — Pre-compute mel cache")
    print("=" * 72)

    if args.merge:
        print("\nMode: MERGE")
        merge_all(args.output_dir)
        return

    if not args.audio_dir or not args.date:
        print("LOI: Can --audio-dir va --date (hoac --merge)")
        sys.exit(1)

    print(f"\nMode: PROCESS DAY")
    print(f"Date: {args.date}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Meta CSV: {args.meta_csv}")
    print(f"Output dir: {args.output_dir}")

    process_day(args.audio_dir, args.meta_csv, args.date, args.output_dir)

    print("\nOK")


if __name__ == '__main__':
    main()