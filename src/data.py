"""CARVE — Data pipeline cho DeepFeat (mel-spectrogram + SpecAugment).

Config chot truoc (implementation_notes.md E.5b):
- Sample rate: 16 kHz
- n_fft: 1024, hop_length: 160 (10 ms), n_mels: 128
- fmin: 20 Hz, fmax: 8000 Hz
- T = 500 frames (~5s), crop tu dau (0-5s), pad zero neu ngan hon
- Log-mel + mean/std normalization (fit tren train fold)
- SpecAugment standard preset: freq_mask=27, time_mask=40, 2 masks moi loai
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


# ==== Config chot truoc ====
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128
FMIN = 20
FMAX = 8000
T_FRAMES = 500  # 5s
CROP_FROM_START = True  # Cat tu dau


def load_audio(wav_path, target_sr=SAMPLE_RATE):
    """Load wav, resample ve target_sr, tra ve mono tensor [n_samples].

    Raises RuntimeError neu file corrupt/silent (0 samples).
    """
    try:
        waveform, sr = torchaudio.load(str(wav_path))
    except Exception as e:
        raise RuntimeError(f"torchaudio.load failed: {e}") from e

    if waveform.numel() == 0:
        raise RuntimeError(f"Audio file 0 samples: {wav_path}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [n_samples]


class MelTransform:
    """Mel-spectrogram + log + pad/crop ve T co dinh."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                 hop_length=HOP_LENGTH, n_mels=N_MELS,
                 fmin=FMIN, fmax=FMAX, T=T_FRAMES):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.T = T
        self.hop_length = hop_length

    def __call__(self, waveform):
        """waveform: [n_samples] -> mel [1, n_mels, T]."""
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-8)

        T_cur = mel.size(1)
        if T_cur >= self.T:
            mel = mel[:, :self.T]
        else:
            pad = torch.zeros(mel.size(0), self.T - T_cur)
            mel = torch.cat([mel, pad], dim=1)

        return mel.unsqueeze(0)


class SpecAugment(nn.Module):
    """SpecAugment voi preset chot truoc (Muc III.8 CARVE v5.1).

    Standard preset: freq_mask=27, time_mask=40, n_freq=2, n_time=2.
    """

    def __init__(self, freq_mask_param=27, time_mask_param=40,
                 n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, mel):
        """mel: [1, n_mels, T] -> masked mel."""
        for _ in range(self.n_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.n_time_masks):
            mel = self.time_mask(mel)
        return mel


class VowelEDataset(Dataset):
    """Dataset vowel-e cho DeepFeat.

    Args:
        df: pandas DataFrame voi cot ['wav_path', 'label']
        mel_transform: MelTransform instance
        spec_augment: SpecAugment hoac None (None cho val/test)
        mel_mean, mel_std: scalar hoac None (None = khong normalize)
    """

    def __init__(self, df, mel_transform=None, spec_augment=None,
                 mel_mean=None, mel_std=None):
        self.df = df.reset_index(drop=True)
        self.mel_transform = mel_transform or MelTransform()
        self.spec_augment = spec_augment
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['wav_path']
        label = float(row['label'])

        waveform = load_audio(wav_path)
        mel = self.mel_transform(waveform)

        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean) / (self.mel_std + 1e-8)

        if self.spec_augment is not None:
            mel = self.spec_augment(mel)

        return mel, torch.tensor(label, dtype=torch.float32)


def compute_mel_stats(df, mel_transform=None, max_samples=None):
    """Tinh mean/std cua log-mel tren train fold."""
    mel_transform = mel_transform or MelTransform()
    df_ = df if max_samples is None else df.head(max_samples)

    all_vals = []
    for _, row in df_.iterrows():
        try:
            waveform = load_audio(row['wav_path'])
            mel = mel_transform(waveform)
            all_vals.append(mel.flatten())
        except Exception:
            continue

    if not all_vals:
        return 0.0, 1.0

    vals = torch.cat(all_vals)
    return float(vals.mean()), float(vals.std())


if __name__ == '__main__':
    print("=== Test MelTransform ===")
    mt = MelTransform()
    wav = torch.randn(SAMPLE_RATE * 3)
    mel = mt(wav)
    print(f"Input: 3s audio ({wav.shape})")
    print(f"Mel shape: {tuple(mel.shape)}")
    assert mel.shape == (1, N_MELS, T_FRAMES)

    wav_long = torch.randn(SAMPLE_RATE * 10)
    mel_long = mt(wav_long)
    print(f"\nInput: 10s audio ({wav_long.shape})")
    print(f"Mel shape: {tuple(mel_long.shape)} (crop tu dau)")
    assert mel_long.shape == (1, N_MELS, T_FRAMES)

    wav_short = torch.randn(SAMPLE_RATE * 2)
    mel_short = mt(wav_short)
    print(f"\nInput: 2s audio ({wav_short.shape})")
    print(f"Mel shape: {tuple(mel_short.shape)} (pad zero)")

    print("\n=== Test SpecAugment ===")
    sa = SpecAugment()
    mel_aug = sa(mel.clone())
    print(f"Before: mean={mel.mean():.4f}, std={mel.std():.4f}")
    print(f"After:  mean={mel_aug.mean():.4f}, std={mel_aug.std():.4f}")
    print(f"Change: {(mel_aug != mel).sum().item()} values masked")
    assert (mel_aug != mel).any()

    print("\nOK")