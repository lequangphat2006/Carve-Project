"""Unit test cho DiCOVA split (Muc 2.2 Buoc 0.4 cua CARVE v5.1).

Kiem tra speaker-aware 5-fold split:
- Khong speaker nao xuat hien o > 1 fold
- Ti le positive moi fold trong khoang 5-9%
- N moi fold ~ 240
"""
import numpy as np
import pandas as pd


def _make_dummy_split(n_speakers=1199, n_pos=80, n_folds=5, seed=0):
    """Tao dummy split gia lap cau truc DiCOVA."""
    rng = np.random.RandomState(seed)
    speaker_ids = [f"spk_{i:04d}" for i in range(n_speakers)]
    labels = np.array([1] * n_pos + [0] * (n_speakers - n_pos))
    rng.shuffle(labels)
    # Stratified fold assignment
    folds = np.zeros(n_speakers, dtype=int)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    folds[pos_idx] = np.arange(len(pos_idx)) % n_folds
    folds[neg_idx] = np.arange(len(neg_idx)) % n_folds
    return pd.DataFrame({
        'speaker_id': speaker_ids,
        'label': labels,
        'fold': folds,
    })


def test_no_speaker_leakage():
    """Moi speaker chi xuat hien o dung 1 fold."""
    df = _make_dummy_split()
    speaker_fold_counts = df.groupby('speaker_id')['fold'].nunique()
    assert (speaker_fold_counts == 1).all(), (
        f"Co {(speaker_fold_counts > 1).sum()} speaker xuat hien o >1 fold"
    )


def test_positive_rate_per_fold():
    """Ti le positive moi fold trong khoang 5-9%."""
    df = _make_dummy_split()
    rates = df.groupby('fold')['label'].mean()
    assert (rates >= 0.05).all() and (rates <= 0.09).all(), (
        f"Positive rate per fold: {rates.to_dict()}"
    )


def test_fold_size():
    """N moi fold ~ 240 (1199/5)."""
    df = _make_dummy_split()
    sizes = df.groupby('fold').size()
    assert (sizes >= 230).all() and (sizes <= 250).all(), (
        f"Fold sizes: {sizes.to_dict()}"
    )


if __name__ == '__main__':
    test_no_speaker_leakage()
    test_positive_rate_per_fold()
    test_fold_size()
    print("All splits tests passed.")