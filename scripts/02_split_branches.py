#!/usr/bin/env python3
"""CARVE — Stage 1: Tao nhanh du lieu tu CSV feature + metadata.

Input:
- data/processed/handfeat_coswara_full.csv (2746 rows, tu Kaggle)
- data/raw/combined_data.csv (metadata Coswara)

Output:
- data/processed/handfeat_coswara_branch_B.csv (acute-only)
- Branch A = handfeat_coswara_full.csv (giu nguyen, da co san)

Dinh nghia nhom (Muc B.1 implementation_notes.md):
- ACUTE_STRICT (A1) = {positive_mild, positive_moderate}
- ACUTE_BROAD (A2)  = A1 ∪ {positive_asymp}
- NEGATIVE          = {healthy}
- RECOVERED         = {recovered_full}
- EXPOSED           = {no_resp_illness_exposed}
- EXCLUDED          = {resp_illness_not_identified, under_validation}
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ==== Dinh nghia nhom (Muc B.1) ====
ACUTE_STRICT = {'positive_mild', 'positive_moderate'}
ACUTE_BROAD = ACUTE_STRICT | {'positive_asymp'}
NEGATIVE = {'healthy'}
RECOVERED = {'recovered_full'}
EXPOSED = {'no_resp_illness_exposed'}
EXCLUDED = {'resp_illness_not_identified', 'under_validation'}

KEEP_B = ACUTE_BROAD | NEGATIVE  # nhanh B: acute + healthy


def main():
    feat_path = Path('data/processed/handfeat_coswara_full.csv')
    meta_path = Path('data/raw/combined_data.csv')
    out_B = Path('data/processed/handfeat_coswara_branch_B.csv')

    print("=" * 60)
    print("CARVE — Stage 1: Split branches")
    print("=" * 60)

    # Load
    print(f"\nDoc feature CSV: {feat_path}")
    feat = pd.read_csv(feat_path)
    print(f"  Rows: {len(feat)}, unique speakers: {feat['speaker_id'].nunique()}")

    print(f"\nDoc metadata: {meta_path}")
    meta = pd.read_csv(meta_path)
    print(f"  Rows: {len(meta)}")

    # Join
    print("\nJoin feature <-> metadata qua speaker_id <-> id...")
    merged = feat.merge(
        meta[['id', 'covid_status', 'a', 'g']],
        left_on='speaker_id', right_on='id',
        how='left', validate='many_to_one'
    )
    n_missing_meta = merged['covid_status'].isna().sum()
    print(f"  Joined rows: {len(merged)}")
    print(f"  Missing covid_status sau join: {n_missing_meta}")
    if n_missing_meta > 0:
        missing_ids = merged[merged['covid_status'].isna()]['speaker_id'].head(5).tolist()
        print(f"  Sample missing: {missing_ids}")

    # Verify N khong doi
    assert len(merged) == len(feat), f"Join lam thay doi so rows: {len(feat)} -> {len(merged)}"

    # Branch A (full) — them covid_status
    merged.to_csv(feat_path, index=False)
    print(f"\nBranch A (full): {len(merged)} rows (da cap nhat covid_status)")

    # Branch B (acute-only)
    print("\nTao Branch B (acute-only)...")
    branch_B = merged[merged['covid_status'].isin(KEEP_B)].copy()
    branch_B['label'] = branch_B['covid_status'].isin(ACUTE_BROAD).astype(int)

    # Verify
    n_pos = int(branch_B['label'].sum())
    n_neg = int((branch_B['label'] == 0).sum())
    print(f"\n=== Branch B ===")
    print(f"  Rows: {len(branch_B)}")
    print(f"  Positive (acute): {n_pos}")
    print(f"  Negative (healthy): {n_neg}")
    print(f"  Prevalence: {branch_B['label'].mean():.2%}")

    # N_B A1 vs A2 sensitivity
    n_a1 = branch_B['covid_status'].isin(ACUTE_STRICT).sum()
    n_a2 = branch_B['covid_status'].isin(ACUTE_BROAD).sum()
    print(f"\n=== Sensitivity ===")
    print(f"  A1 (strict): {n_a1} acute + {n_neg} healthy = {n_a1 + n_neg}")
    print(f"  A2 (broad):  {n_a2} acute + {n_neg} healthy = {n_a2 + n_neg}")

    # QC distribution
    print(f"\n=== Branch B feature distribution (OK only) ===")
    ok_B = branch_B[branch_B['extraction_status'] == 'ok']
    for col in ['F0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean', 'pct_voiced']:
        v = ok_B[col].dropna()
        print(f"  {col:16s}: min={v.min():8.4f}, median={v.median():8.4f}, max={v.max():8.4f}, missing={ok_B[col].isna().sum()}")

    # Status breakdown
    print(f"\n=== Branch B extraction_status ===")
    print(f"  {branch_B['extraction_status'].value_counts().to_dict()}")

    # Luu
    branch_B.to_csv(out_B, index=False)
    print(f"\nLuu: {out_B} ({out_B.stat().st_size / 1024:.1f} KB)")
    print(f"Luu: {feat_path} (da them covid_status, {feat_path.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("XONG.")
    print("=" * 60)


if __name__ == '__main__':
    main()