#!/usr/bin/env python3
"""CARVE — Stage 5a: Commonality analysis (fixed formula)."""
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ACUTE_STRICT = {'positive_mild', 'positive_moderate'}
ACUTE_BROAD = ACUTE_STRICT | {'positive_asymp'}
RECOVERED = {'recovered_full'}
NEGATIVE = {'healthy'}
FEATURES = ['F0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean']

BRANCH_DEFS = {
    'A': {'positive': ACUTE_BROAD | RECOVERED, 'negative': NEGATIVE,
          'desc': 'Coswara full (acute + recovered vs healthy)'},
    'B': {'positive': ACUTE_BROAD, 'negative': NEGATIVE,
          'desc': 'Coswara acute-only (acute vs healthy)'},
}


def load_data():
    df = pd.read_csv('data/processed/handfeat_coswara_full.csv')
    df = df[df['extraction_status'] == 'ok'].copy()
    df = df.rename(columns={'a': 'age', 'g': 'gender'})
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    gender_map = {'male': 1, 'female': 0, 'Male': 1, 'Female': 0,
                  'M': 1, 'F': 0, 'm': 1, 'f': 0}
    df['gender_bin'] = df['gender'].astype(str).str.strip().map(gender_map)
    df = df.dropna(subset=FEATURES + ['age', 'gender_bin']).copy()
    df = df[df['age'] > 1]
    return df


def r2_ols(X, y):
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    y_pred = Xc @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def commonality_coefs(X, y, k):
    """Recursive commonality coefficients C_S cho moi subset khong rong."""
    r2_cache = {}
    for r in range(1, k + 1):
        for combo in combinations(range(k), r):
            r2_cache[combo] = r2_ols(X[:, list(combo)], y)

    C = {}
    for r in range(1, k + 1):
        for combo in combinations(range(k), r):
            # Sum C_T cho proper subset T (khong rong)
            sum_proper = 0.0
            for rr in range(1, r):
                for sub in combinations(combo, rr):
                    sum_proper += C[sub]
            C[combo] = r2_cache[combo] - sum_proper

    return r2_cache, C


def name_of(combo):
    short = {'F0_mean': 'F0', 'jitter_local': 'jit',
             'shimmer_local': 'shim', 'hnr_mean': 'hnr'}
    return '+'.join(short[FEATURES[i]] for i in combo)


def print_branch_report(branch, df, r2_cache, C):
    print(f"\n{'=' * 72}")
    print(f"BRANCH {branch}: {BRANCH_DEFS[branch]['desc']}")
    print(f"{'=' * 72}")
    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    r2_full = r2_cache[tuple(range(len(FEATURES)))]
    print(f"N = {len(df)} (pos={n_pos}, neg={n_neg})")
    print(f"R²_full = {r2_full:.6f}")

    print(f"\n--- Commonality coefficients C_S ---")
    print(f"{'Subset':20s} {'|S|':>3s} {'C_S':>12s}  Note")
    print("-" * 60)

    for r in range(1, len(FEATURES) + 1):
        for combo in combinations(range(len(FEATURES)), r):
            c = C[combo]
            note = ''
            if r >= 2 and c < -1e-4:
                note = 'AM (suppression)'
            elif r >= 2 and abs(c) < 1e-4:
                note = '~0'
            print(f"{name_of(combo):20s} {r:>3d} {c:+12.6f}  {note}")

    total = sum(C.values())
    print("-" * 60)
    print(f"{'Sum C_S':20s} {'':>3s} {total:+12.6f}")
    print(f"{'R²_full':20s} {'':>3s} {r2_full:+12.6f}")
    print(f"{'Diff':20s} {'':>3s} {total - r2_full:+.2e}")

    # Summary: bao nhieu % R² la unique (singleton) vs shared
    sum_single = sum(C[combo] for combo in combinations(range(len(FEATURES)), 1))
    sum_shared = total - sum_single
    print(f"\n--- TONG KET ---")
    print(f"  Unique (singleton C_i): {sum_single:.6f}  ({100*sum_single/r2_full:.1f}% R²_full)")
    print(f"  Shared (|S|>=2):        {sum_shared:.6f}  ({100*sum_shared/r2_full:.1f}% R²_full)")

    # Suppression count
    neg_pairs = sum(1 for combo in combinations(range(len(FEATURES)), 2)
                    if C[combo] < -1e-4)
    print(f"  Pairs co C_S < 0 (suppression): {neg_pairs}/6")


def main():
    print("=" * 72)
    print("CARVE — Stage 5a: Commonality analysis (fixed)")
    print("=" * 72)

    df_all = load_data()
    print(f"\nLoaded: {len(df_all)}")

    Path('results').mkdir(exist_ok=True)
    all_rows = []

    for branch in ['A', 'B']:
        spec = BRANCH_DEFS[branch]
        keep = spec['positive'] | spec['negative']
        sub = df_all[df_all['covid_status'].isin(keep)].copy()
        sub['label'] = sub['covid_status'].isin(spec['positive']).astype(int)

        X = sub[FEATURES].values
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        y = sub['label'].values

        r2_cache, C = commonality_coefs(X_std, y, len(FEATURES))
        print_branch_report(branch, sub, r2_cache, C)

        for combo, val in C.items():
            all_rows.append({
                'branch': branch,
                'size': len(combo),
                'subset': name_of(combo),
                'C_S': val,
            })

    out_csv = 'results/audit_rq1_commonality.csv'
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nLuu: {out_csv}")

    out_json = 'results/audit_rq1_commonality.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'features': FEATURES, 'results': all_rows},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"Luu: {out_json}")

    print(f"\n{'=' * 72}\nXONG.\n{'=' * 72}")


if __name__ == '__main__':
    main()