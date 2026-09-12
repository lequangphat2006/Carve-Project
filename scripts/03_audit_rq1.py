#!/usr/bin/env python3
"""CARVE — Stage 5a: Audit RQ1 tren nhanh A va B (Coswara-only).

3 lop audit (Muc 5.1.d CARVE v5.1):
  Lop 1: Raw test (Mann-Whitney U) + rank-biserial effect size
  Lop 2: Covariate-controlled (partial correlation, control age + gender)
  Lop 3: VIF check (threshold chinh 5, sensitivity 10)
  Lop 4: General dominance analysis (chi chay neu max VIF > 5)

Multi-comparison: Holm-Bonferroni trong tung family (4 features/family).

Branch definitions (chot truoc, xem implementation_notes.md B.1b):
  Branch A: positive = ACUTE_BROAD ∪ RECOVERED; negative = NEGATIVE
  Branch B: positive = ACUTE_BROAD; negative = NEGATIVE

Output:
  results/audit_rq1_branch_A.csv
  results/audit_rq1_branch_B.csv
  results/audit_rq1_summary.json
"""
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ==== Dinh nghia nhom (dong bo implementation_notes.md B.1) ====
ACUTE_STRICT = {'positive_mild', 'positive_moderate'}
ACUTE_BROAD = ACUTE_STRICT | {'positive_asymp'}
RECOVERED = {'recovered_full'}
NEGATIVE = {'healthy'}
FEATURES = ['F0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean']

BRANCH_DEFS = {
    'A': {
        'positive': ACUTE_BROAD | RECOVERED,
        'negative': NEGATIVE,
        'desc': 'Coswara full (acute + recovered vs healthy, exclude exposed)',
    },
    'B': {
        'positive': ACUTE_BROAD,
        'negative': NEGATIVE,
        'desc': 'Coswara acute-only (acute vs healthy)',
    },
}


def load_data():
    df = pd.read_csv('data/processed/handfeat_coswara_full.csv')
    df = df[df['extraction_status'] == 'ok'].copy()
    df = df.rename(columns={'a': 'age', 'g': 'gender'})
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    gender_map = {'male': 1, 'female': 0, 'Male': 1, 'Female': 0,
                  'M': 1, 'F': 0, 'm': 1, 'f': 0}
    df['gender_bin'] = df['gender'].astype(str).str.strip().map(gender_map)
    print(f"\nGender distribution (raw): {df['gender'].value_counts().to_dict()}")
    print(f"Gender encoded: male={int((df['gender_bin']==1).sum())}, "
          f"female={int((df['gender_bin']==0).sum())}, "
          f"unknown={int(df['gender_bin'].isna().sum())}")
    print(f"Age: min={df['age'].min()}, max={df['age'].max()}, "
          f"missing={df['age'].isna().sum()}")
    return df


def filter_branch(df, branch):
    spec = BRANCH_DEFS[branch]
    keep = spec['positive'] | spec['negative']
    sub = df[df['covid_status'].isin(keep)].copy()
    sub['label'] = sub['covid_status'].isin(spec['positive']).astype(int)
    return sub


def rank_biserial(x, y):
    """Rank-biserial correlation cho Mann-Whitney (duong: x > y)."""
    x = np.asarray(x); x = x[~np.isnan(x)]
    y = np.asarray(y); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    return r_rb, p_val


def partial_corr(x, y, covariates):
    """Partial correlation giua x va y, control covariates (them intercept)."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(covariates).any(axis=1))
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    cov = np.asarray(covariates)[mask]
    if len(x) < 10:
        return np.nan, np.nan, 0
    cov_ = np.column_stack([np.ones(len(cov)), cov])
    beta_x, *_ = np.linalg.lstsq(cov_, x, rcond=None)
    resid_x = x - cov_ @ beta_x
    beta_y, *_ = np.linalg.lstsq(cov_, y, rcond=None)
    resid_y = y - cov_ @ beta_y
    if resid_x.std() == 0 or resid_y.std() == 0:
        return np.nan, np.nan, len(x)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(x)


def compute_vif(X):
    """VIF cho tung cot cua X (2D numpy)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    vif = np.zeros(p)
    for i in range(p):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_others = np.column_stack([np.ones(n), X_others])
        beta, *_ = np.linalg.lstsq(X_others, y, rcond=None)
        y_pred = X_others @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[i] = 1 / (1 - r2) if r2 < 1 else np.inf
    return vif


def dominance_analysis(X, y):
    """General dominance analysis (Azen & Budescu 2003)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    def r2_subset(idx):
        if not idx:
            return 0.0
        Xs = X[:, list(idx)]
        Xs = np.column_stack([np.ones(n), Xs])
        beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        y_pred = Xs @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    contributions = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        marginal_sum = 0.0
        count = 0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                r2_without = r2_subset(subset)
                r2_with = r2_subset(tuple(subset) + (i,))
                marginal_sum += (r2_with - r2_without)
                count += 1
        contributions[i] = marginal_sum / count if count > 0 else 0.0
    return contributions


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down. Tra ve adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def audit_branch(df_branch, branch_name):
    print(f"\n{'=' * 72}")
    print(f"BRANCH {branch_name}: {BRANCH_DEFS[branch_name]['desc']}")
    print(f"{'=' * 72}")
    n_pos = int((df_branch['label'] == 1).sum())
    n_neg = int((df_branch['label'] == 0).sum())
    print(f"N: {len(df_branch)} (positive={n_pos}, negative={n_neg})")

    # Lop 1: Raw
    print(f"\n--- Lop 1: Raw test (Mann-Whitney, two-sided) ---")
    raw_p, raw_r = [], []
    for feat in FEATURES:
        pos = df_branch.loc[df_branch['label'] == 1, feat].dropna().values
        neg = df_branch.loc[df_branch['label'] == 0, feat].dropna().values
        r, p = rank_biserial(pos, neg)
        raw_p.append(p); raw_r.append(r)
        print(f"  {feat:16s}: r_rb={r:+.4f}, p={p:.4e}, n_pos={len(pos)}, n_neg={len(neg)}")
    raw_adj = holm_bonferroni(raw_p)
    for feat, p_adj in zip(FEATURES, raw_adj):
        print(f"  {feat:16s}: p_holm={p_adj:.4e} {'*' if p_adj < 0.05 else ''}")

    # Lop 2: Covariate-controlled
    print(f"\n--- Lop 2: Covariate-controlled (partial corr, control age + gender) ---")
    cov = df_branch[['age', 'gender_bin']].values
    cov_p, cov_r = [], []
    for feat in FEATURES:
        r, p, n_used = partial_corr(df_branch[feat].values, df_branch['label'].values, cov)
        cov_p.append(p if not np.isnan(p) else 1.0); cov_r.append(r)
        print(f"  {feat:16s}: r_partial={r:+.4f}, p={p:.4e}, n_used={n_used}")
    cov_adj = holm_bonferroni(cov_p)
    for feat, p_adj in zip(FEATURES, cov_adj):
        print(f"  {feat:16s}: p_holm={p_adj:.4e} {'*' if p_adj < 0.05 else ''}")

    # Lop 3: VIF
    print(f"\n--- Lop 3: VIF ---")
    df_mv = df_branch.dropna(subset=FEATURES + ['age', 'gender_bin'])
    print(f"  Multivariate sample size: {len(df_mv)}")
    if len(df_mv) >= 20:
        X = df_mv[FEATURES].values
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        vif = compute_vif(X_std)
        for feat, v in zip(FEATURES, vif):
            print(f"  {feat:16s}: VIF={v:.3f} {('>5' if v > 5 else '')}")
        max_vif = float(vif.max())
        print(f"  max(VIF) = {max_vif:.3f}")
    else:
        vif = np.full(len(FEATURES), np.nan)
        max_vif = np.nan
        print("  Khong du mau de tinh VIF")

    # Lop 4: Dominance (conditional on VIF rule)
    print(f"\n--- Lop 4: Dominance analysis (rule: max VIF > 5) ---")
    if not np.isnan(max_vif) and max_vif > 5:
        print(f"  max VIF = {max_vif:.3f} > 5 -> CHAY dominance analysis")
        X = df_mv[FEATURES].values
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        y = df_mv['label'].values
        dom = dominance_analysis(X_std, y)
        for feat, d in zip(FEATURES, dom):
            print(f"  {feat:16s}: dominance={d:.4f} ({d*100:.1f}%)")
        print(f"  Total R^2 = {dom.sum():.4f}")
    else:
        print(f"  max VIF <= 5 hoac khong tinh duoc -> DOC TRUC TIEP he so hoi quy")
        dom = np.full(len(FEATURES), np.nan)

    rows = []
    for i, feat in enumerate(FEATURES):
        rows.append({
            'branch': branch_name,
            'feature': feat,
            'n_pos': n_pos, 'n_neg': n_neg,
            'raw_r_rb': raw_r[i], 'raw_p': raw_p[i],
            'raw_p_holm': raw_adj[i], 'raw_sig_holm': bool(raw_adj[i] < 0.05),
            'cov_r_partial': cov_r[i], 'cov_p': cov_p[i],
            'cov_p_holm': cov_adj[i], 'cov_sig_holm': bool(cov_adj[i] < 0.05),
            'vif': vif[i],
            'dominance': dom[i],
        })
    return rows


def main():
    print("=" * 72)
    print("CARVE — Stage 5a: Audit RQ1 (nhanh A va B)")
    print("=" * 72)

    df = load_data()
    print(f"\nLoaded (extraction_status == ok): {len(df)} samples")
    print(f"covid_status distribution:")
    for v, n in df['covid_status'].value_counts().items():
        print(f"  {v}: {n}")

    Path('results').mkdir(exist_ok=True)
    all_results = []
    for branch in ['A', 'B']:
        df_b = filter_branch(df, branch)
        rows = audit_branch(df_b, branch)
        all_results.extend(rows)
        out = Path(f'results/audit_rq1_branch_{branch}.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nLuu: {out}")

    out_json = Path('results/audit_rq1_summary.json')
    def _ser(d):
        return {k: (list(v) if isinstance(v, set) else v) for k, v in d.items()}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'branch_defs': {k: _ser(v) for k, v in BRANCH_DEFS.items()},
            'features': FEATURES,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"Luu: {out_json}")

    print(f"\n{'=' * 72}\nXONG.\n{'=' * 72}")


if __name__ == '__main__':
    main()