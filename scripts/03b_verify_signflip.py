#!/usr/bin/env python3
"""CARVE — Stage 5a verification: Confirm sign flip bang logistic regression.

So sanh he so logreg (raw) vs logreg (adjusted age+gender).
Feature duoc StandardScaler truoc khi fit de he so so sanh duoc.

Neu raw coef va adjusted coef khac dau → confirm sign flip.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


ACUTE_STRICT = {'positive_mild', 'positive_moderate'}
ACUTE_BROAD = ACUTE_STRICT | {'positive_asymp'}
RECOVERED = {'recovered_full'}
NEGATIVE = {'healthy'}
FEATURES = ['F0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean']

BRANCH_DEFS = {
    'A': {'positive': ACUTE_BROAD | RECOVERED, 'negative': NEGATIVE},
    'B': {'positive': ACUTE_BROAD, 'negative': NEGATIVE},
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
    df = df[df['age'] > 1]  # loai age=1
    return df


def fit_logreg(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
    clf.fit(Xs, y)
    return clf.coef_[0]


def analyze_branch(df, branch):
    spec = BRANCH_DEFS[branch]
    keep = spec['positive'] | spec['negative']
    sub = df[df['covid_status'].isin(keep)].copy()
    sub['label'] = sub['covid_status'].isin(spec['positive']).astype(int)

    X_feat = sub[FEATURES].values
    y = sub['label'].values
    X_cov = sub[['age', 'gender_bin']].values

    # Raw logreg (chi features)
    coef_raw = fit_logreg(X_feat, y)

    # Adjusted logreg (features + age + gender)
    X_adj = np.column_stack([X_feat, X_cov])
    coef_adj_all = fit_logreg(X_adj, y)
    coef_adj = coef_adj_all[:len(FEATURES)]  # chi lay features

    print(f"\n{'=' * 72}")
    print(f"BRANCH {branch}: N={len(sub)} (pos={int(y.sum())}, neg={int((1-y).sum())})")
    print(f"{'=' * 72}")

    print(f"\n{'Feature':16s} {'Raw coef':>12s} {'Adj coef':>12s} {'Sign flip':>12s}")
    print("-" * 56)
    results = []
    for i, feat in enumerate(FEATURES):
        raw = coef_raw[i]
        adj = coef_adj[i]
        flip = (raw * adj) < 0
        print(f"{feat:16s} {raw:+12.4f} {adj:+12.4f} {'YES' if flip else 'no':>12s}")
        results.append({'feature': feat, 'raw_coef': raw, 'adj_coef': adj,
                        'sign_flip': flip})

    n_flip = sum(r['sign_flip'] for r in results)
    print(f"\n{'-' * 56}")
    print(f"Sign flip: {n_flip}/{len(FEATURES)} features")

    # Covariate coefficients
    print(f"\nCovariate coef (adjusted model):")
    print(f"  age       : {coef_adj_all[len(FEATURES)]:+.4f}")
    print(f"  gender_bin: {coef_adj_all[len(FEATURES) + 1]:+.4f}")

    return results, coef_adj_all


def main():
    print("=" * 72)
    print("CARVE — Stage 5a verification: sign flip via logistic regression")
    print("=" * 72)

    df = load_data()
    print(f"\nLoaded (after dropna + age>1): {len(df)}")

    all_results = []
    for branch in ['A', 'B']:
        res, _ = analyze_branch(df, branch)
        for r in res:
            r['branch'] = branch
            all_results.append(r)

    Path('results').mkdir(exist_ok=True)
    out = 'results/audit_rq1_logreg_verify.csv'
    pd.DataFrame(all_results).to_csv(out, index=False)
    print(f"\nLuu: {out}")

    # Verdict
    n_flip_total = sum(r['sign_flip'] for r in all_results)
    n_total = len(all_results)
    print(f"\n{'=' * 72}")
    print(f"VERDICT: {n_flip_total}/{n_total} (feature x branch) confirm sign flip")
    if n_flip_total == n_total:
        print("=> CONFIRM: sign flip la that, khong phai artifact cua partial corr.")
        print("=> Finding co the viet vao paper.")
    else:
        print("=> CANH BAO: khong phai tat ca deu flip. Can dieu tra them.")
    print("=" * 72)


if __name__ == '__main__':
    main()