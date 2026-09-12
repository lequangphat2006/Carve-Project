#!/usr/bin/env python3
"""CARVE — Stage 5a: Visualization cho audit RQ1.

Tao 3 figure cho paper:
  Fig 1: Effect size comparison (forest plot) — raw vs multivariate
  Fig 2: Commonality decomposition (stacked bar) — unique vs shared
  Fig 3: R² progression (forward selection)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path('results')
OUT = Path('results/figures')
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['F0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean']
FEAT_SHORT = {'F0_mean': 'F0', 'jitter_local': 'Jitter',
              'shimmer_local': 'Shimmer', 'hnr_mean': 'HNR'}
COLORS = {'A': '#1f77b4', 'B': '#ff7f0e'}


def load_audit():
    a = pd.read_csv(RESULTS / 'audit_rq1_branch_A.csv')
    b = pd.read_csv(RESULTS / 'audit_rq1_branch_B.csv')
    return pd.concat([a, b], ignore_index=True)


def load_logreg():
    return pd.read_csv(RESULTS / 'audit_rq1_logreg_verify.csv')


def load_commonality():
    return pd.read_csv(RESULTS / 'audit_rq1_commonality.csv')


def fig1_forest(audit, logreg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, branch in zip(axes, ['A', 'B']):
        df_a = audit[audit['branch'] == branch].set_index('feature')
        df_l = logreg[logreg['branch'] == branch].set_index('feature')

        y = np.arange(len(FEATURES))
        raw = [df_a.loc[f, 'raw_r_rb'] for f in FEATURES]
        adj = [df_l.loc[f, 'adj_coef'] for f in FEATURES]

        ax.barh(y - 0.2, raw, height=0.35, color='#2ca02c',
                alpha=0.8, label='Univariate r_rb')
        ax.barh(y + 0.2, adj, height=0.35, color='#d62728',
                alpha=0.8, label='Multivariate adj coef (logreg)')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_yticks(y)
        ax.set_yticklabels([FEAT_SHORT[f] for f in FEATURES])
        ax.set_xlabel('Effect size')
        ax.set_title(f'Branch {branch}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Fig 1 — Univariate vs Multivariate effect sizes\n'
                 '(sign flip = suppression effect)', fontsize=12)
    plt.tight_layout()
    out = OUT / 'fig1_effect_sizes.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Luu: {out}')


def fig2_commonality(comm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, branch in zip(axes, ['A', 'B']):
        df = comm[comm['branch'] == branch].copy()
        df = df.sort_values('size')

        labels = df['subset'].tolist()
        vals = df['C_S'].tolist()
        colors = ['#2ca02c' if v > 0 else '#d62728' for v in vals]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('C_S (commonality coefficient)')
        ax.set_title(f'Branch {branch}')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Fig 2 — Commonality decomposition\n'
                 '(red = negative shared variance = suppression)', fontsize=12)
    plt.tight_layout()
    out = OUT / 'fig2_commonality.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Luu: {out}')


def fig3_progression():
    """R² progression forward selection cho Branch A + B."""
    from itertools import combinations

    def r2_ols(X, y):
        n = len(y)
        Xc = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        y_pred = Xc @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    df_all = pd.read_csv('data/processed/handfeat_coswara_full.csv')
    df_all = df_all[df_all['extraction_status'] == 'ok'].copy()
    df_all = df_all.rename(columns={'a': 'age', 'g': 'gender'})
    df_all['age'] = pd.to_numeric(df_all['age'], errors='coerce')
    df_all = df_all.dropna(subset=FEATURES + ['age']).copy()
    df_all = df_all[df_all['age'] > 1]

    ACUTE_BROAD = {'positive_mild', 'positive_moderate', 'positive_asymp'}
    RECOVERED = {'recovered_full'}
    NEGATIVE = {'healthy'}

    branches = {
        'A': ACUTE_BROAD | RECOVERED,
        'B': ACUTE_BROAD,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_sel = {'F0_mean': '#1f77b4', 'jitter_local': '#ff7f0e',
                  'shimmer_local': '#2ca02c', 'hnr_mean': '#d62728'}

    for branch, pos_set in branches.items():
        keep = pos_set | NEGATIVE
        sub = df_all[df_all['covid_status'].isin(keep)].copy()
        sub['label'] = sub['covid_status'].isin(pos_set).astype(int)
        X = sub[FEATURES].values
        y = sub['label'].values

        # Forward selection theo R² gain
        remaining = list(FEATURES)
        selected = []
        r2_seq = [0.0]
        feat_seq = []
        while remaining:
            best_gain, best_feat = -1, None
            for f in remaining:
                cols = [FEATURES.index(s) for s in selected + [f]]
                r2_new = r2_ols(X[:, cols], y)
                gain = r2_new - r2_seq[-1]
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
            selected.append(best_feat)
            cols = [FEATURES.index(s) for s in selected]
            r2_seq.append(r2_ols(X[:, cols], y))
            feat_seq.append(best_feat)
            remaining.remove(best_feat)

        xs = np.arange(len(r2_seq))
        ax.plot(xs, r2_seq, marker='o', linewidth=2,
                label=f'Branch {branch}', color=COLORS[branch])
        for i, f in enumerate(feat_seq):
            ax.annotate(FEAT_SHORT[f], (i + 1, r2_seq[i + 1]),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8,
                        color=colors_sel[f])

    ax.set_xlabel('So features (thu tu forward selection)')
    ax.set_ylabel('R²')
    ax.set_title('Fig 3 — R² progression (forward selection)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT / 'fig3_r2_progression.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Luu: {out}')


def main():
    print("=" * 60)
    print("CARVE — Stage 5a visualization")
    print("=" * 60)

    print("\nLoad data...")
    audit = load_audit()
    logreg = load_logreg()
    comm = load_commonality()
    print(f"  audit: {len(audit)} rows")
    print(f"  logreg: {len(logreg)} rows")
    print(f"  commonality: {len(comm)} rows")

    print("\nFigure 1...")
    fig1_forest(audit, logreg)

    print("\nFigure 2...")
    fig2_commonality(comm)

    print("\nFigure 3...")
    fig3_progression()

    print(f"\n{'=' * 60}")
    print(f"XONG. Figures: {OUT}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()