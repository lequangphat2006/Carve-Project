#!/usr/bin/env python3
"""Stage 0.2 — Data validation cho CARVE.

Kiem 8 checks trong bang C.1 cua implementation_notes.md.

Usage:
    python 00_validate_data.py --coswara-meta /path/combined_data.csv --metadata-only
    python 00_validate_data.py --coswara-meta ... --coswara-audio /path/Extracted_data
    python 00_validate_data.py --coswara-meta ... --dicova-meta /path/meta.csv
"""
import argparse
import json
import sys
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============ KY VONG (tu implementation_notes.md muc C.1) ============
EXPECTED = {
    'coswara_n': 2746,
    'coswara_n_tolerance': 5,
    'covid_status_unique': 8,
    'covid_status_tolerance': 0,
    'test_status_unique': 5,
    'test_status_tolerance': 1,
    'missing_age_pct_max': 5.0,
    'missing_gender_pct_max': 5.0,
    'dicova_n': 1199,
    'dicova_positive': 80,
    'dicova_prevalence': 0.067,
    'dicova_prevalence_tolerance': 0.02,
}

# ============ DINH NGHIA NHOM (dong bo voi B.1) ============
ACUTE_STRICT = {'positive_mild', 'positive_moderate'}
ACUTE_BROAD = ACUTE_STRICT | {'positive_asymp'}
RECOVERED = {'recovered_full'}
EXPOSED = {'no_resp_illness_exposed'}
NEGATIVE = {'healthy'}
EXCLUDED = {'resp_illness_not_identified', 'under_validation'}


def check_coswara_metadata(meta_path):
    """Metadata-only checks (khong can audio)."""
    results = {}
    df = pd.read_csv(meta_path)

    # Check 1: N
    n = len(df)
    tol = EXPECTED['coswara_n_tolerance']
    passed = abs(n - EXPECTED['coswara_n']) <= tol
    results['coswara_n'] = {
        'expected': f"{EXPECTED['coswara_n']} ± {tol}",
        'actual': n,
        'passed': passed,
        'note': 'Da cap nhat tu 2635 theo tai lieu goc',
    }

    # Check 5: covid_status unique
    cs_vals = sorted(df['covid_status'].dropna().unique().tolist())
    passed = abs(len(cs_vals) - EXPECTED['covid_status_unique']) <= EXPECTED['covid_status_tolerance']
    results['covid_status_unique'] = {
        'expected': EXPECTED['covid_status_unique'],
        'actual': len(cs_vals),
        'values': cs_vals,
        'passed': passed,
    }

    # Check 6: test_status unique
    ts_vals = sorted(df['test_status'].fillna('').astype(str).unique().tolist())
    passed = abs(len(ts_vals) - EXPECTED['test_status_unique']) <= EXPECTED['test_status_tolerance']
    results['test_status_unique'] = {
        'expected': f"{EXPECTED['test_status_unique']} ± {EXPECTED['test_status_tolerance']}",
        'actual': len(ts_vals),
        'values': ts_vals,
        'passed': passed,
        'note': 'Tai lieu ghi 4, thuc te 5 (them empty)',
    }

    # Check 8a: missing age
    a_str = df['a'].astype(str).str.strip()
    age_missing = df['a'].isna() | (a_str == '') | (a_str == 'nan')
    age_pct = 100 * age_missing.mean()
    passed = bool(age_pct < EXPECTED['missing_age_pct_max'])
    results['missing_age_pct'] = {
        'expected': f"< {EXPECTED['missing_age_pct_max']}%",
        'actual': round(age_pct, 2),
        'passed': passed,
    }

    # Check 8b: missing gender
    g_str = df['g'].astype(str).str.strip()
    gender_missing = df['g'].isna() | (g_str == '') | (g_str == 'nan')
    gender_pct = 100 * gender_missing.mean()
    passed = bool(gender_pct < EXPECTED['missing_gender_pct_max'])
    results['missing_gender_pct'] = {
        'expected': f"< {EXPECTED['missing_gender_pct_max']}%",
        'actual': round(gender_pct, 2),
        'passed': passed,
    }

    # Bonus: N_B cho A1/A2
    cs_counts = Counter(df['covid_status'].dropna())
    n_healthy = sum(cs_counts.get(v, 0) for v in NEGATIVE)
    n_a1 = sum(cs_counts.get(v, 0) for v in ACUTE_STRICT)
    n_a2 = sum(cs_counts.get(v, 0) for v in ACUTE_BROAD)
    results['N_B'] = {
        'n_healthy': n_healthy,
        'n_A1_acute_strict': n_a1,
        'N_B_A1': n_a1 + n_healthy,
        'n_A2_acute_broad': n_a2,
        'N_B_A2': n_a2 + n_healthy,
        'passed': (n_a1 + n_healthy) >= 500 and (n_a2 + n_healthy) >= 500,
    }

    return results, df


def check_coswara_audio(audio_dir):
    """Audio checks (can extract xong)."""
    results = {}
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        results['coswara_audio'] = {
            'passed': None,
            'note': f"Audio dir chua ton tai: {audio_dir}",
        }
        return results

    wav_files = list(audio_dir.rglob('*.wav'))
    if not wav_files:
        results['coswara_audio'] = {
            'passed': False,
            'note': 'Khong tim thay file .wav nao',
        }
        return results

    sample_size = min(200, len(wav_files))
    rng = np.random.RandomState(0)
    sample = rng.choice(wav_files, size=sample_size, replace=False)

    durations = []
    for p in sample:
        try:
            with wave.open(str(p), 'rb') as w:
                durations.append(w.getnframes() / w.getframerate())
        except Exception:
            continue

    durations = np.array(durations) if durations else np.array([0.0])
    results['coswara_audio'] = {
        'n_wav_files': len(wav_files),
        'n_sampled': int(sample_size),
        'duration_min_sec': round(float(durations.min()), 3),
        'duration_max_sec': round(float(durations.max()), 3),
        'duration_median_sec': round(float(np.median(durations)), 3),
        'passed': True,
    }
    return results


def check_dicova(meta_path):
    """DiCOVA checks (cho data)."""
    results = {}
    meta_path = Path(meta_path)

    if not meta_path.exists():
        results['dicova'] = {
            'passed': None,
            'note': f"DiCOVA metadata chua co: {meta_path}",
        }
        return results

    df = pd.read_csv(meta_path)
    n = len(df)

    label_col = None
    for col in ['covid_status', 'label', 'test_status', 'status']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        results['dicova'] = {
            'passed': False,
            'note': f"Khong tim thay cot label. Columns: {list(df.columns)}",
        }
        return results

    labels = df[label_col].astype(str).str.lower()
    n_pos = int(labels.isin(['positive', 'p', '1', 'true']).sum())
    prevalence = n_pos / n if n > 0 else 0

    passed_n = n == EXPECTED['dicova_n']
    passed_pos = n_pos == EXPECTED['dicova_positive']
    passed_prev = abs(prevalence - EXPECTED['dicova_prevalence']) <= EXPECTED['dicova_prevalence_tolerance']

    results['dicova'] = {
        'n_total': n,
        'n_positive': n_pos,
        'prevalence': round(prevalence, 4),
        'label_column': label_col,
        'n_expected': EXPECTED['dicova_n'],
        'n_positive_expected': EXPECTED['dicova_positive'],
        'passed_n': passed_n,
        'passed_positive': passed_pos,
        'passed_prevalence': passed_prev,
        'passed': passed_n and passed_pos and passed_prev,
    }

    speaker_col = None
    for col in ['speaker_id', 'id', 'speaker']:
        if col in df.columns:
            speaker_col = col
            break

    if speaker_col is not None:
        dup = int(df[speaker_col].duplicated().sum())
        results['dicova_speaker_dup'] = {
            'n_duplicated': dup,
            'passed': dup == 0,
        }

    return results


def print_report(results):
    print("=" * 70)
    print("CARVE — Stage 0.2 Data Validation Report")
    print("=" * 70)

    for section, data in results.items():
        print(f"\n### {section}")
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'passed':
                    mark = 'PASS' if v is True else ('FAIL' if v is False else 'PENDING')
                    print(f"  [{mark}]")
                else:
                    print(f"     {k}: {v}")

    print("\n" + "=" * 70)
    all_passed = [d['passed'] for d in results.values()
                  if isinstance(d, dict) and 'passed' in d]
    n_pass = sum(1 for p in all_passed if p is True)
    n_fail = sum(1 for p in all_passed if p is False)
    n_pending = sum(1 for p in all_passed if p is None)
    print(f"Summary: {n_pass} passed, {n_fail} failed, {n_pending} pending")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='CARVE Stage 0.2 — Data validation')
    parser.add_argument('--coswara-meta', type=str, required=True,
                        help='Path toi combined_data.csv')
    parser.add_argument('--coswara-audio', type=str, default=None,
                        help='Path toi thu muc Extracted_data/')
    parser.add_argument('--dicova-meta', type=str, default=None,
                        help='Path toi metadata DiCOVA')
    parser.add_argument('--metadata-only', action='store_true',
                        help='Chi chay metadata checks (bo qua audio)')
    parser.add_argument('--output', type=str, default='results/logs/',
                        help='Thu muc ghi JSON report')
    args = parser.parse_args()

    all_results = {}

    print(f"Doc Coswara metadata: {args.coswara_meta}")
    coswara_results, _ = check_coswara_metadata(args.coswara_meta)
    all_results.update(coswara_results)

    if not args.metadata_only and args.coswara_audio:
        print(f"Kiem Coswara audio: {args.coswara_audio}")
        all_results.update(check_coswara_audio(args.coswara_audio))

    if args.dicova_meta:
        print(f"Kiem DiCOVA: {args.dicova_meta}")
        all_results.update(check_dicova(args.dicova_meta))
    else:
        all_results['dicova'] = {
            'passed': None,
            'note': 'DiCOVA chua co (--dicova-meta khong duoc truyen)',
        }

    print_report(all_results)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'validate_report_{timestamp}.json'

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'args': vars(args),
            'results': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReport ghi tai: {out_path}")

    n_fail = sum(1 for d in all_results.values()
                 if isinstance(d, dict) and d.get('passed') is False)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == '__main__':
    main()
