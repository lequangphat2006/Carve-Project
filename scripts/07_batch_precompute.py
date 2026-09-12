#!/usr/bin/env python3
"""CARVE — Batch pre-compute mel cho tat ca 42 ngay Coswara.

Loop qua tung ngay:
  1. Sparse-checkout folder date tu repo
  2. Extract tar.gz parts
  3. Chay 06_precompute_mel.py tren audio vua extract
  4. Cleanup folder date (tiet kiem dung luong)
  5. Tiep tuc ngay ke

Resume-able: skip ngay nao da co mel_<date>.npz.

Usage:
    # Chay tat ca
    python scripts/07_batch_precompute.py \\
        --repo-dir /kaggle/working/Coswara-Data-sparse \\
        --meta-csv data/raw/combined_data.csv \\
        --output-dir /kaggle/working/mel_cache

    # Chay 1 vai ngay cu the
    python scripts/07_batch_precompute.py --dates 20200413 20200415 ...
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_URL = "https://github.com/iiscleap/Coswara-Data.git"
WORK = Path("/kaggle/working")


def setup_sparse_repo(repo_dir):
    repo_dir = Path(repo_dir)
    if (repo_dir / ".git").exists():
        print(f"Repo da san: {repo_dir}")
        return
    print("Clone metadata-only...")
    subprocess.run([
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--no-checkout", REPO_URL, str(repo_dir)
    ], check=True)
    subprocess.run([
        "git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"
    ], check=True)
    print("Setup OK")


def list_dates(repo_dir):
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-d", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True
    )
    dates = [x.strip() for x in r.stdout.split("\n")
             if x.strip().startswith("20") and len(x.strip()) == 8]
    return sorted(dates)


def process_one_date(date, repo_dir, meta_csv, output_dir):
    """Extract + pre-compute cho 1 ngay. Tra ve True neu thanh cong."""
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)

    out_path = output_dir / f"mel_{date}.npz"
    if out_path.exists():
        print(f"  [{date}] Da co {out_path.name} — skip")
        return True

    # 1. Sparse checkout ngay nay
    subprocess.run(["git", "-C", str(repo_dir), "sparse-checkout",
                    "set", date], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "HEAD"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "clean", "-fdx"],
                   check=True, capture_output=True)

    folder = repo_dir / date
    if not folder.exists():
        print(f"  [{date}] Folder khong ton tai — skip")
        return False

    # 2. Ghep tar parts
    parts = sorted(folder.glob(f"{date}.tar.gz.*"))
    tar_path = folder / f"{date}.tar.gz"
    if not parts and not tar_path.exists():
        print(f"  [{date}] Khong co tar file — skip")
        subprocess.run(["rm", "-rf", str(folder)], check=False)
        return False

    if parts and not tar_path.exists():
        print(f"  [{date}] Ghep {len(parts)} phan tar...")
        with open(tar_path, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())

    # 3. Extract
    ext_dir = WORK / f"ext_{date}"
    if ext_dir.exists():
        subprocess.run(["rm", "-rf", str(ext_dir)], check=False)
    ext_dir.mkdir(exist_ok=True)
    print(f"  [{date}] Extract tar...")
    try:
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(ext_dir)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  [{date}] Extract fail: {e}")
        subprocess.run(["rm", "-rf", str(ext_dir), str(folder)], check=False)
        return False

    # 4. Pre-compute mel (call 06 script)
    print(f"  [{date}] Pre-compute mel...")
    r = subprocess.run([
        sys.executable, "scripts/06_precompute_mel.py",
        "--audio-dir", str(ext_dir),
        "--meta-csv", meta_csv,
        "--date", date,
        "--output-dir", str(output_dir),
    ], capture_output=True, text=True)

    if r.returncode != 0:
        print(f"  [{date}] Pre-compute fail:")
        print(r.stdout[-500:] if r.stdout else "")
        print(r.stderr[-500:] if r.stderr else "")
        subprocess.run(["rm", "-rf", str(ext_dir), str(folder)], check=False)
        return False

    # In vài dòng cuối output
    for line in r.stdout.strip().split("\n")[-5:]:
        print(f"    {line}")

    # 5. Cleanup ext_dir + folder date (tiet kiem dung luong)
    print(f"  [{date}] Cleanup...")
    subprocess.run(["rm", "-rf", str(ext_dir)], check=False)
    subprocess.run(["rm", "-rf", str(folder)], check=False)

    return out_path.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-dir',
                        default='/kaggle/working/Coswara-Data-sparse')
    parser.add_argument('--meta-csv', default='data/raw/combined_data.csv')
    parser.add_argument('--output-dir', default='/kaggle/working/mel_cache')
    parser.add_argument('--dates', nargs='+', default=None)
    parser.add_argument('--skip-setup', action='store_true')
    args = parser.parse_args()

    print("=" * 72)
    print("CARVE — Batch pre-compute mel")
    print("=" * 72)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_setup:
        setup_sparse_repo(args.repo_dir)

    # Danh sach ngay
    if args.dates:
        dates = args.dates
    else:
        dates = list_dates(args.repo_dir)

    print(f"\nCo {len(dates)} ngay can xu ly:")
    print(f"  {dates[:3]} ... {dates[-3:]}")

    # Filter ngay da co
    done = set()
    for d in dates:
        if (output_dir / f"mel_{d}.npz").exists():
            done.add(d)
    todo = [d for d in dates if d not in done]
    print(f"Da xong: {len(done)}, can lam: {len(todo)}")

    # Check dung luong
    r = subprocess.run(["df", "-h", "/kaggle/working"],
                       capture_output=True, text=True)
    print(f"\nDung luong:\n{r.stdout}")

    # Loop
    n_ok, n_fail = 0, 0
    for i, d in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] === {d} ===")
        try:
            if process_one_date(d, args.repo_dir, args.meta_csv, output_dir):
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"  Loi: {e}")
            n_fail += 1

        # In dung luong moi 5 ngay
        if i % 5 == 0:
            r = subprocess.run(["df", "-h", "/kaggle/working"],
                               capture_output=True, text=True)
            print(f"\n  Dung luong hien tai:\n{r.stdout}")

    print(f"\n{'=' * 72}")
    print(f"XONG: {n_ok} OK, {n_fail} fail")
    print(f"Output: {output_dir}")
    print("=" * 72)


if __name__ == '__main__':
    main()