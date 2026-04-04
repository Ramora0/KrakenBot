"""Scan saved self-play rounds and print diversity metrics for each.

Usage:
    python -m tools.diversity_history [--data-dir training/data/selfplay]
"""

import argparse
import json
import math
import glob
import os

import pandas as pd
import numpy as np
from model.symmetry import PERMS

N = 25


def compute_round_stats(path):
    r = int(os.path.basename(path).split("_")[1].split(".")[0])
    df = pd.read_parquet(path)
    has_fs = "full_search" in df.columns

    entropies = []
    canonical = set()

    for row in df.itertuples():
        is_fs = bool(row.full_search) if has_fs else True
        if is_fs:
            pv = json.loads(row.pair_visits)
            total = sum(pv.values())
            if total > 0:
                ent = -sum(
                    (c / total) * math.log(c / total)
                    for c in pv.values()
                    if c > 0
                )
                entropies.append(ent)

        board = json.loads(row.board)
        cells = sorted(
            (int(k.split(",")[0]) * N + int(k.split(",")[1]), v)
            for k, v in board.items()
        )
        canon = tuple(cells)
        for k in range(1, 12):
            t = tuple(sorted((int(PERMS[k][f]), v) for f, v in cells))
            if t < canon:
                canon = t
        canonical.add(canon)

    n = len(df)
    em = sum(entropies) / len(entropies) if entropies else 0
    es = (
        (sum((e - em) ** 2 for e in entropies) / len(entropies)) ** 0.5
        if len(entropies) > 1
        else 0
    )
    ur = len(canonical) / n if n > 0 else 0

    return {
        "round": r,
        "entropy_mean": em,
        "entropy_std": es,
        "unique_ratio": ur,
        "unique": len(canonical),
        "total": n,
        "fs_positions": len(entropies),
    }


def main():
    parser = argparse.ArgumentParser(description="Diversity history for self-play rounds")
    parser.add_argument(
        "--data-dir",
        default="training/data/selfplay",
        help="Directory containing round_*.parquet files",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.data_dir, "round_*.parquet")),
        key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
    )

    if not files:
        print(f"No round_*.parquet files found in {args.data_dir}")
        return

    print(
        f"{'Round':>6}  {'Entropy':>9}  {'+-':>7}  "
        f"{'Unique%':>8}  {'Unique':>7} / {'Total':>7}  {'FS pos':>7}"
    )
    print("-" * 70)

    for path in files:
        s = compute_round_stats(path)
        print(
            f"{s['round']:>6}  {s['entropy_mean']:>9.3f}  {s['entropy_std']:>7.3f}  "
            f"{100 * s['unique_ratio']:>7.1f}%  {s['unique']:>7} / {s['total']:>7}  "
            f"{s['fs_positions']:>7}"
        )


if __name__ == "__main__":
    main()
