"""Estimate simulation savings from keeping the MCTS tree between moves.

When a move is chosen, its subtree already has accumulated visits from the
search. If we kept the tree and reused that subtree as the new root, those
visits would carry forward — meaning fewer new simulations needed.

This script analyzes pair_visits from self-play data to estimate:
  - How many sims the chosen move typically gets
  - What fraction of total sims that represents (potential savings)

Usage:
    python tools/tree_reuse_analysis.py [--data-dir training/data/selfplay]
                                        [--rounds 80-109]
"""

import argparse
import json
import os
import sys

import pandas as pd


def parse_pair_visits(pv_str: str) -> dict[tuple[int, int], int]:
    raw = json.loads(pv_str)
    return {
        tuple(int(x) for x in k.split(",")): v
        for k, v in raw.items()
    }


def analyze_position(pair_visits: dict[tuple[int, int], int]):
    """Return (total_sims, max_visits, top3_visits) for one position."""
    if not pair_visits:
        return None
    total = sum(pair_visits.values())
    if total == 0:
        return None
    sorted_counts = sorted(pair_visits.values(), reverse=True)
    top1 = sorted_counts[0]
    top3 = sum(sorted_counts[:3])
    return total, top1, top3


def analyze_round(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    results = []
    for _, row in df.iterrows():
        pv = parse_pair_visits(row["pair_visits"])
        stats = analyze_position(pv)
        if stats is None:
            continue
        total, top1, top3 = stats
        results.append({
            "total_sims": total,
            "chosen_sims": top1,
            "top3_sims": top3,
            "n_pairs": len(pv),
            "full_search": bool(row.get("full_search", True)),
            "move_count": int(row.get("move_count", 0)),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Tree reuse savings analysis")
    parser.add_argument("--data-dir", default="training/data/selfplay")
    parser.add_argument("--rounds", default=None,
                        help="Range of rounds e.g. '80-109' or 'all'")
    args = parser.parse_args()

    # Find parquet files
    parquets = sorted(
        f for f in os.listdir(args.data_dir)
        if f.startswith("round_") and f.endswith(".parquet")
    )
    if not parquets:
        print(f"No round_*.parquet files found in {args.data_dir}")
        sys.exit(1)

    # Filter by round range
    if args.rounds and args.rounds != "all":
        lo, hi = args.rounds.split("-")
        lo, hi = int(lo), int(hi)
        parquets = [
            f for f in parquets
            if lo <= int(f.replace("round_", "").replace(".parquet", "")) <= hi
        ]

    print(f"Analyzing {len(parquets)} round files from {args.data_dir}\n")

    all_results = []
    for pf in parquets:
        path = os.path.join(args.data_dir, pf)
        results = analyze_round(path)
        all_results.extend(results)

    if not all_results:
        print("No positions with visit data found.")
        sys.exit(1)

    # Overall stats
    total_positions = len(all_results)
    avg_sims = sum(r["total_sims"] for r in all_results) / total_positions
    avg_chosen = sum(r["chosen_sims"] for r in all_results) / total_positions
    avg_top3 = sum(r["top3_sims"] for r in all_results) / total_positions
    avg_pairs = sum(r["n_pairs"] for r in all_results) / total_positions

    avg_frac = sum(
        r["chosen_sims"] / r["total_sims"] for r in all_results
    ) / total_positions
    avg_frac_top3 = sum(
        r["top3_sims"] / r["total_sims"] for r in all_results
    ) / total_positions

    print(f"{'Positions analyzed:':<30} {total_positions:>10,}")
    print(f"{'Avg total sims/position:':<30} {avg_sims:>10.1f}")
    print(f"{'Avg distinct pairs visited:':<30} {avg_pairs:>10.1f}")
    print()
    print("--- Chosen move (max-visited pair) ---")
    print(f"{'Avg visits to chosen move:':<30} {avg_chosen:>10.1f}")
    print(f"{'Avg fraction of total sims:':<30} {avg_frac:>10.1%}")
    print(f"{'=> Potential savings:':<30} {avg_frac:>10.1%}")
    print()
    print("--- Top 3 pairs ---")
    print(f"{'Avg visits to top 3:':<30} {avg_top3:>10.1f}")
    print(f"{'Avg fraction of total sims:':<30} {avg_frac_top3:>10.1%}")

    # Breakdown by full vs quick search
    full = [r for r in all_results if r["full_search"]]
    quick = [r for r in all_results if not r["full_search"]]

    for label, subset in [("Full search", full), ("Quick search", quick)]:
        if not subset:
            continue
        n = len(subset)
        s = sum(r["total_sims"] for r in subset) / n
        c = sum(r["chosen_sims"] for r in subset) / n
        f = sum(r["chosen_sims"] / r["total_sims"] for r in subset) / n
        print(f"\n--- {label} ({n:,} positions) ---")
        print(f"  Avg sims: {s:.1f}  |  Avg chosen: {c:.1f}  |  Fraction: {f:.1%}")

    # Breakdown by game phase (move_count)
    print("\n--- By game phase ---")
    phases = [
        ("Early (moves 0-10)", lambda r: r["move_count"] <= 10),
        ("Mid   (moves 11-30)", lambda r: 11 <= r["move_count"] <= 30),
        ("Late  (moves 31+)", lambda r: r["move_count"] >= 31),
    ]
    for label, pred in phases:
        subset = [r for r in all_results if pred(r)]
        if not subset:
            continue
        n = len(subset)
        s = sum(r["total_sims"] for r in subset) / n
        c = sum(r["chosen_sims"] for r in subset) / n
        f = sum(r["chosen_sims"] / r["total_sims"] for r in subset) / n
        print(f"  {label}: avg sims={s:.0f}, chosen={c:.0f}, "
              f"fraction={f:.1%}  ({n:,} positions)")

    # Distribution of chosen fraction
    fracs = sorted(r["chosen_sims"] / r["total_sims"] for r in all_results)
    n = len(fracs)
    print(f"\n--- Distribution of chosen-move fraction ---")
    for pct_label, idx in [("10th", n // 10), ("25th", n // 4),
                           ("50th", n // 2), ("75th", 3 * n // 4),
                           ("90th", 9 * n // 10)]:
        print(f"  {pct_label} percentile: {fracs[idx]:.1%}")


if __name__ == "__main__":
    main()
