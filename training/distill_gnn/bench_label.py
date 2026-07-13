"""Benchmark teacher labeling throughput to size the generation run.

Run with the hexo-strix venv python from the KrakenBot project root:
    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.bench_label
"""

from __future__ import annotations

import os
import pickle
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

from game import Player
from training.distill.generate_distill import _board_has_win
from training.distill_gnn.teacher import Teacher

CKPT = os.environ.get("TEACHER_CKPT",
                      r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")


def load_positions(n):
    path = os.path.join(_KRAKEN_ROOT, "training", "distill", "data",
                        "positions_human_labelled.pkl")
    import __main__
    __main__.Player = Player
    with open(path, "rb") as f:
        pos = pickle.load(f)
    out = []
    for entry in pos:
        board, cp = entry[0], entry[1]
        if not board or _board_has_win(board):
            continue
        out.append((dict(board), cp.value if hasattr(cp, "value") else int(cp)))
        if len(out) >= n:
            break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--m-actions", type=int, default=16)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--positions", type=int, default=1024)
    args = ap.parse_args()

    t = Teacher(CKPT, n_simulations=args.sims, m_actions=args.m_actions)
    print(f"teacher: {t.train_steps} steps, sims={args.sims}, m={args.m_actions}, "
          f"dev={t.device}")

    positions = load_positions(args.positions)
    print(f"loaded {len(positions)} pre-turn, non-won human positions\n")

    for bs in args.batch_sizes:
        batch = positions[:bs]
        # warmup
        t.label_batch(batch[:min(16, bs)], seed=0)
        t0 = time.perf_counter()
        labs = t.label_batch(batch, seed=0)
        dt = time.perf_counter() - t0
        pps = len(batch) / dt
        pair2 = sum(1 for l in labs if len(l.best_pair) == 2)
        print(f"batch={bs:4d}: {dt:6.2f}s  {pps:6.1f} pos/s  "
              f"(full-pairs {pair2}/{len(batch)})")
        for scale in (100_000, 500_000, 1_000_000):
            hrs = scale / pps / 3600
            print(f"           -> {scale:>9,} positions = {hrs:5.2f} h")
        print()


if __name__ == "__main__":
    main()
