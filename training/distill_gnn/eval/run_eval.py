"""Orchestrate a full evaluation of one student checkpoint (run in KRAKEN venv).

  1. start the strix teacher at a fixed sim budget;
  2. calibrate: measure strix's avg ms/turn on sample boards -> set the kraken
     side's per-turn wall-clock budget to match (roughly equal time controls);
  3. play student-vs-strix and student-vs-baseline (and baseline-vs-strix as an
     anchor) at that matched budget;
  4. write a JSON report + print an Elo summary.

Example:
  python -m training.distill_gnn.eval.run_eval \
      --student training/distill_gnn/results/attempt1_b10f128/distill_gnn_best.pt \
      --baseline training/mcts_results/best.pt --strix-sims 128 --games 30 \
      --out training/distill_gnn/results/attempt1_b10f128/eval.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics
import sys

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

from training.distill_gnn.eval.referee import (  # noqa: E402
    StrixClient, run_match, DEFAULT_HEXO_PY, DEFAULT_TEACHER)
from training.distill_gnn.eval.kraken_agent import KrakenAgent  # noqa: E402


def _sample_boards(shard_dir, n=8):
    """A few mid-game pre-turn boards {(q,r):int} + current_player for calibration."""
    out = []
    for path in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % 997 != 0:      # spread across the shard
                    continue
                rec = json.loads(line)
                board = {tuple(int(x) for x in k.split(",")): int(v)
                         for k, v in json.loads(rec["board"]).items()}
                if 6 <= len(board) <= 40:
                    out.append((board, int(rec["current_player"])))
                if len(out) >= n:
                    return out
        if out:
            return out
    return out


def calibrate_strix_ms(strix, boards):
    ms = []
    for board, cp in boards:
        strix.choose(board, cp, 2)
        ms.append(strix.turn_ms[-1])
    return statistics.mean(ms) if ms else 500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-name", default="student")
    ap.add_argument("--baseline", default="training/mcts_results/best.pt")
    ap.add_argument("--strix-sims", type=int, default=128)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hexo-python", default=DEFAULT_HEXO_PY)
    ap.add_argument("--teacher-ckpt", default=DEFAULT_TEACHER)
    ap.add_argument("--shard-dir", default="training/distill_gnn/data/labeled_joint")
    ap.add_argument("--time-ms", type=float, default=None,
                    help="override kraken per-turn budget (else = calibrated strix ms/turn)")
    ap.add_argument("--min-time-ms", type=float, default=150.0)
    ap.add_argument("--max-time-ms", type=float, default=3000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--anchor", action="store_true", help="also play baseline vs strix")
    args = ap.parse_args()

    report = {"student": args.student, "baseline": args.baseline,
              "strix_sims": args.strix_sims, "games": args.games}

    strix = StrixClient(args.hexo_python, sims=args.strix_sims, device=args.device,
                        teacher_ckpt=args.teacher_ckpt, name=f"strix({args.strix_sims})",
                        log_path=(args.out + ".strix.log") if args.out else None)
    try:
        if args.time_ms is not None:
            kraken_ms = args.time_ms
        else:
            boards = _sample_boards(args.shard_dir)
            kraken_ms = calibrate_strix_ms(strix, boards)
        kraken_ms = max(args.min_time_ms, min(args.max_time_ms, kraken_ms))
        report["kraken_time_ms"] = kraken_ms
        print(f"[calib] strix sims={args.strix_sims} -> kraken budget {kraken_ms:.0f} ms/turn",
              flush=True)

        student = KrakenAgent(args.student, time_budget_ms=kraken_ms, device=args.device,
                              name=args.student_name)
        baseline = KrakenAgent(args.baseline, time_budget_ms=kraken_ms, device=args.device,
                               name="kraken_base")

        def _save():
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=2)

        print("\n### student vs strix ###", flush=True)
        report["vs_strix"] = run_match(student, strix, args.games, seed=1000,
                                       label1=args.student_name, label2=strix.name)
        _save()
        print("\n### student vs baseline ###", flush=True)
        report["vs_baseline"] = run_match(student, baseline, args.games, seed=2000,
                                          label1=args.student_name, label2="kraken_base")
        _save()
        if args.anchor:
            print("\n### baseline vs strix (anchor) ###", flush=True)
            report["baseline_vs_strix"] = run_match(baseline, strix, args.games, seed=3000,
                                                    label1="kraken_base", label2=strix.name)
    finally:
        strix.close()

    print("\n=== SUMMARY ===")
    for key in ("vs_strix", "vs_baseline", "baseline_vs_strix"):
        r = report.get(key)
        if r:
            print(f"{key}: {r['label1']} score={r['score1']:.3f} "
                  f"({r['w1']}-{r['w2']}, {r['draws']}d)  Elo={r['elo1_vs_2']:+.0f}  "
                  f"ms {r['a1_avg_ms']:.0f}/{r['a2_avg_ms']:.0f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
