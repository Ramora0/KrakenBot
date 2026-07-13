"""Parallel match runner: split N games across W referee workers, merge stats.

Each worker is a separate process (own strix server / kraken contexts), playing
games/W games with a distinct seed block, writing a partial JSON. This changes
nothing about measurement conditions — same engines, same per-turn budgets —
it just plays W games at once.

Screening protocol (fast, for ranking attempts): --strix-sims 32 and a
time-matched kraken budget; confirm any new best at the full 128/703ms setting.

Run in the KrakenBot venv:
  python -m training.distill_gnn.eval.fast_eval \
      --a-ckpt <student.pt> --a-time-ms 200 --b-strix --b-sims 32 \
      --games 30 --workers 3 --out eval_fast.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def sample_openings(shard_dir, n, seed=0, min_stones=4, max_stones=12):
    """Sample n distinct early-game positions from labeled shards as openings."""
    rng = random.Random(seed)
    pool = []
    for path in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % 61 != 0:
                    continue
                rec = json.loads(line)
                board = json.loads(rec["board"])
                if min_stones <= len(board) <= max_stones:
                    pool.append({"board": board, "cp": int(rec["current_player"])})
        if len(pool) >= n * 20:
            break
    rng.shuffle(pool)
    seen, out = set(), []
    for o in pool:
        key = frozenset(o["board"].items())
        if key not in seen:
            seen.add(key)
            out.append(o)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ckpt", required=True)
    ap.add_argument("--a-name", default="student")
    ap.add_argument("--a-time-ms", type=float, required=True)
    ap.add_argument("--a-log-temp", type=float, default=None)
    ap.add_argument("--b-log-temp", type=float, default=None)
    ap.add_argument("--b-strix", action="store_true")
    ap.add_argument("--b-sims", type=int, default=32)
    ap.add_argument("--b-ckpt", default=None)
    ap.add_argument("--b-time-ms", type=float, default=None)
    ap.add_argument("--b-name", default=None)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--openings", action="store_true",
                    help="start games from sampled human positions (each played "
                         "once per color) instead of the bare (0,0) start")
    ap.add_argument("--openings-shards", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill_gnn", "data", "labeled_joint_k8"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-lost-states", default=None,
                    help="pickle path: merge every worker's lost-game pre-turn "
                         "states here (blunder mining)")
    args = ap.parse_args()

    per = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        per[i] += 1
    # keep per-worker counts even so colors stay balanced
    for i in range(len(per)):
        if per[i] % 2 and per[i] > 1:
            per[i] -= 1
            per[(i + 1) % len(per)] += 1

    openings = None
    if args.openings:
        openings = sample_openings(args.openings_shards,
                                   (args.games + 1) // 2, seed=args.seed)
        print(f"sampled {len(openings)} openings", flush=True)

    procs, outs = [], []
    t0 = time.perf_counter()
    o_taken = 0
    for w, n in enumerate(per):
        if n <= 0:
            continue
        out = (args.out or "fast_eval") + f".w{w}.json"
        outs.append(out)
        cmd = [sys.executable, "-u", "-m", "training.distill_gnn.eval.referee",
               "--a-ckpt", args.a_ckpt, "--a-name", args.a_name,
               "--a-time-ms", str(args.a_time_ms),
               "--games", str(n), "--seed", str(args.seed + 10000 * (w + 1)),
               "--out", out]
        if openings:
            need = (n + 1) // 2
            of = out + ".openings.json"
            with open(of, "w") as f:
                json.dump(openings[o_taken:o_taken + need], f)
            o_taken += need
            cmd += ["--openings-file", of]
        if args.dump_lost_states:
            cmd += ["--dump-lost-states", out + ".lost.pkl"]
        if args.a_log_temp is not None:
            cmd += ["--a-log-temp", str(args.a_log_temp)]
        if args.b_strix:
            cmd += ["--b-strix", "--b-sims", str(args.b_sims)]
        else:
            cmd += ["--b-ckpt", args.b_ckpt, "--b-name", args.b_name or "opp"]
            if args.b_time_ms is not None:
                cmd += ["--b-time-ms", str(args.b_time_ms)]
            if args.b_log_temp is not None:
                cmd += ["--b-log-temp", str(args.b_log_temp)]
        log = open(out + ".log", "w")
        procs.append((subprocess.Popen(cmd, cwd=_KRAKEN_ROOT, stdout=log,
                                       stderr=subprocess.STDOUT), log))
        print(f"worker {w}: {n} games (seed {args.seed + 10000 * (w + 1)})", flush=True)

    for p, log in procs:
        p.wait()
        log.close()

    w1 = w2 = draws = games = 0
    for out in outs:
        with open(out) as f:
            r = json.load(f)
        w1 += r["w1"]; w2 += r["w2"]; draws += r["draws"]; games += r["games"]

    if args.dump_lost_states:
        import pickle
        mined, seen = [], set()
        for out in outs:
            p = out + ".lost.pkl"
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                for (bd, cp, extra, gid) in pickle.load(f):
                    key = (frozenset(bd.items()), cp)
                    if key not in seen:
                        seen.add(key)
                        mined.append((bd, cp, extra, gid))
        with open(args.dump_lost_states, "wb") as f:
            pickle.dump(mined, f)
        print(f"mined {len(mined):,} unique lost-game states "
              f"-> {args.dump_lost_states}", flush=True)
    score = (w1 + 0.5 * draws) / max(games, 1)
    s = min(max(score, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / max(games, 1))
    lo = min(max(s - 1.96 * se, 1e-6), 1 - 1e-6)
    hi = min(max(s + 1.96 * se, 1e-6), 1 - 1e-6)
    result = {"a": args.a_name, "games": games, "w1": w1, "w2": w2,
              "draws": draws, "score": score, "elo": elo,
              "elo_ci": [-400.0 * math.log10(1.0 / hi - 1.0),
                         -400.0 * math.log10(1.0 / lo - 1.0)],
              "wall_s": time.perf_counter() - t0}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
