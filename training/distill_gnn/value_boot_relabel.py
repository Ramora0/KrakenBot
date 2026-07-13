"""1-turn TD bootstrap of the value column: v(s) <- -v_net(s after teacher pair).

The raw v_net is positionally smooth but tactically soft. Looking ahead one
full turn along the teacher's own greedy pair sharpens exactly the states MCTS
leans on: if the pair completes a win the target is +1 exactly; if it walks
into a lost position the sign flips. Everything is teacher forwards — no
student self-play values, no AlphaZero.

Reads joint-k8 shards, writes a copy with `value` replaced:
  * proven==1                -> keep +1 (solver truth)
  * pair wins on the spot    -> +1
  * else                     -> -v_net(board + pair, opponent to move)
  * no/partial pair          -> keep original value

Run with the hexo venv from the KrakenBot root:
  python -m training.distill_gnn.value_boot_relabel \
      --in-dir training/distill_gnn/data/merged_k8_dagger3 \
      --out-dir training/distill_gnn/data/merged_k8_boot --device cuda
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

CKPT = os.environ.get("TEACHER_CKPT",
                      r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")


def _parse_board(board_json):
    return {tuple(int(x) for x in k.split(",")): int(v)
            for k, v in json.loads(board_json).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from training.distill.generate_distill import _board_has_win
    from training.distill_gnn.teacher import Teacher

    t = Teacher(CKPT, device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    n_done = n_boot = n_win = n_kept = 0
    t0 = time.perf_counter()
    for path in sorted(glob.glob(os.path.join(args.in_dir, "shard_*.jsonl.gz"))):
        recs = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))

        # build after-pair states for records with a full 2-stone pair
        pend_idx, pend_states = [], []
        for i, r in enumerate(recs):
            if r.get("proven") == 1 or len(r.get("pair") or []) < 2:
                n_kept += 1
                continue
            board = _parse_board(r["board"])
            cp = int(r["current_player"])
            (q1, r1), (q2, r2) = [tuple(int(x) for x in m) for m in r["pair"]]
            if (q1, r1) in board or (q2, r2) in board or (q1, r1) == (q2, r2):
                n_kept += 1
                continue
            board[(q1, r1)] = cp
            board[(q2, r2)] = cp
            if _board_has_win(board):
                r["value"] = 1.0
                n_win += 1
                continue
            opp = 2 if cp == 1 else 1
            pend_idx.append(i)
            pend_states.append(t.make_state(board, opp, 2))

        for j in range(0, len(pend_states), args.batch):
            chunk = pend_states[j:j + args.batch]
            _logits, values = t._eval_fn(chunk)
            for k, v in enumerate(values):
                recs[pend_idx[j + k]]["value"] = -float(v)
            n_boot += len(chunk)

        out_path = os.path.join(args.out_dir, os.path.basename(path))
        with gzip.open(out_path + ".tmp", "wt", encoding="utf-8") as out:
            for r in recs:
                out.write(json.dumps(r) + "\n")
        os.replace(out_path + ".tmp", out_path)
        n_done += len(recs)
        el = time.perf_counter() - t0
        print(f"  {n_done:,} done ({n_boot:,} boot, {n_win:,} pair-wins, "
              f"{n_kept:,} kept)  {n_done/el:5.0f} rec/s", flush=True)

    print(f"DONE: {n_done:,} records -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
