"""Strix teacher player server — run with the HEXO-STRIX venv, from KrakenBot root:

    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.eval.strix_server

Speaks newline-delimited JSON on stdin/stdout so a referee in the KrakenBot venv
can play full Connect-6 games against the teacher. Each request asks the teacher
to play ONE full turn (2 stones, or fewer if a stone wins) with Gumbel MCTS.

Protocol (one JSON object per line):
  <- {"board": {"q,r": player_int, ...}, "current_player": int, "moves_remaining": 2, "seed": int?}
  -> {"pair": [[q,r], ...], "ms": float}          # stones in play order
  <- {"cmd": "quit"}                               # shut down
First line emitted is {"ready": true, "sims": N, "device": "..."}.

Env: TEACHER_CKPT (default Desktop checkpoint), STRIX_SIMS (default 128),
     STRIX_M (m_actions, default 16), STRIX_DEVICE (default cuda).
"""
from __future__ import annotations

import json
import os
import sys
import time

# teacher.py inserts the KrakenBot root + hexo_a0 src onto sys.path on import.
from training.distill_gnn.teacher import Teacher


def main():
    ckpt = os.environ.get("TEACHER_CKPT",
                          r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")
    sims = int(os.environ.get("STRIX_SIMS", "128"))
    m_actions = int(os.environ.get("STRIX_M", "16"))
    device = os.environ.get("STRIX_DEVICE", "cuda")

    t = Teacher(ckpt, device=device, n_simulations=sims, m_actions=m_actions)
    print(json.dumps({"ready": True, "sims": sims, "m": m_actions,
                      "device": str(t.device), "train_steps": t.train_steps}),
          flush=True)
    sys.stderr.write(f"[strix] ready sims={sims} m={m_actions} device={t.device}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        board = {tuple(int(x) for x in k.split(",")): int(v)
                 for k, v in req["board"].items()}
        cp = int(req["current_player"])
        mr = int(req.get("moves_remaining", 2))
        seed = req.get("seed")
        t0 = time.perf_counter()
        lab = t.label_position(board, current_player=cp, moves_remaining=mr, seed=seed)
        ms = (time.perf_counter() - t0) * 1000.0
        pair = [[int(a), int(b)] for (a, b) in lab.best_pair]
        print(json.dumps({"pair": pair, "ms": ms, "value": float(lab.value)}),
              flush=True)


if __name__ == "__main__":
    main()
