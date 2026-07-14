"""Benchmark search throughput per checkpoint: sims/turn at a wall-clock budget.

For the model-size ablation: converts each model's Elo-vs-sims curve into
Elo-vs-time. Plays no games — runs the batched MCTS on a fixed set of midgame
positions sampled from labeled shards and records KrakenAgent.last_sims.

Run in the KrakenBot venv:
  python -m training.distill_gnn.eval.bench_sims \
      --ckpts a.pt b.pt --time-ms 703 --positions 12 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

from training.distill_gnn.eval.fast_eval import sample_openings  # noqa: E402
from training.distill_gnn.eval.kraken_agent import KrakenAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--time-ms", type=float, default=703)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--min-stones", type=int, default=8)
    ap.add_argument("--max-stones", type=int, default=24)
    ap.add_argument("--shards", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill_gnn", "data", "labeled_joint_k8"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    poss = sample_openings(args.shards, args.positions, seed=args.seed,
                           min_stones=args.min_stones, max_stones=args.max_stones)
    results = {}
    for ck in args.ckpts:
        agent = KrakenAgent(ck, time_budget_ms=args.time_ms, name=ck)
        sims = []
        for o in poss:
            board = {tuple(int(x) for x in k.split(",")): int(v)
                     for k, v in o["board"].items()}
            agent.choose(board, o["cp"], 2)
            sims.append(agent.last_sims)
        sims.sort()
        n_params = sum(p.numel() for p in agent.base.parameters())
        results[ck] = {"params": n_params,
                       "sims_median": sims[len(sims) // 2],
                       "sims_min": sims[0], "sims_max": sims[-1],
                       "ms_per_sim": args.time_ms / max(sims[len(sims) // 2], 1)}
        print(f"{ck}: {n_params/1e6:.2f}M params, median {sims[len(sims)//2]} "
              f"sims/{args.time_ms:.0f}ms (min {sims[0]}, max {sims[-1]})", flush=True)
        del agent
        import torch
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
