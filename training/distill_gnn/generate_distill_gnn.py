"""Phase 1 + 1b: raw-policy teacher labeling + forced-win solver overlay.

Distills the hexo-strix teacher's RAW policy+value (one forward, no MCTS) over a
set of positions, and overlays exact forced-win labels from hexo_rs.solve_forcing.

Runs with the hexo-strix venv python from the KrakenBot project root:

    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.generate_distill_gnn --limit 5000

Output: JSONL.gz shards in <output-dir>, one record per position:
  board            json {"q,r": player_int}   (player_int 1=A/P1, 2=B/P2)
  current_player   int (1 or 2)
  game_id          int (stable per source human game -> leak-free train/val split)
  value            float in [-1,1]   teacher v_net  (or +1 if solver-proven win)
  pair             [[q1,r1],[q2,r2]] teacher greedy pair (hard joint-policy target)
  pi1_qr           list[[q,r]]       top-K soft policy cells (teacher pi_net)
  pi1_p            list[float]       soft policy mass, aligned with pi1_qr
  winning_singles  list[[q,r]]       depth-1 immediate 6-completions (finishing eval)
  winning_pairs    list[[[q,r],[q,r]]]
  forced_first_move [q,r] | null     solver VCF first move (depth-k), if any
  proven           int               +1 solver forced win, else 0

The scalar `value` is exported as-is; the categorical WDL transform lives in the
Phase 2 trainer (two-hot so the WDL mean regresses to `value`; solver-proven ->
one-hot win).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

import hexo_rs

from game import Player
from training.distill.generate_distill import _find_winning_moves, _board_has_win
from training.distill_gnn.teacher import Teacher

CKPT = os.environ.get("TEACHER_CKPT",
                      r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")


def load_human_positions(path, limit=None):
    """Yield (board_int, current_player_int, game_id_int) for non-won positions.

    game_id is a stable int per source human game (uuid), so all positions from
    one game land in the same train/val split downstream.
    """
    import __main__
    __main__.Player = Player  # human pkl pickled Player under __main__
    with open(path, "rb") as f:
        pos = pickle.load(f)

    gid_map: dict = {}
    out = []
    for entry in pos:
        board, cp = entry[0], entry[1]
        if not board or _board_has_win(board):
            continue
        human_gid = entry[3] if len(entry) == 4 else entry[4]
        gid = gid_map.setdefault(human_gid, len(gid_map))
        board_int = {(int(q), int(r)): (p.value if hasattr(p, "value") else int(p))
                     for (q, r), p in board.items()}
        cp_int = cp.value if hasattr(cp, "value") else int(cp)
        out.append((board_int, cp_int, gid))
        if limit and len(out) >= limit:
            break
    return out


def _board_to_json(board_int):
    return json.dumps({f"{q},{r}": p for (q, r), p in board_int.items()})


def _solve_forced_win(board_int, cp_int, cfg, depth_cap, node_budget):
    """Return the solver's VCF first move [q,r] for the side to move, or None."""
    stones = [((q, r), "P1" if p == 1 else "P2") for (q, r), p in board_int.items()]
    cp = "P1" if cp_int == 1 else "P2"
    st = hexo_rs.GameState.from_state(stones, cp, 2, cfg)
    if st.is_terminal() or st.legal_move_count() == 0:
        return None
    res = hexo_rs.solve_forcing(st, depth_cap=depth_cap, node_budget=node_budget)
    if res is None:
        return None
    first_move, _pv = res
    return [int(first_move[0]), int(first_move[1])]


def _write_shard(path, records):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill", "data", "positions_human_labelled.pkl"))
    ap.add_argument("--output-dir", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill_gnn", "data", "labeled"))
    ap.add_argument("--limit", type=int, default=None, help="max positions (default all)")
    ap.add_argument("--batch", type=int, default=256, help="teacher forward batch")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=32, help="soft-policy cells kept per position")
    ap.add_argument("--solve", dest="solve", action="store_true", default=True)
    ap.add_argument("--no-solve", dest="solve", action="store_false")
    ap.add_argument("--solve-depth", type=int, default=8)
    ap.add_argument("--solve-nodes", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"loading positions from {args.input} ...")
    positions = load_human_positions(args.input, args.limit)
    n_games = len({g for _, _, g in positions})
    print(f"{len(positions):,} non-won positions from {n_games:,} human games")

    t = Teacher(CKPT, device="cuda")
    cfg = t.game_config
    print(f"teacher: {t.train_steps} steps, graph={t.mc.graph_type}, dev={t.device}, "
          f"solve={'on' if args.solve else 'off'}")

    shard_idx = 0
    shard_records: list[dict] = []
    n_done = 0
    n_forced = 0
    n_win1 = 0
    solve_time = 0.0
    t0 = time.perf_counter()

    def flush():
        nonlocal shard_idx, shard_records
        if not shard_records:
            return
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.jsonl.gz")
        _write_shard(path, shard_records)
        shard_idx += 1
        shard_records = []

    for i in range(0, len(positions), args.batch):
        chunk = positions[i:i + args.batch]
        labs = t.label_batch_raw([(b, cp) for (b, cp, _g) in chunk])

        for (board_int, cp_int, gid), lab in zip(chunk, labs):
            cp_player = Player(cp_int)
            wsingles, wpairs = _find_winning_moves(board_int_to_enum(board_int), cp_player)
            if wsingles:
                n_win1 += 1

            forced = None
            if args.solve:
                st = time.perf_counter()
                forced = _solve_forced_win(board_int, cp_int, cfg,
                                           args.solve_depth, args.solve_nodes)
                solve_time += time.perf_counter() - st
                if forced is not None:
                    n_forced += 1

            proven = 1 if forced is not None else 0
            value = 1.0 if proven == 1 else float(lab.value)

            # top-K soft policy
            top = sorted(lab.pi1, key=lambda x: -x[1])[:args.topk]
            pi1_qr = [[int(q), int(r)] for (q, r), _p in top]
            pi1_p = [float(p) for _c, p in top]

            pair = [[int(q), int(r)] for (q, r) in lab.best_pair]

            shard_records.append({
                "board": _board_to_json(board_int),
                "current_player": cp_int,
                "game_id": gid,
                "value": value,
                "pair": pair,
                "pi1_qr": pi1_qr,
                "pi1_p": pi1_p,
                "winning_singles": wsingles,
                "winning_pairs": wpairs,
                "forced_first_move": forced,
                "proven": proven,
            })
            n_done += 1
            if len(shard_records) >= args.shard_size:
                flush()

        if (i // args.batch) % 20 == 0:
            el = time.perf_counter() - t0
            pps = n_done / el if el else 0
            print(f"  {n_done:,}/{len(positions):,}  {pps:5.0f} pos/s  "
                  f"forced={n_forced:,} ({100*n_forced/max(n_done,1):.1f}%)  "
                  f"win1={n_win1:,}  solve={solve_time:.0f}s", flush=True)

    flush()
    el = time.perf_counter() - t0
    print(f"\nDONE: {n_done:,} positions in {el:.0f}s ({n_done/el:.0f} pos/s), "
          f"{shard_idx} shards -> {args.output_dir}")
    print(f"  forced wins (solver): {n_forced:,} ({100*n_forced/max(n_done,1):.1f}%), "
          f"depth-1 wins: {n_win1:,}, solve overhead: {solve_time:.0f}s")


def board_int_to_enum(board_int):
    return {c: Player(p) for c, p in board_int.items()}


if __name__ == "__main__":
    main()
