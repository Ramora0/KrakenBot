"""Fast JOINT relabel: teacher pi1 marginal + pi2 over top-k first moves.

Parallelizes the (GIL-held, single-core) graph builder across CPU workers, and
REUSES the forced-win/solver columns from an existing label run (no re-solve),
so this is many times faster than the original inline-solver pass.

Run with the hexo-strix venv python from the KrakenBot project root:
    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.relabel_joint --workers 6 --k 8 --m 16

Output: JSONL.gz shards in <output-dir>. Per-record schema adds `joint`:
  joint = [ {"a":[q,r], "pa":float, "b_qr":[[q,r]...], "b_p":[float...]}, ... ]
one entry per top-k first move; `b_*` is pi2 top-m for that first move.
Solver columns (winning_singles/pairs, forced_first_move, proven) are copied
from the reused run.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

CKPT = os.environ.get("TEACHER_CKPT",
                      r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")

# per-worker global (set in initializer)
_W = {}


def _init_worker(ckpt, k, m, device):
    import torch
    torch.set_num_threads(1)  # each worker is one core; avoid thread oversubscription
    from training.distill_gnn.teacher import Teacher
    _W["t"] = Teacher(ckpt, device=device)
    _W["k"] = k
    _W["m"] = m


def _board_to_json(board_int):
    return json.dumps({f"{q},{r}": p for (q, r), p in board_int.items()})


def _label_chunk(chunk):
    """chunk: list of (board_int, cp_int, game_id, solver_dict). Returns records."""
    t = _W["t"]; k = _W["k"]; m = _W["m"]
    labs = t.label_batch_topk([(b, cp) for (b, cp, _g, _s) in chunk], k=k, m=m)
    out = []
    for (board, cp, gid, solver), lab in zip(chunk, labs):
        proven = int(solver.get("proven", 0))
        value = 1.0 if proven == 1 else float(lab.value)
        joint = []
        for (a, pa, pi2) in lab.joint:
            joint.append({"a": [int(a[0]), int(a[1])], "pa": float(pa),
                          "b_qr": [[int(q), int(r)] for (q, r), _p in pi2],
                          "b_p": [float(p) for _c, p in pi2]})
        # greedy pair for metrics: top first move + its top second move
        pair = [[int(joint[0]["a"][0]), int(joint[0]["a"][1])]] if joint else []
        if joint and joint[0]["b_qr"]:
            pair.append(joint[0]["b_qr"][0])
        out.append({
            "board": _board_to_json(board),
            "current_player": cp,
            "game_id": gid,
            "value": value,
            "pair": pair,
            "pi1_qr": [[int(q), int(r)] for (q, r), _p in lab.pi1],
            "pi1_p": [float(p) for _c, p in lab.pi1],
            "joint": joint,
            "winning_singles": solver.get("winning_singles", []),
            "winning_pairs": solver.get("winning_pairs", []),
            "forced_first_move": solver.get("forced_first_move"),
            "proven": proven,
        })
    return out


def load_positions_and_solver(input_pkl, reuse_dir, limit=None):
    """Load positions (deterministic order) + aligned solver columns from shards."""
    from game import Player
    from training.distill.generate_distill import _board_has_win
    import pickle
    import __main__
    __main__.Player = Player

    with open(input_pkl, "rb") as f:
        pos = pickle.load(f)
    gid_map = {}
    positions = []
    for entry in pos:
        board, cp = entry[0], entry[1]
        if not board or _board_has_win(board):
            continue
        human_gid = entry[3] if len(entry) == 4 else entry[4]
        gid = gid_map.setdefault(human_gid, len(gid_map))
        board_int = {(int(q), int(r)): (p.value if hasattr(p, "value") else int(p))
                     for (q, r), p in board.items()}
        cp_int = cp.value if hasattr(cp, "value") else int(cp)
        positions.append((board_int, cp_int, gid))
        if limit and len(positions) >= limit:
            break

    # reuse solver columns from existing shards, in order
    solver = []
    for path in sorted(glob.glob(os.path.join(reuse_dir, "shard_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                solver.append({
                    "proven": r.get("proven", 0),
                    "forced_first_move": r.get("forced_first_move"),
                    "winning_singles": r.get("winning_singles", []),
                    "winning_pairs": r.get("winning_pairs", []),
                })
    if limit:
        solver = solver[:limit]
    if len(solver) != len(positions):
        raise SystemExit(
            f"alignment error: {len(positions)} positions vs {len(solver)} reused "
            f"solver rows — reuse-dir must be a label run over the same input.")
    return [(b, cp, g, s) for (b, cp, g), s in zip(positions, solver)]


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
    ap.add_argument("--reuse-dir", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill_gnn", "data", "labeled"),
        help="existing label run to copy solver columns from")
    ap.add_argument("--output-dir", default=os.path.join(
        _KRAKEN_ROOT, "training", "distill_gnn", "data", "labeled_joint"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="worker device: 'cpu' (safe, no VRAM/context risk) or 'cuda'. "
                         "The teacher is tiny; CPU across workers parallelizes the "
                         "GIL-held graph builder without multiple CUDA contexts.")
    ap.add_argument("--k", type=int, default=8, help="top-k first moves for pi2")
    ap.add_argument("--m", type=int, default=16, help="top-m second moves per first move")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--maxtasks", type=int, default=12,
                    help="chunks per worker before it is recycled — the graph "
                         "labeling path leaks memory (worker commit grows to "
                         "10GB+ over hundreds of chunks), so recycle workers")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip positions already present in output-dir shards "
                         "(matched by board json + current_player; shard order "
                         "is arbitrary under imap_unordered, so match content)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"loading positions + reusing solver from {args.reuse_dir} ...", flush=True)
    data = load_positions_and_solver(args.input, args.reuse_dir, args.limit)
    n_forced = sum(1 for d in data if d[3].get("proven") == 1)
    print(f"{len(data):,} positions ({n_forced:,} reused forced wins), "
          f"workers={args.workers} k={args.k} m={args.m}", flush=True)

    shard = 0
    if args.resume:
        done_keys = set()
        existing = sorted(glob.glob(os.path.join(args.output_dir, "shard_*.jsonl.gz")))
        for path in existing:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    done_keys.add((r["board"], r["current_player"]))
        before = len(data)
        data = [t for t in data if (_board_to_json(t[0]), t[1]) not in done_keys]
        shard = len(existing)
        print(f"resume: {before - len(data):,} already labeled in {len(existing)} "
              f"shards; {len(data):,} to go", flush=True)

    chunks = [data[i:i + args.chunk] for i in range(0, len(data), args.chunk)]

    buf, done = [], 0
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(CKPT, args.k, args.m, args.device),
              maxtasksperchild=args.maxtasks) as pool:
        for recs in pool.imap_unordered(_label_chunk, chunks):
            buf.extend(recs); done += len(recs)
            while len(buf) >= args.shard_size:
                _write_shard(os.path.join(args.output_dir, f"shard_{shard:05d}.jsonl.gz"),
                             buf[:args.shard_size])
                buf = buf[args.shard_size:]; shard += 1
            el = time.perf_counter() - t0
            print(f"  {done:,}/{len(data):,}  {done/el:5.0f} pos/s  {el:.0f}s", flush=True)
    if buf:
        _write_shard(os.path.join(args.output_dir, f"shard_{shard:05d}.jsonl.gz"), buf)
        shard += 1

    el = time.perf_counter() - t0
    print(f"DONE: {done:,} positions in {el:.0f}s ({done/el:.0f} pos/s), "
          f"{shard} shards -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
