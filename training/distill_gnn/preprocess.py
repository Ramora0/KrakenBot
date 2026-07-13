"""Phase 2a preprocessing: JSONL.gz teacher-label shards -> numpy cache.

Runs in the KrakenBot venv (numpy only). Centers each board into the 25x25
torus and maps every coordinate (greedy pair, pi1 soft-policy cells, winning
sets, solver forced move) through the same offset so all flat grid indices are
consistent with the planes.

Positions whose stone bounding box does not fit in BOARD_SIZE are skipped;
individual target cells that fall outside the grid after centering are dropped.

Cache arrays (n = kept positions):
  planes.npy          [n,2,25,25] uint8   plane0=current player, plane1=opp
  moves.npy           [n,2] int16         greedy pair, flat grid idx (-1 missing)
  value.npy           [n] float32         teacher scalar value (proven -> +/-1)
  proven.npy          [n] int8            +1 solver forced win else 0
  pi1_idx.npy         [n,K] int16         soft-policy cells, flat idx (-1 pad)
  pi1_p.npy           [n,K] float32       soft-policy mass (aligned)
  forced_idx.npy      [n] int16           solver forced first move, flat idx (-1)
  winning_singles.npy [n,MAX_S] int16     depth-1 completions (-1 pad)
  winning_pairs.npy   [n,MAX_P,2] int16   (-1 pad)
  game_ids.npy        [n] int32           source human game (leak-free split)
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np

from model.resnet import BOARD_SIZE

CACHE_VERSION = "gnn-v3"   # bumped: top-k joint (K first moves, each with pi2|a)
MAX_PI1 = 32
MAX_K = 8                  # top-k first moves kept in the joint target
MAX_PI2 = 16               # top-m second moves per first move
MAX_WIN_SINGLES = 10
MAX_WIN_PAIRS = 10
BS = BOARD_SIZE


def _iter_records(shard_dir):
    for path in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _parse_board(board_json):
    return {tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(board_json).items()}


def build_cache(shard_dir, cache_dir):
    records = list(_iter_records(shard_dir))
    n_in = len(records)
    print(f"  {n_in:,} records from {shard_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    planes = np.zeros((n_in, 2, BS, BS), dtype=np.uint8)
    moves = np.full((n_in, 2), -1, dtype=np.int16)
    value = np.zeros(n_in, dtype=np.float32)
    proven = np.zeros(n_in, dtype=np.int8)
    pi1_idx = np.full((n_in, MAX_PI1), -1, dtype=np.int16)
    pi1_p = np.zeros((n_in, MAX_PI1), dtype=np.float32)
    # top-k joint: K first moves a_k (with pi1 mass pa_k), each with pi2|a_k
    joint_a_idx = np.full((n_in, MAX_K), -1, dtype=np.int16)
    joint_a_p = np.zeros((n_in, MAX_K), dtype=np.float32)
    joint_b_idx = np.full((n_in, MAX_K, MAX_PI2), -1, dtype=np.int16)
    joint_b_p = np.zeros((n_in, MAX_K, MAX_PI2), dtype=np.float32)
    forced_idx = np.full(n_in, -1, dtype=np.int16)
    wsingles = np.full((n_in, MAX_WIN_SINGLES), -1, dtype=np.int16)
    wpairs = np.full((n_in, MAX_WIN_PAIRS, 2), -1, dtype=np.int16)
    game_ids = np.zeros(n_in, dtype=np.int32)

    kept = 0
    skipped_span = 0
    for rec in records:
        board = _parse_board(rec["board"])
        if not board:
            continue
        qs = [q for q, _ in board]
        rs = [r for _, r in board]
        min_q, max_q = min(qs), max(qs)
        min_r, max_r = min(rs), max(rs)
        if (max_q - min_q + 1) > BS or (max_r - min_r + 1) > BS:
            skipped_span += 1
            continue
        off_q = (BS - (max_q - min_q + 1)) // 2 - min_q
        off_r = (BS - (max_r - min_r + 1)) // 2 - min_r

        def to_flat(q, r):
            gq, gr = q + off_q, r + off_r
            if 0 <= gq < BS and 0 <= gr < BS:
                return gq * BS + gr
            return -1

        cp = int(rec["current_player"])
        i = kept
        for (q, r), p in board.items():
            gq, gr = q + off_q, r + off_r
            planes[i, 0 if p == cp else 1, gq, gr] = 1

        # greedy pair
        pr = rec["pair"]
        for j, m in enumerate(pr[:2]):
            moves[i, j] = to_flat(int(m[0]), int(m[1]))

        value[i] = float(rec["value"])
        proven[i] = int(rec["proven"])
        game_ids[i] = int(rec["game_id"])

        # soft policy pi1 (drop out-of-grid cells, keep order)
        k = 0
        for (qr, p) in zip(rec["pi1_qr"], rec["pi1_p"]):
            fl = to_flat(int(qr[0]), int(qr[1]))
            if fl >= 0:
                pi1_idx[i, k] = fl
                pi1_p[i, k] = float(p)
                k += 1
                if k >= MAX_PI1:
                    break

        # top-k joint: one slot per teacher first move a_k (skip out-of-grid a_k)
        slot = 0
        for ent in (rec.get("joint") or []):
            if slot >= MAX_K or not ent.get("b_qr"):
                continue
            a = ent["a"]
            afl = to_flat(int(a[0]), int(a[1]))
            if afl < 0:
                continue
            joint_a_idx[i, slot] = afl
            joint_a_p[i, slot] = float(ent.get("pa", 1.0))
            kk = 0
            for (bqr, bp) in zip(ent["b_qr"], ent["b_p"]):
                fl = to_flat(int(bqr[0]), int(bqr[1]))
                if fl >= 0:
                    joint_b_idx[i, slot, kk] = fl
                    joint_b_p[i, slot, kk] = float(bp)
                    kk += 1
                    if kk >= MAX_PI2:
                        break
            if kk == 0:              # no in-grid second moves -> free the slot
                joint_a_idx[i, slot] = -1
                joint_a_p[i, slot] = 0.0
            else:
                slot += 1

        ff = rec.get("forced_first_move")
        if ff is not None:
            forced_idx[i] = to_flat(int(ff[0]), int(ff[1]))

        for k2, s in enumerate(rec.get("winning_singles", [])[:MAX_WIN_SINGLES]):
            wsingles[i, k2] = to_flat(int(s[0]), int(s[1]))
        for k2, pr2 in enumerate(rec.get("winning_pairs", [])[:MAX_WIN_PAIRS]):
            wpairs[i, k2, 0] = to_flat(int(pr2[0][0]), int(pr2[0][1]))
            wpairs[i, k2, 1] = to_flat(int(pr2[1][0]), int(pr2[1][1]))

        kept += 1

    # trim to kept
    sl = slice(0, kept)
    arrays = {
        "planes": planes[sl], "moves": moves[sl], "value": value[sl],
        "proven": proven[sl], "pi1_idx": pi1_idx[sl], "pi1_p": pi1_p[sl],
        "joint_a_idx": joint_a_idx[sl], "joint_a_p": joint_a_p[sl],
        "joint_b_idx": joint_b_idx[sl], "joint_b_p": joint_b_p[sl],
        "forced_idx": forced_idx[sl], "winning_singles": wsingles[sl],
        "winning_pairs": wpairs[sl], "game_ids": game_ids[sl],
    }
    for name, arr in arrays.items():
        np.save(os.path.join(cache_dir, name + ".npy"), arr)
    with open(os.path.join(cache_dir, "DONE"), "w") as f:
        f.write(f"{CACHE_VERSION}:{kept}")

    mb = sum(a.nbytes for a in arrays.values()) / 1e6
    print(f"  kept {kept:,} / {n_in:,} ({skipped_span:,} skipped: bbox > {BS}), "
          f"cache {mb:.0f} MB -> {cache_dir}")
    return kept


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=os.path.join(
        os.path.dirname(__file__), "data", "labeled"))
    ap.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(__file__), "data", "cache"))
    args = ap.parse_args()
    build_cache(args.shard_dir, args.cache_dir)


if __name__ == "__main__":
    main()
