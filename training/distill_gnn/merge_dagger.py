"""Merge human-position joint shards with DAgger joint shards for training.

- Human shards are hardlinked (or copied) into the output dir unchanged.
- DAgger records are dropped if their (board, current_player) already appears
  in the human set (identical positions in different splits would leak), and
  their game_id is offset by --gid-base so the game-id split never collides
  with human game ids.

Run in either venv (stdlib only):
  python -m training.distill_gnn.merge_dagger \
      --human-dir training/distill_gnn/data/labeled_joint_k8 \
      --dagger-dir training/distill_gnn/data/dagger_joint_k8 \
      --out-dir training/distill_gnn/data/merged_k8_dagger
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--dagger-dir", required=True, nargs="+",
                    help="one or more dagger joint-shard dirs (each gets its "
                         "own gid offset block and output shard)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gid-base", type=int, default=1_000_000)
    ap.add_argument("--repeat", type=int, default=1,
                    help="write each kept dagger record N times (upweight small "
                         "targeted sets, e.g. blunder-mined states; copies share "
                         "a game_id so the split never leaks them)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    human_keys = set()
    n_human = 0
    for path in sorted(glob.glob(os.path.join(args.human_dir, "shard_*.jsonl.gz"))):
        dst = os.path.join(args.out_dir, os.path.basename(path))
        if not os.path.exists(dst):
            try:
                os.link(path, dst)
            except OSError:
                import shutil
                shutil.copy2(path, dst)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                human_keys.add((r["board"], r["current_player"]))
                n_human += 1
    print(f"human: {n_human:,} records, {len(human_keys):,} unique keys")

    seen = set(human_keys)
    # new dagger shards must not collide with shards hardlinked from the human
    # dir (which may itself be a previous merge containing shard_9000N files)
    def _idx(p):
        stem = os.path.basename(p)[len("shard_"):-len(".jsonl.gz")]
        return int(stem) if stem.isdigit() else -1
    next_idx = 1 + max(
        [_idx(p) for p in glob.glob(os.path.join(args.out_dir, "shard_*.jsonl.gz"))]
        + [89999])
    for d_i, ddir in enumerate(args.dagger_dir):
        kept = dropped = 0
        gid_off = args.gid_base * (d_i + 1)
        out_path = os.path.join(args.out_dir, f"shard_{next_idx + d_i:05d}.jsonl.gz")
        with gzip.open(out_path + ".tmp", "wt", encoding="utf-8") as out:
            for path in sorted(glob.glob(os.path.join(ddir, "shard_*.jsonl.gz"))):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        key = (r["board"], r["current_player"])
                        if key in seen:
                            dropped += 1
                            continue
                        seen.add(key)
                        r["game_id"] = int(r["game_id"]) + gid_off
                        line_out = json.dumps(r) + "\n"
                        for _ in range(args.repeat):
                            out.write(line_out)
                        kept += 1
        os.replace(out_path + ".tmp", out_path)
        print(f"dagger[{ddir}]: kept {kept:,}, dropped {dropped:,} (dups) -> {out_path}")
    print(f"merged dir: {args.out_dir}")


if __name__ == "__main__":
    main()
