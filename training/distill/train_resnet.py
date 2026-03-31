"""Train HexResNet with pair attention policy on distillation data.

Trains a multi-head ResNet: value (win rate) + pair policy (attention over cell
embeddings producing N×N pair logits) + moves-left + chain (auxiliary heads).

For positions with known winning moves the policy loss pushes all probability
mass onto the set of winning pairs (any pair containing a winning single, or
any explicitly listed winning pair) without caring about the distribution
within that set.  Non-winning positions use standard pair cross-entropy.

Usage:
  python -m training.distill.train_resnet --epochs 5 --batch-size 256 --amp
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from game import HEX_DIRECTIONS
from model.resnet import BOARD_SIZE, HexResNet

CACHE_VERSION = "v3"  # bumped: winning_singles + winning_pairs, all double-move

MAX_WIN_SINGLES = 10  # max winning single cells per example
MAX_WIN_PAIRS = 10    # max winning cell pairs per example


# ---------------------------------------------------------------------------
# Chain target computation (vectorized, GPU-friendly)
# ---------------------------------------------------------------------------

def _precompute_chain_tables(N=BOARD_SIZE, win_len=6):
    """Precompute window cell indices and per-direction cell membership."""
    n_windows = N * N * 3  # 3 hex directions
    win_qs = np.zeros((n_windows, win_len), dtype=np.int64)
    win_rs = np.zeros((n_windows, win_len), dtype=np.int64)

    w_idx = 0
    for dq, dr in HEX_DIRECTIONS:
        for sq in range(N):
            for sr in range(N):
                for i in range(win_len):
                    win_qs[w_idx, i] = (sq + i * dq) % N
                    win_rs[w_idx, i] = (sr + i * dr) % N
                w_idx += 1

    # Per-direction cell-to-windows membership
    n_per_dir = N * N
    cw_per_dir = []
    cm_per_dir = []
    for d in range(3):
        start = d * n_per_dir
        end = start + n_per_dir
        membership = [[] for _ in range(N * N)]
        for w in range(start, end):
            for i in range(win_len):
                flat = win_qs[w, i] * N + win_rs[w, i]
                membership[flat].append(w)
        max_per = max(len(m) for m in membership)
        cw = np.zeros((N * N, max_per), dtype=np.int64)
        cm = np.zeros((N * N, max_per), dtype=np.bool_)
        for idx, m in enumerate(membership):
            cw[idx, :len(m)] = m
            cm[idx, :len(m)] = True
        cw_per_dir.append(torch.from_numpy(cw))
        cm_per_dir.append(torch.from_numpy(cm))

    return torch.from_numpy(win_qs), torch.from_numpy(win_rs), cw_per_dir, cm_per_dir


_WIN_QS, _WIN_RS, _CW_PER_DIR, _CM_PER_DIR = _precompute_chain_tables()
_chain_tables_cache: dict = {}


def _get_chain_tables(device):
    """Get chain tables on the given device (cached)."""
    if device not in _chain_tables_cache:
        _chain_tables_cache[device] = (
            _WIN_QS.to(device), _WIN_RS.to(device),
            [cw.to(device) for cw in _CW_PER_DIR],
            [cm.to(device) for cm in _CM_PER_DIR],
        )
    return _chain_tables_cache[device]


def compute_chain_targets_batch(planes):
    """Compute per-direction chain targets from planes tensor.

    Args:
        planes: [B, 2, N, N] float tensor (0/1 values)
    Returns:
        targets: [B, 6, N, N] float — per-dir unblocked chain per cell per player
        mask: [B, 6, N, N] float — 0 on cells occupied by the other player
    """
    device = planes.device
    wq, wr, cw_list, cm_list = _get_chain_tables(device)
    B, _, N, _ = planes.shape

    cur = planes[:, 0]  # [B, N, N]
    opp = planes[:, 1]

    cur_in = cur[:, wq, wr]  # [B, n_windows, 6]
    opp_in = opp[:, wq, wr]

    targets = planes.new_zeros(B, 6, N, N)

    for ch_base, (p_in, b_in) in enumerate([(cur_in, opp_in), (opp_in, cur_in)]):
        counts = p_in.sum(dim=2)               # [B, n_windows]
        blocked = b_in.sum(dim=2) > 0          # [B, n_windows]
        unblocked = counts * (~blocked).float()

        for d in range(3):
            cw = cw_list[d]
            cm_float = cm_list[d].unsqueeze(0).float()
            vals = unblocked[:, cw] * cm_float
            targets[:, ch_base * 3 + d] = vals.max(dim=2).values.reshape(B, N, N)

    mask = planes.new_ones(B, 6, N, N)
    mask[:, 0:3] = (1 - opp).unsqueeze(1).expand(B, 3, N, N)
    mask[:, 3:6] = (1 - cur).unsqueeze(1).expand(B, 3, N, N)

    return targets, mask


# ---------------------------------------------------------------------------
# Preprocessing: parquet → .npy cache (single-process, ~5 GB RAM)
# ---------------------------------------------------------------------------

def preprocess_to_cache(parquet_path, cache_dir):
    """Parse parquet into .npy files. Single-process to keep RAM low."""
    df = pd.read_parquet(parquet_path)
    n = len(df)
    bs = BOARD_SIZE
    print(f"  Preprocessing {n:,} rows to {cache_dir} (one-time)...")
    os.makedirs(cache_dir, exist_ok=True)

    planes_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "planes.npy"), mode="w+",
        dtype=np.uint8, shape=(n, 2, bs, bs),
    )
    moves_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "moves.npy"), mode="w+",
        dtype=np.int16, shape=(n, 2),
    )
    wins_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "wins.npy"), mode="w+",
        dtype=np.int8, shape=(n,),
    )
    gids_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "game_ids.npy"), mode="w+",
        dtype=np.int32, shape=(n,),
    )
    moves_left_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "moves_left.npy"), mode="w+",
        dtype=np.int16, shape=(n,),
    )
    ws_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "winning_singles.npy"), mode="w+",
        dtype=np.int16, shape=(n, MAX_WIN_SINGLES),
    )
    ws_mm[:] = -1
    wp_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "winning_pairs.npy"), mode="w+",
        dtype=np.int16, shape=(n, MAX_WIN_PAIRS, 2),
    )
    wp_mm[:] = -1

    boards = df["board"].values
    cps = df["current_player"].values
    moves_col = df["moves"].values
    win_col = df["win_score"].values
    gid_col = df["game_id"].values
    has_winning = "winning_singles" in df.columns
    if has_winning:
        ws_col = df["winning_singles"].values
        wp_col = df["winning_pairs"].values
    del df

    # Track stone counts for moves_left computation
    stone_counts = np.zeros(n, dtype=np.int16)
    total_after = np.zeros(n, dtype=np.int16)

    for i in tqdm(range(n), desc="  Parsing", unit="pos", mininterval=2):
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(boards[i]).items()
        }
        cp = int(cps[i])
        off_q, off_r = 0, 0

        if board_dict:
            qs = [q for q, _r in board_dict]
            rs = [r for _q, r in board_dict]
            min_q, max_q = min(qs), max(qs)
            min_r, max_r = min(rs), max(rs)
            off_q = (bs - (max_q - min_q + 1)) // 2 - min_q
            off_r = (bs - (max_r - min_r + 1)) // 2 - min_r

            for (q, r), player in board_dict.items():
                gq, gr = q + off_q, r + off_r
                if player == cp:
                    planes_mm[i, 0, gq, gr] = 1
                else:
                    planes_mm[i, 1, gq, gr] = 1

        raw_moves = moves_col[i]
        move_indices = []
        for m in raw_moves:
            gq = int(m[0]) + off_q
            gr = int(m[1]) + off_r
            if 0 <= gq < bs and 0 <= gr < bs:
                move_indices.append(gq * bs + gr)

        moves_mm[i, 0] = move_indices[0] if len(move_indices) >= 1 else 0
        moves_mm[i, 1] = move_indices[1] if len(move_indices) >= 2 else -1
        wins_mm[i] = 1 if float(win_col[i]) > 0 else -1
        gids_mm[i] = int(gid_col[i])

        stone_counts[i] = len(board_dict)
        total_after[i] = len(board_dict) + len(move_indices)

        # Parse winning moves (apply same centering offset)
        if has_winning:
            for k, (sq, sr) in enumerate(json.loads(ws_col[i])):
                if k >= MAX_WIN_SINGLES:
                    break
                gq, gr = int(sq) + off_q, int(sr) + off_r
                if 0 <= gq < bs and 0 <= gr < bs:
                    ws_mm[i, k] = gq * bs + gr
            for k, pair in enumerate(json.loads(wp_col[i])):
                if k >= MAX_WIN_PAIRS:
                    break
                gq1, gr1 = int(pair[0][0]) + off_q, int(pair[0][1]) + off_r
                gq2, gr2 = int(pair[1][0]) + off_q, int(pair[1][1]) + off_r
                if (0 <= gq1 < bs and 0 <= gr1 < bs and
                        0 <= gq2 < bs and 0 <= gr2 < bs):
                    wp_mm[i, k, 0] = gq1 * bs + gr1
                    wp_mm[i, k, 1] = gq2 * bs + gr2

    # Compute moves_left: game_total_stones - current_stone_count
    gids_arr = np.array(gids_mm[:])
    max_gid = int(gids_arr.max())
    game_totals = np.zeros(max_gid + 1, dtype=np.int16)
    np.maximum.at(game_totals, gids_arr, total_after)
    moves_left_mm[:] = game_totals[gids_arr] - stone_counts

    for mm in (planes_mm, moves_mm, wins_mm, gids_mm, moves_left_mm,
               ws_mm, wp_mm):
        mm.flush()

    with open(os.path.join(cache_dir, "DONE"), "w") as f:
        f.write(f"{CACHE_VERSION}:{n}")

    size_mb = (planes_mm.nbytes + moves_mm.nbytes + wins_mm.nbytes
               + moves_left_mm.nbytes + ws_mm.nbytes + wp_mm.nbytes) / 1e6
    print(f"  Cache written: {size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Loading: cache → CPU tensors, split by game_id
# ---------------------------------------------------------------------------

def load_data(parquet_path, cache_dir, val_fraction=0.2, seed=42):
    """Load cache and split by game_id."""
    done_path = os.path.join(cache_dir, "DONE")
    need_reprocess = True
    if os.path.exists(done_path):
        with open(done_path) as f:
            content = f.read().strip()
        if content.startswith(CACHE_VERSION + ":"):
            need_reprocess = False

    if need_reprocess:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
        preprocess_to_cache(parquet_path, cache_dir)

    print("Loading cache...")
    gids_mm = np.load(os.path.join(cache_dir, "game_ids.npy"), mmap_mode="r")
    n_total = len(gids_mm)

    # Split by game_id
    gids_arr = np.array(gids_mm[:])
    unique_gids = sorted(set(gids_arr.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_gids)
    n_val_games = max(1, int(len(unique_gids) * val_fraction))
    val_set = set(unique_gids[:n_val_games])

    val_mask = np.isin(gids_arr, list(val_set))
    train_sub = np.where(~val_mask)[0]
    val_sub = np.where(val_mask)[0]
    del gids_arr, gids_mm
    print(f"  {n_total:,} rows, {len(train_sub):,} train / {len(val_sub):,} val "
          f"({len(unique_gids) - n_val_games} / {n_val_games} games)")

    planes_mm = np.load(os.path.join(cache_dir, "planes.npy"), mmap_mode="r")
    moves_mm = np.load(os.path.join(cache_dir, "moves.npy"), mmap_mode="r")
    wins_mm = np.load(os.path.join(cache_dir, "wins.npy"), mmap_mode="r")
    ml_mm = np.load(os.path.join(cache_dir, "moves_left.npy"), mmap_mode="r")
    ws_mm = np.load(os.path.join(cache_dir, "winning_singles.npy"), mmap_mode="r")
    wp_mm = np.load(os.path.join(cache_dir, "winning_pairs.npy"), mmap_mode="r")

    def make_dataset(idx):
        idx = np.sort(idx)
        p = torch.from_numpy(np.array(planes_mm[idx], dtype=np.float32))
        m = torch.from_numpy(np.array(moves_mm[idx], dtype=np.int64))
        w = torch.from_numpy(np.array(wins_mm[idx], dtype=np.float32))
        ml = torch.from_numpy(np.array(ml_mm[idx], dtype=np.float32))
        ws = torch.from_numpy(np.array(ws_mm[idx], dtype=np.int64))
        wp = torch.from_numpy(np.array(wp_mm[idx], dtype=np.int64))
        return TensorDataset(p, m, w, ml, ws, wp)

    print("  Loading train split...")
    train_ds = make_dataset(train_sub)
    print("  Loading val split...")
    val_ds = make_dataset(val_sub)

    n_win = int((ws_mm[train_sub, 0] >= 0).sum() +
                (wp_mm[train_sub, 0, 0] >= 0).sum())
    train_gb = sum(t.nbytes for t in train_ds.tensors) / 1e9
    val_gb = sum(t.nbytes for t in val_ds.tensors) / 1e9
    print(f"  Examples with winning moves: ~{n_win:,}")
    print(f"  CPU RAM: {train_gb:.1f} GB train + {val_gb:.1f} GB val")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _winning_policy_loss(pair_logits, w_singles, w_pairs):
    """Compute -log P(any winning pair) for examples with winning moves.

    Builds a mask of all winning pairs (any pair containing a winning single,
    plus explicitly listed winning pairs), then returns
    logsumexp(all) - logsumexp(winning)  per example.

    Only materializes the [Bw, NC, NC] mask for the subset of examples that
    actually have winning moves, keeping memory small.

    Returns [B] loss (0 for examples without winning moves).
    """
    B, NC, _ = pair_logits.shape
    device = pair_logits.device

    has_win = (w_singles[:, 0] >= 0) | (w_pairs[:, 0, 0] >= 0)
    loss = torch.zeros(B, device=device)
    win_idx = has_win.nonzero(as_tuple=True)[0]
    if len(win_idx) == 0:
        return loss, has_win

    Bw = len(win_idx)
    wl = pair_logits[win_idx]             # [Bw, NC, NC]
    ws = w_singles[win_idx]               # [Bw, max_s]
    wp = w_pairs[win_idx]                 # [Bw, max_p, 2]

    # Cell mask: 1 for winning single cells
    cell_mask = torch.zeros(Bw, NC, device=device, dtype=torch.bool)
    for k in range(ws.shape[1]):
        valid = ws[:, k] >= 0
        if not valid.any():
            break
        cell_mask[valid, ws[valid, k]] = True

    # Pair mask: (i,j) wins if cell_mask[i] or cell_mask[j]
    pair_mask = cell_mask.unsqueeze(2) | cell_mask.unsqueeze(1)  # [Bw, NC, NC]

    # Add explicit winning pairs
    for k in range(wp.shape[1]):
        valid = wp[:, k, 0] >= 0
        if not valid.any():
            break
        v = valid.nonzero(as_tuple=True)[0]
        a, b = wp[v, k, 0], wp[v, k, 1]
        pair_mask[v, a, b] = True
        pair_mask[v, b, a] = True

    # loss = log Z - log(sum over winning pairs)
    masked = wl.masked_fill(~pair_mask, float('-inf'))
    log_Z = wl.reshape(Bw, -1).logsumexp(dim=-1)
    log_win = masked.reshape(Bw, -1).logsumexp(dim=-1)

    loss[win_idx] = log_Z - log_win
    return loss, has_win


def compute_loss(value_pred, pair_logits, ml_pred, chain_pred,
                 wins, moves, ml_target, chain_target, chain_mask,
                 w_singles, w_pairs,
                 value_weight=1.0, policy_weight=1.0, entropy_weight=0.01,
                 ml_weight=0.1, chain_weight=0.1):
    """Pair policy CE + value MSE + entropy + moves-left MSE + chain MSE.

    For examples with winning moves, the policy loss is replaced by
    -log P(winning pair) which pushes all mass onto the winning set.

    pair_logits: [B, N, N] — symmetrized pair scores
    moves: [B, 2] — (m1, m2) flat cell indices
    w_singles: [B, max_s] int — winning single-cell indices, -1 padded
    w_pairs: [B, max_p, 2] int — winning pair indices, -1 padded
    """
    B, N, _ = pair_logits.shape
    m1 = moves[:, 0]
    m2 = moves[:, 1]

    value_loss = F.mse_loss(value_pred, wins)

    # Policy: CE for normal examples, winning-set loss for finishing examples
    flat_logits = pair_logits.reshape(B, -1)  # [B, N²]
    pair_target = m1 * N + m2
    ce_loss = F.cross_entropy(flat_logits, pair_target, reduction='none')  # [B]

    win_loss, has_win = _winning_policy_loss(pair_logits, w_singles, w_pairs)
    policy_per = torch.where(has_win, win_loss, ce_loss)
    policy_loss = policy_per.mean()

    # Entropy regularization (only on non-winning examples)
    if entropy_weight > 0:
        probs = F.softmax(flat_logits.float(), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean()
        entropy_loss = -entropy
    else:
        entropy = torch.tensor(0.0, device=flat_logits.device)
        entropy_loss = 0.0

    # Moves-left MSE (normalized by 150 to keep scale similar to value)
    ml_pred_norm = ml_pred / 150.0
    ml_tgt_norm = ml_target / 150.0
    ml_loss = F.mse_loss(ml_pred_norm, ml_tgt_norm)

    # Chain masked MSE (values 0-6)
    chain_diff_sq = (chain_pred - chain_target) ** 2
    masked = chain_diff_sq * chain_mask
    chain_loss = masked.sum() / chain_mask.sum().clamp(min=1)

    total = (value_weight * value_loss
             + policy_weight * policy_loss
             + entropy_weight * entropy_loss
             + ml_weight * ml_loss
             + chain_weight * chain_loss)

    return total, value_loss, policy_loss, entropy, ml_loss, chain_loss


# ---------------------------------------------------------------------------
# Finishing evaluation
# ---------------------------------------------------------------------------

def _find_finishing_indices(w_singles, w_pairs, wins):
    """Return indices of val examples that have winning moves and win_score > 0."""
    has_singles = w_singles[:, 0] >= 0
    has_pairs = w_pairs[:, 0, 0] >= 0
    has_win = has_singles | has_pairs
    return (has_win & (wins > 0)).nonzero(as_tuple=True)[0]


@torch.no_grad()
def evaluate_finishing(model, device, planes, moves, finish_indices,
                       batch_size=512):
    """Check if model's top pair prediction completes 6-in-a-row.

    Returns (model_rate, gt_rate, n_positions).
    """
    N = BOARD_SIZE
    NC = N * N
    wq, wr = _get_chain_tables(torch.device('cpu'))[:2]

    total = len(finish_indices)
    if total == 0:
        return 0.0, 0.0, 0

    model_wins = 0
    gt_wins = 0

    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        idx = finish_indices[i:j]
        B = len(idx)
        b_range = torch.arange(B)

        batch = planes[idx].to(device)
        _, pair_logits, _, _ = model(batch)

        top = pair_logits.reshape(B, -1).argmax(dim=-1).cpu()
        pred_s1, pred_s2 = top // NC, top % NC

        gt = moves[idx]
        gt_s1, gt_s2 = gt[:, 0], gt[:, 1]

        base = planes[idx, 0]

        for s1, s2, label in [(pred_s1, pred_s2, 'model'),
                               (gt_s1, gt_s2, 'gt')]:
            board = base.clone()
            board[b_range, s1 // N, s1 % N] = 1
            board[b_range, s2 // N, s2 % N] = 1
            won = (board[:, wq, wr].sum(dim=2) >= 6).any(dim=1).sum().item()
            if label == 'model':
                model_wins += won
            else:
                gt_wins += won

    return model_wins / total, gt_wins / total, total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = os.path.splitext(args.input)[0] + "_cache"
    train_ds, val_ds = load_data(args.input, cache_dir, args.val_fraction)

    if args.overfit_batches > 0:
        n = min(args.overfit_batches * args.batch_size, len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, range(n))
        val_ds = torch.utils.data.Subset(val_ds, range(min(n, len(val_ds))))
        print(f"  Overfit mode: {len(train_ds)} train / {len(val_ds)} val")

    # Extract val tensors for finishing eval
    if hasattr(val_ds, 'tensors'):
        _vp, _vm, _vw, _, _vws, _vwp = val_ds.tensors
    else:
        _si = list(val_ds.indices)
        _vp = val_ds.dataset.tensors[0][_si]
        _vm = val_ds.dataset.tensors[1][_si]
        _vw = val_ds.dataset.tensors[2][_si]
        _vws = val_ds.dataset.tensors[4][_si]
        _vwp = val_ds.dataset.tensors[5][_si]
    finish_idx = _find_finishing_indices(_vws, _vwp, _vw)
    print(f"  Finishing eval: {len(finish_idx):,} val positions with winning moves")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = HexResNet(
        num_blocks=args.num_blocks, num_filters=args.num_filters,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_blocks} blocks, {args.num_filters} filters, "
          f"{n_params:,} params")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")

    # wandb
    use_wandb = HAS_WANDB
    if use_wandb:
        try:
            wandb.init(
                project="hex-tictactoe-resnet",
                config=vars(args),
                name=f"pair_b{args.num_blocks}_f{args.num_filters}_lr{args.lr}",
            )
            wandb.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without wandb")
            use_wandb = False

    # Resume
    start_epoch = 0
    global_step = 0
    resume_path = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {resume_path} (epoch {start_epoch}, "
              f"step {global_step}, best_val={best_val_loss:.4f})")

    # Scheduler (created after subsetting so step count is correct)
    n_batches = len(train_loader)
    total_steps = args.epochs * n_batches

    def lr_lambda(step):
        progress = step / max(total_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_path and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    log_every = args.log_every
    n_train = len(train_ds)
    n_val = len(val_ds)
    N = BOARD_SIZE * BOARD_SIZE
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_v = 0.0
        train_p = 0.0
        train_ml = 0.0
        train_ch = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for planes, moves, wins, ml_target, w_singles, w_pairs in pbar:
            planes = planes.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            wins = wins.to(device, non_blocking=True)
            ml_target = ml_target.to(device, non_blocking=True)
            w_singles = w_singles.to(device, non_blocking=True)
            w_pairs = w_pairs.to(device, non_blocking=True)

            # Compute chain targets on-the-fly from board planes
            with torch.no_grad():
                chain_target, chain_mask = compute_chain_targets_batch(planes)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    v_pred, pair_logits, ml_pred, chain_pred = model(planes)
                    loss, v_loss, p_loss, ent, ml_loss, ch_loss = compute_loss(
                        v_pred, pair_logits, ml_pred, chain_pred,
                        wins, moves, ml_target, chain_target, chain_mask,
                        w_singles, w_pairs,
                        args.value_weight, args.policy_weight,
                        args.entropy_weight, args.ml_weight, args.chain_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                v_pred, pair_logits, ml_pred, chain_pred = model(planes)
                loss, v_loss, p_loss, ent, ml_loss, ch_loss = compute_loss(
                    v_pred, pair_logits, ml_pred, chain_pred,
                    wins, moves, ml_target, chain_target, chain_mask,
                    w_singles, w_pairs,
                    args.value_weight, args.policy_weight,
                    args.entropy_weight, args.ml_weight, args.chain_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                optimizer.step()

            bs = len(wins)
            v_val = v_loss.item()
            p_val = p_loss.item()
            ent_val = ent.item()
            ml_val = ml_loss.item()
            ch_val = ch_loss.item()
            loss_val = loss.item()

            # NaN detection with diagnostic dump
            if not (loss_val == loss_val):  # fast NaN check
                logit_fin = pair_logits[pair_logits.isfinite()]
                nan_parts = []
                if not (v_val == v_val): nan_parts.append("value")
                if not (p_val == p_val): nan_parts.append("policy")
                if not (ent_val == ent_val): nan_parts.append("entropy")
                if not (ml_val == ml_val): nan_parts.append("moves_left")
                if not (ch_val == ch_val): nan_parts.append("chain")
                print(
                    f"\n  NaN at step {global_step} | "
                    f"nan_in: {','.join(nan_parts) or 'total_only'} | "
                    f"v={v_val:.4f} p={p_val:.4f} H={ent_val:.4f} "
                    f"ml={ml_val:.4f} ch={ch_val:.4f} | "
                    f"logits: [{pair_logits.min().item():.1f}, "
                    f"{pair_logits.max().item():.1f}] "
                    f"inf={pair_logits.isinf().sum().item()} "
                    f"nan={pair_logits.isnan().sum().item()}"
                )
                continue  # skip accumulation to avoid poisoning averages

            train_v += v_val * bs
            train_p += p_val * bs
            train_ml += ml_val * bs
            train_ch += ch_val * bs
            n_seen += bs
            global_step += 1

            # Periodic health check
            if global_step % 200 == 0:
                logit_abs_max = pair_logits.detach().abs().max().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                ).item()
                qk_norms = {}
                for name, param in model.named_parameters():
                    if "pair_head" in name and "proj" in name:
                        qk_norms[name.split(".")[-1]] = (
                            f"{param.data.norm().item():.2f}"
                        )
                pbar.set_postfix(
                    v=f"{train_v/n_seen:.4f}",
                    p=f"{train_p/n_seen:.4f}",
                    ml=f"{train_ml/n_seen:.4f}",
                    ch=f"{train_ch/n_seen:.4f}",
                    H=f"{ent_val:.2f}",
                    logit_max=f"{logit_abs_max:.1f}",
                    gnorm=f"{grad_norm:.1f}",
                    **qk_norms,
                )
            elif global_step % 50 == 0:
                pbar.set_postfix(
                    v=f"{train_v/n_seen:.4f}",
                    p=f"{train_p/n_seen:.4f}",
                    ml=f"{train_ml/n_seen:.4f}",
                    ch=f"{train_ch/n_seen:.4f}",
                    H=f"{ent_val:.2f}",
                )

            scheduler.step()

            if use_wandb and global_step % log_every == 0:
                logit_abs_max = pair_logits.detach().abs().max().item()
                wandb.log({
                    "step": global_step,
                    "train/value_loss_step": v_val,
                    "train/policy_loss_step": p_val,
                    "train/ml_loss_step": ml_val,
                    "train/chain_loss_step": ch_val,
                    "train/total_loss_step": loss_val,
                    "train/entropy": ent_val,
                    "train/logit_abs_max": logit_abs_max,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # Validation
        model.eval()
        val_v = 0.0
        val_p = 0.0
        val_ml = 0.0
        val_ch = 0.0
        v_correct = 0
        pair_correct = 0
        either_correct = 0
        ml_abs_err = 0.0

        with torch.no_grad():
            for planes, moves, wins, ml_target, w_singles, w_pairs in val_loader:
                planes = planes.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                wins = wins.to(device, non_blocking=True)
                ml_target = ml_target.to(device, non_blocking=True)
                w_singles = w_singles.to(device, non_blocking=True)
                w_pairs = w_pairs.to(device, non_blocking=True)

                chain_target, chain_mask = compute_chain_targets_batch(planes)

                v_pred, pair_logits, ml_pred, chain_pred = model(planes)
                _, vl, pl, _, mll, chl = compute_loss(
                    v_pred, pair_logits, ml_pred, chain_pred,
                    wins, moves, ml_target, chain_target, chain_mask,
                    w_singles, w_pairs,
                    args.value_weight, args.policy_weight,
                    0.0, args.ml_weight, args.chain_weight,
                )
                bs = len(wins)
                val_v += vl.item() * bs
                val_p += pl.item() * bs
                val_ml += mll.item() * bs
                val_ch += chl.item() * bs
                v_correct += ((v_pred > 0) == (wins > 0)).sum().item()
                ml_abs_err += (ml_pred - ml_target).abs().sum().item()

                # Pair accuracy: top predicted pair matches target (either order)
                m1 = moves[:, 0]
                m2 = moves[:, 1]
                flat = pair_logits.reshape(bs, -1)
                top_flat = flat.argmax(dim=-1)
                pred_a = top_flat // N
                pred_b = top_flat % N
                exact = ((pred_a == m1) & (pred_b == m2))
                flipped = ((pred_a == m2) & (pred_b == m1))
                pair_correct += (exact | flipped).sum().item()

                # Either-cell accuracy: at least one predicted cell is correct
                a_in = (pred_a == m1) | (pred_a == m2)
                b_in = (pred_b == m1) | (pred_b == m2)
                either_correct += (a_in | b_in).sum().item()

        val_combined = val_v / n_val + val_p / n_val
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  train v={train_v/n_train:.4f} p={train_p/n_train:.4f} "
            f"ml={train_ml/n_train:.4f} ch={train_ch/n_train:.4f} | "
            f"val v={val_v/n_val:.4f} p={val_p/n_val:.4f} "
            f"ml={val_ml/n_val:.4f} ch={val_ch/n_val:.4f} | "
            f"v_acc={v_correct/n_val:.3f} pair_acc={pair_correct/n_val:.3f} "
            f"either={either_correct/n_val:.3f} "
            f"ml_mae={ml_abs_err/n_val:.1f} | "
            f"lr={lr_now:.6f} | {elapsed:.0f}s"
        )

        # Finishing eval
        f_model, f_gt, f_n = evaluate_finishing(
            model, device, _vp, _vm, finish_idx)
        print(f"  finish: model={f_model:.1%} gt={f_gt:.1%} (n={f_n:,})")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/value_loss": train_v / n_train,
                "train/policy_loss": train_p / n_train,
                "train/ml_loss": train_ml / n_train,
                "train/chain_loss": train_ch / n_train,
                "train/total_loss": (train_v + train_p) / n_train,
                "val/value_loss": val_v / n_val,
                "val/policy_loss": val_p / n_val,
                "val/ml_loss": val_ml / n_val,
                "val/chain_loss": val_ch / n_val,
                "val/total_loss": val_combined,
                "val/value_acc": v_correct / n_val,
                "val/pair_acc": pair_correct / n_val,
                "val/either_cell_acc": either_correct / n_val,
                "val/ml_mae": ml_abs_err / n_val,
                "val/finish_model": f_model,
                "val/finish_gt": f_gt,
            }, step=global_step)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_loss": min(best_val_loss, val_combined),
            "global_step": global_step,
            "args": vars(args),
        }
        tmp_path = ckpt_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            print(f"  -> New best (val_loss={val_combined:.4f})")

    print(f"\nDone in {time.time()-t0:.0f}s. Best val_loss={best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train HexResNet (pair policy)")
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "data", "distill_100k.parquet"))
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(__file__), "resnet_results"))
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01,
                        help="Entropy regularization weight (default: 0.01)")
    parser.add_argument("--ml-weight", type=float, default=0.1,
                        help="Moves-left auxiliary loss weight (default: 0.1)")
    parser.add_argument("--chain-weight", type=float, default=0.1,
                        help="Chain auxiliary loss weight (default: 0.1)")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="Train on N batches only (debug)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log to wandb every N steps (default: 50)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
