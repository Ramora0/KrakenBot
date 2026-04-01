"""MCTS self-play training loop: generate → train → evaluate.

Usage:
    python -m training.selfplay.train_loop --amp
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from game import HexGame, HEX_DIRECTIONS, Player
from model.resnet import BOARD_SIZE, HexResNet, board_to_planes_torus
from model.symmetry import (
    apply_symmetry_planes, apply_symmetry_chain, PERMS, N as SYM_N,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Chain target precomputation (vectorized with numpy)
# ---------------------------------------------------------------------------

def _precompute_chain_tables(N=BOARD_SIZE, win_len=6):
    """Precompute window cell indices and per-direction cell membership."""
    n_windows = N * N * 3  # 3 directions
    win_qs = np.zeros((n_windows, win_len), dtype=np.int32)
    win_rs = np.zeros((n_windows, win_len), dtype=np.int32)

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
    cell_windows_per_dir = []
    cell_masks_per_dir = []
    for d in range(3):
        start = d * n_per_dir
        end = start + n_per_dir
        membership = [[] for _ in range(N * N)]
        for w in range(start, end):
            for i in range(win_len):
                flat = win_qs[w, i] * N + win_rs[w, i]
                membership[flat].append(w)
        max_per = max(len(m) for m in membership)
        cw = np.zeros((N * N, max_per), dtype=np.int32)
        cm = np.zeros((N * N, max_per), dtype=bool)
        for idx, m in enumerate(membership):
            cw[idx, :len(m)] = m
            cm[idx, :len(m)] = True
        cell_windows_per_dir.append(cw)
        cell_masks_per_dir.append(cm)

    return win_qs, win_rs, cell_windows_per_dir, cell_masks_per_dir


_WIN_QS, _WIN_RS, _CW_PER_DIR, _CM_PER_DIR = _precompute_chain_tables()


CHAIN_VERSION = 3  # bump when cached fields change (added current_players)


def compute_chain_targets(board_dict, current_player):
    """Compute per-direction chain targets [6, N, N] and mask [6, N, N].

    Channels 0-2: current player, directions 0, 1, 2.
    Channels 3-5: opponent, directions 0, 1, 2.
    Mask: 0 on cells occupied by the other player.
    """
    N = BOARD_SIZE
    cp = int(current_player)

    cur_board = np.zeros((N, N), dtype=np.int8)
    opp_board = np.zeros((N, N), dtype=np.int8)
    for (q, r), p in board_dict.items():
        if p == cp:
            cur_board[q, r] = 1
        else:
            opp_board[q, r] = 1

    # Precompute per-window counts once (shared across directions)
    player_in_cur = cur_board[_WIN_QS, _WIN_RS]    # [n_windows, 6]
    blocker_in_cur = opp_board[_WIN_QS, _WIN_RS]
    player_in_opp = blocker_in_cur                   # opponent's stones
    blocker_in_opp = player_in_cur

    targets = np.zeros((6, N, N), dtype=np.float32)

    for ch_base, (p_in, b_in) in enumerate(
            [(player_in_cur, blocker_in_cur),
             (player_in_opp, blocker_in_opp)]):
        counts = p_in.sum(axis=1)              # [n_windows]
        blocked = b_in.any(axis=1)             # [n_windows]
        unblocked = np.where(blocked, 0, counts).astype(np.float32)

        for d in range(3):
            cw = _CW_PER_DIR[d]
            cm = _CM_PER_DIR[d]
            vals = unblocked[cw]               # [N*N, max_per_dir]
            vals[~cm] = 0
            targets[ch_base * 3 + d] = vals.max(axis=1).reshape(N, N)

    # Loss mask: don't predict on cells occupied by the other player
    mask = np.ones((6, N, N), dtype=np.float32)
    for (q, r), p in board_dict.items():
        if p == cp:
            mask[3, q, r] = 0.0   # mask opp channels on current's cells
            mask[4, q, r] = 0.0
            mask[5, q, r] = 0.0
        else:
            mask[0, q, r] = 0.0   # mask current channels on opp's cells
            mask[1, q, r] = 0.0
            mask[2, q, r] = 0.0

    return torch.from_numpy(targets), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Training dataset with sparse visits, D6 augmentation, per-round weights
# ---------------------------------------------------------------------------

class SelfPlayDataset(torch.utils.data.Dataset):
    """Dataset of self-play positions with D6 augmentation and round weights.

    Visits are stored sparsely (~200 entries per sample) and densified on the
    fly in __getitem__. Chain targets and moves-left are included for
    auxiliary losses. A random D6 symmetry is applied at access time.
    """

    def __init__(self, planes, visit_dicts, values, round_ids,
                 current_players,
                 chain_targets, chain_masks, moves_left, draw_mask,
                 current_round, decay=0.75, augment=True):
        self.planes = planes              # [N, 2, 25, 25]
        self.visit_dicts = visit_dicts    # list of list[(flat_pair_idx, prob)]
        self.values = values              # [N]
        self.chain_targets = chain_targets  # [N, 6, 25, 25]
        self.chain_masks = chain_masks    # [N, 6, 25, 25]
        self.moves_left = moves_left      # [N]
        self.draw_mask = draw_mask        # [N] bool — True = drawn, mask out
        self.augment = augment
        self._NN = BOARD_SIZE * BOARD_SIZE

        ages = current_round - round_ids.float()
        self.weights = decay ** ages

        # --- Outcome-balanced sampling ---
        # Determine absolute game outcome per example:
        #   A-win: (cp==1 and val>0) or (cp==2 and val<0)
        #   B-win: (cp==2 and val>0) or (cp==1 and val<0)
        #   draw:  draw_mask is True
        # Reweight so A-win and B-win examples have equal total weight.
        cp = current_players.float()
        v = values
        a_win = ((cp == 1) & (v > 0)) | ((cp == 2) & (v < 0))
        b_win = ((cp == 2) & (v > 0)) | ((cp == 1) & (v < 0))
        # Draws and SFT (cp==0) get no reweighting
        a_win = a_win & ~draw_mask
        b_win = b_win & ~draw_mask

        w_a = self.weights[a_win].sum()
        w_b = self.weights[b_win].sum()
        if w_a > 0 and w_b > 0:
            # Scale the overrepresented side down to match the underrepresented
            target = (w_a + w_b) / 2
            self.weights[a_win] *= target / w_a
            self.weights[b_win] *= target / w_b
            pct_a = a_win.sum().item() / max(1, (a_win | b_win).sum().item())
            print(f"  Outcome balance: {a_win.sum().item():,} A-win examples, "
                  f"{b_win.sum().item():,} B-win examples "
                  f"({pct_a:.0%} / {1 - pct_a:.0%}) — weights equalized")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        planes = self.planes[idx]
        value = self.values[idx]
        visit_entries = self.visit_dicts[idx]
        chain_t = self.chain_targets[idx]   # [6, 25, 25]
        chain_m = self.chain_masks[idx]     # [6, 25, 25]
        ml = self.moves_left[idx]
        drawn = self.draw_mask[idx]

        if self.augment:
            k = random.randint(0, 11)
        else:
            k = 0

        if k != 0:
            planes = apply_symmetry_planes(planes, k)
            chain_t = apply_symmetry_chain(chain_t, k)
            chain_m = apply_symmetry_chain(chain_m, k)

        # Build dense visit vector, applying symmetry to sparse entries
        NN = self._NN
        visit_vec = torch.zeros(NN * NN)
        if visit_entries:
            if k != 0:
                perm = PERMS[k]
                for flat_idx, prob in visit_entries:
                    a = flat_idx // NN
                    b = flat_idx % NN
                    new_a = int(perm[a])
                    new_b = int(perm[b])
                    visit_vec[new_a * NN + new_b] = prob
            else:
                for flat_idx, prob in visit_entries:
                    visit_vec[flat_idx] = prob

        return planes, visit_vec, value, chain_t, chain_m, ml, drawn


def _load_sft_examples(parquet_path: str, max_examples: int = 50000):
    """Load SFT/distillation data and convert to self-play tensor format.

    Filters to double-move examples, centers boards onto the 25x25 grid,
    and converts hard move targets to one-hot visit vectors.
    """
    df = pd.read_parquet(parquet_path)

    # Filter to double-move examples only
    df = df[df["moves"].apply(lambda m: len(m) >= 2)].copy()
    # Filter out draws (win_score == 0)
    df = df[df["win_score"] != 0.0].copy()

    if max_examples and len(df) > max_examples:
        # Sample by game_id for game-level integrity
        game_ids = df["game_id"].unique()
        rng = np.random.default_rng(42)
        rng.shuffle(game_ids)
        selected = set()
        count = 0
        for gid in game_ids:
            gdf = df[df["game_id"] == gid]
            if count + len(gdf) > max_examples:
                break
            selected.add(gid)
            count += len(gdf)
        df = df[df["game_id"].isin(selected)].copy()

    n = len(df)
    if n == 0:
        return None

    N = BOARD_SIZE * BOARD_SIZE  # 625
    bs = BOARD_SIZE

    planes_tensor = torch.zeros(n, 2, bs, bs)
    visit_dicts: list[list[tuple[int, float]]] = []
    value_tensor = torch.zeros(n)
    chain_targets = torch.zeros(n, 6, bs, bs)
    chain_masks = torch.zeros(n, 6, bs, bs)

    boards = df["board"].values
    cps = df["current_player"].values
    moves_col = df["moves"].values
    win_col = df["win_score"].values

    for i in tqdm(range(n), desc="Loading SFT data", unit="ex", mininterval=2):
        board_raw = json.loads(boards[i])
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in board_raw.items()
        }
        cp_int = int(cps[i])

        # Center board onto 25x25 grid (same logic as train_resnet.py)
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
                if 0 <= gq < bs and 0 <= gr < bs:
                    if player == cp_int:
                        planes_tensor[i, 0, gq, gr] = 1.0
                    else:
                        planes_tensor[i, 1, gq, gr] = 1.0

        # Chain targets using centered board
        centered_board = {}
        cp = Player(cp_int)
        for (q, r), player in board_dict.items():
            gq, gr = q + off_q, r + off_r
            if 0 <= gq < bs and 0 <= gr < bs:
                centered_board[(gq, gr)] = Player(player)
        ct, cm = compute_chain_targets(centered_board, cp)
        chain_targets[i] = ct
        chain_masks[i] = cm

        # Convert move pair to one-hot visit entry
        raw_moves = moves_col[i]
        m1_q, m1_r = int(raw_moves[0][0]) + off_q, int(raw_moves[0][1]) + off_r
        m2_q, m2_r = int(raw_moves[1][0]) + off_q, int(raw_moves[1][1]) + off_r
        m1_flat = m1_q * bs + m1_r
        m2_flat = m2_q * bs + m2_r
        if 0 <= m1_flat < N and 0 <= m2_flat < N:
            visit_dicts.append([(m1_flat * N + m2_flat, 1.0)])
        else:
            visit_dicts.append([])

        value_tensor[i] = 1.0 if float(win_col[i]) > 0 else -1.0

    print(f"Loaded {n:,} SFT examples from {parquet_path}")
    return planes_tensor, visit_dicts, value_tensor, chain_targets, chain_masks


def _preprocess_round(parquet_path: str) -> dict:
    """Preprocess one round's parquet → tensors. Cached as .pt file."""
    cache_path = parquet_path.replace('.parquet', '.pt')

    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(parquet_path):
            d = torch.load(cache_path, weights_only=False)
            if d.get('chain_version', 1) == CHAIN_VERSION:
                n = len(d['values'])
                print(f"  {os.path.basename(parquet_path)}: {n:,} examples (cached)")
                return d
            print(f"  Chain version mismatch, recomputing cache for "
                  f"{os.path.basename(parquet_path)}")

    df = pd.read_parquet(parquet_path)
    n = len(df)
    N = BOARD_SIZE * BOARD_SIZE
    bs = BOARD_SIZE

    planes = torch.zeros(n, 2, bs, bs)
    visit_dicts = []
    values = torch.zeros(n)
    rids = torch.zeros(n, dtype=torch.int64)
    cps = torch.zeros(n, dtype=torch.int8)
    chain_t = torch.zeros(n, 6, bs, bs)
    chain_m = torch.zeros(n, 6, bs, bs)
    ml = torch.zeros(n)
    dm = torch.zeros(n, dtype=torch.bool)
    has_ml = "moves_left" in df.columns

    for i, row in enumerate(tqdm(df.itertuples(), total=n,
                                  desc=f"  Caching {os.path.basename(parquet_path)}",
                                  unit="ex", mininterval=2)):
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(row.board).items()
        }
        cp = int(row.current_player)
        cps[i] = cp
        planes[i] = board_to_planes_torus(board_dict, cp)
        ct, cm = compute_chain_targets(board_dict, cp)
        chain_t[i] = ct
        chain_m[i] = cm

        pair_visits_raw = json.loads(row.pair_visits)
        total_visits = sum(pair_visits_raw.values())
        entries = []
        if total_visits > 0:
            for key, count in pair_visits_raw.items():
                parts = key.split(",")
                a, b = int(parts[0]), int(parts[1])
                entries.append((a * N + b, count / total_visits))
        visit_dicts.append(entries)

        values[i] = row.value_target
        rids[i] = row.round_id
        if has_ml:
            ml[i] = row.moves_left
            dm[i] = bool(row.game_drawn)
        else:
            dm[i] = True

    result = {
        'planes': planes, 'visit_dicts': visit_dicts,
        'values': values, 'round_ids': rids,
        'current_players': cps,
        'chain_targets': chain_t, 'chain_masks': chain_m,
        'moves_left': ml, 'draw_mask': dm,
        'chain_version': CHAIN_VERSION,
    }
    torch.save(result, cache_path)
    return result


def load_selfplay_rounds(data_dir: str, current_round: int,
                         window: int = 4, decay: float = 0.75,
                         augment: bool = True,
                         sft_path: str = None, sft_weight: float = 0.3,
                         sft_max_examples: int = 50000) -> SelfPlayDataset:
    """Load the last `window` rounds of self-play data into a SelfPlayDataset.

    Each round is cached as a .pt file on first load — subsequent loads
    skip all JSON parsing and are near-instant.
    """
    rounds = range(max(0, current_round - window + 1), current_round + 1)

    all_planes, all_visits, all_values = [], [], []
    all_rids, all_cps, all_ct, all_cm, all_ml, all_dm = [], [], [], [], [], []

    for r in rounds:
        path = os.path.join(data_dir, f"round_{r}.parquet")
        if not os.path.exists(path):
            continue
        d = _preprocess_round(path)
        all_planes.append(d['planes'])
        all_visits.extend(d['visit_dicts'])
        all_values.append(d['values'])
        all_rids.append(d['round_ids'])
        all_cps.append(d.get('current_players', torch.ones(len(d['values']),
                                                            dtype=torch.int8)))
        all_ct.append(d['chain_targets'])
        all_cm.append(d['chain_masks'])
        all_ml.append(d['moves_left'])
        all_dm.append(d['draw_mask'])

    if not all_planes:
        raise FileNotFoundError(f"No self-play data for rounds {list(rounds)}")

    planes_tensor = torch.cat(all_planes)
    value_tensor = torch.cat(all_values)
    round_ids = torch.cat(all_rids)
    current_players = torch.cat(all_cps)
    chain_targets = torch.cat(all_ct)
    chain_masks = torch.cat(all_cm)
    moves_left_tensor = torch.cat(all_ml)
    draw_mask = torch.cat(all_dm)
    visit_dicts = all_visits
    n = len(value_tensor)

    print(f"Loaded {n:,} self-play examples from rounds {list(rounds)}")

    n_selfplay = n

    # --- Mix in SFT data ---
    if sft_path:
        sft = _load_sft_examples(sft_path, max_examples=sft_max_examples)
        if sft is not None:
            sft_planes, sft_visits, sft_values, sft_ct, sft_cm = sft
            n_sft = len(sft_values)
            planes_tensor = torch.cat([planes_tensor, sft_planes])
            visit_dicts.extend(sft_visits)
            value_tensor = torch.cat([value_tensor, sft_values])
            # SFT gets current round_id so decay^0 = 1.0
            sft_round_ids = torch.full((n_sft,), current_round, dtype=torch.int64)
            round_ids = torch.cat([round_ids, sft_round_ids])
            chain_targets = torch.cat([chain_targets, sft_ct])
            chain_masks = torch.cat([chain_masks, sft_cm])
            # SFT has no meaningful moves_left; mask from ml loss
            moves_left_tensor = torch.cat([
                moves_left_tensor, torch.zeros(n_sft)])
            draw_mask = torch.cat([
                draw_mask, torch.ones(n_sft, dtype=torch.bool)])
            # SFT: mark as player 0 so outcome balancing skips them
            current_players = torch.cat([
                current_players, torch.zeros(n_sft, dtype=torch.int8)])

    dataset = SelfPlayDataset(
        planes_tensor, visit_dicts, value_tensor, round_ids,
        current_players,
        chain_targets, chain_masks, moves_left_tensor, draw_mask,
        current_round, decay=decay, augment=augment,
    )

    # Scale SFT weights
    if sft_path and n_selfplay < len(dataset):
        dataset.weights[n_selfplay:] *= sft_weight

    return dataset


def compute_selfplay_loss(value_pred, pair_logits, moves_left_pred, chain_pred,
                          visit_dist, value_target,
                          moves_left_target, draw_mask,
                          chain_target, chain_mask,
                          value_weight=1.0):
    """Combined loss: value + policy + moves_left + chain."""
    B, N_sq, _ = pair_logits.shape

    # --- Primary losses ---
    value_loss = F.mse_loss(value_pred, value_target)

    flat_logits = pair_logits.reshape(B, -1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    # nan_to_num: 0 * -inf (diagonal) -> nan -> 0
    policy_loss = -(visit_dist * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

    # --- Auxiliary: moves left (mask drawn games, normalize to [0,1]) ---
    valid = ~draw_mask
    if valid.any():
        ml_pred_norm = moves_left_pred[valid] / 150.0
        ml_tgt_norm = moves_left_target[valid] / 150.0
        ml_loss = F.mse_loss(ml_pred_norm, ml_tgt_norm)
    else:
        ml_loss = torch.zeros(1, device=value_pred.device).squeeze()

    # --- Auxiliary: chain length (masked MSE, values 0-6) ---
    chain_diff_sq = (chain_pred - chain_target) ** 2  # [B, 2, H, W]
    masked = chain_diff_sq * chain_mask
    chain_loss = masked.sum() / chain_mask.sum().clamp(min=1)

    total = (value_weight * value_loss + policy_loss
             + 0.1 * ml_loss + 0.1 * chain_loss)

    return total, value_loss, policy_loss, ml_loss, chain_loss


def train_one_epoch(model, optimizer, dataset, device, batch_size=512,
                    use_amp=False, scaler=None, grad_clip=5.0,
                    value_weight=1.0):
    """Train one epoch on self-play data with weighted sampling."""
    model.train()
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)

    total_loss = 0.0
    total_vloss = 0.0
    total_ploss = 0.0
    total_ml_loss = 0.0
    total_chain_loss = 0.0
    total_entropy = 0.0
    n_batches = 0

    for (planes, visit_dist, value_target,
         chain_target, chain_mask, ml_target, drawn) in tqdm(
            loader, desc="Training", unit="batch"):
        planes = planes.to(device)
        visit_dist = visit_dist.to(device)
        value_target = value_target.to(device)
        chain_target = chain_target.to(device)
        chain_mask = chain_mask.to(device)
        ml_target = ml_target.to(device)
        drawn = drawn.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                value_pred, pair_logits, ml_pred, chain_pred = model(planes)
                loss, vloss, ploss, ml_loss, cl = compute_selfplay_loss(
                    value_pred, pair_logits, ml_pred, chain_pred,
                    visit_dist, value_target, ml_target, drawn,
                    chain_target, chain_mask,
                    value_weight=value_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            value_pred, pair_logits, ml_pred, chain_pred = model(planes)
            loss, vloss, ploss, ml_loss, cl = compute_selfplay_loss(
                value_pred, pair_logits, ml_pred, chain_pred,
                visit_dist, value_target, ml_target, drawn,
                chain_target, chain_mask,
                value_weight=value_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            flat = pair_logits.reshape(pair_logits.size(0), -1).float()
            p = F.softmax(flat, dim=-1)
            ent = -(p * p.clamp(min=1e-8).log()).sum(dim=-1).mean()
            total_entropy += ent.item()

        total_loss += loss.item()
        total_vloss += vloss.item()
        total_ploss += ploss.item()
        total_ml_loss += ml_loss.item()
        total_chain_loss += cl.item()
        n_batches += 1

    d = max(n_batches, 1)
    return (total_loss / d, total_vloss / d, total_ploss / d,
            total_ml_loss / d, total_chain_loss / d, total_entropy / d)


# ---------------------------------------------------------------------------
# Evaluation vs MinimaxBot (batched MCTS, all games run in parallel)
# ---------------------------------------------------------------------------

def _eval_minimax_worker(args):
    """Process pool worker: run minimax get_move on a game."""
    bot, game = args
    try:
        return bot.get_move(game)
    except Exception:
        return None


@torch.no_grad()
def evaluate_vs_minimax(model, device, n_games: int = 100,
                        n_sims: int = 200, minimax_time: float = 0.1) -> dict:
    """Play MCTSBot vs MinimaxBot with batched GPU eval across all games.

    Returns dict with wins, losses, draws, and score (W=1, D=0.5, L=0).
    """
    from mcts.tree import (
        create_trees_batched, select_leaf, expand_and_backprop,
        maybe_expand_leaf, select_move_pair, select_single_move,
    )
    from game import ToroidalHexGame, TORUS_SIZE

    try:
        import ai_cpp
        minimax_bot = ai_cpp.MinimaxBot(time_limit=minimax_time)
    except ImportError:
        from bot import RandomBot
        print("WARNING: ai_cpp not available, using RandomBot for evaluation")
        minimax_bot = RandomBot(time_limit=minimax_time)
    _is_pair = getattr(minimax_bot, 'pair_moves', False)

    model.eval()
    model_dtype = next(model.parameters()).dtype
    ANCHOR = TORUS_SIZE // 2

    # Process pool for parallel minimax (GIL blocks threads)
    from multiprocessing import Pool as ProcPool
    n_workers = min(n_games, os.cpu_count() or 8)
    proc_pool = ProcPool(n_workers)

    # All games run in parallel
    games = [HexGame() for _ in range(n_games)]
    mcts_side = [Player.A if i % 2 == 0 else Player.B for i in range(n_games)]
    active = set(range(n_games))
    move_counts = [0] * n_games
    wins = losses = draws = 0

    pbar = tqdm(total=n_games, desc="Evaluating", unit="game")

    while active:
        # --- Minimax turns: run in parallel via process pool ---
        minimax_needed = [i for i in active
                          if not games[i].game_over
                          and move_counts[i] < 200
                          and games[i].current_player != mcts_side[i]]

        if minimax_needed:
            args = [(minimax_bot, games[i]) for i in minimax_needed]
            results = proc_pool.map(_eval_minimax_worker, args)
            for i, result in zip(minimax_needed, results):
                if result is None:
                    continue
                moves = result if _is_pair else [result]
                for q, r in moves:
                    if not games[i].game_over:
                        if games[i].make_move(q, r):
                            move_counts[i] += 1

        # Check for games finished by minimax
        for i in list(active):
            if games[i].game_over or move_counts[i] >= 200:
                active.discard(i)
                w, l, d = _eval_result(games[i], mcts_side[i])
                wins += w; losses += l; draws += d
                pbar.update(1)

        if not active:
            break

        # --- Batched MCTS turns ---
        mcts_idx = [i for i in sorted(active)
                    if games[i].current_player == mcts_side[i]
                    and not games[i].game_over]

        if not mcts_idx:
            continue

        # Handle empty boards (first move at origin)
        need_mcts = []
        for i in mcts_idx:
            if not games[i].board:
                games[i].make_move(0, 0)
                move_counts[i] += 1
            else:
                need_mcts.append(i)

        if not need_mcts:
            continue

        # Convert to toroidal for MCTS
        torus_games = [
            ToroidalHexGame.from_hex_game(games[i], ANCHOR, ANCHOR)
            for i in need_mcts
        ]
        B = len(need_mcts)

        # Batch create trees (one GPU forward pass)
        trees = create_trees_batched(torus_games, model, device, add_noise=False)

        # Batch sims
        eval_buf = torch.empty(B, 2, BOARD_SIZE, BOARD_SIZE, dtype=model_dtype)

        for _sim in range(n_sims):
            leaves = [select_leaf(trees[j], torus_games[j]) for j in range(B)]

            # Collect non-terminal leaves needing NN eval
            eval_list = [(j, leaves[j]) for j in range(B)
                         if not leaves[j].is_terminal and leaves[j].deltas]

            eval_map = {}
            vals_cpu = []
            expand_data = {}

            if eval_list:
                n_eval = len(eval_list)
                batch = eval_buf[:n_eval]
                for k, (j, leaf) in enumerate(eval_list):
                    rp = trees[j].root_planes
                    if leaf.player_flipped:
                        batch[k, 0] = rp[1]
                        batch[k, 1] = rp[0]
                    else:
                        batch[k] = rp
                    for gq, gr, ch in leaf.deltas:
                        actual_ch = (1 - ch) if leaf.player_flipped else ch
                        batch[k, actual_ch, gq, gr] = 1.0

                batch_gpu = batch.to(device, non_blocking=True)
                values, pair_logits, _, _ = model(batch_gpu)
                vals_cpu = values.cpu().tolist()

                # Expansion data for leaves that need it
                need_expand = [k for k, (_, lf) in enumerate(eval_list)
                               if lf.needs_expansion]
                if need_expand:
                    ne = len(need_expand)
                    exp_logits = pair_logits[need_expand]
                    flat_logits = exp_logits.reshape(ne, -1)
                    top_raw, top_idxs = flat_logits.topk(200, dim=-1)
                    top_vals = F.softmax(top_raw, dim=-1)
                    marg_logits = exp_logits.logsumexp(dim=-1)
                    margs = F.softmax(marg_logits, dim=-1).cpu()
                    top_idxs_cpu = top_idxs.cpu()
                    top_vals_cpu = top_vals.cpu()
                    for kk, idx in enumerate(need_expand):
                        expand_data[idx] = (
                            margs[kk], top_idxs_cpu[kk], top_vals_cpu[kk])
                del pair_logits

                eval_map = {j: k for k, (j, _) in enumerate(eval_list)}

            # Backprop
            for j, leaf in enumerate(leaves):
                k = eval_map.get(j)
                if leaf.is_terminal:
                    expand_and_backprop(trees[j], leaf, 0.0)
                elif k is not None:
                    expand_and_backprop(trees[j], leaf, vals_cpu[k])
                    data = expand_data.get(k)
                    if data is not None:
                        maybe_expand_leaf(trees[j], leaf, *data)
                else:
                    expand_and_backprop(trees[j], leaf, 0.0)

        # Select moves and apply to real games
        for j, i in enumerate(need_mcts):
            if games[i].moves_left_in_turn == 1:
                tq, tr = select_single_move(trees[j])
                real_moves = [(tq - ANCHOR, tr - ANCHOR)]
            else:
                (t1q, t1r), (t2q, t2r) = select_move_pair(
                    trees[j], temperature=0.1)
                real_moves = [(t1q - ANCHOR, t1r - ANCHOR),
                              (t2q - ANCHOR, t2r - ANCHOR)]

            for q, r in real_moves:
                if not games[i].game_over:
                    if games[i].make_move(q, r):
                        move_counts[i] += 1

        # Check for games finished after MCTS moves
        for i in list(active):
            if games[i].game_over or move_counts[i] >= 200:
                active.discard(i)
                w, l, d = _eval_result(games[i], mcts_side[i])
                wins += w; losses += l; draws += d
                pbar.update(1)

    proc_pool.terminate()
    proc_pool.join()
    pbar.close()
    total = max(wins + losses + draws, 1)
    score = (wins + 0.5 * draws) / total
    print(f"  MCTSBot vs Minimax({minimax_time:.3f}s): {wins}W / {losses}L / {draws}D "
          f"= {100 * score:.1f}% score")
    return {"wins": wins, "losses": losses, "draws": draws, "score": score}


def _eval_result(game, mcts_player):
    """Return (wins, losses, draws) tuple for one finished game."""
    if game.game_over and game.winner != Player.NONE:
        if game.winner == mcts_player:
            return 1, 0, 0
        return 0, 1, 0
    return 0, 0, 1


# ---------------------------------------------------------------------------
# Adaptive crossover time evaluation
# ---------------------------------------------------------------------------

def _estimate_crossover(log_times, scores, old_center_log):
    """Estimate minimax time where MCTS score = 0.5 via log-space interpolation.

    Args:
        log_times: list of 3 log(time) values (ascending).
        scores: list of 3 MCTS scores (should generally decrease as time rises).
        old_center_log: log of previous center estimate (fallback).

    Returns:
        (crossover_log, extrapolated): log of estimated crossover time, and
        whether the estimate required extrapolation beyond the bracket.
    """
    target = 0.5

    # Try to find a bracket containing 0.5
    for i in range(len(scores) - 1):
        s_lo, s_hi = scores[i], scores[i + 1]
        t_lo, t_hi = log_times[i], log_times[i + 1]
        if (s_lo >= target >= s_hi) or (s_lo <= target <= s_hi):
            if abs(s_hi - s_lo) < 1e-9:
                return (t_lo + t_hi) / 2, False
            frac = (target - s_lo) / (s_hi - s_lo)
            return t_lo + frac * (t_hi - t_lo), False

    # Not bracketed — extrapolate
    if all(s > target for s in scores):
        # All wins: minimax needs more time. Extrapolate rightward.
        slope = (scores[-1] - scores[-2]) / (log_times[-1] - log_times[-2])
        if abs(slope) < 1e-9:
            return log_times[-1] + math.log(2), True
        return log_times[-1] + (target - scores[-1]) / slope, True

    if all(s < target for s in scores):
        # All losses: minimax needs less time. Extrapolate leftward.
        slope = (scores[1] - scores[0]) / (log_times[1] - log_times[0])
        if abs(slope) < 1e-9:
            return log_times[0] - math.log(2), True
        return log_times[0] + (target - scores[0]) / slope, True

    # Non-monotonic: fall back to old center
    return old_center_log, False


def evaluate_crossover(model, device, n_games=100, n_sims=200,
                       center=0.1, momentum=0.3, max_time=1.0):
    """Adaptive evaluation: find minimax time where MCTS scores 0.5.

    Plays games at 3 log-spaced time limits around `center`, interpolates the
    crossover point, and smooths with momentum. Actual play times are capped
    at `max_time`; crossover estimates can exceed the cap via extrapolation.

    Returns dict with crossover_time, scores at each bracket, etc.
    """
    # Build bracket in log-space
    raw_times = [center * 0.5, center, center * 2.0]

    # If bracket exceeds cap, anchor at the cap
    if raw_times[-1] > max_time:
        actual_times = [max_time * 0.25, max_time * 0.5, max_time]
    else:
        actual_times = raw_times

    # Distribute games across 3 brackets
    games_per = n_games // 3
    games_last = n_games - 2 * games_per  # remainder to middle bracket

    model.eval()
    results = []
    for i, t in enumerate(actual_times):
        n = games_last if i == 1 else games_per
        print(f"\n  Bracket {i+1}/3: minimax_time={t:.4f}s, {n} games")
        r = evaluate_vs_minimax(model, device, n_games=n,
                                n_sims=n_sims, minimax_time=t)
        results.append(r)

    scores = [r["score"] for r in results]
    log_times = [math.log(t) for t in actual_times]
    old_center_log = math.log(center)

    raw_log, extrapolated = _estimate_crossover(log_times, scores, old_center_log)
    raw_crossover = math.exp(raw_log)

    # Momentum smoothing
    smoothed_log = (1 - momentum) * raw_log + momentum * old_center_log
    crossover_time = math.exp(smoothed_log)

    print(f"\n  Crossover estimate: {crossover_time:.4f}s"
          f" (raw={raw_crossover:.4f}s, center={center:.4f}s"
          f"{', extrapolated' if extrapolated else ''})")

    return {
        "crossover_time": crossover_time,
        "raw_crossover": raw_crossover,
        "score_low": scores[0],
        "score_mid": scores[1],
        "score_high": scores[2],
        "times": actual_times,
        "extrapolated": extrapolated,
    }


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, round_num, output_dir,
                    best_win_rate=0.0, crossover_time=None):
    """Save model checkpoint atomically."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "round": round_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_win_rate": best_win_rate,
        "crossover_time": crossover_time,
    }
    path = os.path.join(output_dir, f"round_{round_num}.pt")
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

    # Also save as best.pt
    best_path = os.path.join(output_dir, "best.pt")
    tmp = best_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, best_path)

    print(f"Checkpoint saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCTS self-play training loop")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of training rounds (default: run indefinitely)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of parallel games in self-play")
    parser.add_argument("--n-sims", type=int, default=200,
                        help="MCTS simulations per turn")
    parser.add_argument("--train-batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Evaluation games per round")
    parser.add_argument("--eval-sims", type=int, default=200,
                        help="MCTS sims for evaluation bot")
    parser.add_argument("--minimax-time", type=float, default=0.001,
                        help="Initial center time for crossover estimation")
    parser.add_argument("--window", type=int, default=4,
                        help="Sliding window of rounds for training data")
    parser.add_argument("--decay", type=float, default=0.75,
                        help="Exponential weight decay per round age")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--output-dir", type=str,
                        default="training/mcts_results",
                        help="Output directory for checkpoints")
    parser.add_argument("--data-dir", type=str,
                        default="training/data/selfplay",
                        help="Directory for self-play data")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate vs minimax every N rounds (default: 10)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Disable live game viewer")
    parser.add_argument("--viewer-port", type=int, default=8765,
                        help="Port for game viewer (default 8765)")
    parser.add_argument("--value-weight", type=float, default=1.0,
                        help="Weight for value loss (default: 1.0)")
    parser.add_argument("--draw-penalty", type=float, default=0.1,
                        help="Draw penalty magnitude (draws get value=-penalty)")
    parser.add_argument("--late-temperature", type=float, default=0.3,
                        help="Temperature for turns >= 20 (default: 0.3)")
    parser.add_argument("--sft-path", type=str, default=None,
                        help="Path to SFT/distillation parquet for mixing")
    parser.add_argument("--sft-weight", type=float, default=0.3,
                        help="Sampling weight for SFT data relative to self-play")
    parser.add_argument("--sft-max-examples", type=int, default=50000,
                        help="Max SFT examples to sample per round")
    parser.add_argument("--sft-anneal-rounds", type=int, default=10,
                        help="Linearly anneal SFT weight to 0 over N rounds")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel self-play (use sequential)")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of worker processes for parallel self-play")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation immediately on startup before training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = HexResNet(
        num_blocks=args.num_blocks, num_filters=args.num_filters
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_blocks} blocks, {args.num_filters} filters, "
          f"{n_params:,} params")



    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume
    start_round = 0
    best_win_rate = 0.0
    crossover_time = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Only load optimizer/scaler from own checkpoints (have "round" key),
        # not from distillation checkpoints which use a different optimizer.
        if "round" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler and ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_round = ckpt.get("round", 0) + 1
            best_win_rate = ckpt.get("best_win_rate", 0.0)
            crossover_time = ckpt.get("crossover_time", None)
            print(f"Resumed from {args.resume} (round {start_round})")
        else:
            print(f"Loaded model weights from {args.resume} (fresh optimizer)")
    elif os.path.exists(os.path.join(args.output_dir, "best.pt")):
        # Auto-resume from best checkpoint
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_round = ckpt.get("round", 0) + 1
        best_win_rate = ckpt.get("best_win_rate", 0.0)
        crossover_time = ckpt.get("crossover_time", None)
        print(f"Auto-resumed from {ckpt_path} (round {start_round})")

    os.makedirs(args.output_dir, exist_ok=True)

    # wandb
    use_wandb = (not args.no_wandb) and HAS_WANDB
    if use_wandb:
        # Load or create a stable run ID so restarts resume the same run
        run_id_path = os.path.join(args.data_dir, "wandb_run_id.txt")
        run_id = os.environ.get("WANDB_RUN_ID")
        if not run_id and os.path.exists(run_id_path):
            run_id = open(run_id_path).read().strip()
        wandb.init(
            project="hex-mcts-selfplay",
            config=vars(args),
            name=f"mcts_r{start_round}",
            id=run_id,
            resume="allow",
        )
        # Save the run ID for future restarts
        with open(run_id_path, "w") as f:
            f.write(wandb.run.id)

    # Game viewer
    viewer = None
    if not args.no_viewer:
        from tools.game_viewer import GameViewer
        viewer = GameViewer(port=args.viewer_port)
        viewer.start()

    os.makedirs(args.data_dir, exist_ok=True)

    # Run evaluation immediately if requested
    if args.evaluate:
        center = crossover_time if crossover_time is not None else args.minimax_time
        print(f"\n--- Startup evaluation: {args.eval_games} games, "
              f"center={center:.4f}s ---")
        model.eval()
        eval_result = evaluate_crossover(
            model, device, n_games=args.eval_games,
            n_sims=args.eval_sims, center=center,
        )
        crossover_time = eval_result["crossover_time"]
        win_rate = eval_result["score_mid"]
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            print(f"  New best win rate at mid time: {100 * best_win_rate:.1f}%")
        if use_wandb:
            wandb.log({
                "eval/crossover_time": crossover_time,
                "eval/score_low": eval_result["score_low"],
                "eval/score_mid": eval_result["score_mid"],
                "eval/score_high": eval_result["score_high"],
                "eval/win_rate": win_rate,
                "round": start_round - 1,
            }, commit=True)

    # Create persistent worker pool if using parallel self-play
    use_parallel = (device.type == 'cuda'
                    and args.batch_size >= 16
                    and not args.no_parallel)
    pool = None
    if use_parallel:
        from training.selfplay.parallel_selfplay import (
            ParallelSelfPlayPool, COLD_START_GAMES, COMPLETED_PER_ROUND,
        )
        pending_path = os.path.join(args.data_dir, "pending.json")
        game_dicts = [None] * args.batch_size
        next_game_id = 0
        is_cold_start = True
        if os.path.exists(pending_path):
            with open(pending_path, 'r') as f:
                pd = json.load(f)
            for i, item in enumerate(pd["games"][:args.batch_size]):
                game_dicts[i] = item
            next_game_id = pd["next_game_id"]
            is_cold_start = False
            print(f"Resumed {len(pd['games'])} in-progress games")
        else:
            next_game_id = args.batch_size

        model_dtype = torch.bfloat16  # inference runs in bfloat16
        pool = ParallelSelfPlayPool(
            args.batch_size, args.n_sims, args.n_workers, model_dtype)
        pool.start(game_dicts, next_game_id)

    try:
        round_num = start_round
        while args.rounds is None or round_num < start_round + args.rounds:
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}")
            print(f"{'='*60}")
            t0 = time.time()

            # --- 1. Self-play ---
            from training.selfplay.self_play import SelfPlayManager
            print(f"\n--- Self-play ---")
            model.eval()
            model.bfloat16()

            if use_parallel:
                target = COLD_START_GAMES if (
                    is_cold_start and round_num == start_round
                ) else COMPLETED_PER_ROUND
                examples, draw_rate, a_win_rate, avg_moves = \
                    pool.generate_round(
                        model, device,
                        round_id=round_num,
                        data_dir=args.data_dir,
                        late_temperature=args.late_temperature,
                        draw_penalty=args.draw_penalty,
                        target=target,
                        viewer=viewer,
                    )
                # Save examples
                manager = SelfPlayManager(
                    model, device, data_dir=args.data_dir)
                manager.save_round(examples, round_num, args.data_dir)
            else:
                manager = SelfPlayManager(
                    model, device,
                    batch_size=args.batch_size,
                    n_sims=args.n_sims,
                    data_dir=args.data_dir,
                    viewer=viewer,
                    late_temperature=args.late_temperature,
                    draw_penalty=args.draw_penalty,
                )
                examples, draw_rate, a_win_rate, avg_moves = \
                    manager.generate(round_num)
                manager.save_round(examples, round_num, args.data_dir)

            model.float()
            t_gen = time.time() - t0

            # --- 2. Train ---
            # Anneal SFT weight linearly to 0
            if args.sft_path and args.sft_anneal_rounds > 0:
                progress = min(round_num / args.sft_anneal_rounds, 1.0)
                cur_sft_weight = args.sft_weight * (1.0 - progress)
            else:
                cur_sft_weight = args.sft_weight
            sft_path = args.sft_path if cur_sft_weight > 0 else None

            print(f"\n--- Training (window={args.window}, decay={args.decay}"
                  f"{f', sft_weight={cur_sft_weight:.3f}' if sft_path else ''}) ---")
            t1 = time.time()
            dataset = load_selfplay_rounds(
                args.data_dir, round_num,
                window=args.window, decay=args.decay,
                sft_path=sft_path, sft_weight=cur_sft_weight,
                sft_max_examples=args.sft_max_examples)
            losses = train_one_epoch(
                model, optimizer, dataset, device,
                batch_size=args.train_batch_size,
                use_amp=use_amp,
                scaler=scaler,
                value_weight=args.value_weight,
            )
            avg_loss, avg_vloss, avg_ploss, avg_ml, avg_chain, avg_entropy = losses
            t_train = time.time() - t1
            print(f"  Loss: {avg_loss:.4f} (value={avg_vloss:.4f}, "
                  f"policy={avg_ploss:.4f}, ml={avg_ml:.4f}, "
                  f"chain={avg_chain:.4f}, entropy={avg_entropy:.4f})")

            # --- 3. Checkpoint ---
            ckpt_path = save_checkpoint(model, optimizer, scaler, round_num,
                                        args.output_dir, best_win_rate,
                                        crossover_time=crossover_time)

            # --- 4. Evaluate (every eval_every rounds) ---
            eval_result = None
            win_rate = None
            t_eval = 0.0
            if (round_num + 1) % args.eval_every == 0:
                center = (crossover_time if crossover_time is not None
                          else args.minimax_time)
                print(f"\n--- Evaluation: {args.eval_games} games, "
                      f"center={center:.4f}s ---")
                t2 = time.time()
                eval_result = evaluate_crossover(
                    model, device, n_games=args.eval_games,
                    n_sims=args.eval_sims, center=center,
                )
                t_eval = time.time() - t2
                crossover_time = eval_result["crossover_time"]
                win_rate = eval_result["score_mid"]

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    print(f"  New best win rate at mid time: "
                          f"{100 * best_win_rate:.1f}%")

            t_total = time.time() - t0

            # --- Log ---
            print(f"\n  Round {round_num} summary:")
            print(f"    Examples: {len(examples):,}")
            print(f"    Draw rate: {100 * draw_rate:.1f}%")
            print(f"    Loss: {avg_loss:.4f}")
            if eval_result is not None:
                print(f"    Crossover time: {crossover_time:.4f}s")
                print(f"    Scores: low={eval_result['score_low']:.2f} "
                      f"mid={eval_result['score_mid']:.2f} "
                      f"high={eval_result['score_high']:.2f}")
            print(f"    Time: {t_gen:.0f}s gen + {t_train:.0f}s train "
                  f"+ {t_eval:.0f}s eval = {t_total:.0f}s total")

            if use_wandb:
                log_data = {
                    "round": round_num,
                    "loss": avg_loss,
                    "value_loss": avg_vloss,
                    "policy_loss": avg_ploss,
                    "moves_left_loss": avg_ml,
                    "chain_loss": avg_chain,
                    "policy_entropy": avg_entropy,
                    "best_win_rate": best_win_rate,
                    "draw_rate": draw_rate,
                    "a_win_pct": a_win_rate,
                    "avg_moves": avg_moves,
                    "examples": len(examples),
                    "time_gen": t_gen,
                    "time_train": t_train,
                    "time_eval": t_eval,
                }
                if eval_result is not None:
                    log_data["eval/crossover_time"] = crossover_time
                    log_data["eval/score_low"] = eval_result["score_low"]
                    log_data["eval/score_mid"] = eval_result["score_mid"]
                    log_data["eval/score_high"] = eval_result["score_high"]
                    log_data["eval/win_rate"] = win_rate
                wandb.log(log_data)

            round_num += 1
    finally:
        if pool is not None:
            pool.shutdown()

    print(f"\n{'='*60}")
    if crossover_time is not None:
        print(f"  Training complete. Crossover time: {crossover_time:.4f}s")
    else:
        print(f"  Training complete. Best win rate: {100 * best_win_rate:.1f}%")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
