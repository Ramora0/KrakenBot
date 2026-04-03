"""Parallel self-play: multiprocessing CPU + single-process GPU batching.

Workers own games/trees and do select_leaf + backprop in parallel.
Main process does full batch-256 GPU forward passes.
One-sim-delayed pipeline overlaps CPU and GPU work.

Architecture:
  Workers: backprop(K-1) + select(K) → write deltas to shared buf
  Main:    read deltas → GPU forward(K) → write results
  Pipeline: workers do CPU for sim K+1 while GPU runs sim K.
"""

from __future__ import annotations

import json
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass, field
from threading import BrokenBarrierError

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from game import Player
from mcts.tree import (
    N_CELLS, select_leaf,
    expand_and_backprop, maybe_expand_leaf, get_pair_visits, get_single_visits,
    select_move_pair, select_single_move, _build_tree_from_eval,
    _hex_dist_torus,
)
from model.resnet import BOARD_SIZE

try:
    from mcts._mcts_cy import CyGameState, select_leaf_cy, backprop_cy
    _HAS_CY = True
except ImportError:
    _HAS_CY = False

from training.selfplay.self_play import (
    MAX_GAME_MOVES, COMPLETED_PER_ROUND, COLD_START_GAMES, SelfPlaySlot,
    _CENTER,
)
from game import TORUS_SIZE


# ---------------------------------------------------------------------------
# Shared memory buffers between main and workers
# ---------------------------------------------------------------------------

class SharedBuffers:
    """Pre-allocated shared memory for main↔worker communication."""

    def __init__(self, batch_size, model_dtype=torch.float32):
        BS = BOARD_SIZE
        # Double-buffered delta planes (workers write, main reads)
        self.delta = [
            torch.zeros(batch_size, 2, BS, BS, dtype=model_dtype).share_memory_()
            for _ in range(2)
        ]
        # Which slots need NN eval (workers write, main reads)
        self.needs_eval = [
            torch.zeros(batch_size, dtype=torch.bool).share_memory_()
            for _ in range(2)
        ]
        # Which slots need expansion data (workers write, main reads)
        self.needs_expand_flag = [
            torch.zeros(batch_size, dtype=torch.bool).share_memory_()
            for _ in range(2)
        ]

        # GPU results (main writes, workers read)
        self.values = torch.zeros(batch_size, dtype=torch.float32).share_memory_()
        self.has_expand = torch.zeros(batch_size, dtype=torch.bool).share_memory_()
        self.marginals = torch.zeros(batch_size, N_CELLS,
                                     dtype=torch.float32).share_memory_()
        self.top_indices = torch.zeros(batch_size, 200,
                                       dtype=torch.int64).share_memory_()
        self.top_values = torch.zeros(batch_size, 200,
                                      dtype=torch.float32).share_memory_()

        # Tree creation buffers (workers write planes, main writes NN results)
        self.tree_planes = torch.zeros(
            batch_size, 2, BS, BS, dtype=model_dtype).share_memory_()
        self.tree_needs_init = torch.zeros(
            batch_size, dtype=torch.bool).share_memory_()
        self.tree_values = torch.zeros(
            batch_size, dtype=torch.float32).share_memory_()
        self.tree_marginals = torch.zeros(
            batch_size, N_CELLS, dtype=torch.float32).share_memory_()
        self.tree_pair_probs = torch.zeros(
            batch_size, N_CELLS, N_CELLS, dtype=torch.float16).share_memory_()

        # Per-slot model ID (workers write, main reads)
        # 0 = primary model, 1 = opponent model. All-zero for self-play.
        self.model_id = torch.zeros(
            batch_size, dtype=torch.int8).share_memory_()
        # Per-slot model assignment: model_for_player[gi, player_int]
        # Maps (game, player) → model_id. Set by main at init.
        # player_int: 1=Player.A, 2=Player.B
        self.model_for_player = torch.zeros(
            batch_size, 3, dtype=torch.int8).share_memory_()  # index 0 unused

        # Synchronization
        # worker_barrier: workers sync among themselves
        # deltas_ready[i]: signals main that delta buf[i] is filled
        # results_ready: signals workers that GPU results are available
        self.n_workers = 0  # set later


def _create_sync(n_workers):
    """Create synchronization primitives. Must be called before spawn."""
    return {
        'worker_barrier': mp.Barrier(n_workers),
        'deltas_ready': [mp.Event() for _ in range(2)],
        'results_ready': mp.Event(),
        'error': mp.Value('i', 0),  # error flag
        'stop': mp.Value('i', 0),   # set to 1 when main wants workers to exit
        'tree_request_ready': mp.Event(),   # workers → main: tree planes written
        'tree_results_ready': mp.Event(),   # main → workers: tree NN results ready
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _new_game():
    if _HAS_CY:
        g = CyGameState()
    else:
        from game import ToroidalHexGame
        g = ToroidalHexGame()
    g.make_move(_CENTER, _CENTER)
    return g


def _worker_fn(worker_id, n_workers, batch_size, n_sims,
               game_dicts, next_game_id_start,
               shared_bufs, sync, round_id,
               late_temperature, draw_penalty,
               result_queue):
    """Worker process: owns games/trees, does select + backprop."""
    try:
        _worker_loop(worker_id, n_workers, batch_size, n_sims,
                     game_dicts, next_game_id_start,
                     shared_bufs, sync, round_id,
                     late_temperature, draw_penalty,
                     result_queue)
    except Exception as e:
        sync['error'].value = 1
        result_queue.put(('error', worker_id, traceback.format_exc()))
        # Unblock barriers/events so main doesn't hang
        try:
            sync['worker_barrier'].abort()
        except Exception:
            pass


def _worker_loop(worker_id, n_workers, batch_size, n_sims,
                 game_dicts, next_game_id_start,
                 shared_bufs, sync, round_id,
                 late_temperature, draw_penalty,
                 result_queue):
    """Inner worker loop (unwrapped for clean error handling)."""
    games_per_worker = batch_size // n_workers
    my_start = worker_id * games_per_worker
    my_end = my_start + games_per_worker
    if worker_id == n_workers - 1:
        my_end = batch_size  # last worker takes remainder

    n_mine = my_end - my_start

    # Create/restore games
    slots = []
    for i in range(n_mine):
        global_idx = my_start + i
        if global_idx < len(game_dicts) and game_dicts[global_idx] is not None:
            gd = game_dicts[global_idx]
            if _HAS_CY:
                game = CyGameState.from_dict(gd['game'])
            else:
                from game import ToroidalHexGame
                game = ToroidalHexGame.from_dict(gd['game'])
            slot = SelfPlaySlot(
                game=game, game_id=gd['game_id'],
                turn_number=gd['turn_number'], examples=gd['examples'])
        else:
            game = _new_game()
            slot = SelfPlaySlot(game=game,
                                game_id=next_game_id_start + global_idx)
        slots.append(slot)

    _sel = select_leaf_cy if _HAS_CY else select_leaf
    _bp = backprop_cy if _HAS_CY else expand_and_backprop

    wb = sync['worker_barrier']
    deltas_ready = sync['deltas_ready']
    results_ready = sync['results_ready']
    tree_request_ready = sync['tree_request_ready']
    tree_results_ready = sync['tree_results_ready']

    # ---- Outer turn loop ----
    stop_flag = sync.get('stop')  # mp.Value set by main when done

    while not (stop_flag and stop_flag.value):
        # --- Tree creation via GPU shared-memory protocol ---
        # Step 1: write board planes for slots needing trees
        shared_bufs.tree_needs_init[my_start:my_end] = False
        for i, slot in enumerate(slots):
            gi = my_start + i
            _write_model_id(shared_bufs, slot, gi)
            if slot.tree is None and not slot.game.game_over \
               and slot.game.move_count < MAX_GAME_MOVES:
                if hasattr(slot.game, 'to_planes_tensor'):
                    shared_bufs.tree_planes[gi] = slot.game.to_planes_tensor()
                else:
                    from model.resnet import board_to_planes_torus
                    shared_bufs.tree_planes[gi] = board_to_planes_torus(
                        slot.game.board, slot.game.current_player)
                shared_bufs.tree_needs_init[gi] = True

        # Step 2: barrier + signal main
        wb.wait()
        if worker_id == 0:
            tree_request_ready.set()

        # Step 3: wait for GPU results
        tree_results_ready.wait()

        if stop_flag and stop_flag.value:
            break

        # Step 4: build trees locally from shared memory data
        for i, slot in enumerate(slots):
            gi = my_start + i
            if shared_bufs.tree_needs_init[gi]:
                slot.tree = _build_tree_from_eval(
                    slot.game,
                    shared_bufs.tree_values[gi].item(),
                    shared_bufs.tree_pair_probs[gi].float(),
                    shared_bufs.tree_marginals[gi].clone(),
                    shared_bufs.tree_planes[gi].clone(),
                    add_noise=True,
                )

        # Step 5: barrier + clear events
        wb.wait()
        if worker_id == 0:
            tree_request_ready.clear()
            tree_results_ready.clear()

        # ---- Sim loop with one-sim-delayed pipeline ----
        try:
            # Phase A: initial select (sim 0)
            leaves = [None] * n_mine
            for i, slot in enumerate(slots):
                leaves[i] = _sel(slot.tree, slot.game)
                _write_delta(shared_bufs, leaves[i], slot.tree,
                             my_start + i, 0)

            wb.wait()  # all workers done
            if worker_id == 0:
                deltas_ready[0].set()

            # Phase B: pipelined sims
            for sim in range(n_sims):
                buf_cur = sim % 2
                buf_next = (sim + 1) % 2

                # Select NEXT sim's leaves (overlapped with GPU on current sim)
                next_leaves = [None] * n_mine
                if sim < n_sims - 1:
                    for i, slot in enumerate(slots):
                        next_leaves[i] = _sel(slot.tree, slot.game)
                        _write_delta(shared_bufs, next_leaves[i], slot.tree,
                                     my_start + i, buf_next)

                # Wait for GPU results for current sim
                results_ready.wait()

                # Backprop current sim
                for i, slot in enumerate(slots):
                    gi = my_start + i
                    if not shared_bufs.needs_eval[buf_cur][gi]:
                        _bp(slot.tree, leaves[i], 0.0)
                    else:
                        nn_val = shared_bufs.values[gi].item()
                        _bp(slot.tree, leaves[i], nn_val)
                        if shared_bufs.has_expand[gi]:
                            maybe_expand_leaf(
                                slot.tree, leaves[i],
                                shared_bufs.marginals[gi],
                                shared_bufs.top_indices[gi],
                                shared_bufs.top_values[gi],
                                nn_value=nn_val)

                leaves = next_leaves

                # Sync: all workers done with backprop + next select
                wb.wait()
                if worker_id == 0:
                    results_ready.clear()
                    if sim < n_sims - 1:
                        deltas_ready[buf_next].set()
        except BrokenBarrierError:
            break  # main aborted the barrier during shutdown

        # ---- Move selection + example recording ----
        completed = []
        turn_far_total = 0
        turn_stones_total = 0
        next_gid = next_game_id_start + batch_size + worker_id
        for i, slot in enumerate(slots):
            turn = slot.turn_number
            temp = 1.0 if turn < 20 else late_temperature

            if slot.game.moves_left_in_turn == 1:
                cell = select_single_move(slot.tree)
                moves = [cell]
                pv = get_single_visits(slot.tree)
            else:
                s1, s2 = select_move_pair(slot.tree, temperature=temp)
                moves = [s1, s2]
                pv = get_pair_visits(slot.tree)

            # Record example
            if hasattr(slot.game, 'to_board_dict'):
                bd = slot.game.to_board_dict()
            else:
                bd = slot.game.board
            cp = slot.game.current_player
            cp_val = cp.value if hasattr(cp, 'value') else int(cp)
            ex = {
                "board": json.dumps({
                    f"{q},{r}": v.value if isinstance(v, Player) else int(v)
                    for (q, r), v in bd.items()
                }),
                "current_player": cp_val,
                "pair_visits": json.dumps({
                    f"{a},{b}": c for (a, b), c in pv.items()
                }),
                "value_target": 0.0,
                "move_count": slot.game.move_count,
                "moves_left": 0,
                "game_drawn": False,
                "game_id": slot.game_id,
                "round_id": round_id,
            }
            slot.examples.append(ex)

            # Apply moves (track distance from existing stones)
            for q, r in moves:
                if slot.game.game_over:
                    break
                if hasattr(slot.game, 'get_occupied_set'):
                    occ = slot.game.get_occupied_set()
                else:
                    occ = frozenset(slot.game.board.keys())
                if occ:
                    min_d = min(_hex_dist_torus(q, r, oq, or_)
                                for oq, or_ in occ)
                    turn_stones_total += 1
                    if min_d > 2:
                        turn_far_total += 1
                slot.game.make_move(q, r)

            slot.turn_number += 1
            slot.tree = None
            slot.sims_done = 0

            # Check completion
            if slot.game.game_over or slot.game.move_count >= MAX_GAME_MOVES:
                raw_w = slot.game.winner
                w_val = raw_w.value if hasattr(raw_w, 'value') else int(raw_w)
                if slot.game.game_over and w_val != 0:
                    winner = Player(w_val)
                else:
                    winner = Player.NONE

                total_m = slot.game.move_count
                is_drawn = (winner == Player.NONE)
                for e in slot.examples:
                    e["round_id"] = round_id
                    e["moves_left"] = total_m - e["move_count"]
                    e["game_drawn"] = is_drawn
                    cp_e = Player(e["current_player"])
                    if is_drawn:
                        e["value_target"] = -draw_penalty
                    elif cp_e == winner:
                        e["value_target"] = 1.0
                    else:
                        e["value_target"] = -1.0

                # Capture final board for viewer before replacing slot
                if hasattr(slot.game, 'to_board_dict'):
                    bd_src = slot.game.to_board_dict()
                else:
                    bd_src = slot.game.board
                final_board = {
                    f"{q},{r}": v.value if isinstance(v, Player) else int(v)
                    for (q, r), v in bd_src.items()
                }

                completed.append({
                    'examples': slot.examples,
                    'winner': winner,
                    'move_count': total_m,
                    'game_id': slot.game_id,
                    'turn_number': slot.turn_number,
                    'final_board': final_board,
                })

                # Replace with fresh game
                slot.game = _new_game()
                slot.game_id = next_gid
                next_gid += n_workers
                slot.examples = []
                slot.turn_number = 0

        # Send completed games + far-move stats to main
        result_queue.put(('turn_done', worker_id, completed,
                          [_serialize_slot(s) for s in slots],
                          turn_far_total, turn_stones_total))


def _write_model_id(shared, slot, global_idx):
    """Set model_id for this slot based on current_player and assignment."""
    cp = slot.game.current_player
    cp_int = cp.value if hasattr(cp, 'value') else int(cp)
    shared.model_id[global_idx] = shared.model_for_player[global_idx, cp_int]


def _write_delta(shared, leaf, tree, global_idx, buf_idx):
    """Write one leaf's delta planes to shared buffer."""
    rp = tree.root_planes
    buf = shared.delta[buf_idx]

    if leaf.is_terminal or not leaf.deltas:
        shared.needs_eval[buf_idx][global_idx] = False
        shared.needs_expand_flag[buf_idx][global_idx] = False
        return

    shared.needs_eval[buf_idx][global_idx] = True
    shared.needs_expand_flag[buf_idx][global_idx] = leaf.needs_expansion

    if leaf.player_flipped:
        buf[global_idx, 0] = rp[1]
        buf[global_idx, 1] = rp[0]
    else:
        buf[global_idx] = rp

    for gq, gr, ch in leaf.deltas:
        actual_ch = (1 - ch) if leaf.player_flipped else ch
        buf[global_idx, actual_ch, gq, gr] = 1.0


def _serialize_slot(slot):
    """Serialize slot for pending game save."""
    if slot.game.game_over or slot.game.move_count >= MAX_GAME_MOVES:
        return None
    return {
        'game': slot.game.to_dict(),
        'game_id': slot.game_id,
        'turn_number': slot.turn_number,
        'examples': slot.examples,
    }


# ---------------------------------------------------------------------------
# Main process GPU orchestration
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gpu_forward(models, device, delta_buf, needs_eval, needs_expand_flag,
                 shared):
    """Run GPU forward on the delta buffer, write results to shared.

    Args:
        models: single model or list/tuple of models.
            When multiple models are given, shared.model_id routes each
            slot to the correct model.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    indices = needs_eval.nonzero(as_tuple=True)[0]
    B = len(indices)

    if B == 0:
        return

    if len(models) == 1:
        # Fast path: single model (self-play)
        batch = delta_buf[indices].to(device, non_blocking=True)
        values, pair_logits, _, _ = models[0](batch)
    else:
        # Multi-model: split by model_id, run each, reassemble
        slot_model_ids = shared.model_id[indices]
        values = torch.empty(B, dtype=torch.float32, device=device)
        # Pre-allocate pair_logits on device
        pair_logits = torch.empty(B, N_CELLS, N_CELLS, device=device)
        for mid, m in enumerate(models):
            mask = (slot_model_ids == mid)
            if not mask.any():
                continue
            sub_idx = mask.nonzero(as_tuple=True)[0]
            sub_batch = delta_buf[indices[sub_idx]].to(device, non_blocking=True)
            v, pl, _, _ = m(sub_batch)
            v_flat = v.squeeze(-1) if v.dim() > 1 else v
            values[sub_idx] = v_flat.float()
            pair_logits[sub_idx] = pl.float()

    # Write values (bulk copy, no Python loop)
    vals_cpu = values.cpu()
    idx_list = indices.tolist()
    shared.values[indices] = vals_cpu.float()

    # Only compute expand data for leaves that need it
    shared.has_expand.fill_(False)

    # Map eval-batch indices to global indices for expansion
    expand_global = []
    expand_local = []
    for j, gi in enumerate(idx_list):
        if needs_expand_flag[gi]:
            expand_global.append(gi)
            expand_local.append(j)

    if expand_local:
        local_t = torch.tensor(expand_local, dtype=torch.long, device=device)
        exp_logits = pair_logits[local_t]  # [K, N, N] where K << B
        flat = exp_logits.reshape(len(expand_local), -1)
        top_raw, top_idx = flat.topk(200, dim=-1)
        top_vals = F.softmax(top_raw, dim=-1)
        marg_logits = exp_logits.logsumexp(dim=-1)
        margs = F.softmax(marg_logits, dim=-1)

        margs_cpu = margs.cpu()
        top_idx_cpu = top_idx.cpu()
        top_vals_cpu = top_vals.cpu()

        for j, gi in enumerate(expand_global):
            shared.has_expand[gi] = True
            shared.marginals[gi] = margs_cpu[j]
            shared.top_indices[gi] = top_idx_cpu[j]
            shared.top_values[gi] = top_vals_cpu[j]

    del pair_logits


@torch.no_grad()
def _gpu_tree_forward(models, device, shared):
    """Run GPU forward for tree creation on flagged slots.

    Args:
        models: single model or list/tuple of models.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    indices = shared.tree_needs_init.nonzero(as_tuple=True)[0]
    B = len(indices)
    if B == 0:
        return

    if len(models) == 1:
        # Fast path: single model
        batch = shared.tree_planes[indices].to(device, non_blocking=True)
        values, pair_logits, _, _ = models[0](batch)
    else:
        # Multi-model: split by model_id
        slot_model_ids = shared.model_id[indices]
        values = torch.empty(B, dtype=torch.float32, device=device)
        pair_logits = torch.empty(B, N_CELLS, N_CELLS, device=device)
        for mid, m in enumerate(models):
            mask = (slot_model_ids == mid)
            if not mask.any():
                continue
            sub_idx = mask.nonzero(as_tuple=True)[0]
            sub_batch = shared.tree_planes[indices[sub_idx]].to(
                device, non_blocking=True)
            v, pl, _, _ = m(sub_batch)
            v_flat = v.squeeze(-1) if v.dim() > 1 else v
            values[sub_idx] = v_flat.float()
            pair_logits[sub_idx] = pl.float()

    # Compute pair_probs and marginals in bulk on GPU
    flat = pair_logits.reshape(B, -1)
    pair_probs = F.softmax(flat, dim=-1).reshape(B, N_CELLS, N_CELLS)
    marginals = pair_probs.sum(dim=-1)

    # Write to shared memory
    shared.tree_values[indices] = values.cpu().float()
    shared.tree_marginals[indices] = marginals.cpu().float()
    shared.tree_pair_probs[indices] = pair_probs.cpu().half()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_parallel(model, device, batch_size, n_sims, round_id, data_dir,
                      n_workers=8, late_temperature=0.3, draw_penalty=0.1,
                      model_kwargs=None, viewer=None):
    """Generate self-play games using multiprocessing.

    Returns (all_examples, draw_rate).
    """
    model_dtype = next(model.parameters()).dtype

    # Load or create game data
    pending_path = os.path.join(data_dir, "pending.json")
    game_dicts = [None] * batch_size
    next_game_id = 0
    is_cold_start = True

    if os.path.exists(pending_path):
        with open(pending_path, 'r') as f:
            pd = json.load(f)
        for i, item in enumerate(pd["games"][:batch_size]):
            game_dicts[i] = item
        next_game_id = pd["next_game_id"]
        is_cold_start = False
        print(f"Resumed {len(pd['games'])} in-progress games")
    else:
        next_game_id = batch_size

    target = COLD_START_GAMES if is_cold_start else COMPLETED_PER_ROUND
    if is_cold_start:
        print(f"Cold start: targeting {target} games")

    # Create shared buffers
    shared = SharedBuffers(batch_size, model_dtype)
    sync = _create_sync(n_workers)

    # Result collection
    result_queue = mp.Queue()

    # Spawn workers
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # already set
    workers = []
    for wid in range(n_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(wid, n_workers, batch_size, n_sims,
                  game_dicts, next_game_id,
                  shared, sync, round_id,
                  late_temperature, draw_penalty,
                  result_queue),
        )
        p.start()
        workers.append(p)

    # Main loop: orchestrate GPU + collect results
    all_examples = []
    games_completed = 0
    wins_a = wins_b = draws = 0
    total_turns = 0
    far_stones = 0
    total_stones = 0

    pbar = tqdm(total=target, desc="Games", unit="game", position=0)
    pos_bar = tqdm(desc="Positions", unit="pos", position=1)

    # Timing accumulators
    t_wait = 0.0   # main waiting for workers (CPU bottleneck indicator)
    t_gpu = 0.0    # GPU forward passes
    t_tree_gpu = 0.0  # GPU tree creation forward passes
    t_collect = 0.0  # collecting results from workers
    n_turns = 0

    try:
        while games_completed < target:
            # === Tree creation phase: GPU forward for new trees ===
            _t0 = time.monotonic()
            sync['tree_request_ready'].wait()
            sync['tree_request_ready'].clear()
            t_wait += time.monotonic() - _t0

            if sync['error'].value:
                _drain_errors(result_queue)
                raise RuntimeError("Worker process failed")

            _t0 = time.monotonic()
            _gpu_tree_forward(model, device, shared)
            t_tree_gpu += time.monotonic() - _t0

            sync['tree_results_ready'].set()

            # === Sim loop: GPU forward passes ===
            for sim in range(n_sims):
                buf_idx = sim % 2

                # Wait for workers to finish writing deltas
                _t0 = time.monotonic()
                sync['deltas_ready'][buf_idx].wait()
                sync['deltas_ready'][buf_idx].clear()
                t_wait += time.monotonic() - _t0

                # Check for worker errors
                if sync['error'].value:
                    _drain_errors(result_queue)
                    raise RuntimeError("Worker process failed")

                # GPU forward
                _t0 = time.monotonic()
                _gpu_forward(model, device,
                             shared.delta[buf_idx],
                             shared.needs_eval[buf_idx],
                             shared.needs_expand_flag[buf_idx],
                             shared)
                t_gpu += time.monotonic() - _t0

                # Signal workers: results ready
                sync['results_ready'].set()

            n_turns += 1

            # Periodic timing
            if n_turns % 5 == 0:
                t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
                if t_tot > 0:
                    pbar.write(
                        f"  [turn {n_turns}] "
                        f"wait_cpu {t_wait/t_tot*100:.0f}% "
                        f"gpu {t_gpu/t_tot*100:.0f}% "
                        f"tree_gpu {t_tree_gpu/t_tot*100:.0f}% "
                        f"collect {t_collect/t_tot*100:.0f}% "
                        f"| {n_turns/t_tot:.1f} turns/s"
                    )

            # === Collect turn results from workers ===
            _t0 = time.monotonic()
            pos_bar.update(batch_size)  # one position per slot per turn
            pending_slots = []
            for _ in range(n_workers):
                msg = result_queue.get(timeout=120)
                if msg[0] == 'error':
                    raise RuntimeError(f"Worker {msg[1]} failed:\n{msg[2]}")

                _, wid, completed, slot_data, w_far, w_total = msg
                far_stones += w_far
                total_stones += w_total
                for c in completed:
                    all_examples.extend(c['examples'])
                    w = c['winner']
                    if w == Player.A:
                        wins_a += 1
                    elif w == Player.B:
                        wins_b += 1
                    else:
                        draws += 1
                    games_completed += 1
                    total_turns += c['turn_number']
                    pbar.update(1)
                    n = wins_a + wins_b + draws
                    pbar.set_postfix(
                        A=wins_a, B=wins_b, draw=draws,
                        avg_turns=f"{total_turns/n:.0f}")

                    if viewer:
                        viewer.add_finished_data(
                            gid=c['game_id'],
                            winner=w.value,
                            moves=c['move_count'],
                            turns=c['turn_number'],
                            board=c['final_board'],
                            history=[{"b": ex["board"],
                                      "p": ex["current_player"]}
                                     for ex in c['examples']],
                        )

                pending_slots.extend(
                    s for s in slot_data if s is not None)

            t_collect += time.monotonic() - _t0

            if viewer:
                viewer.update_slots([], games_completed, target, round_id)

            if games_completed >= target:
                break

    except Exception:
        raise
    else:
        # Print final timing
        t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
        if t_tot > 0 and n_turns > 0:
            print(f"\n  Timing ({n_turns} turns, {t_tot:.1f}s):")
            for label, t in [("wait_cpu", t_wait), ("gpu_fwd", t_gpu),
                             ("tree_gpu", t_tree_gpu), ("collect", t_collect)]:
                print(f"    {label:>10s}: {t:6.1f}s ({100*t/t_tot:4.1f}%) "
                      f" {1000*t/n_turns:6.1f}ms/turn")
            print(f"    {'total':>10s}: {t_tot:6.1f}s  {n_turns/t_tot:.1f} turns/s")
    finally:
        # Signal workers to stop and break any barrier deadlocks
        sync['stop'].value = 1
        sync['results_ready'].set()
        sync['tree_results_ready'].set()
        for ev in sync['deltas_ready']:
            ev.set()
        try:
            sync['worker_barrier'].abort()
        except Exception:
            pass
        for p in workers:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        pos_bar.close()
        pbar.close()

    # Save pending games
    _save_pending(pending_slots, next_game_id + games_completed * 2, data_dir)

    total_games = wins_a + wins_b + draws
    draw_rate = draws / max(total_games, 1)
    decisive = wins_a + wins_b
    a_win_rate = wins_a / max(decisive, 1)
    far_pct = 100 * far_stones / max(total_stones, 1)
    return all_examples, draw_rate, a_win_rate, far_pct


def _drain_errors(queue):
    """Print any error messages from workers."""
    while not queue.empty():
        try:
            msg = queue.get_nowait()
            if msg[0] == 'error':
                print(f"Worker {msg[1]} error:\n{msg[2]}")
        except Exception:
            break


def _save_pending(slots_data, next_game_id, data_dir):
    """Save pending games for next round."""
    data = {"games": slots_data, "next_game_id": next_game_id}
    path = os.path.join(data_dir, "pending.json")
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)
    print(f"Saved {len(slots_data)} in-progress games")


# ---------------------------------------------------------------------------
# Persistent worker pool
# ---------------------------------------------------------------------------

class ParallelSelfPlayPool:
    """Persistent pool of worker processes for multi-round self-play.

    Workers stay alive across rounds. Tree creation happens on GPU via
    shared memory, so workers never need the model.
    """

    def __init__(self, batch_size, n_sims, n_workers=8,
                 model_dtype=torch.float32,
                 n_sims_full=600, full_search_prob=0.25):
        self.batch_size = batch_size
        self.n_sims = n_sims
        self.n_sims_full = n_sims_full
        self.full_search_prob = full_search_prob
        self.n_workers = n_workers

        # Shared memory (allocated once, reused across rounds)
        self.shared = SharedBuffers(batch_size, model_dtype)
        self.sync = None
        self.workers = []
        self.result_queue = None
        self._alive = False

        # Round control (shared values readable by workers)
        self._round_id = mp.Value('i', -1)
        self._late_temperature = mp.Value('f', 0.3)
        self._draw_penalty = mp.Value('f', 0.1)
        self._round_stop = mp.Value('i', 0)
        self._new_round = mp.Event()
        # Barrier for workers + main to sync at round end
        self._round_end_barrier = mp.Barrier(n_workers + 1)
        self._next_game_id = mp.Value('i', 0)
        # Eval mode flags (shared with workers)
        self._eval_mode = mp.Value('i', 0)    # 1 = eval mode (no replace, no noise)
        self._eval_temperature = mp.Value('f', 0.1)
        # Playout cap randomization (main writes, workers read per turn)
        self._turn_n_sims = mp.Value('i', n_sims)
        self._is_full_search = mp.Value('i', 0)

    def start(self, game_dicts, next_game_id):
        """Spawn workers with initial game data. Call once."""
        self._next_game_id.value = next_game_id
        self.sync = _create_sync(self.n_workers)
        self.result_queue = mp.Queue()

        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        for wid in range(self.n_workers):
            p = mp.Process(
                target=_pool_worker_fn,
                args=(wid, self.n_workers, self.batch_size, self.n_sims,
                      game_dicts, next_game_id,
                      self.shared, self.sync,
                      self._round_id, self._late_temperature,
                      self._draw_penalty,
                      self._round_stop,
                      self._new_round, self._round_end_barrier,
                      self._next_game_id,
                      self._eval_mode, self._eval_temperature,
                      self._turn_n_sims, self._is_full_search,
                      self.result_queue),
            )
            p.start()
            self.workers.append(p)
        self._alive = True

    def generate_round(self, model, device, round_id, data_dir,
                       late_temperature=0.3, draw_penalty=0.1,
                       target=None, viewer=None):
        """Run one round of self-play. Returns (examples, draw_rate)."""
        if not self._alive:
            raise RuntimeError("Pool not started")

        if target is None:
            target = COMPLETED_PER_ROUND

        # Set round parameters
        self._round_id.value = round_id
        self._late_temperature.value = late_temperature
        self._draw_penalty.value = draw_penalty
        self._round_stop.value = 0

        # Reset sync state for fresh round
        self.sync['error'].value = 0

        shared = self.shared
        sync = self.sync
        result_queue = self.result_queue
        n_sims = self.n_sims
        batch_size = self.batch_size
        n_workers = self.n_workers

        # Signal workers to start
        self._new_round.set()

        # Main loop: orchestrate GPU + collect results
        all_examples = []
        games_completed = 0
        wins_a = wins_b = draws = 0
        total_moves = 0
        far_stones = 0
        total_stones = 0
        n_full_turns = 0
        n_quick_turns = 0

        pbar = tqdm(total=target, desc="Games", unit="game", position=0)
        pos_bar = tqdm(desc="Positions", unit="pos", position=1)

        t_wait = 0.0
        t_gpu = 0.0
        t_tree_gpu = 0.0
        t_collect = 0.0
        n_turns = 0
        pending_slots = []

        try:
            while games_completed < target:
                # === Tree creation phase ===
                _t0 = time.monotonic()
                sync['tree_request_ready'].wait()
                sync['tree_request_ready'].clear()
                t_wait += time.monotonic() - _t0

                if sync['error'].value:
                    _drain_errors(result_queue)
                    raise RuntimeError("Worker process failed")

                _t0 = time.monotonic()
                _gpu_tree_forward(model, device, shared)
                t_tree_gpu += time.monotonic() - _t0

                # Playout cap: decide sim count for this turn before workers proceed
                is_full = random.random() < self.full_search_prob
                turn_sims = self.n_sims_full if is_full else n_sims
                self._turn_n_sims.value = turn_sims
                self._is_full_search.value = int(is_full)
                if is_full:
                    n_full_turns += 1
                else:
                    n_quick_turns += 1

                sync['tree_results_ready'].set()

                # === Sim loop ===
                for sim in range(turn_sims):
                    buf_idx = sim % 2

                    _t0 = time.monotonic()
                    sync['deltas_ready'][buf_idx].wait()
                    sync['deltas_ready'][buf_idx].clear()
                    t_wait += time.monotonic() - _t0

                    if sync['error'].value:
                        _drain_errors(result_queue)
                        raise RuntimeError("Worker process failed")

                    _t0 = time.monotonic()
                    _gpu_forward(model, device,
                                 shared.delta[buf_idx],
                                 shared.needs_eval[buf_idx],
                                 shared.needs_expand_flag[buf_idx],
                                 shared)
                    t_gpu += time.monotonic() - _t0

                    sync['results_ready'].set()

                n_turns += 1

                if n_turns % 5 == 0:
                    t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
                    if t_tot > 0:
                        pbar.write(
                            f"  [turn {n_turns}] "
                            f"wait_cpu {t_wait/t_tot*100:.0f}% "
                            f"gpu {t_gpu/t_tot*100:.0f}% "
                            f"tree_gpu {t_tree_gpu/t_tot*100:.0f}% "
                            f"collect {t_collect/t_tot*100:.0f}% "
                            f"| {n_turns/t_tot:.1f} turns/s"
                        )

                # === Collect results ===
                _t0 = time.monotonic()
                pos_bar.update(batch_size)
                pending_slots = []
                for _ in range(n_workers):
                    # Poll with short timeout so we can check worker health
                    while True:
                        try:
                            msg = result_queue.get(timeout=10)
                            break
                        except queue.Empty:
                            # Check if any worker process died
                            dead = [
                                (i, p.exitcode)
                                for i, p in enumerate(self.workers)
                                if not p.is_alive()
                            ]
                            if dead:
                                info = ", ".join(
                                    f"worker {i} exit={c}"
                                    for i, c in dead
                                )
                                raise RuntimeError(
                                    f"Worker process(es) died: {info}. "
                                    f"Likely a segfault in Cython code "
                                    f"or out-of-memory kill."
                                )
                    if msg[0] == 'error':
                        raise RuntimeError(
                            f"Worker {msg[1]} failed:\n{msg[2]}")

                    _, wid, completed, slot_data, w_far, w_total = msg
                    far_stones += w_far
                    total_stones += w_total
                    for c in completed:
                        all_examples.extend(c['examples'])
                        w = c['winner']
                        if w == Player.A:
                            wins_a += 1
                        elif w == Player.B:
                            wins_b += 1
                        else:
                            draws += 1
                        games_completed += 1
                        total_moves += c['move_count']
                        pbar.update(1)
                        n = wins_a + wins_b + draws
                        pbar.set_postfix(
                            A=wins_a, B=wins_b, draw=draws,
                            avg_moves=f"{total_moves/n:.0f}")

                        if viewer:
                            viewer.add_finished_data(
                                gid=c['game_id'],
                                winner=w.value,
                                moves=c['move_count'],
                                turns=c['turn_number'],
                                board=c['final_board'],
                                history=[{"b": ex["board"],
                                          "p": ex["current_player"]}
                                         for ex in c['examples']],
                            )

                    pending_slots.extend(
                        s for s in slot_data if s is not None)

                t_collect += time.monotonic() - _t0

                if viewer:
                    viewer.update_slots([], games_completed, target, round_id)

                if games_completed >= target:
                    break

        except Exception:
            raise
        else:
            t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
            if t_tot > 0 and n_turns > 0:
                print(f"\n  Timing ({n_turns} turns, {t_tot:.1f}s):")
                for label, t in [("wait_cpu", t_wait), ("gpu_fwd", t_gpu),
                                 ("tree_gpu", t_tree_gpu),
                                 ("collect", t_collect)]:
                    print(f"    {label:>10s}: {t:6.1f}s ({100*t/t_tot:4.1f}%) "
                          f" {1000*t/n_turns:6.1f}ms/turn")
                print(f"    {'total':>10s}: {t_tot:6.1f}s  "
                      f"{n_turns/t_tot:.1f} turns/s")
        finally:
            # Signal workers to stop the current round (not permanently)
            self._round_stop.value = 1
            self._new_round.clear()
            # Unblock workers waiting on any event
            sync['results_ready'].set()
            sync['tree_results_ready'].set()
            for ev in sync['deltas_ready']:
                ev.set()
            # Abort worker barrier to unblock workers stuck in wb.wait()
            try:
                sync['worker_barrier'].abort()
            except Exception:
                pass
            # Wait for all workers to reach round-end barrier
            try:
                self._round_end_barrier.wait(timeout=10)
            except Exception:
                pass
            # Reset barrier for next round
            try:
                sync['worker_barrier'].reset()
            except Exception:
                pass
            # Reset sync state for next round
            sync['results_ready'].clear()
            sync['tree_results_ready'].clear()
            for ev in sync['deltas_ready']:
                ev.clear()
            sync['tree_request_ready'].clear()
            pos_bar.close()
            pbar.close()

        # Save pending games for crash recovery
        _save_pending(pending_slots,
                      self._next_game_id.value, data_dir)

        total_games = wins_a + wins_b + draws
        draw_rate = draws / max(total_games, 1)
        decisive = wins_a + wins_b
        a_win_rate = wins_a / max(decisive, 1)
        avg_moves = total_moves / max(total_games, 1)
        far_pct = 100 * far_stones / max(total_stones, 1)
        full_search_pct = n_full_turns / max(n_full_turns + n_quick_turns, 1)
        print(f"  Full-search turns: {n_full_turns}/{n_full_turns + n_quick_turns} "
              f"({100 * full_search_pct:.1f}%)")
        return all_examples, draw_rate, a_win_rate, avg_moves, far_pct, \
            full_search_pct

    def evaluate(self, models, device, n_games=256, n_sims=200,
                 temperature=0.1):
        """Play model-vs-model evaluation using the parallel infrastructure.

        Args:
            models: tuple of (current_model, anchor_model).
            device: torch device.
            n_games: total games (split 50/50 by side).
            n_sims: MCTS simulations per turn.
            temperature: move selection temperature.

        Returns:
            dict with wins, losses, draws, score (from current_model's POV).
        """
        if not self._alive:
            raise RuntimeError("Pool not started")

        current_model, anchor_model = models

        # Configure eval mode
        self._eval_mode.value = 1
        self._eval_temperature.value = temperature
        self._round_id.value = -1  # sentinel for eval
        self._late_temperature.value = temperature
        self._draw_penalty.value = 0.0
        self._round_stop.value = 0

        # Set model_for_player: first half current=A, second half current=B
        half = n_games // 2
        shared = self.shared
        for gi in range(n_games):
            if gi < half:
                # current model plays A, anchor plays B
                shared.model_for_player[gi, 1] = 0  # Player.A → model 0
                shared.model_for_player[gi, 2] = 1  # Player.B → model 1
            else:
                # current model plays B, anchor plays A
                shared.model_for_player[gi, 1] = 1  # Player.A → model 1
                shared.model_for_player[gi, 2] = 0  # Player.B → model 0

        # Reset sync state
        self.sync['error'].value = 0

        sync = self.sync
        result_queue = self.result_queue

        # Signal workers to start
        self._new_round.set()

        # Track results
        wins = losses = draws = 0
        games_completed = 0
        total_target = n_games

        pbar = tqdm(total=total_target, desc="Anchor eval", unit="game")

        # Map game_id → which side current_model played
        # Workers report game_id in completed results
        current_side = {}
        # Games 0..half-1: current=A; half..n_games-1: current=B
        # game_ids match global_idx at start (slot.game_id = next_game_id + gi)

        # Must use pool's n_sims to stay in sync with workers
        n_sims = self.n_sims
        model_list = [current_model, anchor_model]

        t_wait = 0.0
        t_gpu = 0.0
        t_tree_gpu = 0.0
        t_collect = 0.0
        n_turns = 0

        try:
            while games_completed < total_target:
                # === Tree creation phase ===
                _t0 = time.monotonic()
                sync['tree_request_ready'].wait()
                sync['tree_request_ready'].clear()
                t_wait += time.monotonic() - _t0

                if sync['error'].value:
                    _drain_errors(result_queue)
                    raise RuntimeError("Worker process failed")

                _t0 = time.monotonic()
                _gpu_tree_forward(model_list, device, shared)
                t_tree_gpu += time.monotonic() - _t0

                # Eval: always use fixed sim count, no playout cap
                self._turn_n_sims.value = n_sims
                self._is_full_search.value = 0

                sync['tree_results_ready'].set()

                # === Sim loop ===
                for sim in range(n_sims):
                    buf_idx = sim % 2

                    _t0 = time.monotonic()
                    sync['deltas_ready'][buf_idx].wait()
                    sync['deltas_ready'][buf_idx].clear()
                    t_wait += time.monotonic() - _t0

                    if sync['error'].value:
                        _drain_errors(result_queue)
                        raise RuntimeError("Worker process failed")

                    _t0 = time.monotonic()
                    _gpu_forward(model_list, device,
                                 shared.delta[buf_idx],
                                 shared.needs_eval[buf_idx],
                                 shared.needs_expand_flag[buf_idx],
                                 shared)
                    t_gpu += time.monotonic() - _t0

                    sync['results_ready'].set()

                n_turns += 1
                if n_turns % 10 == 0:
                    t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
                    if t_tot > 0:
                        pbar.write(
                            f"  [eval turn {n_turns}] "
                            f"wait_cpu {t_wait/t_tot*100:.0f}% "
                            f"gpu {t_gpu/t_tot*100:.0f}% "
                            f"tree_gpu {t_tree_gpu/t_tot*100:.0f}% "
                            f"collect {t_collect/t_tot*100:.0f}% "
                            f"| {n_turns/t_tot:.1f} turns/s"
                        )

                # === Collect results ===
                _t0 = time.monotonic()
                for _ in range(self.n_workers):
                    while True:
                        try:
                            msg = result_queue.get(timeout=10)
                            break
                        except queue.Empty:
                            dead = [
                                (i, p.exitcode)
                                for i, p in enumerate(self.workers)
                                if not p.is_alive()
                            ]
                            if dead:
                                info = ", ".join(
                                    f"worker {i} exit={c}" for i, c in dead)
                                raise RuntimeError(
                                    f"Worker process(es) died: {info}")
                    if msg[0] == 'error':
                        raise RuntimeError(
                            f"Worker {msg[1]} failed:\n{msg[2]}")

                    _, wid, completed, slot_data, _, _ = msg
                    for c in completed:
                        gid = c['game_id']
                        w = c['winner']
                        # Determine if current_model won
                        # game_id was set at init: gi = game_id (for fresh
                        # games). gi < half means current=A.
                        gi = gid  # game_id == global_idx for eval
                        is_current_a = (gi < half)
                        if w == Player.NONE:
                            draws += 1
                        elif (w == Player.A) == is_current_a:
                            wins += 1
                        else:
                            losses += 1
                        games_completed += 1
                        pbar.update(1)

                t_collect += time.monotonic() - _t0

                if games_completed >= total_target:
                    break

        finally:
            # Signal workers to stop the eval round
            self._round_stop.value = 1
            self._new_round.clear()
            sync['results_ready'].set()
            sync['tree_results_ready'].set()
            for ev in sync['deltas_ready']:
                ev.set()
            try:
                sync['worker_barrier'].abort()
            except Exception:
                pass
            try:
                self._round_end_barrier.wait(timeout=10)
            except Exception:
                pass
            try:
                sync['worker_barrier'].reset()
            except Exception:
                pass
            sync['results_ready'].clear()
            sync['tree_results_ready'].clear()
            for ev in sync['deltas_ready']:
                ev.clear()
            sync['tree_request_ready'].clear()
            pbar.close()

            # Restore self-play mode and re-init games for next round
            self._eval_mode.value = 0
            # Reset model_for_player to all-zero (single model)
            shared.model_for_player.fill_(0)

        # Print timing summary
        t_tot = t_wait + t_gpu + t_tree_gpu + t_collect
        if t_tot > 0 and n_turns > 0:
            print(f"\n  Eval timing ({n_turns} turns, {t_tot:.1f}s):")
            for label, t in [("wait_cpu", t_wait), ("gpu_fwd", t_gpu),
                             ("tree_gpu", t_tree_gpu),
                             ("collect", t_collect)]:
                print(f"    {label:>10s}: {t:6.1f}s ({100*t/t_tot:4.1f}%) "
                      f" {1000*t/n_turns:6.1f}ms/turn")
            print(f"    {'total':>10s}: {t_tot:6.1f}s  "
                  f"{n_turns/t_tot:.1f} turns/s")

        total = max(wins + losses + draws, 1)
        score = (wins + 0.5 * draws) / total
        print(f"  vs Anchor: {wins}W / {losses}L / {draws}D "
              f"= {100 * score:.1f}% score")
        return {"wins": wins, "losses": losses, "draws": draws,
                "score": score}

    def shutdown(self):
        """Stop all workers permanently and clean up."""
        if not self._alive:
            return
        self.sync['stop'].value = 1
        self._round_stop.value = 1
        self._new_round.set()
        self.sync['results_ready'].set()
        self.sync['tree_results_ready'].set()
        for ev in self.sync['deltas_ready']:
            ev.set()
        try:
            self.sync['worker_barrier'].abort()
        except Exception:
            pass
        try:
            self._round_end_barrier.abort()
        except Exception:
            pass
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.workers.clear()
        self._alive = False


def _pool_worker_fn(worker_id, n_workers, batch_size, n_sims,
                    game_dicts, next_game_id_start,
                    shared_bufs, sync,
                    round_id_val, late_temp_val, draw_penalty_val,
                    round_stop,
                    new_round_event, round_end_barrier,
                    next_game_id_shared,
                    eval_mode_val, eval_temp_val,
                    turn_n_sims_val, is_full_search_val,
                    result_queue):
    """Persistent worker: stays alive across rounds."""
    try:
        _pool_worker_loop(
            worker_id, n_workers, batch_size, n_sims,
            game_dicts, next_game_id_start,
            shared_bufs, sync,
            round_id_val, late_temp_val, draw_penalty_val,
            round_stop,
            new_round_event, round_end_barrier,
            next_game_id_shared,
            eval_mode_val, eval_temp_val,
            turn_n_sims_val, is_full_search_val,
            result_queue)
    except Exception:
        sync['error'].value = 1
        result_queue.put(('error', worker_id, traceback.format_exc()))
        try:
            sync['worker_barrier'].abort()
        except Exception:
            pass
        try:
            round_end_barrier.abort()
        except Exception:
            pass


def _pool_worker_loop(worker_id, n_workers, batch_size, n_sims,
                      game_dicts, next_game_id_start,
                      shared_bufs, sync,
                      round_id_val, late_temp_val, draw_penalty_val,
                      round_stop,
                      new_round_event, round_end_barrier,
                      next_game_id_shared,
                      eval_mode_val, eval_temp_val,
                      turn_n_sims_val, is_full_search_val,
                      result_queue):
    """Inner loop for persistent pool worker."""
    games_per_worker = batch_size // n_workers
    my_start = worker_id * games_per_worker
    my_end = my_start + games_per_worker
    if worker_id == n_workers - 1:
        my_end = batch_size
    n_mine = my_end - my_start

    # One-time game restoration
    slots = []
    for i in range(n_mine):
        global_idx = my_start + i
        if global_idx < len(game_dicts) and game_dicts[global_idx] is not None:
            gd = game_dicts[global_idx]
            if _HAS_CY:
                game = CyGameState.from_dict(gd['game'])
            else:
                from game import ToroidalHexGame
                game = ToroidalHexGame.from_dict(gd['game'])
            slot = SelfPlaySlot(
                game=game, game_id=gd['game_id'],
                turn_number=gd['turn_number'], examples=gd['examples'])
        else:
            game = _new_game()
            slot = SelfPlaySlot(game=game,
                                game_id=next_game_id_start + global_idx)
        slots.append(slot)

    _sel = select_leaf_cy if _HAS_CY else select_leaf
    _bp = backprop_cy if _HAS_CY else expand_and_backprop

    wb = sync['worker_barrier']
    deltas_ready = sync['deltas_ready']
    results_ready = sync['results_ready']
    tree_request_ready = sync['tree_request_ready']
    tree_results_ready = sync['tree_results_ready']
    stop_flag = sync['stop']

    next_gid = next_game_id_start + batch_size + worker_id

    # ---- Outer loop: one iteration per round ----
    while not stop_flag.value:
        new_round_event.wait()
        if stop_flag.value:
            break

        round_id = round_id_val.value
        late_temperature = late_temp_val.value
        draw_penalty = draw_penalty_val.value
        is_eval = bool(eval_mode_val.value)
        eval_temperature = eval_temp_val.value

        # In eval mode, save self-play state and create fresh games
        saved_slots = None
        if is_eval:
            saved_slots = [(s.game, s.game_id, s.examples, s.turn_number,
                            s.tree) for s in slots]
            for i, slot in enumerate(slots):
                gi = my_start + i
                slot.game = _new_game()
                slot.game_id = gi  # use global index as game_id
                slot.examples = []
                slot.turn_number = 0
                slot.tree = None

        # ---- Inner turn loop ----
        while not stop_flag.value and not round_stop.value:
            # --- Tree creation via GPU shared-memory protocol ---
            shared_bufs.tree_needs_init[my_start:my_end] = False
            for i, slot in enumerate(slots):
                gi = my_start + i
                _write_model_id(shared_bufs, slot, gi)
                if slot.tree is None and not slot.game.game_over \
                   and slot.game.move_count < MAX_GAME_MOVES:
                    if hasattr(slot.game, 'to_planes_tensor'):
                        shared_bufs.tree_planes[gi] = \
                            slot.game.to_planes_tensor()
                    else:
                        from model.resnet import board_to_planes_torus
                        shared_bufs.tree_planes[gi] = \
                            board_to_planes_torus(
                                slot.game.board, slot.game.current_player)
                    shared_bufs.tree_needs_init[gi] = True

            try:
                wb.wait()
            except BrokenBarrierError:
                break
            if worker_id == 0:
                tree_request_ready.set()

            tree_results_ready.wait()

            if stop_flag.value or round_stop.value:
                break

            # Read per-turn sim count set by main before tree_results_ready
            turn_n_sims = turn_n_sims_val.value
            is_full = bool(is_full_search_val.value)

            for i, slot in enumerate(slots):
                gi = my_start + i
                if shared_bufs.tree_needs_init[gi]:
                    slot.tree = _build_tree_from_eval(
                        slot.game,
                        shared_bufs.tree_values[gi].item(),
                        shared_bufs.tree_pair_probs[gi].float(),
                        shared_bufs.tree_marginals[gi].clone(),
                        shared_bufs.tree_planes[gi].clone(),
                        add_noise=(not is_eval),
                    )

            try:
                wb.wait()
            except BrokenBarrierError:
                break
            if worker_id == 0:
                tree_request_ready.clear()
                tree_results_ready.clear()

            # --- Sim loop with one-sim-delayed pipeline ---
            try:
                leaves = [None] * n_mine
                for i, slot in enumerate(slots):
                    gi = my_start + i
                    if slot.tree is None:
                        shared_bufs.needs_eval[0][gi] = False
                        continue
                    leaves[i] = _sel(slot.tree, slot.game)
                    _write_delta(shared_bufs, leaves[i], slot.tree, gi, 0)

                wb.wait()
                if worker_id == 0:
                    deltas_ready[0].set()

                for sim in range(turn_n_sims):
                    buf_cur = sim % 2
                    buf_next = (sim + 1) % 2

                    next_leaves = [None] * n_mine
                    if sim < turn_n_sims - 1:
                        for i, slot in enumerate(slots):
                            gi = my_start + i
                            if slot.tree is None:
                                shared_bufs.needs_eval[buf_next][gi] = False
                                continue
                            next_leaves[i] = _sel(slot.tree, slot.game)
                            _write_delta(shared_bufs, next_leaves[i],
                                         slot.tree, gi, buf_next)

                    results_ready.wait()

                    for i, slot in enumerate(slots):
                        gi = my_start + i
                        if leaves[i] is None:
                            continue
                        if not shared_bufs.needs_eval[buf_cur][gi]:
                            _bp(slot.tree, leaves[i], 0.0)
                        else:
                            nn_val = shared_bufs.values[gi].item()
                            _bp(slot.tree, leaves[i], nn_val)
                            if shared_bufs.has_expand[gi]:
                                maybe_expand_leaf(
                                    slot.tree, leaves[i],
                                    shared_bufs.marginals[gi],
                                    shared_bufs.top_indices[gi],
                                    shared_bufs.top_values[gi],
                                    nn_value=nn_val)

                    leaves = next_leaves

                    wb.wait()
                    if worker_id == 0:
                        results_ready.clear()
                        if sim < turn_n_sims - 1:
                            deltas_ready[buf_next].set()
            except BrokenBarrierError:
                break

            # --- Move selection + example recording ---
            completed = []
            turn_far_total = 0
            turn_stones_total = 0
            for i, slot in enumerate(slots):
                if slot.tree is None:
                    continue  # game already finished (eval mode)
                turn = slot.turn_number
                if is_eval:
                    temp = eval_temperature
                else:
                    temp = 1.0 if turn < 20 else late_temperature

                if slot.game.moves_left_in_turn == 1:
                    cell = select_single_move(slot.tree)
                    moves = [cell]
                    pv = get_single_visits(slot.tree)
                else:
                    s1, s2 = select_move_pair(slot.tree, temperature=temp)
                    moves = [s1, s2]
                    pv = get_pair_visits(slot.tree)

                # Record training example (skip in eval mode)
                if not is_eval:
                    if hasattr(slot.game, 'to_board_dict'):
                        bd = slot.game.to_board_dict()
                    else:
                        bd = slot.game.board
                    cp = slot.game.current_player
                    cp_val = cp.value if hasattr(cp, 'value') else int(cp)
                    ex = {
                        "board": json.dumps({
                            f"{q},{r}": v.value if isinstance(v, Player)
                            else int(v)
                            for (q, r), v in bd.items()
                        }),
                        "current_player": cp_val,
                        "pair_visits": json.dumps({
                            f"{a},{b}": c for (a, b), c in pv.items()
                        }),
                        "value_target": 0.0,
                        "move_count": slot.game.move_count,
                        "moves_left": 0,
                        "game_drawn": False,
                        "full_search": is_full,
                        "game_id": slot.game_id,
                        "round_id": round_id,
                    }
                    slot.examples.append(ex)

                # Apply moves (track distance from existing stones)
                for q, r in moves:
                    if slot.game.game_over:
                        break
                    if not is_eval:
                        if hasattr(slot.game, 'get_occupied_set'):
                            occ = slot.game.get_occupied_set()
                        else:
                            occ = frozenset(slot.game.board.keys())
                        if occ:
                            min_d = min(_hex_dist_torus(q, r, oq, or_)
                                        for oq, or_ in occ)
                            turn_stones_total += 1
                            if min_d > 2:
                                turn_far_total += 1
                    slot.game.make_move(q, r)

                slot.turn_number += 1
                slot.tree = None
                slot.sims_done = 0

                if slot.game.game_over or \
                   slot.game.move_count >= MAX_GAME_MOVES:
                    raw_w = slot.game.winner
                    w_val = (raw_w.value if hasattr(raw_w, 'value')
                             else int(raw_w))
                    if slot.game.game_over and w_val != 0:
                        winner = Player(w_val)
                    else:
                        winner = Player.NONE

                    if not is_eval:
                        total_m = slot.game.move_count
                        is_drawn = (winner == Player.NONE)
                        for e in slot.examples:
                            e["round_id"] = round_id
                            e["moves_left"] = total_m - e["move_count"]
                            e["game_drawn"] = is_drawn
                            cp_e = Player(e["current_player"])
                            if is_drawn:
                                e["value_target"] = -draw_penalty
                            elif cp_e == winner:
                                e["value_target"] = 1.0
                            else:
                                e["value_target"] = -1.0

                    if hasattr(slot.game, 'to_board_dict'):
                        bd_src = slot.game.to_board_dict()
                    else:
                        bd_src = slot.game.board
                    final_board = {
                        f"{q},{r}": v.value if isinstance(v, Player)
                        else int(v)
                        for (q, r), v in bd_src.items()
                    }

                    completed.append({
                        'examples': slot.examples,
                        'winner': winner,
                        'move_count': slot.game.move_count,
                        'game_id': slot.game_id,
                        'turn_number': slot.turn_number,
                        'final_board': final_board,
                    })

                    if is_eval:
                        # Don't replace — leave slot dead
                        pass
                    else:
                        slot.game = _new_game()
                        slot.game_id = next_gid
                        next_gid += n_workers
                        slot.examples = []
                        slot.turn_number = 0

            result_queue.put(('turn_done', worker_id, completed,
                              [_serialize_slot(s) for s in slots],
                              turn_far_total, turn_stones_total))

        # In eval mode, restore self-play game state
        if is_eval and saved_slots is not None:
            for i, (game, gid, exs, tn, tree) in enumerate(saved_slots):
                slots[i].game = game
                slots[i].game_id = gid
                slots[i].examples = exs
                slots[i].turn_number = tn
                slots[i].tree = tree
            saved_slots = None

        # Round done -- update shared next_game_id and wait at barrier
        if not is_eval:
            next_game_id_shared.value = max(
                next_game_id_shared.value, next_gid)
        try:
            round_end_barrier.wait(timeout=10)
        except (BrokenBarrierError, Exception):
            if stop_flag.value:
                return
