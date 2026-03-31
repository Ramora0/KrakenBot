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
    N_CELLS, create_trees_batched, select_leaf,
    expand_and_backprop, maybe_expand_leaf, get_pair_visits, get_single_visits,
    select_move_pair, select_single_move, _build_tree_from_eval,
)
from model.resnet import BOARD_SIZE, HexResNet

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
               model_kwargs, model_state_dict,
               game_dicts, next_game_id_start,
               shared_bufs, sync, round_id,
               late_temperature, draw_penalty,
               result_queue):
    """Worker process: owns games/trees, does select + backprop."""
    try:
        _worker_loop(worker_id, n_workers, batch_size, n_sims,
                     model_kwargs, model_state_dict,
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
                 model_kwargs, model_state_dict,
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

    # Create CPU model for tree creation
    cpu_model = HexResNet(**model_kwargs)
    cpu_model.load_state_dict(model_state_dict)
    cpu_model.eval()

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

    # ---- Outer turn loop ----
    stop_flag = sync.get('stop')  # mp.Value set by main when done

    while not (stop_flag and stop_flag.value):
        # Create trees for games that need them (CPU forward)
        needs = [i for i, s in enumerate(slots)
                 if s.tree is None and not s.game.game_over
                 and s.game.move_count < MAX_GAME_MOVES]
        if needs:
            if stop_flag and stop_flag.value:
                break
            games = [slots[i].game for i in needs]
            trees = create_trees_batched(games, cpu_model,
                                         torch.device('cpu'), add_noise=True)
            for i, tree in zip(needs, trees):
                slots[i].tree = tree

        if stop_flag and stop_flag.value:
            break

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
                                shared_bufs.top_values[gi])

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

            # Apply moves
            for q, r in moves:
                if slot.game.game_over:
                    break
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

        # Send completed games to main
        result_queue.put(('turn_done', worker_id, completed,
                          [_serialize_slot(s) for s in slots]))


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
def _gpu_forward(model, device, delta_buf, needs_eval, needs_expand_flag,
                 shared):
    """Run GPU forward on the delta buffer, write results to shared."""
    indices = needs_eval.nonzero(as_tuple=True)[0]
    B = len(indices)

    if B == 0:
        return

    batch = delta_buf[indices].to(device, non_blocking=True)
    values, pair_logits, _, _ = model(batch)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_parallel(model, device, batch_size, n_sims, round_id, data_dir,
                      n_workers=8, late_temperature=0.3, draw_penalty=0.1,
                      model_kwargs=None, viewer=None):
    """Generate self-play games using multiprocessing.

    Returns (all_examples, draw_rate).
    """
    if model_kwargs is None:
        model_kwargs = {'num_blocks': 10, 'num_filters': 128}

    model_dtype = next(model.parameters()).dtype
    model_state = {k: v.cpu().float() for k, v in model.state_dict().items()}

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
                  model_kwargs, model_state,
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
    total_moves = 0

    pbar = tqdm(total=target, desc="Games", unit="game", position=0)
    pos_bar = tqdm(desc="Positions", unit="pos", position=1)

    # Timing accumulators
    t_wait = 0.0   # main waiting for workers (CPU bottleneck indicator)
    t_gpu = 0.0    # GPU forward passes
    t_collect = 0.0  # collecting results from workers
    n_turns = 0

    try:
        while games_completed < target:
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
                t_tot = t_wait + t_gpu + t_collect
                if t_tot > 0:
                    pbar.write(
                        f"  [turn {n_turns}] "
                        f"wait_cpu {t_wait/t_tot*100:.0f}% "
                        f"gpu {t_gpu/t_tot*100:.0f}% "
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

                _, wid, completed, slot_data = msg
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
        # Print final timing
        t_tot = t_wait + t_gpu + t_collect
        if t_tot > 0 and n_turns > 0:
            print(f"\n  Timing ({n_turns} turns, {t_tot:.1f}s):")
            for label, t in [("wait_cpu", t_wait), ("gpu_fwd", t_gpu),
                             ("collect", t_collect)]:
                print(f"    {label:>10s}: {t:6.1f}s ({100*t/t_tot:4.1f}%) "
                      f" {1000*t/n_turns:6.1f}ms/turn")
            print(f"    {'total':>10s}: {t_tot:6.1f}s  {n_turns/t_tot:.1f} turns/s")
    finally:
        # Signal workers to stop and break any barrier deadlocks
        sync['stop'].value = 1
        sync['results_ready'].set()
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
    return all_examples, draw_rate


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
