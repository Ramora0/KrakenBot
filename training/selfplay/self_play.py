"""Batched self-play game generation for MCTS training.

Runs 256 games in lockstep on a toroidal board: all slots search
simultaneously, batch NN evals on GPU, then advance games together.
Each round generates exactly COMPLETED_PER_ROUND (256) completed games.
In-progress games are saved and resumed across rounds so no work is wasted.

Multi-ply MCTS: after enough visits to a pair, child PosNodes are created
from the NN's pair logits, allowing the tree to search deeper.

Output: list of training examples saved as parquet.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from game import Player
import torch.nn.functional as F

from mcts.tree import (
    MCTSTree, N_CELLS, NON_ROOT_TOP_K, create_trees_batched, select_leaf,
    expand_and_backprop, maybe_expand_leaf, get_pair_visits, get_single_visits,
    select_move_pair, select_single_move,
)
from model.resnet import BOARD_SIZE
from game import ToroidalHexGame, TORUS_SIZE

try:
    from mcts._mcts_cy import CyGameState, select_leaf_cy, backprop_cy
    _HAS_CY = True
except ImportError:
    _HAS_CY = False

MAX_GAME_MOVES = 150
COMPLETED_PER_ROUND = 256
COLD_START_GAMES = 1024

# Center of torus — first move always here
_CENTER = TORUS_SIZE // 2


@dataclass
class SelfPlaySlot:
    game: ToroidalHexGame
    tree: MCTSTree | None = None
    sims_done: int = 0
    turn_number: int = 0
    game_id: int = 0
    examples: list[dict] = field(default_factory=list)


class SelfPlayManager:
    def __init__(self, model, device, batch_size=256, n_sims=200,
                 data_dir="training/data/selfplay", viewer=None,
                 late_temperature=0.3, draw_penalty=0.1):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_sims = n_sims
        self.data_dir = data_dir
        self.viewer = viewer
        self.late_temperature = late_temperature
        self.draw_penalty = draw_penalty

    def _load_or_create_slots(self) -> tuple[list[SelfPlaySlot], int, bool]:
        """Load pending games from previous round, or create all fresh slots.

        Returns (slots, next_game_id, is_cold_start).
        """
        batch_size = self.batch_size
        pending_path = os.path.join(self.data_dir, "pending.json")

        if os.path.exists(pending_path):
            with open(pending_path, 'r') as f:
                pending_data = json.load(f)

            slots = []
            for item in pending_data["games"]:
                if _HAS_CY:
                    game = CyGameState.from_dict(item["game"])
                else:
                    game = ToroidalHexGame.from_dict(item["game"])
                slot = SelfPlaySlot(
                    game=game,
                    game_id=item["game_id"],
                    turn_number=item["turn_number"],
                    examples=item["examples"],
                )
                slots.append(slot)

            next_game_id = pending_data["next_game_id"]
            n_resumed = len(slots)

            while len(slots) < batch_size:
                slots.append(self._new_slot(next_game_id))
                next_game_id += 1

            print(f"Resumed {n_resumed} in-progress games")
            return slots, next_game_id, False
        else:
            slots = []
            next_game_id = 0
            for _ in range(batch_size):
                slots.append(self._new_slot(next_game_id))
                next_game_id += 1
            return slots, next_game_id, True

    def _save_pending(self, slots: list[SelfPlaySlot], next_game_id: int):
        """Save in-progress games for next round."""
        pending = []
        for slot in slots:
            if not slot.game.game_over and slot.game.move_count < MAX_GAME_MOVES:
                pending.append({
                    "game": slot.game.to_dict(),
                    "game_id": slot.game_id,
                    "turn_number": slot.turn_number,
                    "examples": slot.examples,
                })

        data = {"games": pending, "next_game_id": next_game_id}
        path = os.path.join(self.data_dir, "pending.json")
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
        print(f"Saved {len(pending)} in-progress games to {path}")

    def generate(self, round_id: int) -> list[dict]:
        """Generate completed games. Returns example dicts.

        On cold start (no pending games), targets COLD_START_GAMES to build
        a representative distribution of decisive and drawn games. On warm
        start, targets COMPLETED_PER_ROUND. Saves pending games at the end.
        """
        model = self.model
        device = self.device
        n_sims = self.n_sims
        batch_size = self.batch_size

        all_examples: list[dict] = []
        games_completed = 0
        wins_a = 0
        wins_b = 0
        draws = 0
        total_positions = 0
        total_moves_in_completed = 0

        slots, next_game_id, is_cold_start = self._load_or_create_slots()
        target = COLD_START_GAMES if is_cold_start else COMPLETED_PER_ROUND
        if is_cold_start:
            print(f"Cold start: targeting {target} games to build distribution")

        # Pre-allocate eval buffers for double-buffered GPU overlap
        model_dtype = next(model.parameters()).dtype
        half = batch_size // 2
        use_stream = device.type == 'cuda'
        eval_buf_A = torch.empty(half, 2, BOARD_SIZE, BOARD_SIZE,
                                 dtype=model_dtype)
        eval_buf_B = torch.empty(batch_size - half, 2, BOARD_SIZE, BOARD_SIZE,
                                 dtype=model_dtype)
        stream = torch.cuda.Stream() if use_stream else None

        pbar = tqdm(total=target, desc="Games", unit="game", position=0)
        pos_bar = tqdm(desc="Positions", unit="pos", position=1)

        # Timing accumulators (seconds)
        t_tree_create = 0.0
        t_select = 0.0
        t_collect = 0.0
        t_batch_eval = 0.0
        t_backprop = 0.0
        t_move = 0.0
        n_turns = 0
        self._t_delta = 0.0    # delta plane construction
        self._t_forward = 0.0  # transfer + model forward + transfer back

        _sel = select_leaf_cy if _HAS_CY else select_leaf
        _bp = backprop_cy if _HAS_CY else expand_and_backprop

        while games_completed < target:
            # --- Phase 1: Create trees for slots that need them ---
            needs_tree = [i for i, s in enumerate(slots) if s.tree is None]
            if needs_tree:
                _t0 = time.monotonic()
                self._batch_create_trees(slots, needs_tree, model, device)
                t_tree_create += time.monotonic() - _t0

            # --- Phase 2: Run n_sims with double-buffered GPU overlap ---
            slots_A = slots[:half]
            slots_B = slots[half:]

            for _sim in range(n_sims):
                # -- Select + prepare group A (CPU) --
                _t0 = time.monotonic()
                leaves_A = [_sel(s.tree, s.game) for s in slots_A]
                t_select += time.monotonic() - _t0

                _t0 = time.monotonic()
                eval_A = [(i, leaves_A[i]) for i in range(len(leaves_A))
                          if not leaves_A[i].is_terminal and leaves_A[i].deltas]
                batch_A_vals, batch_A_exp, raw_A = None, {}, None
                if eval_A:
                    el = [lf for _, lf in eval_A]
                    et = [slots_A[i].tree for i, _ in eval_A]
                    self._prepare_delta(el, et, eval_buf_A)
                    # Launch GPU forward async
                    raw_A = self._forward_async(
                        eval_buf_A[:len(el)], stream)
                t_batch_eval += time.monotonic() - _t0

                # -- While GPU processes A: select + prepare B (CPU) --
                _t0 = time.monotonic()
                leaves_B = [_sel(s.tree, s.game) for s in slots_B]
                t_select += time.monotonic() - _t0

                _t0 = time.monotonic()
                eval_B = [(i, leaves_B[i]) for i in range(len(leaves_B))
                          if not leaves_B[i].is_terminal and leaves_B[i].deltas]
                t_collect += time.monotonic() - _t0

                # -- Sync A, backprop A --
                _t0 = time.monotonic()
                if raw_A is not None:
                    batch_A_vals, batch_A_exp = self._collect_results(
                        raw_A, [lf for _, lf in eval_A], stream)
                t_batch_eval += time.monotonic() - _t0

                _t0 = time.monotonic()
                self._backprop_group(
                    slots_A, leaves_A, eval_A, batch_A_vals, batch_A_exp, _bp)
                t_backprop += time.monotonic() - _t0

                # -- Launch B on GPU --
                _t0 = time.monotonic()
                batch_B_vals, batch_B_exp, raw_B = None, {}, None
                if eval_B:
                    el = [lf for _, lf in eval_B]
                    et = [slots_B[i].tree for i, _ in eval_B]
                    self._prepare_delta(el, et, eval_buf_B)
                    raw_B = self._forward_async(
                        eval_buf_B[:len(el)], stream)
                t_batch_eval += time.monotonic() - _t0

                # -- Sync B, backprop B --
                _t0 = time.monotonic()
                if raw_B is not None:
                    batch_B_vals, batch_B_exp = self._collect_results(
                        raw_B, [lf for _, lf in eval_B], stream)
                t_batch_eval += time.monotonic() - _t0

                _t0 = time.monotonic()
                self._backprop_group(
                    slots_B, leaves_B, eval_B, batch_B_vals, batch_B_exp, _bp)
                t_backprop += time.monotonic() - _t0

            n_turns += 1

            # Periodic timing breakdown
            if n_turns % 10 == 0:
                t_tot = t_tree_create + t_select + t_collect + t_batch_eval + t_backprop + t_move
                if t_tot > 0:
                    pbar.write(
                        f"  [turn {n_turns}] "
                        f"tree {t_tree_create/t_tot*100:.0f}% "
                        f"select {t_select/t_tot*100:.0f}% "
                        f"delta {self._t_delta/t_tot*100:.0f}% "
                        f"fwd {self._t_forward/t_tot*100:.0f}% "
                        f"| {n_turns/t_tot:.1f} turns/s"
                    )

            # --- Phase 3: Pick moves, record examples, advance games ---
            _t0 = time.monotonic()

            for slot in slots:
                turn_number = slot.turn_number
                temperature = 1.0 if turn_number < 20 else self.late_temperature

                if slot.game.moves_left_in_turn == 1:
                    cell = select_single_move(slot.tree)
                    moves = [cell]
                    pair_visits = get_single_visits(slot.tree)
                else:
                    s1, s2 = select_move_pair(slot.tree, temperature=temperature)
                    moves = [s1, s2]
                    pair_visits = get_pair_visits(slot.tree)

                # Record training example
                if hasattr(slot.game, 'to_board_dict'):
                    board_dict = slot.game.to_board_dict()
                else:
                    board_dict = slot.game.board
                cp = slot.game.current_player
                cp_val = cp.value if hasattr(cp, 'value') else int(cp)
                example = {
                    "board": json.dumps({
                        f"{q},{r}": v.value if isinstance(v, Player) else int(v)
                        for (q, r), v in board_dict.items()
                    }),
                    "current_player": cp_val,
                    "pair_visits": json.dumps({
                        f"{a},{b}": c for (a, b), c in pair_visits.items()
                    }),
                    "value_target": 0.0,  # backfilled after game ends
                    "move_count": slot.game.move_count,
                    "moves_left": 0,      # backfilled after game ends
                    "game_drawn": False,   # backfilled after game ends
                    "game_id": slot.game_id,
                    "round_id": round_id,
                }
                slot.examples.append(example)
                total_positions += 1
                pos_bar.update(1)

                # Apply moves
                for q, r in moves:
                    if slot.game.game_over:
                        break
                    slot.game.make_move(q, r)

                slot.turn_number += 1
                slot.tree = None  # will be re-created next iteration
                slot.sims_done = 0

            t_move += time.monotonic() - _t0

            # --- Phase 4: Check for finished games, backfill values ---
            for i, slot in enumerate(slots):
                game_done = slot.game.game_over or slot.game.move_count >= MAX_GAME_MOVES

                if game_done:
                    # Determine outcome (handle both Player enum and int)
                    raw_winner = slot.game.winner
                    winner_val = raw_winner.value if hasattr(raw_winner, 'value') else int(raw_winner)
                    if slot.game.game_over and winner_val != 0:
                        winner = Player(winner_val)
                    else:
                        winner = Player.NONE  # draw

                    # Backfill value_target, moves_left, game_drawn
                    total_moves = slot.game.move_count
                    is_drawn = (winner == Player.NONE)
                    for ex in slot.examples:
                        ex["round_id"] = round_id
                        ex["moves_left"] = total_moves - ex["move_count"]
                        ex["game_drawn"] = is_drawn
                        cp = Player(ex["current_player"])
                        if is_drawn:
                            ex["value_target"] = -self.draw_penalty
                        elif cp == winner:
                            ex["value_target"] = 1.0
                        else:
                            ex["value_target"] = -1.0

                    all_examples.extend(slot.examples)
                    if self.viewer:
                        self.viewer.add_finished(slot)
                    if winner == Player.A:
                        wins_a += 1
                    elif winner == Player.B:
                        wins_b += 1
                    else:
                        draws += 1
                    games_completed += 1
                    pbar.update(1)
                    total_moves_in_completed += slot.game.move_count
                    n = wins_a + wins_b + draws
                    avg_moves = total_moves_in_completed / max(n, 1)
                    pbar.set_postfix(
                        avg_moves=f"{avg_moves:.0f}",
                        A=f"{wins_a}",
                        B=f"{wins_b}",
                        draw=f"{draws}",
                    )

                    # Always replace with a fresh game
                    slots[i] = self._new_slot(next_game_id)
                    next_game_id += 1

            # Update viewer (just 4 attribute sets — ~0 cost)
            if self.viewer:
                self.viewer.update_slots(
                    slots, games_completed, target, round_id)

            # Check if all needed games are done
            if games_completed >= target:
                break

        pos_bar.close()
        pbar.close()

        # Save in-progress games for next round
        self._save_pending(slots, next_game_id)

        # Timing breakdown
        t_total = t_tree_create + t_select + t_collect + t_batch_eval + t_backprop + t_move
        if t_total > 0 and n_turns > 0:
            print(f"\n  Timing breakdown ({n_turns} turns, {t_total:.1f}s total):")
            for label, t in [
                ("tree_create", t_tree_create),
                ("select_leaf", t_select),
                ("collect",     t_collect),
                ("batch_eval",  t_batch_eval),
                ("  delta",      self._t_delta),
                ("  forward+xfer", self._t_forward),
                ("backprop",    t_backprop),
                ("move+record", t_move),
            ]:
                pct = 100 * t / t_total
                per_turn = 1000 * t / n_turns
                print(f"    {label:>15s}: {t:7.1f}s ({pct:5.1f}%)  {per_turn:6.1f}ms/turn")

        total_games = wins_a + wins_b + draws
        draw_rate = draws / max(total_games, 1)
        decisive = wins_a + wins_b
        a_win_rate = wins_a / max(decisive, 1)
        avg_moves = total_moves_in_completed / max(total_games, 1)
        return all_examples, draw_rate, a_win_rate, avg_moves

    def _new_slot(self, game_id: int) -> SelfPlaySlot:
        """Create a new game slot on a toroidal board. First move at center."""
        if _HAS_CY:
            game = CyGameState()
        else:
            game = ToroidalHexGame()
        game.make_move(_CENTER, _CENTER)  # First move at torus center
        return SelfPlaySlot(game=game, game_id=game_id)

    def _batch_create_trees(
        self,
        slots: list[SelfPlaySlot],
        indices: list[int],
        model: torch.nn.Module,
        device: torch.device,
    ):
        """Batch-create trees with a single forward pass."""
        active = [i for i in indices
                  if not slots[i].game.game_over
                  and slots[i].game.move_count < MAX_GAME_MOVES]
        if not active:
            return
        games = [slots[i].game for i in active]
        trees = create_trees_batched(games, model, device, add_noise=True)
        for i, tree in zip(active, trees):
            slots[i].tree = tree

    def _prepare_delta(self, leaves, trees, eval_buf):
        """Fill eval buffer with root planes + deltas (CPU only)."""
        _t0 = time.monotonic()
        B = len(leaves)
        batch = eval_buf[:B]
        for i, (leaf, tree) in enumerate(zip(leaves, trees)):
            rp = tree.root_planes
            if leaf.player_flipped:
                batch[i, 0] = rp[1]
                batch[i, 1] = rp[0]
            else:
                batch[i] = rp
            for gq, gr, ch in leaf.deltas:
                actual_ch = (1 - ch) if leaf.player_flipped else ch
                batch[i, actual_ch, gq, gr] = 1.0
        self._t_delta += time.monotonic() - _t0

    @torch.no_grad()
    def _forward_async(self, batch, stream):
        """Launch model forward, optionally on a CUDA stream."""
        if stream is not None:
            with torch.cuda.stream(stream):
                batch_gpu = batch.to(self.device, non_blocking=True)
                vals, pair_logits, _, _ = self.model(batch_gpu)
        else:
            batch_gpu = batch.to(self.device)
            vals, pair_logits, _, _ = self.model(batch_gpu)
        return vals, pair_logits

    @torch.no_grad()
    def _collect_results(self, raw, leaves, stream):
        """Sync stream, extract values and expand_data."""
        _t0 = time.monotonic()
        if stream is not None:
            stream.synchronize()
        vals, pair_logits = raw
        result = vals.cpu().tolist()

        expand_data = {}
        need_expand = [i for i, lf in enumerate(leaves) if lf.needs_expansion]
        if need_expand:
            ne = len(need_expand)
            exp_logits = pair_logits[need_expand]
            flat_logits = exp_logits.reshape(ne, -1)
            top_raw, top_idxs = flat_logits.topk(200, dim=-1)
            top_vals = F.softmax(top_raw, dim=-1)
            marginal_logits = exp_logits.logsumexp(dim=-1)
            marginals = F.softmax(marginal_logits, dim=-1)
            del exp_logits, flat_logits, top_raw, marginal_logits
            marginals_cpu = marginals.cpu()
            top_idxs_cpu = top_idxs.cpu()
            top_vals_cpu = top_vals.cpu()
            del marginals, top_vals, top_idxs
            for j, i in enumerate(need_expand):
                expand_data[i] = (
                    marginals_cpu[j], top_idxs_cpu[j], top_vals_cpu[j])

        del pair_logits
        self._t_forward += time.monotonic() - _t0
        return result, expand_data

    def _backprop_group(self, group_slots, leaves, eval_list,
                        eval_values, expand_data, _bp):
        """Backprop for a group of slots."""
        eval_map = {li: j for j, (li, _) in enumerate(eval_list)}
        for i, leaf in enumerate(leaves):
            j = eval_map.get(i)
            if leaf.is_terminal:
                _bp(group_slots[i].tree, leaf, 0.0)
            elif j is not None:
                nn_val = eval_values[j]
                _bp(group_slots[i].tree, leaf, nn_val)
                data = expand_data.get(j)
                if data is not None:
                    maybe_expand_leaf(group_slots[i].tree, leaf, *data)
            else:
                _bp(group_slots[i].tree, leaf, 0.0)


    def save_round(self, examples: list[dict], round_id: int, output_dir: str):
        """Save examples as parquet + pre-built .pt cache for instant training."""
        import pandas as pd
        from training.selfplay.train_loop import compute_chain_targets

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"round_{round_id}.parquet")
        df = pd.DataFrame(examples)
        df.to_parquet(path, index=False)

        # Build .pt cache directly from in-memory data (no JSON re-parse)
        n = len(examples)
        N = BOARD_SIZE * BOARD_SIZE
        planes = torch.zeros(n, 2, BOARD_SIZE, BOARD_SIZE)
        visit_dicts = []
        values = torch.zeros(n)
        rids = torch.zeros(n, dtype=torch.int64)
        chain_t = torch.zeros(n, 6, BOARD_SIZE, BOARD_SIZE)
        chain_m = torch.zeros(n, 6, BOARD_SIZE, BOARD_SIZE)
        ml = torch.zeros(n)
        dm = torch.zeros(n, dtype=torch.bool)

        for i, ex in enumerate(examples):
            board_raw = json.loads(ex["board"])
            board_dict = {
                tuple(int(x) for x in k.split(",")): v
                for k, v in board_raw.items()
            }
            cp = int(ex["current_player"])

            # Planes
            for (q, r), player in board_dict.items():
                if player == cp:
                    planes[i, 0, q, r] = 1.0
                else:
                    planes[i, 1, q, r] = 1.0

            # Chain targets
            ct, cm = compute_chain_targets(board_dict, cp)
            chain_t[i] = ct
            chain_m[i] = cm

            # Sparse visits
            pair_visits_raw = json.loads(ex["pair_visits"])
            total_visits = sum(pair_visits_raw.values())
            entries = []
            if total_visits > 0:
                for key, count in pair_visits_raw.items():
                    parts = key.split(",")
                    a, b = int(parts[0]), int(parts[1])
                    entries.append((a * N + b, count / total_visits))
            visit_dicts.append(entries)

            values[i] = ex["value_target"]
            rids[i] = ex["round_id"]
            ml[i] = ex.get("moves_left", 0)
            dm[i] = bool(ex.get("game_drawn", False))

        from training.selfplay.train_loop import CHAIN_VERSION
        cache_path = path.replace('.parquet', '.pt')
        torch.save({
            'planes': planes, 'visit_dicts': visit_dicts,
            'values': values, 'round_ids': rids,
            'chain_targets': chain_t, 'chain_masks': chain_m,
            'moves_left': ml, 'draw_mask': dm,
            'chain_version': CHAIN_VERSION,
        }, cache_path)

        print(f"Saved {n:,} examples to {path} (+cache)")
        return path
