"""Play actual selfplay games and track when far moves occur.

Usage:
  python diagnostic_games.py --checkpoint training/distill/resnet_results/checkpoint.pt --n-games 10
"""

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from game import ToroidalHexGame, Player, TORUS_SIZE
from model.resnet import HexResNet, board_to_planes_torus, BOARD_SIZE
from mcts.tree import (
    create_tree, select_leaf, expand_and_backprop, maybe_expand_leaf,
    get_pair_visits, get_single_visits, select_move_pair, select_single_move,
    _idx_to_cell, _cell_to_idx, N_CELLS,
)

try:
    from mcts._mcts_cy import CyGameState, select_leaf_cy, backprop_cy
    _HAS_CY = True
except ImportError:
    _HAS_CY = False

_sel = select_leaf_cy if _HAS_CY else select_leaf
_bp = backprop_cy if _HAS_CY else expand_and_backprop

CENTER = TORUS_SIZE // 2


def hex_dist(q1, r1, q2, r2):
    N = BOARD_SIZE
    dq = min(abs(q1 - q2), N - abs(q1 - q2))
    dr = min(abs(r1 - r2), N - abs(r1 - r2))
    s1, s2 = -q1 - r1, -q2 - r2
    ds = min(abs(s1 - s2) % N, N - abs(s1 - s2) % N)
    return max(dq, dr, ds)


def min_dist_to_stones(q, r, occupied):
    if not occupied:
        return 0
    return min(hex_dist(q, r, oq, or_) for oq, or_ in occupied)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    block_ids = [int(k.split('.')[1]) for k in state if k.startswith("blocks.")]
    num_blocks = max(block_ids) + 1 if block_ids else 10
    num_filters = state["stem_conv.weight"].shape[0]
    model = HexResNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_sims(tree, game, model, device, n_sims):
    """Run n_sims MCTS simulations on the tree."""
    for _ in range(n_sims):
        leaf = _sel(tree, game)

        if leaf.is_terminal:
            _bp(tree, leaf, 0.0)
            continue

        if leaf.needs_expansion and leaf.deltas:
            # Build board at leaf
            tmp_board = dict(game.board)
            cp = game.current_player
            for q, r, ch in leaf.deltas:
                p = cp if ch == 0 else (Player.B if cp == Player.A else Player.A)
                tmp_board[(q, r)] = p
            next_cp = leaf.current_player
            planes = board_to_planes_torus(tmp_board, next_cp)
            x = planes.unsqueeze(0).to(device)
            with torch.no_grad():
                v, pl, _, _ = model(x)
            nn_value = v[0].item()

            flat_logits = pl[0].reshape(-1)
            top_raw, top_idxs = flat_logits.topk(200)
            top_vals = F.softmax(top_raw, dim=-1).cpu()
            marginal_logits = pl[0].logsumexp(dim=-1)
            marg = F.softmax(marginal_logits.reshape(-1), dim=-1).cpu()

            _bp(tree, leaf, nn_value)
            maybe_expand_leaf(tree, leaf, marg, top_idxs.cpu(), top_vals)
        else:
            _bp(tree, leaf, 0.0)


def play_game(model, device, n_sims=200, game_id=0, verbose=True):
    """Play a full game, return move log."""
    game = ToroidalHexGame()
    game.make_move(CENTER, CENTER)  # Player A's first (single) move

    move_log = []
    turn = 0
    max_turns = 75  # 150 individual stones / 2

    while not game.game_over and turn < max_turns:
        tree = create_tree(game, model, device, add_noise=True)
        run_sims(tree, game, model, device, n_sims)

        occupied = set(game.board.keys())
        turn_info = {"turn": turn, "move_count": game.move_count,
                     "player": game.current_player.name}

        if game.moves_left_in_turn == 1:
            cell = select_single_move(tree)
            moves = [cell]
        else:
            s1, s2 = select_move_pair(tree, temperature=1.0 if turn < 20 else 0.3)
            moves = [s1, s2]

        # Measure distances
        for i, (q, r) in enumerate(moves):
            d = min_dist_to_stones(q, r, occupied)
            stone_label = f"s{i+1}"
            turn_info[f"{stone_label}"] = (q, r)
            turn_info[f"{stone_label}_dist"] = d
            # After placing s1, update occupied for s2 distance calc
            occupied.add((q, r))

        # Get visit info for this turn
        if len(moves) == 2:
            pair_visits = get_pair_visits(tree)
            total_v = sum(pair_visits.values())
            n_pairs = len(pair_visits)
            top_visit = max(pair_visits.values()) if pair_visits else 0
            turn_info["n_pairs_visited"] = n_pairs
            turn_info["total_visits"] = total_v
            turn_info["top_pair_visits"] = top_visit
            if total_v > 0:
                turn_info["top_pair_frac"] = top_visit / total_v

        move_log.append(turn_info)

        # Check for far moves
        is_far = any(turn_info.get(f"s{i}_dist", 0) > 2 for i in [1, 2])
        if verbose and is_far:
            print(f"  Game {game_id} turn {turn} (move {game.move_count}): "
                  f"FAR MOVE  {turn_info}")

        # Apply moves
        for q, r in moves:
            if game.game_over:
                break
            game.make_move(q, r)

        turn += 1

    result = "draw"
    if game.winner == Player.A:
        result = "A"
    elif game.winner == Player.B:
        result = "B"

    return move_log, result, game.move_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--n-sims", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Device: {device}")
    print(f"Loaded: {args.checkpoint}")
    print(f"Playing {args.n_games} games with {args.n_sims} sims each\n")

    all_far_moves = []  # (game_id, turn, move_count, dist)
    games_with_far = 0

    for g in range(args.n_games):
        print(f"--- Game {g} ---")
        log, result, n_moves = play_game(model, device, args.n_sims, g)

        # Analyze
        far_turns = []
        for entry in log:
            for i in [1, 2]:
                d = entry.get(f"s{i}_dist", 0)
                if d > 2:
                    far_turns.append((entry["turn"], entry["move_count"], d))
                    all_far_moves.append((g, entry["turn"], entry["move_count"], d))

        has_far = len(far_turns) > 0
        if has_far:
            games_with_far += 1
            first_far = far_turns[0]
            print(f"  Result: {result} in {n_moves} moves. "
                  f"Far moves: {len(far_turns)}, "
                  f"first at turn {first_far[0]} (move {first_far[1]})")
        else:
            print(f"  Result: {result} in {n_moves} moves. No far moves.")

        # Visit concentration stats
        flat_entries = [e for e in log if "n_pairs_visited" in e]
        if flat_entries:
            avg_pairs = sum(e["n_pairs_visited"] for e in flat_entries) / len(flat_entries)
            avg_top_frac = sum(e.get("top_pair_frac", 0) for e in flat_entries) / len(flat_entries)
            print(f"  Avg pairs visited: {avg_pairs:.1f}, "
                  f"avg top-pair fraction: {avg_top_frac:.2f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.n_games} games")
    print(f"  Games with far moves (dist>2): {games_with_far}/{args.n_games} "
          f"({games_with_far/args.n_games*100:.0f}%)")
    print(f"  Total far move events: {len(all_far_moves)}")
    if all_far_moves:
        first_turns = [t for _, t, _, _ in all_far_moves]
        first_moves = [m for _, _, m, _ in all_far_moves]
        dists = [d for _, _, _, d in all_far_moves]
        print(f"  Far move turn range: {min(first_turns)}-{max(first_turns)}")
        print(f"  Far move dist range: {min(dists)}-{max(dists)}")
        # First far move per game
        first_per_game = {}
        for g, t, m, d in all_far_moves:
            if g not in first_per_game:
                first_per_game[g] = (t, m, d)
        first_turns_per_game = [t for t, m, d in first_per_game.values()]
        print(f"  First far move per game — turn: "
              f"median={sorted(first_turns_per_game)[len(first_turns_per_game)//2]}, "
              f"min={min(first_turns_per_game)}, max={max(first_turns_per_game)}")


if __name__ == "__main__":
    main()
