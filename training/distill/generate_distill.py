"""Generate ResNet distillation data from bot self-play.

Plays games from early human-game positions using the C++ bot on both sides.
Each double move generates 3 training examples (original + 2 sub-move positions).
Games where the board exceeds 19x19 bounding box are discarded.

Output: Parquet file with columns:
  board          - JSON string of board state {"q,r": player_int, ...}
  current_player - int (1=Player.A, 2=Player.B)
  moves          - list of [q,r] pairs: [[q0,r0],[q1,r1]] or [[q0,r0]]
  eval_score     - float, bot.last_score / SCORE_SCALE
  win_score      - float, +1.0 win / -1.0 loss / 0.0 draw from current_player POV
  game_id        - int, unique per generated game

Usage: python -m training.distill.generate_distill [--num-games 100000]
"""

import json
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from game import HexGame, Player, HEX_DIRECTIONS

SCORE_SCALE = 20_000
MAX_MOVES = 200
MAX_BOARD_SPAN = 19
_WIN_LENGTH = 6


def _board_bbox_ok(board):
    """Return True if board fits within 19x19 bounding box."""
    if not board:
        return True
    qs = [q for q, _r in board]
    rs = [r for _q, r in board]
    return (max(qs) - min(qs) + 1 <= MAX_BOARD_SPAN and
            max(rs) - min(rs) + 1 <= MAX_BOARD_SPAN)


def _board_has_win(board):
    """Check if any player already has 6 in a row."""
    for (q, r), player in board.items():
        for dq, dr in HEX_DIRECTIONS:
            count = 1
            for i in range(1, _WIN_LENGTH):
                if board.get((q + dq * i, r + dr * i)) == player:
                    count += 1
                else:
                    break
            if count >= _WIN_LENGTH:
                return True
    return False


def _board_to_json(board):
    """Convert board dict to JSON string with 'q,r' keys and int values."""
    return json.dumps({f"{q},{r}": p.value for (q, r), p in board.items()})


def _load_starting_positions(path, max_stones=11, seed=42):
    """Load early-game positions from human games.

    Filters to odd stone counts <= max_stones (start-of-turn positions),
    deduplicates, and removes positions with existing wins.
    """
    with open(path, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {path}")

    # Filter to early positions with odd stone counts (start of turn)
    early = [p for p in positions
             if len(p[0]) <= max_stones and len(p[0]) % 2 == 1]
    print(f"Filtered to {len(early)} early positions (odd stones <= {max_stones})")

    # Deduplicate by board state
    seen = set()
    unique = []
    for p in early:
        key = frozenset(p[0].items())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) < len(early):
        print(f"Deduplicated: {len(early)} -> {len(unique)}")
    early = unique

    # Remove positions with existing wins
    before = len(early)
    early = [p for p in early if not _board_has_win(p[0])]
    if len(early) < before:
        print(f"Removed {before - len(early)} already-won positions")

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(early)

    # Return (board, current_player, human_game_id)
    result = []
    for p in early:
        board, cp = p[0], p[1]
        human_gid = p[3] if len(p) == 4 else p[4]
        result.append((board, cp, human_gid))

    print(f"Final: {len(result)} starting positions")
    return result


def _play_one_game(args):
    """Play one game from a starting position, collecting training examples.

    Returns list of dicts (one per training example), or None if game
    should be discarded (board exceeded 19x19).
    """
    board, cp, _human_gid, time_limit, time_jitter, pattern_path, game_id, seed = args

    rng = random.Random(seed)
    tl_a = time_limit + rng.uniform(-time_jitter, time_jitter)
    tl_b = time_limit + rng.uniform(-time_jitter, time_jitter)
    tl_a = max(0.01, tl_a)
    tl_b = max(0.01, tl_b)

    game = HexGame(win_length=6)
    game.board = dict(board)
    game.current_player = cp
    game.move_count = len(board)
    game.moves_left_in_turn = 2

    from ai_cpp import MinimaxBot
    bot_a = MinimaxBot(tl_a, pattern_path)
    bot_b = MinimaxBot(tl_b, pattern_path)
    bots = {Player.A: bot_a, Player.B: bot_b}

    # Collect per-turn data: (board_snap, player, moves_played, eval_score)
    # moves_played tracks only moves actually executed (not m2 if m1 won)
    turn_records = []
    total_moves = 0
    forfeit_player = None

    while not game.game_over and total_moves < MAX_MOVES:
        player = game.current_player
        bot = bots[player]

        board_snap = dict(game.board)
        moves = list(bot.get_move(game))
        eval_score = bot.last_score / SCORE_SCALE

        moves_played = []
        for q, r in moves:
            if game.game_over:
                break  # game ended on previous move, skip rest
            if not game.make_move(q, r):
                forfeit_player = player
                break
            moves_played.append((q, r))
            total_moves += 1

        turn_records.append((board_snap, player, moves, moves_played, eval_score))

        if forfeit_player is not None:
            break

        if not _board_bbox_ok(game.board):
            return None, 0

    if forfeit_player is not None:
        winner = Player.B if forfeit_player == Player.A else Player.A
    else:
        winner = game.winner

    # Build training examples
    examples = []
    for board_snap, player, moves, moves_played, eval_score in turn_records:
        if winner == Player.NONE:
            win_score = 0.0
        elif winner == player:
            win_score = 1.0
        else:
            win_score = -1.0

        board_json = _board_to_json(board_snap)

        if len(moves) == 2 and len(moves_played) == 2:
            m1, m2 = moves[0], moves[1]

            # Example 1: full turn - both moves
            examples.append({
                "board": board_json,
                "current_player": player.value,
                "moves": [list(m1), list(m2)],
                "eval_score": eval_score,
                "win_score": win_score,
                "game_id": game_id,
            })

            # Example 2: after m1, target is m2
            board_with_m1 = dict(board_snap)
            board_with_m1[m1] = player
            if _board_bbox_ok(board_with_m1):
                examples.append({
                    "board": _board_to_json(board_with_m1),
                    "current_player": player.value,
                    "moves": [list(m2)],
                    "eval_score": eval_score,
                    "win_score": win_score,
                    "game_id": game_id,
                })

            # Example 3: after m2, target is m1
            board_with_m2 = dict(board_snap)
            board_with_m2[m2] = player
            if _board_bbox_ok(board_with_m2):
                examples.append({
                    "board": _board_to_json(board_with_m2),
                    "current_player": player.value,
                    "moves": [list(m1)],
                    "eval_score": eval_score,
                    "win_score": win_score,
                    "game_id": game_id,
                })
        else:
            # Single move (game ended on m1, or bot returned only 1 move)
            m = moves_played[0] if moves_played else moves[0]
            examples.append({
                "board": board_json,
                "current_player": player.value,
                "moves": [list(m)],
                "eval_score": eval_score,
                "win_score": win_score,
                "game_id": game_id,
            })

    return examples, total_moves


def main():
    import argparse
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Generate ResNet distillation data from bot self-play.")
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "data", "positions_human_labelled.pkl"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "data", "distill_100k.parquet"))
    parser.add_argument("--time-limit", type=float, default=0.04,
                        help="Base seconds per bot move (default: 0.04)")
    parser.add_argument("--time-jitter", type=float, default=0.015,
                        help="Random jitter +/- around time-limit per bot (default: 0.015)")
    parser.add_argument("--pattern-path", type=str, default=None,
                        help="Pattern values JSON for ai_cpp (default: built-in)")
    parser.add_argument("--max-stones", type=int, default=11,
                        help="Max stones in starting position (default: 11 = first 6 turns)")
    parser.add_argument("--num-games", type=int, default=100_000,
                        help="Total games to generate (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Checkpoint save every N games (default: 1000)")
    args = parser.parse_args()

    # Load starting positions
    starts = _load_starting_positions(args.input, args.max_stones, args.seed)
    if not starts:
        print("No starting positions found!")
        sys.exit(1)

    # Build task list
    tasks = []
    for game_id in range(args.num_games):
        start_idx = game_id % len(starts)
        board, cp, human_gid = starts[start_idx]
        tasks.append((board, cp, human_gid, args.time_limit, args.time_jitter,
                       args.pattern_path, game_id, args.seed + game_id))

    workers = os.cpu_count() or 1
    print(f"Generating {args.num_games} games with {workers} workers "
          f"(time_limit={args.time_limit}+/-{args.time_jitter}s)")

    all_examples = []
    games_completed = 0
    games_discarded = 0
    total_game_moves = 0
    wins = 0
    losses = 0
    draws = 0
    games_since_save = 0
    t0 = time.time()

    try:
        with Pool(workers) as pool:
            pbar = tqdm(pool.imap(_play_one_game, tasks, chunksize=4),
                        total=len(tasks), desc="Games", unit="game")
            for examples, move_count in pbar:
                games_completed += 1

                if examples is None:
                    games_discarded += 1
                else:
                    all_examples.extend(examples)
                    total_game_moves += move_count
                    # Count outcome from first example
                    ws = examples[0]["win_score"]
                    if ws > 0:
                        wins += 1
                    elif ws < 0:
                        losses += 1
                    else:
                        draws += 1

                kept = games_completed - games_discarded
                avg_moves = total_game_moves / kept if kept else 0
                pbar.set_postfix(W=wins, L=losses, D=draws,
                                 avg_mv=f"{avg_moves:.0f}")

                games_since_save += 1
                if games_since_save >= args.save_interval:
                    _save_parquet(all_examples, args.output)
                    games_since_save = 0

    except KeyboardInterrupt:
        print(f"\nInterrupted after {games_completed} games! Saving...")

    elapsed = time.time() - t0

    # Final save
    _save_parquet(all_examples, args.output)

    # Stats
    kept = games_completed - games_discarded
    print(f"\nDone in {elapsed:.1f}s ({games_completed / elapsed:.1f} games/s)")
    print(f"  Games: {games_completed} completed, {games_discarded} discarded "
          f"({100 * games_discarded / max(games_completed, 1):.1f}% bbox exceeded)")
    print(f"  Outcomes (kept): {wins}W / {losses}L / {draws}D")
    print(f"  Training examples: {len(all_examples):,}")
    if kept > 0:
        print(f"  Avg examples/game: {len(all_examples) / kept:.1f}")
    if all_examples:
        evals = [e["eval_score"] for e in all_examples]
        print(f"  Eval range: [{min(evals):.3f}, {max(evals):.3f}], "
              f"mean={sum(evals) / len(evals):.4f}")
    print(f"Saved to {args.output}")


def _save_parquet(examples, path):
    """Save examples as a Parquet file."""
    import pandas as pd

    if not examples:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(examples)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
