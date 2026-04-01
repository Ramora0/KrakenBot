"""Generate ResNet distillation data from bot self-play.

Plays games from early human-game positions using the C++ bot on both sides.
Each turn generates one training example with the bot's full double move.
Games where the board exceeds 19x19 bounding box are discarded.

Output: Parquet file with columns:
  board            - JSON string of board state {"q,r": player_int, ...}
  current_player   - int (1=Player.A, 2=Player.B)
  moves            - list of [q,r] pairs: always [[q0,r0],[q1,r1]]
  eval_score       - float, bot.last_score / SCORE_SCALE
  win_score        - float, +1.0 win / -1.0 loss / 0.0 draw from current_player POV
  game_id          - int, unique per generated game
  winning_singles  - JSON list of [q,r] cells that each win alone (empty if N/A)
  winning_pairs    - JSON list of [[q1,r1],[q2,r2]] pairs that win together (empty if N/A)

Usage: python -m training.distill.generate_distill [--num-games 100000]
"""

import json
import os
import pickle
import random
import signal
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from game import HexGame, Player, HEX_DIRECTIONS

SCORE_SCALE = 20_000
MAX_MOVES = 200
MOVE_WALL_TIMEOUT = 10.0  # seconds; skip game if a single get_move exceeds this
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


def _find_winning_moves(board, player):
    """Find all single-cell and pair wins for player on the current board.

    Scans every 6-cell window (3 hex directions) that contains at least one
    of *player*'s stones.  Windows with 5 friendly / 0 enemy yield a winning
    single (the one empty cell).  Windows with 4 friendly / 0 enemy yield a
    winning pair (the two empty cells).

    Returns (winning_singles, winning_pairs):
        winning_singles: list of [q, r]
        winning_pairs:   list of [[q1, r1], [q2, r2]]
    """
    seen_singles = set()
    seen_pairs = set()
    winning_singles = []
    winning_pairs = []

    player_cells = {pos for pos, p in board.items() if p == player}

    for dq, dr in HEX_DIRECTIONS:
        checked = set()
        for bq, br in player_cells:
            for offset in range(_WIN_LENGTH):
                sq = bq - dq * offset
                sr = br - dr * offset
                key = (sq, sr, dq, dr)
                if key in checked:
                    continue
                checked.add(key)

                my = 0
                opp = 0
                empties = []
                for i in range(_WIN_LENGTH):
                    c = (sq + dq * i, sr + dr * i)
                    v = board.get(c)
                    if v == player:
                        my += 1
                    elif v is not None:
                        opp += 1
                        break          # blocked — skip rest
                    else:
                        empties.append(c)

                if opp > 0:
                    continue

                if my == 5 and len(empties) == 1:
                    c = empties[0]
                    if c not in seen_singles:
                        seen_singles.add(c)
                        winning_singles.append(list(c))

                elif my == 4 and len(empties) == 2:
                    pair = tuple(sorted(empties))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        winning_pairs.append(
                            [list(empties[0]), list(empties[1])])

    return winning_singles, winning_pairs


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

    Returns (examples, total_moves, finish_type) where finish_type is
    'single' (stone 1 won), 'double' (stone 2 won), or None (discarded).
    """
    try:
        return _play_one_game_inner(args)
    except Exception as e:
        game_id = args[6]
        stones = len(args[0])
        print(f"[ERROR] game {game_id} ({stones} stones): {type(e).__name__}: {e}",
              flush=True)
        return None, 0, None


def _play_one_game_inner(args):
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

    # Collect per-turn data
    turn_records = []
    total_moves = 0
    forfeit_player = None

    while not game.game_over and total_moves < MAX_MOVES:
        player = game.current_player
        bot = bots[player]

        board_snap = dict(game.board)
        t_move = time.time()
        moves = list(bot.get_move(game))
        elapsed = time.time() - t_move
        if elapsed > MOVE_WALL_TIMEOUT:
            print(f"[SLOW] game {game_id}: get_move took {elapsed:.1f}s "
                  f"(limit={bot.time_limit:.3f}s, depth={bot.last_depth}, "
                  f"nodes={bot._nodes}, stones={len(game.board)}, "
                  f"move#{total_moves})", flush=True)
            return None, 0, None
        eval_score = bot.last_score / SCORE_SCALE

        moves_played = []
        for q, r in moves:
            if game.game_over:
                break
            if not game.make_move(q, r):
                forfeit_player = player
                break
            moves_played.append((q, r))
            total_moves += 1

        turn_records.append((board_snap, player, moves, moves_played, eval_score))

        if forfeit_player is not None:
            break

        if not _board_bbox_ok(game.board):
            return None, 0, None

    if forfeit_player is not None:
        winner = Player.B if forfeit_player == Player.A else Player.A
    else:
        winner = game.winner

    # Determine finish type from last turn
    finish_type = None
    if winner != Player.NONE and turn_records:
        last_played = turn_records[-1][3]  # moves_played
        if len(last_played) == 1:
            finish_type = 'single'
        elif len(last_played) == 2:
            finish_type = 'double'

    # Build training examples — always record bot's full recommended pair
    examples = []
    n_turns = len(turn_records)
    for idx, (board_snap, player, moves, moves_played, eval_score) in enumerate(turn_records):
        if winner == Player.NONE:
            win_score = 0.0
        elif winner == player:
            win_score = 1.0
        else:
            win_score = -1.0

        if len(moves) < 2:
            continue  # skip if bot returned only 1 move (shouldn't happen)

        m1, m2 = moves[0], moves[1]
        ex = {
            "board": _board_to_json(board_snap),
            "current_player": player.value,
            "moves": [list(m1), list(m2)],
            "eval_score": eval_score,
            "win_score": win_score,
            "game_id": game_id,
        }

        # For game-ending turn of the winner, enumerate all winning options
        is_last = (idx == n_turns - 1)
        if is_last and win_score > 0 and winner != Player.NONE:
            ws, wp = _find_winning_moves(board_snap, player)
            ex["winning_singles"] = json.dumps(ws)
            ex["winning_pairs"] = json.dumps(wp)
        else:
            ex["winning_singles"] = "[]"
            ex["winning_pairs"] = "[]"

        examples.append(ex)

    return examples, total_moves, finish_type


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
    single_finishes = 0
    double_finishes = 0
    games_since_save = 0
    t0 = time.time()

    pool = Pool(workers)
    try:
        pbar = tqdm(pool.imap_unordered(_play_one_game, tasks, chunksize=1),
                    total=len(tasks), desc="Games", unit="game")
        for examples, move_count, finish_type in pbar:
            games_completed += 1

            if examples is None:
                games_discarded += 1
            else:
                all_examples.extend(examples)
                total_game_moves += move_count
                ws = examples[0]["win_score"]
                if ws > 0:
                    wins += 1
                elif ws < 0:
                    losses += 1
                else:
                    draws += 1
                if finish_type == 'single':
                    single_finishes += 1
                elif finish_type == 'double':
                    double_finishes += 1

            kept = games_completed - games_discarded
            avg_moves = total_game_moves / kept if kept else 0
            pbar.set_postfix(W=wins, L=losses, D=draws,
                             s1=single_finishes, s2=double_finishes,
                             avg_mv=f"{avg_moves:.0f}")

            games_since_save += 1
            if games_since_save >= args.save_interval:
                _save_parquet(all_examples, args.output)
                games_since_save = 0

    except KeyboardInterrupt:
        print(f"\nInterrupted after {games_completed} games! Saving...")
    except Exception as e:
        print(f"\nError after {games_completed} games: {e}\nSaving...")
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0

    # Final save
    _save_parquet(all_examples, args.output)

    # Stats
    kept = games_completed - games_discarded
    print(f"\nDone in {elapsed:.1f}s ({games_completed / elapsed:.1f} games/s)")
    print(f"  Games: {games_completed} completed, {games_discarded} discarded "
          f"({100 * games_discarded / max(games_completed, 1):.1f}% bbox exceeded)")
    print(f"  Outcomes (kept): {wins}W / {losses}L / {draws}D")
    print(f"  Finishes: {single_finishes} stone-1 / {double_finishes} stone-2 "
          f"({100 * single_finishes / max(kept, 1):.1f}% / "
          f"{100 * double_finishes / max(kept, 1):.1f}%)")
    print(f"  Training examples: {len(all_examples):,}")
    if kept > 0:
        print(f"  Avg examples/game: {len(all_examples) / kept:.1f}")
    if all_examples:
        n_with_wins = sum(1 for e in all_examples
                          if e["winning_singles"] != "[]"
                          or e["winning_pairs"] != "[]")
        evals = [e["eval_score"] for e in all_examples]
        print(f"  Examples with winning moves: {n_with_wins:,}")
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
