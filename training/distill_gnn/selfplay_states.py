"""DAgger state generation: student self-play -> visited turn-start states.

Plays student-vs-student games with the distilled checkpoint (KrakenBot MCTS,
small sim budget) and collects every PRE-TURN position the students actually
visit. Those states are then labeled by the TEACHER (generate_distill_gnn.py
--input <out.pkl>, then relabel_joint.py --input <out.pkl> --reuse-dir <labeled>)
— classic DAgger: student chooses the states, teacher supplies the targets.
No MCTS-derived targets are used anywhere (this is NOT AlphaZero self-play).

Diversity: the first `--explore-turns` turns of each game are sampled from the
pair-softmax prior at temperature 1 (no search), after which each side runs its
normal MCTS with a per-turn sim budget. Positions are deduped by (board, cp).
game_id starts at --gid-base (default 1_000_000) so DAgger games never collide
with human game ids in the leak-free split.

Run in the KrakenBot venv:
  python -m training.distill_gnn.selfplay_states \
      --ckpt training/distill_gnn/results/attempt1_b10f128/distill_gnn_best.pt \
      --games 200 --sims 64 --out training/distill_gnn/data/dagger_states.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

import torch
import torch.nn.functional as F

from game import HexGame, Player
from model.resnet import BOARD_SIZE
from training.distill_gnn.eval.kraken_agent import KrakenAgent

MAX_STONES = 160


def _board_int(game):
    return {(q, r): p.value for (q, r), p in game.board.items()}


@torch.no_grad()
def _sample_pair_from_prior(agent, game, temp=1.0):
    """Sample a stone pair (or single) from the raw pair-softmax — no search."""
    from mcts.tree import create_tree_dynamic
    agent.model.set_padding_mode('zeros')
    tree, off_q, off_r = create_tree_dynamic(
        game, agent.model, agent.device, add_noise=False, min_size=BOARD_SIZE)
    x = tree.root_planes.unsqueeze(0).to(agent.device)
    _v, pair_logits, _ml, _c = agent.model(x)
    agent.model.set_padding_mode('circular')
    logits = pair_logits[0].float()
    occ = (tree.root_planes[0] + tree.root_planes[1]).reshape(-1).to(agent.device) > 0
    logits[occ, :] = float("-inf")
    logits[:, occ] = float("-inf")
    logits.fill_diagonal_(float("-inf"))
    flat = logits.reshape(-1) / temp
    probs = F.softmax(flat, dim=0)
    idx = torch.multinomial(probs, 1).item()
    n = BOARD_SIZE
    a, b = idx // (n * n), idx % (n * n)
    p1 = (a // n - off_q, a % n - off_r)
    p2 = (b // n - off_q, b % n - off_r)
    if game.moves_left_in_turn == 1:
        return [p1]
    if p1 == p2:
        p2 = (p2[0] + 1, p2[1])
    return [p1, p2]


def play_selfplay_game(agent_a, agent_b, explore_turns, rng, states_out, seen):
    game = HexGame(win_length=6)
    game.make_move(0, 0)
    turn = 0
    while not game.game_over and game.move_count < MAX_STONES:
        cp = game.current_player
        agent = agent_a if cp == Player.A else agent_b
        board = _board_int(game)
        key = (frozenset(board.items()), cp.value)
        if key not in seen:
            seen.add(key)
            states_out.append((dict(board), cp.value))
        if turn < explore_turns:
            stones = _sample_pair_from_prior(agent, game, temp=1.0)
        else:
            stones = agent.choose(board, cp.value, game.moves_left_in_turn)
        for (q, r) in stones:
            if game.game_over:
                break
            if not game.make_move(q, r):
                # fallback: random legal near stones
                for _ in range(200):
                    q2 = rng.randint(-8, BOARD_SIZE)
                    r2 = rng.randint(-8, BOARD_SIZE)
                    if (q2, r2) not in game.board and game.make_move(q2, r2):
                        break
        turn += 1
    return game.winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-b", default=None,
                    help="optional different opponent checkpoint (default: same)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=64,
                    help="per-turn MCTS sims for non-explore turns")
    ap.add_argument("--explore-turns", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gid-base", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a = KrakenAgent(args.ckpt, n_sims=args.sims, device=args.device, name="sp_a")
    b = (KrakenAgent(args.ckpt_b, n_sims=args.sims, device=args.device, name="sp_b")
         if args.ckpt_b else a)

    all_entries = []
    seen = set()
    t0 = time.perf_counter()
    for g in range(args.games):
        states = []
        try:
            winner = play_selfplay_game(a, b, args.explore_turns, rng, states, seen)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  game {g+1}/{args.games}: CRASHED, skipping (kept "
                  f"{len(states)} states)", flush=True)
            all_entries.extend((bd, cp, None, args.gid_base + g)
                               for (bd, cp) in states)
            continue
        gid = args.gid_base + g
        # 4-tuple (board, cp, None, gid) matches the labelers' loaders
        all_entries.extend((bd, cp, None, gid) for (bd, cp) in states)
        el = time.perf_counter() - t0
        print(f"  game {g+1}/{args.games}: +{len(states)} states "
              f"(total {len(all_entries):,} uniq)  winner={winner}  {el:.0f}s",
              flush=True)
        if (g + 1) % 10 == 0:      # incremental save (survive kills)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out + ".tmp", "wb") as f:
                pickle.dump(all_entries, f)
            os.replace(args.out + ".tmp", args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(all_entries, f)
    print(f"DONE: {len(all_entries):,} unique states from {args.games} games "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
