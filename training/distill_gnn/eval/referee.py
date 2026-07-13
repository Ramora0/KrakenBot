"""Head-to-head referee (run in the KRAKEN venv).

Plays full Connect-6-on-hex games between two agents and reports win rate + Elo.
Agents expose .choose(board_int, current_player, moves_remaining) -> [(q,r), ...]
and .name. Kraken agents (HexResNet) run in-process; the strix teacher runs as a
subprocess under the hexo venv (StrixClient).

The authoritative game is game.HexGame (infinite grid). A opens forced at (0,0);
thereafter 2 stones/turn; 6-in-a-row wins. Sides alternate across games.

Examples:
  # distilled student vs strix teacher, student time-matched to strix's ms/turn
  python -m training.distill_gnn.eval.referee \
      --a-ckpt training/distill_gnn/results/attempt1_b10f128/distill_gnn_best.pt \
      --a-time-ms 400 --b-strix --b-sims 128 --games 40

  # distilled student vs best kraken checkpoint (both in-process)
  python -m training.distill_gnn.eval.referee \
      --a-ckpt .../distill_gnn_best.pt --a-time-ms 400 \
      --b-ckpt training/mcts_results/best.pt --b-time-ms 400 --games 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

from game import HexGame, Player  # noqa: E402

MAX_STONES = 200
DEFAULT_HEXO_PY = r"C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe"
DEFAULT_TEACHER = r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt"


class StrixClient:
    """Drives the strix teacher as a subprocess under the hexo venv."""

    def __init__(self, hexo_python, sims, device="cuda", m_actions=16,
                 teacher_ckpt=DEFAULT_TEACHER, name=None, log_path=None):
        env = dict(os.environ)
        env.update(STRIX_SIMS=str(sims), STRIX_M=str(m_actions),
                   STRIX_DEVICE=device, TEACHER_CKPT=teacher_ckpt)
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [hexo_python, "-u", "-m", "training.distill_gnn.eval.strix_server"],
            cwd=_KRAKEN_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, env=env, bufsize=1)
        ready = json.loads(self._readline())
        self.info = ready
        self.name = name or f"strix(sims={sims})"
        self.sims = sims
        self.turn_ms = []

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("strix server closed unexpectedly (see strix log)")
        return line

    def choose(self, board_int, current_player, moves_remaining):
        req = {"board": {f"{q},{r}": int(v) for (q, r), v in board_int.items()},
               "current_player": int(current_player),
               "moves_remaining": int(moves_remaining)}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self._readline())
        self.turn_ms.append(resp["ms"])
        return [(int(q), int(r)) for (q, r) in resp["pair"]]

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()


def _random_legal(game):
    qs = [q for q, _ in game.board] or [0]
    rs = [r for _, r in game.board] or [0]
    for _ in range(500):
        q = random.randint(min(qs) - 6, max(qs) + 6)
        r = random.randint(min(rs) - 6, max(rs) + 6)
        if (q, r) not in game.board:
            return q, r
    return max(qs) + 1, max(rs) + 1


def play_game(agent_a, agent_b, seed=0, opening=None, state_log=None):
    """agent_a plays Player.A, agent_b plays Player.B. Returns (winner, n_stones,
    a_turn_ms_list, b_turn_ms_list). winner in {'A','B','draw'}.

    opening: optional (board_int, current_player_int) to start from a mid-game
    position (e.g. sampled from human games) instead of the bare (0,0) start —
    decorrelates games in a match. Play each opening twice with agents swapped.

    state_log: optional list; every pre-turn (board_int, cp_int) the game visits
    is appended (blunder-mining: label these with the teacher, DAgger-style).
    """
    rng = random.Random(seed)
    game = HexGame(win_length=6)
    if opening is not None:
        board_int, cp = opening
        game.board = {(q, r): Player(v) for (q, r), v in board_int.items()}
        game.current_player = Player(int(cp))
        game.moves_left_in_turn = 2
        game.move_count = len(game.board)
    else:
        game.make_move(0, 0)  # forced opening by A
    a_ms, b_ms = [], []
    while not game.game_over and game.move_count < MAX_STONES:
        cp = game.current_player
        agent = agent_a if cp == Player.A else agent_b
        board_int = {(q, r): p.value for (q, r), p in game.board.items()}
        if state_log is not None and game.moves_left_in_turn == 2 and board_int:
            state_log.append((dict(board_int), cp.value))
        t0 = time.perf_counter()
        stones = agent.choose(board_int, cp.value, game.moves_left_in_turn)
        dt = (time.perf_counter() - t0) * 1000.0
        (a_ms if cp == Player.A else b_ms).append(dt)
        if not stones:
            stones = [_random_legal(game)]
        for (q, r) in stones:
            if game.game_over:
                break
            if not game.make_move(q, r):
                q2, r2 = _random_legal(game)
                game.make_move(q2, r2)
    if game.winner == Player.A:
        return "A", game.move_count, a_ms, b_ms
    if game.winner == Player.B:
        return "B", game.move_count, a_ms, b_ms
    return "draw", game.move_count, a_ms, b_ms


def elo_diff(score, n):
    """Elo of A relative to B from A's score fraction; +/- from Wilson-ish CI."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    e = -400.0 * math.log10(1.0 / score - 1.0)
    se = math.sqrt(score * (1 - score) / max(n, 1))
    lo = min(max(score - 1.96 * se, 1e-6), 1 - 1e-6)
    hi = min(max(score + 1.96 * se, 1e-6), 1 - 1e-6)
    return e, -400.0 * math.log10(1.0 / lo - 1.0), -400.0 * math.log10(1.0 / hi - 1.0)


def run_match(agent1, agent2, games, seed=0, label1="A1", label2="A2",
              openings=None, dump_lost_states=None):
    """Play `games`, alternating which agent is Player.A. Returns stats dict.

    openings: optional list of (board_int, cp) — game g starts from
    openings[g//2], so each opening is played once per color assignment.

    dump_lost_states: optional pickle path — every pre-turn state from games
    agent1 LOST is saved as (board_int, cp_int, None, gid) tuples (the
    selfplay_states format the teacher labelers load). gid = seed + g.
    """
    w1 = w2 = draws = 0
    a1_ms, a2_ms = [], []
    mined, mined_seen = [], set()
    for g in range(games):
        op = openings[(g // 2) % len(openings)] if openings else None
        states = [] if dump_lost_states else None
        if g % 2 == 0:
            winner, n, ms_a, ms_b = play_game(agent1, agent2, seed + g,
                                              opening=op, state_log=states)
            a1_ms += ms_a; a2_ms += ms_b
            if winner == "A": w1 += 1
            elif winner == "B": w2 += 1
            else: draws += 1
            agent1_lost = winner == "B"
        else:
            winner, n, ms_a, ms_b = play_game(agent2, agent1, seed + g,
                                              opening=op, state_log=states)
            a2_ms += ms_a; a1_ms += ms_b
            if winner == "A": w2 += 1
            elif winner == "B": w1 += 1
            else: draws += 1
            agent1_lost = winner == "A"
        if dump_lost_states:
            if agent1_lost:
                for (bd, cp) in states:
                    key = (frozenset(bd.items()), cp)
                    if key not in mined_seen:
                        mined_seen.add(key)
                        mined.append((bd, cp, None, seed + g))
            if (g + 1) % 10 == 0 or g == games - 1:
                import pickle
                with open(dump_lost_states + ".tmp", "wb") as f:
                    pickle.dump(mined, f)
                os.replace(dump_lost_states + ".tmp", dump_lost_states)
        s1 = (w1 + 0.5 * draws) / (g + 1)
        print(f"  game {g+1}/{games}: {label1} {w1} - {w2} {label2} "
              f"(draws {draws})  {label1}_score={s1:.3f}  last={winner} in {n} stones",
              flush=True)
    score1 = (w1 + 0.5 * draws) / max(games, 1)
    e, elo_lo, elo_hi = elo_diff(score1, games)
    return {"label1": label1, "label2": label2, "games": games,
            "w1": w1, "w2": w2, "draws": draws, "score1": score1,
            "elo1_vs_2": e, "elo_ci": [elo_hi, elo_lo],
            "a1_avg_ms": sum(a1_ms) / max(len(a1_ms), 1),
            "a2_avg_ms": sum(a2_ms) / max(len(a2_ms), 1)}


def _make_kraken(ckpt, time_ms, sims, device, name, log_temp=None):
    from training.distill_gnn.eval.kraken_agent import KrakenAgent
    return KrakenAgent(ckpt, n_sims=sims, time_budget_ms=time_ms, device=device,
                       name=name, log_temp=log_temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ckpt", required=True)
    ap.add_argument("--a-time-ms", type=float, default=None)
    ap.add_argument("--a-sims", type=int, default=200)
    ap.add_argument("--a-name", default=None)
    ap.add_argument("--a-log-temp", type=float, default=None,
                    help="override pair_head.log_temp (softer pair policy < 3.22)")
    ap.add_argument("--b-log-temp", type=float, default=None)
    ap.add_argument("--b-ckpt", default=None)
    ap.add_argument("--b-time-ms", type=float, default=None)
    ap.add_argument("--b-sims", type=int, default=200)
    ap.add_argument("--b-name", default=None)
    ap.add_argument("--b-strix", action="store_true", help="opponent is the strix teacher")
    ap.add_argument("--hexo-python", default=DEFAULT_HEXO_PY)
    ap.add_argument("--teacher-ckpt", default=DEFAULT_TEACHER)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--openings-file", default=None,
                    help="JSON [{'board': {'q,r': int}, 'cp': int}, ...]; game g "
                         "starts from opening g//2 (each opening once per color)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strix-log", default=None)
    ap.add_argument("--out", default=None, help="write result JSON here")
    ap.add_argument("--dump-lost-states", default=None,
                    help="pickle path: save pre-turn states from games agent A "
                         "lost (blunder mining, selfplay_states 4-tuple format)")
    args = ap.parse_args()

    a = _make_kraken(args.a_ckpt, args.a_time_ms, args.a_sims, args.device,
                     args.a_name or "student", log_temp=args.a_log_temp)
    if args.b_strix:
        b = StrixClient(args.hexo_python, sims=args.b_sims, device=args.device,
                        teacher_ckpt=args.teacher_ckpt,
                        name=args.b_name or f"strix({args.b_sims})",
                        log_path=args.strix_log)
    else:
        b = _make_kraken(args.b_ckpt, args.b_time_ms, args.b_sims, args.device,
                         args.b_name or "kraken_base", log_temp=args.b_log_temp)
    openings = None
    if args.openings_file:
        with open(args.openings_file) as f:
            openings = [({tuple(int(x) for x in k.split(",")): int(v)
                          for k, v in o["board"].items()}, int(o["cp"]))
                        for o in json.load(f)]
    try:
        print(f"Match: {a.name} vs {b.name}  ({args.games} games, "
              f"{len(openings) if openings else 0} openings)", flush=True)
        stats = run_match(a, b, args.games, args.seed, a.name, b.name,
                          openings=openings,
                          dump_lost_states=args.dump_lost_states)
    finally:
        if args.b_strix:
            b.close()
    print("\n=== RESULT ===")
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
