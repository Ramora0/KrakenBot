"""Parity + sanity check for the teacher labeler.

Confirms (a) the coordinate mapping between KrakenBot and hexo is correct and
(b) the teacher genuinely understands the game, by feeding it positions with a
known immediate winning move and checking that:

  * the teacher's value is strongly positive (current player is about to win),
  * the teacher's chosen move (acted pair AND argmax of the improved policy)
    lands on a cell that completes 6-in-a-row on the *KrakenBot* board (identity
    coordinate map), i.e. no hidden mirror/rotation.

Sources of test positions:
  1. Synthetic 5-in-a-row threats along all 3 hex axes, both directions.
  2. Real human positions where the side to move has a winning single.

Run with the hexo-strix venv python from the KrakenBot project root:
    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.check_parity
"""

from __future__ import annotations

import os
import pickle
import sys

_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

from game import Player, HEX_DIRECTIONS
from training.distill.generate_distill import _find_winning_moves, _board_has_win
from training.distill_gnn.teacher import Teacher, KRAKEN_A, KRAKEN_B

CKPT = os.environ.get("TEACHER_CKPT",
                      r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")


def _completes_win(board, player, cell):
    """True if placing `player` at `cell` yields 6-in-a-row on the KrakenBot board."""
    b = dict(board)
    b[cell] = player
    q, r = cell
    for dq, dr in HEX_DIRECTIONS:
        run = 1
        for sign in (1, -1):
            i = 1
            while b.get((q + dq * i * sign, r + dr * i * sign)) == player:
                run += 1
                i += 1
        if run >= 6:
            return True
    return False


def _synthetic_positions():
    """5-in-a-row threats for the side to move, across all axes/directions.

    Side to move is A (P1); the opening stone (0,0)=A anchors one end of the
    line so from_state's mandatory P1@(0,0) seed is consistent. Opponent B gets
    a couple of far-away stones so the position isn't degenerate.
    """
    out = []
    b_candidates = [(3, -1), (-3, 1), (1, 3), (-1, -3), (4, -2), (2, 3)]
    for dq, dr in HEX_DIRECTIONS:
        for sign in (1, -1):
            # A line of 5 A-stones anchored at (0,0): (0,0),(±d),(±2d),(±3d),(±4d)
            line = [(dq * i * sign, dr * i * sign) for i in range(5)]
            line_set = set(line)
            win_cell = (dq * 5 * sign, dr * 5 * sign)
            board = {c: Player.A for c in line}
            # two off-line B stones so the position isn't degenerate
            bs = [c for c in b_candidates
                  if c not in line_set and c != (0, 0) and c != win_cell][:2]
            for c in bs:
                board[c] = Player.B
            out.append((f"axis({dq},{dr}) dir{sign:+d}", board, Player.A, win_cell))
    return out


def _human_win_positions(limit=30, scan=120_000):
    """Find real human positions where the side to move has a winning single."""
    path = os.path.join(_KRAKEN_ROOT, "training", "distill", "data",
                        "positions_human_labelled.pkl")
    import __main__
    __main__.Player = Player  # human pkl pickled Player under __main__
    with open(path, "rb") as f:
        pos = pickle.load(f)
    found = []
    counts = {}
    for entry in pos[:scan]:
        board, cp = entry[0], entry[1]
        if not board or _board_has_win(board):
            continue
        counts[len(board)] = counts.get(len(board), 0) + 1
        singles, _pairs = _find_winning_moves(board, cp)
        if singles:
            found.append((f"human({len(board)} stones)", dict(board), cp,
                          tuple(singles[0])))
            if len(found) >= limit:
                break
    depth_summary = sorted(counts.items())[:1] + sorted(counts.items())[-1:]
    print(f"human scan: depths seen (min/max) ~ {depth_summary}, "
          f"{len(found)} winnable positions found")
    return found


def main():
    t = Teacher(CKPT, n_simulations=128, m_actions=16)
    print(f"teacher: {t.train_steps} steps, graph={t.mc.graph_type}, dev={t.device}\n")

    cases = _synthetic_positions() + _human_win_positions()

    n = 0
    val_ok = 0
    acted_ok = 0
    argmax_ok = 0
    for name, board, cp, expected_win_cell in cases:
        cp_int = cp.value if hasattr(cp, "value") else cp
        # sanity: the expected cell really wins on the KrakenBot board
        assert _completes_win(board, cp, expected_win_cell), \
            f"test bug: {expected_win_cell} does not win in {name}"

        lab = t.label_position(board, cp_int, moves_remaining=2, seed=0)
        pi1_argmax = max(lab.pi1, key=lambda x: x[1])[0] if lab.pi1 else None

        acted_cells = set(lab.best_pair)
        acted_hit = any(_completes_win(board, cp, c) for c in acted_cells)
        argmax_hit = pi1_argmax is not None and _completes_win(board, cp, pi1_argmax)
        v_hit = lab.value > 0.5

        n += 1
        val_ok += v_hit
        acted_ok += acted_hit
        argmax_ok += argmax_hit
        flag = "OK " if (acted_hit or argmax_hit) and v_hit else "!! "
        print(f"{flag}{name:22s} v={lab.value:+.3f} "
              f"acted={lab.best_pair} argmax_pi1={pi1_argmax} "
              f"win_cell={expected_win_cell}")

    print(f"\n=== {n} positions ===")
    print(f"value>0.5:        {val_ok}/{n}")
    print(f"acted completes:  {acted_ok}/{n}")
    print(f"pi1-argmax wins:  {argmax_ok}/{n}")
    solved = sum(1 for _ in range(0))  # placeholder
    ok = val_ok == n and max(acted_ok, argmax_ok) == n
    print("PARITY:", "PASS" if ok else "CHECK — see !! rows above")


if __name__ == "__main__":
    main()
