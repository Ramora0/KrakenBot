"""Minimax bot using learned pattern evaluation.

Iterative deepening, alpha-beta, TT, incremental eval with learned pattern
values from a pattern_values.json file.

Uses two window systems:
  - 6-cell windows: win detection, threat analysis (hot windows, instant win)
  - N-cell windows: evaluation scoring via learned pattern lookup (N from model meta)
"""

import json
import math
import os
import random
import time
from itertools import combinations
from bot import Bot
from game import Player, HEX_DIRECTIONS

# ── Hyperparameters ──────────────────────────────────────────────────
# Pair masks: number = search order (1=first, 2=second, ...), 0 = skip.
# Row = rank of m1 (0=best delta), Col = rank of m2. Upper triangle only.
# Edit these directly to control which combinations are searched and in what order.
#                          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
_ROOT_PAIR_MASK = (
    (0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),  #  0
    (0, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),  #  1
    (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),  #  2
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),  #  3
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  4
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  5
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  6
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  7
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  8
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  9
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 10
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 11
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 12
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 13
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 14
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 15
)
#                          0   1   2   3   4   5   6   7   8   9  10
_INNER_PAIR_MASK = (
    (0, 2, 3, 4, 6, 8, 11, 12, 13, 14, 15),  #  0
    (0, 0, 5, 7, 9, 0, 0, 0, 0, 0, 0),  #  1
    (0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0),  #  2
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  3
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  4
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  5
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  6
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  7
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  8
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  #  9
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 10
)
_ROOT_CANDIDATE_CAP = len(_ROOT_PAIR_MASK)
_CANDIDATE_CAP = len(_INNER_PAIR_MASK)

def _build_pairs(mask):
    """Build ordered pair list from mask.
    0 = skip, 1 = auto-order (by i+j then i), 2+ = explicit priority (searched first)."""
    explicit = []  # (priority, i, j)
    auto = []      # (i+j, i, j)
    for i, row in enumerate(mask):
        for j, v in enumerate(row):
            if v >= 2:
                explicit.append((v, i, j))
            elif v == 1:
                auto.append((i + j, i, j))
    explicit.sort()
    auto.sort()
    return tuple([(i, j) for _, i, j in explicit] + [(i, j) for _, i, j in auto])

_ROOT_PAIRS = _build_pairs(_ROOT_PAIR_MASK)
_INNER_PAIRS = _build_pairs(_INNER_PAIR_MASK)
_NEIGHBOR_DIST = 2           # hex distance for candidate generation
_DELTA_WEIGHT = 1.5          # weight of eval delta vs history in move ordering
_MAX_QDEPTH = 16             # max depth for quiescence threat search

# 6 unit hex directions for colony candidate selection
_COLONY_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

_WIN_LENGTH = 6


class TimeUp(Exception):
    pass


def hex_distance(dq, dr):
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


# Zobrist hash table
_zobrist_rng = random.Random(42)
_zobrist = {}

# Direction vectors
_DIR_VECTORS = tuple(HEX_DIRECTIONS)

# 6-cell window offsets (for win detection)
_WIN_OFFSETS = tuple(
    (d_idx, k * dq, k * dr)
    for d_idx, (dq, dr) in enumerate(HEX_DIRECTIONS)
    for k in range(_WIN_LENGTH)
)


def _build_eval_offsets(eval_length):
    return tuple(
        (d_idx, k, k * dq, k * dr)
        for d_idx, (dq, dr) in enumerate(HEX_DIRECTIONS)
        for k in range(eval_length)
    )

# Neighbor offsets
_NEIGHBOR_OFFSETS_2 = tuple(
    (dq, dr)
    for dq in range(-_NEIGHBOR_DIST, _NEIGHBOR_DIST + 1)
    for dr in range(-_NEIGHBOR_DIST, _NEIGHBOR_DIST + 1)
    if hex_distance(dq, dr) <= _NEIGHBOR_DIST and (dq, dr) != (0, 0)
)

# TT flags
_EXACT = 0
_LOWER = 1
_UPPER = 2

_WIN_SCORE = 100000000

# Default pattern values path
_DEFAULT_PATTERN_PATH = os.path.join(
    os.path.dirname(__file__), "data", "pattern_values.json")


def _load_pattern_values(path):
    """Load pattern_values.json and build a flat lookup array.

    Values in JSON are normalized; multiplied by _meta.score_scale to recover
    eval-scale units (comparable to LINE_SCORES).
    Returns (pat_value, eval_length) where pat_value[pattern_int] = eval value.
    """
    with open(path) as f:
        raw = json.load(f)

    meta = raw.get("_meta", {})
    score_scale = meta.get("score_scale", 1)
    eval_length = meta.get("window_length", 6)

    # Build pattern tables for this window length
    from mcts.pattern_table import build_arrays
    piece_swap = meta.get("piece_swap_symmetry", False)
    canon_patterns, canon_index, canon_sign, num_canon, num_patterns = build_arrays(eval_length, enforce_piece_swap=piece_swap)

    # Parse the canonical pattern values
    canon_by_str = {}
    for pat_str, val in raw.items():
        if pat_str != "_meta":
            canon_by_str[pat_str] = val * score_scale

    params = [0.0] * num_canon
    for i, pat in enumerate(canon_patterns):
        pat_str = "".join(str(c) for c in pat)
        if pat_str in canon_by_str:
            params[i] = canon_by_str[pat_str]

    # Build PAT_VALUE: pattern_int -> value (me=1, opp=2 perspective)
    pat_value = [0.0] * num_patterns
    for pi in range(num_patterns):
        ci = canon_index[pi]
        cs = canon_sign[pi]
        if cs != 0 and ci >= 0:
            pat_value[pi] = cs * params[ci]

    return pat_value, eval_length


def get_candidates(game):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    occupied = list(game.board)
    if not occupied:
        return [(0, 0)]
    candidates = set()
    board = game.board
    for q, r in occupied:
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            if nb not in board:
                candidates.add(nb)
    return list(candidates)


class MinimaxBot(Bot):
    """Minimax bot with learned pattern evaluation."""

    pair_moves = True

    def __init__(self, time_limit=0.05, pattern_path=None):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0
        self._tt = {}
        self._hash = 0
        self._rc_stack = []
        self._history = {}
        self.last_ebf = 0
        self.last_score = 0

        if pattern_path is None:
            pattern_path = _DEFAULT_PATTERN_PATH
        self._pv, self._eval_length = _load_pattern_values(pattern_path)
        self._eval_offsets = _build_eval_offsets(self._eval_length)
        self._pow3 = tuple(3 ** k for k in range(self._eval_length))

    def get_move(self, game):
        if not game.board:
            return [(0, 0)]

        self._deadline = time.time() + self.time_limit * 2
        if game.current_player != getattr(self, '_player', None):
            self._tt.clear()
            self._history.clear()
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        self.last_score = 0
        self.last_ebf = 0
        self.last_score = 0
        if len(self._tt) > 1_000_000:
            self._tt.clear()

        # Zobrist hash
        self._hash = 0
        for (q, r), p in game.board.items():
            zkey = (q, r, p)
            v = _zobrist.get(zkey)
            if v is None:
                v = _zobrist_rng.getrandbits(64)
                _zobrist[zkey] = v
            self._hash ^= v

        # Cell value mapping: which raw cell value (1 or 2) corresponds to
        # "me" and "opp" in the pattern encoding
        if self._player == Player.A:
            self._cell_a = 1  # A stones = me = 1
            self._cell_b = 2  # B stones = opp = 2
        else:
            self._cell_a = 2  # A stones = opp = 2
            self._cell_b = 1  # B stones = me = 1

        board = game.board

        # ── 6-cell windows: counts for win/threat detection ──
        self._wc = {}
        seen6 = set()
        for (q, r) in board:
            for d_idx, oq, or_ in _WIN_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                if wkey in seen6:
                    continue
                seen6.add(wkey)
                dq, dr = _DIR_VECTORS[d_idx]
                sq, sr = wkey[1], wkey[2]
                a_count = 0
                b_count = 0
                for j in range(_WIN_LENGTH):
                    cp = board.get((sq + j * dq, sr + j * dr))
                    if cp == Player.A:
                        a_count += 1
                    elif cp == Player.B:
                        b_count += 1
                if a_count > 0 or b_count > 0:
                    self._wc[wkey] = [a_count, b_count]

        self._hot_a = set()
        self._hot_b = set()
        for wkey, counts in self._wc.items():
            if counts[0] >= 4:
                self._hot_a.add(wkey)
            if counts[1] >= 4:
                self._hot_b.add(wkey)

        # ── N-cell windows: pattern_ints for eval ──
        self._wp = {}  # wkey -> pattern_int (me=1, opp=2 encoding)
        pv = self._pv
        cell_a = self._cell_a
        cell_b = self._cell_b
        pow3 = self._pow3
        seen8 = set()
        self._eval_score = 0.0
        for (q, r) in board:
            for d_idx, k, oq, or_ in self._eval_offsets:
                wkey8 = (3 + d_idx, q - oq, r - or_)  # offset key space from 6-cell
                if wkey8 in seen8:
                    continue
                seen8.add(wkey8)
                dq, dr = _DIR_VECTORS[d_idx]
                sq, sr = q - oq, r - or_
                pat_int = 0
                has_piece = False
                for j in range(self._eval_length):
                    cp = board.get((sq + j * dq, sr + j * dr))
                    if cp == Player.A:
                        pat_int += cell_a * pow3[j]
                        has_piece = True
                    elif cp == Player.B:
                        pat_int += cell_b * pow3[j]
                        has_piece = True
                if has_piece:
                    self._wp[wkey8] = pat_int
                    self._eval_score += pv[pat_int]

        # ── Candidate set ──
        self._cand_refcount = {}
        for (q, r) in board:
            for dq, dr in _NEIGHBOR_OFFSETS_2:
                nb = (q + dq, r + dr)
                if nb not in board:
                    self._cand_refcount[nb] = self._cand_refcount.get(nb, 0) + 1
        self._cand_set = set(self._cand_refcount)

        if not self._cand_set:
            return [(0, 0)]

        maximizing = game.current_player == self._player
        turns = self._generate_turns(game)
        if not turns:
            return [(0, 0)]

        best_move = list(turns[0])

        # Save state for TimeUp rollback
        saved_board = dict(game.board)
        saved_state = (game.current_player, game.moves_left_in_turn,
                       game.winner, game.game_over)
        saved_move_count = game.move_count
        saved_hash = self._hash
        saved_eval = self._eval_score
        saved_wc = {k: v[:] for k, v in self._wc.items()}
        saved_wp = dict(self._wp)
        saved_cand_set = set(self._cand_set)
        saved_cand_rc = dict(self._cand_refcount)
        saved_hot_a = set(self._hot_a)
        saved_hot_b = set(self._hot_b)

        for depth in range(1, 200):
            try:
                nodes_before = self._nodes
                result, scores = self._search_root(game, turns, depth)
                best_move = list(result)
                self.last_depth = depth
                self.last_score = scores.get(result, 0)
                nodes_this_depth = self._nodes - nodes_before
                if nodes_this_depth > 1:
                    self.last_ebf = round(nodes_this_depth ** (1.0 / depth), 1)
                self.last_score = scores.get(result, 0)
                turns.sort(key=lambda t: scores.get(t, 0), reverse=maximizing)
                if abs(scores.get(result, 0)) >= _WIN_SCORE:
                    break
            except TimeUp:
                game.board = saved_board
                game.move_count = saved_move_count
                (game.current_player, game.moves_left_in_turn,
                 game.winner, game.game_over) = saved_state
                self._hash = saved_hash
                self._eval_score = saved_eval
                self._wc = saved_wc
                self._wp = saved_wp
                self._cand_set = saved_cand_set
                self._cand_refcount = saved_cand_rc
                self._hot_a = saved_hot_a
                self._hot_b = saved_hot_b
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 1024 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _make(self, game, q, r):
        """Make move: update Zobrist, 6-cell win windows, 8-cell eval windows, candidates."""
        player = game.current_player

        # Zobrist hash
        zkey = (q, r, player)
        v = _zobrist.get(zkey)
        if v is None:
            v = _zobrist_rng.getrandbits(64)
            _zobrist[zkey] = v
        self._hash ^= v

        # Cell value for pattern encoding
        cell_val = self._cell_a if player == Player.A else self._cell_b

        # ── 6-cell windows: update counts for win detection ──
        wc = self._wc
        won = False
        if player == Player.A:
            hot_a = self._hot_a
            for d_idx, oq, or_ in _WIN_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc.get(wkey)
                if counts is None:
                    counts = [0, 0]
                    wc[wkey] = counts
                counts[0] += 1
                if counts[0] >= 4:
                    hot_a.add(wkey)
                if counts[0] == _WIN_LENGTH and counts[1] == 0:
                    won = True
        else:
            hot_b = self._hot_b
            for d_idx, oq, or_ in _WIN_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc.get(wkey)
                if counts is None:
                    counts = [0, 0]
                    wc[wkey] = counts
                counts[1] += 1
                if counts[1] >= 4:
                    hot_b.add(wkey)
                if counts[1] == _WIN_LENGTH and counts[0] == 0:
                    won = True

        # ── 8-cell windows: update pattern_ints for eval ──
        pv = self._pv
        wp = self._wp
        pow3 = self._pow3
        for d_idx, k, oq, or_ in self._eval_offsets:
            wkey8 = (3 + d_idx, q - oq, r - or_)
            old_pi = wp.get(wkey8, 0)
            new_pi = old_pi + cell_val * pow3[k]
            self._eval_score += pv[new_pi] - pv[old_pi]
            wp[wkey8] = new_pi

        # ── Candidates ──
        self._cand_set.discard((q, r))
        rc = self._cand_refcount
        self._rc_stack.append(rc.pop((q, r), 0))
        board = game.board
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            rc[nb] = rc.get(nb, 0) + 1
            if nb not in board:
                self._cand_set.add(nb)

        # Place stone
        game.board[(q, r)] = player
        game.move_count += 1
        if won:
            game.winner = player
            game.game_over = True
        else:
            game.moves_left_in_turn -= 1
            if game.moves_left_in_turn <= 0:
                game.current_player = Player.B if player == Player.A else Player.A
                game.moves_left_in_turn = 2

    def _undo(self, game, q, r, state, player):
        """Undo move: restore Zobrist, 6-cell windows, 8-cell eval windows, candidates."""
        del game.board[(q, r)]
        game.move_count -= 1
        game.current_player, game.moves_left_in_turn, game.winner, game.game_over = state

        # Zobrist
        self._hash ^= _zobrist[(q, r, player)]

        cell_val = self._cell_a if player == Player.A else self._cell_b

        # ── 6-cell windows ──
        wc = self._wc
        if player == Player.A:
            hot_a = self._hot_a
            for d_idx, oq, or_ in _WIN_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc[wkey]
                counts[0] -= 1
                if counts[0] < 4:
                    hot_a.discard(wkey)
        else:
            hot_b = self._hot_b
            for d_idx, oq, or_ in _WIN_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc[wkey]
                counts[1] -= 1
                if counts[1] < 4:
                    hot_b.discard(wkey)

        # ── 8-cell windows ──
        pv = self._pv
        wp = self._wp
        pow3 = self._pow3
        for d_idx, k, oq, or_ in self._eval_offsets:
            wkey8 = (3 + d_idx, q - oq, r - or_)
            old_pi = wp[wkey8]
            new_pi = old_pi - cell_val * pow3[k]
            self._eval_score += pv[new_pi] - pv[old_pi]
            if new_pi == 0:
                del wp[wkey8]
            else:
                wp[wkey8] = new_pi

        # ── Candidates ──
        rc = self._cand_refcount
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            c = rc[nb] - 1
            if c == 0:
                del rc[nb]
                self._cand_set.discard(nb)
            else:
                rc[nb] = c
        saved_rc = self._rc_stack.pop()
        if saved_rc > 0:
            rc[(q, r)] = saved_rc
            self._cand_set.add((q, r))

    def _tt_key(self, game):
        return (self._hash, game.current_player, game.moves_left_in_turn)

    def _move_delta(self, q, r, is_a):
        """Eval delta from placing at (q,r) using N-cell patterns — read-only."""
        cell_val = self._cell_a if is_a else self._cell_b
        pv = self._pv
        wp = self._wp
        pow3 = self._pow3
        delta = 0.0
        for d_idx, k, oq, or_ in self._eval_offsets:
            wkey8 = (3 + d_idx, q - oq, r - or_)
            old_pi = wp.get(wkey8, 0)
            new_pi = old_pi + cell_val * pow3[k]
            delta += pv[new_pi] - pv[old_pi]
        return delta

    # ── Win/threat detection (6-cell windows, identical to ai.py) ──

    def _find_instant_win(self, game, player):
        p_idx = 0 if player == Player.A else 1
        o_idx = 1 - p_idx
        hot = self._hot_a if player == Player.A else self._hot_b
        wc = self._wc
        board = game.board
        for wkey in hot:
            counts = wc[wkey]
            if counts[p_idx] >= _WIN_LENGTH - 2 and counts[o_idx] == 0:
                d_idx, sq, sr = wkey
                dq, dr = _DIR_VECTORS[d_idx]
                cells = []
                for j in range(_WIN_LENGTH):
                    cell = (sq + j * dq, sr + j * dr)
                    if cell not in board:
                        cells.append(cell)
                if len(cells) == 1:
                    other = next((c for c in self._cand_set if c != cells[0]), cells[0])
                    return (min(cells[0], other), max(cells[0], other))
                elif len(cells) == 2:
                    return (min(cells[0], cells[1]), max(cells[0], cells[1]))
        return None

    def _find_threat_cells(self, game, player):
        threat_cells = set()
        p_idx = 0 if player == Player.A else 1
        o_idx = 1 - p_idx
        hot = self._hot_a if player == Player.A else self._hot_b
        wc = self._wc
        board = game.board
        for wkey in hot:
            counts = wc[wkey]
            if counts[o_idx] == 0:
                d_idx, sq, sr = wkey
                dq, dr = _DIR_VECTORS[d_idx]
                for j in range(_WIN_LENGTH):
                    cell = (sq + j * dq, sr + j * dr)
                    if cell not in board:
                        threat_cells.add(cell)
        return threat_cells

    def _filter_turns_by_threats(self, game, turns):
        """Filter turns to only those that block ALL opponent threat windows.

        Own threats are not checked here — _find_instant_win already
        catches and short-circuits those before we reach this point.
        """
        current = game.current_player
        opponent = Player.B if current == Player.A else Player.A

        p_idx = 0 if opponent == Player.A else 1
        o_idx = 1 - p_idx
        hot = self._hot_a if opponent == Player.A else self._hot_b
        wc = self._wc
        board = game.board

        # Collect per-window empty sets — each window must be hit
        must_hit = []
        for wkey in hot:
            counts = wc[wkey]
            if counts[p_idx] >= _WIN_LENGTH - 2 and counts[o_idx] == 0:
                d_idx, sq, sr = wkey
                dq, dr = _DIR_VECTORS[d_idx]
                empties = frozenset(
                    (sq + j * dq, sr + j * dr)
                    for j in range(_WIN_LENGTH)
                    if (sq + j * dq, sr + j * dr) not in board
                )
                must_hit.append(empties)

        if not must_hit:
            return turns

        # Only keep turns where the two stones cover every threat window
        return [t for t in turns
                if all(t[0] in w or t[1] in w for w in must_hit)]

    # ── Turn management (identical to ai.py) ──

    def _make_turn(self, game, turn):
        m1, m2 = turn
        p1 = game.current_player
        s1 = (p1, game.moves_left_in_turn, game.winner, game.game_over)
        self._make(game, m1[0], m1[1])
        if game.game_over:
            return [(m1, s1, p1)]
        p2 = game.current_player
        s2 = (p2, game.moves_left_in_turn, game.winner, game.game_over)
        self._make(game, m2[0], m2[1])
        return [(m1, s1, p1), (m2, s2, p2)]

    def _undo_turn(self, game, undo_info):
        for cell, state, player in reversed(undo_info):
            self._undo(game, cell[0], cell[1], state, player)

    def _generate_turns(self, game):
        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            return [win_turn]

        candidates = list(self._cand_set)
        if len(candidates) < 2:
            if candidates:
                return [(candidates[0], candidates[0])]
            return []

        is_a = game.current_player == Player.A
        move_delta = self._move_delta
        maximizing = game.current_player == self._player

        candidates.sort(key=lambda c: move_delta(c[0], c[1], is_a), reverse=maximizing)
        candidates = candidates[:_ROOT_CANDIDATE_CAP]

        # Colony candidate: a random hex at max distance from the board cluster.
        # Represents starting a separate group far from the main action.
        occupied = list(game.board)
        cq = sum(q for q, r in occupied) // len(occupied)
        cr = sum(r for q, r in occupied) // len(occupied)
        max_r = max(hex_distance(q - cq, r - cr) for q, r in occupied)
        colony_dist = max_r + 3
        dq, dr = random.choice(_COLONY_DIRS)
        colony = (cq + dq * colony_dist, cr + dr * colony_dist)
        if colony not in game.board:
            candidates.append(colony)

        n = len(candidates)
        turns = [(candidates[i], candidates[j]) for i, j in _ROOT_PAIRS
                 if i < n and j < n]
        return self._filter_turns_by_threats(game, turns)

    def _generate_threat_turns(self, game, my_threats, opp_threats):
        """Generate threat turns: block opponent threats first, else make own.

        When blocking: pairs of blocking cells (need 2 stones) + each blocking
        cell with greedy best companion (need 1 stone).
        When attacking: pairs of own threat cells + each with greedy companion.
        Greedy companion chosen by _move_delta for the single-threat-cell case.
        """
        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            return [win_turn]

        is_a = game.current_player == Player.A
        maximizing = game.current_player == self._player
        sign = 1 if maximizing else -1

        opp_cells = [c for c in opp_threats if c in self._cand_set]
        my_cells = [c for c in my_threats if c in self._cand_set]

        if opp_cells:
            primary = opp_cells
        elif my_cells:
            primary = my_cells
        else:
            return []

        if len(primary) >= 2:
            # All pairs of threat/block cells — sorted by move_delta sum
            pairs = list(combinations(primary, 2))
            pairs.sort(
                key=lambda p: self._move_delta(p[0][0], p[0][1], is_a)
                            + self._move_delta(p[1][0], p[1][1], is_a),
                reverse=maximizing,
            )
            return pairs

        # Single threat cell — pair with greedy best companion by move_delta
        tc = primary[0]
        cand_list = list(self._cand_set)
        best_comp = None
        best_delta = -math.inf
        for c in cand_list:
            if c != tc:
                d = self._move_delta(c[0], c[1], is_a) * sign
                if d > best_delta:
                    best_delta = d
                    best_comp = c
        if best_comp is None:
            return []
        return [(min(tc, best_comp), max(tc, best_comp))]

    def _quiescence(self, game, alpha, beta, qdepth):
        """Extend search while threats exist, considering only threat moves."""
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return _WIN_SCORE
            elif game.winner != Player.NONE:
                return -_WIN_SCORE
            return 0

        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            undo_info = self._make_turn(game, win_turn)
            score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            self._undo_turn(game, undo_info)
            return score

        stand_pat = self._eval_score
        current = game.current_player
        opponent = Player.B if current == Player.A else Player.A
        my_threats = self._find_threat_cells(game, current)
        opp_threats = self._find_threat_cells(game, opponent)

        if (not my_threats and not opp_threats) or qdepth <= 0:
            return stand_pat

        maximizing = current == self._player

        if maximizing:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return stand_pat
            beta = min(beta, stand_pat)

        threat_turns = self._generate_threat_turns(game, my_threats, opp_threats)
        if not threat_turns:
            return stand_pat

        if maximizing:
            value = stand_pat
            for turn in threat_turns:
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._quiescence(game, alpha, beta, qdepth - 1)
                self._undo_turn(game, undo_info)
                if child_val > value:
                    value = child_val
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = stand_pat
            for turn in threat_turns:
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._quiescence(game, alpha, beta, qdepth - 1)
                self._undo_turn(game, undo_info)
                if child_val < value:
                    value = child_val
                beta = min(beta, value)
                if alpha >= beta:
                    break

        return value

    # ── Search ──

    def _search_root(self, game, turns, depth):
        maximizing = game.current_player == self._player
        best_turn = turns[0]
        alpha = -math.inf
        beta = math.inf

        scores = {}
        for turn in turns:
            self._check_time()
            undo_info = self._make_turn(game, turn)
            if game.game_over:
                score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            else:
                score = self._minimax(game, depth - 1, alpha, beta)
            self._undo_turn(game, undo_info)
            scores[turn] = score

            if maximizing and score > alpha:
                alpha = score
                best_turn = turn
            elif not maximizing and score < beta:
                beta = score
                best_turn = turn

        best_score = alpha if maximizing else beta
        self._tt[self._tt_key(game)] = (depth, best_score, _EXACT, best_turn)
        return best_turn, scores

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return _WIN_SCORE
            elif game.winner != Player.NONE:
                return -_WIN_SCORE
            return 0

        tt_key = self._tt_key(game)
        tt_entry = self._tt.get(tt_key)
        tt_move = None
        if tt_entry:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    return tt_score
                elif tt_flag == _LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if depth == 0:
            score = self._quiescence(game, alpha, beta, _MAX_QDEPTH)
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            undo_info = self._make_turn(game, win_turn)
            score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            self._undo_turn(game, undo_info)
            self._tt[tt_key] = (depth, score, _EXACT, win_turn)
            return score

        opponent = Player.B if game.current_player == Player.A else Player.A
        opp_win = self._find_instant_win(game, opponent)
        if opp_win:
            opp_threats = self._find_threat_cells(game, opponent)
            p_idx = 0 if opponent == Player.A else 1
            o_idx = 1 - p_idx
            must_hit = []
            board = game.board
            hot = self._hot_a if opponent == Player.A else self._hot_b
            wc = self._wc
            for wkey in hot:
                counts = wc[wkey]
                if counts[p_idx] >= _WIN_LENGTH - 2 and counts[o_idx] == 0:
                    d_idx, sq, sr = wkey
                    dq, dr = _DIR_VECTORS[d_idx]
                    empties = frozenset(
                        (sq + j * dq, sr + j * dr)
                        for j in range(_WIN_LENGTH)
                        if (sq + j * dq, sr + j * dr) not in board
                    )
                    must_hit.append(empties)
            if len(must_hit) > 1:
                can_block = False
                all_cells = set()
                for s in must_hit:
                    all_cells |= s
                for c1 in all_cells:
                    for c2 in all_cells:
                        if all(c1 in w or c2 in w for w in must_hit):
                            can_block = True
                            break
                    if can_block:
                        break
                if not can_block:
                    score = -_WIN_SCORE if opponent != self._player else _WIN_SCORE
                    self._tt[tt_key] = (depth, score, _EXACT, None)
                    return score

        orig_alpha = alpha
        orig_beta = beta
        maximizing = game.current_player == self._player

        candidates = list(self._cand_set)
        if len(candidates) < 2:
            if not candidates:
                score = self._eval_score
                self._tt[tt_key] = (depth, score, _EXACT, None)
                return score
            c = candidates[0]
            turns = [(c, c)]
        else:
            is_a = game.current_player == Player.A
            history = self._history
            move_delta = self._move_delta

            delta_sign = _DELTA_WEIGHT if maximizing else -_DELTA_WEIGHT
            candidates.sort(
                key=lambda c: history.get(c, 0) + move_delta(c[0], c[1], is_a) * delta_sign,
                reverse=True)
            candidates = candidates[:_CANDIDATE_CAP]

            n = len(candidates)
            turns = [(candidates[i], candidates[j]) for i, j in _INNER_PAIRS
                     if i < n and j < n]
            turns = self._filter_turns_by_threats(game, turns)

        if not turns:
            score = self._eval_score
            self._tt[tt_key] = (depth, score, _EXACT, None)
            return score

        if tt_move is not None:
            try:
                idx = turns.index(tt_move)
                turns[0], turns[idx] = turns[idx], turns[0]
            except ValueError:
                pass

        best_move = None

        if maximizing:
            value = -math.inf
            for turn in turns:
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo_turn(game, undo_info)
                if child_val > value:
                    value = child_val
                    best_move = turn
                alpha = max(alpha, value)
                if alpha >= beta:
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break
        else:
            value = math.inf
            for turn in turns:
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo_turn(game, undo_info)
                if child_val < value:
                    value = child_val
                    best_move = turn
                beta = min(beta, value)
                if alpha >= beta:
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break

        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self._tt[tt_key] = (depth, value, flag, best_move)
        return value
