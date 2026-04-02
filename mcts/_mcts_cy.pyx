# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated game state, select_leaf, and backprop for MCTS.

CyGameState replaces ToroidalHexGame with a flat int[625] board for ~100x
faster _check_win (C array lookups vs Python dict.get).  select_leaf_cy
and backprop_cy mirror the Python versions exactly but use C-level game
ops, eliminating the main CPU bottleneck in self-play.
"""

from libc.string cimport memset, memcpy
from libc.math cimport sqrt

import torch
from game import Player

# Import PUCT from existing Cython extension
from mcts._puct_cy import puct_select as _puct_select

# Import Python-level helpers that we call infrequently
from mcts.tree import (
    _expand_level2, LeafInfo, EXPAND_VISITS, MAX_DEPTH, N_CELLS,
    FPU_REDUCTION,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

cdef int TORUS = 25
cdef int CELLS = 625  # 25 * 25
cdef int WIN_LEN = 6

# HEX_DIRECTIONS = [(1, 0), (0, 1), (1, -1)]
cdef int DQ[3]
cdef int DR[3]
DQ[0] = 1; DQ[1] = 0; DQ[2] = 1
DR[0] = 0; DR[1] = 1; DR[2] = -1

# Max depth for state stacks
cdef int MAX_STATES = 100

# Player constants matching game.Player enum
cdef int NONE = 0
cdef int PLAYER_A = 1
cdef int PLAYER_B = 2


# ---------------------------------------------------------------------------
# CyGameState
# ---------------------------------------------------------------------------

cdef class CyGameState:
    """Fast game state with flat C array board for MCTS traversal."""

    cdef int board[625]
    cdef public int current_player
    cdef public int moves_left_in_turn
    cdef public int move_count
    cdef public int winner
    cdef public bint game_over

    def __cinit__(self):
        memset(self.board, 0, sizeof(self.board))
        self.current_player = PLAYER_A
        self.moves_left_in_turn = 1
        self.move_count = 0
        self.winner = NONE
        self.game_over = False

    # --- C-level methods (hot path) ---

    cdef inline bint _check_win_c(self, int q, int r) noexcept nogil:
        """Check if placing at (q,r) wins. ~100x faster than dict.get."""
        cdef int player = self.board[q * TORUS + r]
        cdef int d, i, count, nq, nr

        for d in range(3):
            count = 1
            # Forward
            for i in range(1, WIN_LEN):
                nq = (q + DQ[d] * i + TORUS) % TORUS
                nr = (r + DR[d] * i + TORUS) % TORUS
                if self.board[nq * TORUS + nr] == player:
                    count += 1
                else:
                    break
            # Backward
            for i in range(1, WIN_LEN):
                nq = (q - DQ[d] * i + TORUS) % TORUS
                nr = (r - DR[d] * i + TORUS) % TORUS
                if self.board[nq * TORUS + nr] == player:
                    count += 1
                else:
                    break
            if count >= WIN_LEN:
                return True
        return False

    cdef inline bint make_move_c(self, int q, int r) noexcept nogil:
        """Place stone, check win, update turn. Returns True on success."""
        cdef int wq = q % TORUS
        cdef int wr = r % TORUS

        if self.game_over or self.board[wq * TORUS + wr] != 0:
            return False

        self.board[wq * TORUS + wr] = self.current_player
        self.move_count += 1

        if self._check_win_c(wq, wr):
            self.winner = self.current_player
            self.game_over = True
            return True

        self.moves_left_in_turn -= 1
        if self.moves_left_in_turn <= 0:
            # Switch player
            if self.current_player == PLAYER_A:
                self.current_player = PLAYER_B
            else:
                self.current_player = PLAYER_A
            self.moves_left_in_turn = 2

        return True

    cdef inline void undo_move_c(self, int q, int r,
                                  int s_cp, int s_mlit,
                                  int s_winner, bint s_go) noexcept nogil:
        """Undo a move, restoring saved state."""
        cdef int wq = q % TORUS
        cdef int wr = r % TORUS
        self.board[wq * TORUS + wr] = 0
        self.move_count -= 1
        self.current_player = s_cp
        self.moves_left_in_turn = s_mlit
        self.winner = s_winner
        self.game_over = s_go

    # --- Python wrappers (cold path: serialization, tree creation) ---

    def make_move(self, int q, int r):
        """Python-callable make_move."""
        return self.make_move_c(q, r)

    def save_state(self):
        """Return state tuple compatible with ToroidalHexGame."""
        return (Player(self.current_player), self.moves_left_in_turn,
                Player(self.winner), self.game_over)

    def undo_move(self, int q, int r, state):
        """Python-callable undo_move from state tuple."""
        cp, mlit, winner, go = state
        self.undo_move_c(q, r,
                         cp.value if hasattr(cp, 'value') else int(cp),
                         int(mlit),
                         winner.value if hasattr(winner, 'value') else int(winner),
                         bool(go))

    def is_valid_move(self, int q, int r):
        cdef int wq = q % TORUS
        cdef int wr = r % TORUS
        if self.game_over:
            return False
        return self.board[wq * TORUS + wr] == 0

    def to_board_dict(self):
        """Return {(q,r): player_int} dict for occupied cells."""
        cdef int q, r, v
        result = {}
        for q in range(TORUS):
            for r in range(TORUS):
                v = self.board[q * TORUS + r]
                if v != 0:
                    result[(q, r)] = Player(v)

        return result

    def to_planes_tensor(self):
        """Return [2, 25, 25] tensor for NN input."""
        cdef int q, r, v
        planes = torch.zeros(2, TORUS, TORUS)
        for q in range(TORUS):
            for r in range(TORUS):
                v = self.board[q * TORUS + r]
                if v == self.current_player:
                    planes[0, q, r] = 1.0
                elif v != 0:
                    planes[1, q, r] = 1.0
        return planes

    def get_occupied_set(self):
        """Return set of (q,r) for occupied cells."""
        cdef int q, r
        result = set()
        for q in range(TORUS):
            for r in range(TORUS):
                if self.board[q * TORUS + r] != 0:
                    result.add((q, r))
        return result

    def to_dict(self):
        """Serialize to JSON-compatible dict (same format as ToroidalHexGame)."""
        board_dict = {}
        cdef int q, r, v
        for q in range(TORUS):
            for r in range(TORUS):
                v = self.board[q * TORUS + r]
                if v != 0:
                    board_dict[f"{q},{r}"] = v
        return {
            "board": board_dict,
            "current_player": self.current_player,
            "moves_left_in_turn": self.moves_left_in_turn,
            "move_count": self.move_count,
            "winner": self.winner,
            "game_over": self.game_over,
        }

    @staticmethod
    def from_dict(d):
        """Restore from serialized dict."""
        cdef CyGameState g = CyGameState()
        for k, v in d["board"].items():
            parts = k.split(",")
            q, r = int(parts[0]), int(parts[1])
            g.board[q * TORUS + r] = int(v)
        g.current_player = int(d["current_player"])
        g.moves_left_in_turn = int(d["moves_left_in_turn"])
        g.move_count = int(d["move_count"])
        g.winner = int(d["winner"])
        g.game_over = bool(d["game_over"])
        return g

    @staticmethod
    def from_toroidal_game(tg):
        """Convert a ToroidalHexGame to CyGameState."""
        cdef CyGameState g = CyGameState()
        for (q, r), player in tg.board.items():
            g.board[q * TORUS + r] = player.value if hasattr(player, 'value') else int(player)
        g.current_player = tg.current_player.value if hasattr(tg.current_player, 'value') else int(tg.current_player)
        g.moves_left_in_turn = tg.moves_left_in_turn
        g.move_count = tg.move_count
        g.winner = tg.winner.value if hasattr(tg.winner, 'value') else int(tg.winner)
        g.game_over = tg.game_over
        return g


# ---------------------------------------------------------------------------
# select_leaf_cy
# ---------------------------------------------------------------------------

cdef inline tuple _idx_to_cell(int idx):
    return (idx // TORUS, idx % TORUS)


def select_leaf_cy(tree, game):
    """Cython select_leaf using CyGameState for fast game ops.

    Mirrors mcts.select_leaf exactly — same tree traversal, same return type.
    """
    cdef CyGameState cg = <CyGameState>game

    # State stack (C arrays instead of Python list of tuples)
    cdef int st_q[100]
    cdef int st_r[100]
    cdef int st_cp[100]
    cdef int st_mlit[100]
    cdef int st_winner[100]
    cdef bint st_go[100]
    cdef int n_states = 0

    # Typed locals
    cdef int depth = 0
    cdef int s1_idx, s2_idx, s1_q, s1_r, s2_q, s2_r
    cdef int ch, local, local_s2, local_pair
    cdef bint needs_exp
    cdef int root_cp_int = tree.root_player.value
    cdef int cp_int
    cdef double root_fpu = tree.root_value - FPU_REDUCTION
    cdef double child_fpu

    path = []
    pair_depths = []
    deltas = []
    pos = tree.root_pos

    while depth < MAX_DEPTH:
        if pos.is_root:
            # ---- Root: two-level (stone_1 -> stone_2) ----

            s1_idx = _puct_select(pos.move_node, fpu=root_fpu)
            s1_q = s1_idx // TORUS
            s1_r = s1_idx % TORUS

            path.append((pos.move_node, s1_idx))
            pair_depths.append(depth)

            # Save state to C stack
            st_q[n_states] = s1_q
            st_r[n_states] = s1_r
            st_cp[n_states] = cg.current_player
            st_mlit[n_states] = cg.moves_left_in_turn
            st_winner[n_states] = cg.winner
            st_go[n_states] = cg.game_over
            n_states += 1

            cg.make_move_c(s1_q, s1_r)

            ch = depth % 2
            deltas.append((s1_q, s1_r, ch))

            # Terminal after stone_1?
            if cg.game_over:
                local = pos.move_node.action_map[s1_idx]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Single-move turn?
            if cg.moves_left_in_turn == 0:
                cp_int = cg.current_player
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(
                    path=path, pair_depths=pair_depths,
                    current_player=Player(cp_int), deltas=deltas,
                    player_flipped=(cp_int != root_cp_int))

            # Level 2: expand lazily, select stone_2
            l2_node = (pos.move_node.level2 or {}).get(s1_idx)
            if l2_node is None:
                l2_node = _expand_level2(
                    tree, pos, s1_idx, game, add_noise=(depth == 0))

            if l2_node is None or l2_node.actions is None:
                cp_int = cg.current_player
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(
                    path=path, pair_depths=pair_depths,
                    current_player=Player(cp_int), deltas=deltas,
                    player_flipped=(cp_int != root_cp_int))

            s2_idx = _puct_select(l2_node, fpu=root_fpu)
            s2_q = s2_idx // TORUS
            s2_r = s2_idx % TORUS

            path.append((l2_node, s2_idx))
            pair_depths.append(depth)

            # Save state
            st_q[n_states] = s2_q
            st_r[n_states] = s2_r
            st_cp[n_states] = cg.current_player
            st_mlit[n_states] = cg.moves_left_in_turn
            st_winner[n_states] = cg.winner
            st_go[n_states] = cg.game_over
            n_states += 1

            cg.make_move_c(s2_q, s2_r)
            deltas.append((s2_q, s2_r, ch))

            # Terminal after stone_2?
            if cg.game_over:
                local = l2_node.action_map[s2_idx]
                l2_node.terminals[local] = True
                l2_node.term_vals[local] = 1.0
                l2_node._has_terminal = True
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Check for child PosNode
            pair_key = (s1_idx, s2_idx)
            child = (pos.children or {}).get(pair_key)

            if child is not None:
                pos = child
                depth += 1
                continue

            # Leaf -- check expansion threshold
            local_s2 = l2_node.action_map[s2_idx]
            needs_exp = (l2_node.visits[local_s2] + 1 >= EXPAND_VISITS)

            cp_int = cg.current_player
            _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
            return LeafInfo(
                path=path, pair_depths=pair_depths,
                current_player=Player(cp_int), deltas=deltas,
                player_flipped=(cp_int != root_cp_int),
                needs_expansion=needs_exp,
                expand_parent=pos, expand_pair=pair_key)

        else:
            # ---- Non-root: flat pair selection ----

            child_fpu = pos.value - FPU_REDUCTION
            pair_action = _puct_select(pos.move_node, fpu=child_fpu)
            s1_idx = pair_action // CELLS
            s2_idx = pair_action % CELLS
            s1_q = s1_idx // TORUS
            s1_r = s1_idx % TORUS
            s2_q = s2_idx // TORUS
            s2_r = s2_idx % TORUS

            path.append((pos.move_node, pair_action))
            pair_depths.append(depth)

            ch = depth % 2

            # Make stone_1
            st_q[n_states] = s1_q
            st_r[n_states] = s1_r
            st_cp[n_states] = cg.current_player
            st_mlit[n_states] = cg.moves_left_in_turn
            st_winner[n_states] = cg.winner
            st_go[n_states] = cg.game_over
            n_states += 1

            cg.make_move_c(s1_q, s1_r)
            deltas.append((s1_q, s1_r, ch))

            if cg.game_over:
                local = pos.move_node.action_map[pair_action]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Make stone_2
            st_q[n_states] = s2_q
            st_r[n_states] = s2_r
            st_cp[n_states] = cg.current_player
            st_mlit[n_states] = cg.moves_left_in_turn
            st_winner[n_states] = cg.winner
            st_go[n_states] = cg.game_over
            n_states += 1

            cg.make_move_c(s2_q, s2_r)
            deltas.append((s2_q, s2_r, ch))

            if cg.game_over:
                local = pos.move_node.action_map[pair_action]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Check for child PosNode
            child = (pos.children or {}).get(pair_action)

            if child is not None:
                pos = child
                depth += 1
                continue

            # Leaf -- check expansion threshold
            local_pair = pos.move_node.action_map[pair_action]
            needs_exp = (pos.move_node.visits[local_pair] + 1 >= EXPAND_VISITS)

            cp_int = cg.current_player
            _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
            return LeafInfo(
                path=path, pair_depths=pair_depths,
                current_player=Player(cp_int), deltas=deltas,
                player_flipped=(cp_int != root_cp_int),
                needs_expansion=needs_exp,
                expand_parent=pos, expand_pair=pair_action)

    # MAX_DEPTH reached
    cp_int = cg.current_player
    _undo_all_c(cg, st_q, st_r, st_cp, st_mlit, st_winner, st_go, n_states)
    return LeafInfo(
        path=path, pair_depths=pair_depths,
        current_player=Player(cp_int), deltas=deltas,
        player_flipped=(cp_int != root_cp_int))


cdef inline void _undo_all_c(CyGameState cg,
                              int* sq, int* sr,
                              int* scp, int* smlit,
                              int* swinner, bint* sgo,
                              int n) noexcept:
    """Undo all moves from the C state stack."""
    cdef int i, j
    for j in range(n):
        i = n - 1 - j
        cg.undo_move_c(sq[i], sr[i], scp[i], smlit[i], swinner[i], sgo[i])


# ---------------------------------------------------------------------------
# backprop_cy
# ---------------------------------------------------------------------------

def backprop_cy(tree, leaf, double nn_value):
    """Cython backprop — typed locals for speed."""
    cdef double value_for_mover
    cdef int d, k, sign, local

    if leaf.is_terminal:
        value_for_mover = leaf.terminal_value
    else:
        value_for_mover = -nn_value

    if not leaf.path:
        return

    _pd = leaf.pair_depths
    d = _pd[len(_pd) - 1]

    for (node, action_idx), k in zip(leaf.path, leaf.pair_depths):
        sign = 1 if (d - k) % 2 == 0 else -1

        local = node.action_map[action_idx]
        node.visits[local] += 1
        node.values[local] += sign * value_for_mover
        node.visit_count += 1
