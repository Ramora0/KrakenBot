"""MCTS engine for HexTicTacToe with multi-ply tree search.

Root positions use a two-level tree (stone_1 -> stone_2) with full conditional
priors from the NN's pair attention matrix and ALL empty cells as candidates.
Non-root positions use flat top-K pair selection for efficiency.  The tree
grows deeper as simulations accumulate: after EXPAND_VISITS visits to a pair,
a child PosNode is created for the resulting position.

Children are stored as parallel Python lists for fast PUCT selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from game import Player
from model.resnet import BOARD_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUCT_C = 0.8             # low exploration for tactical connect-6
FPU_REDUCTION = 0.25     # parent-relative first play urgency (KataGo-style)
EXPAND_VISITS = 1       # expand on first visit (standard AlphaZero)
MAX_DEPTH = 50          # safety limit on pair-move depth
NON_ROOT_TOP_K = 50     # candidate pairs for non-root flat selection
DIRICHLET_ALPHA = 0.06  # slightly higher for more uniform noise across candidates
DIRICHLET_FRAC = 0.10   # reduced from 0.25; high noise causes missed defenses
N_CELLS = BOARD_SIZE * BOARD_SIZE
_ALL_CELLS = frozenset((q, r) for q in range(BOARD_SIZE) for r in range(BOARD_SIZE))

# Distance gating: candidate cells must be within max_cand_dist of an
# existing stone.  Starts at 2 (matching minimax teacher) and ramps up
# over self-play rounds so the model gradually learns longer-range play.
DEFAULT_MAX_CAND_DIST = None  # None = no limit (legacy behaviour)
DIST_GATE_BASE = 8           # starting max distance
DIST_GATE_RAMP_ROUNDS = 50   # rounds per +1 distance step

# ---------------------------------------------------------------------------
# Precomputed neighbor table (built once at import time)
# ---------------------------------------------------------------------------

def _build_neighbor_table():
    """Precompute NEIGHBORS_WITHIN[dist][cell_idx] = frozenset of cell indices."""
    max_d = BOARD_SIZE // 2  # max meaningful distance on torus
    N = BOARD_SIZE
    table = [None] * (max_d + 1)  # table[0] unused

    # First compute pairwise distances
    dist_from = {}  # dist_from[idx] = {other_idx: dist, ...}
    for idx in range(N_CELLS):
        q1, r1 = idx // N, idx % N
        neighbors_by_dist = {}
        for other in range(N_CELLS):
            if other == idx:
                continue
            q2, r2 = other // N, other % N
            dq = min(abs(q1 - q2), N - abs(q1 - q2))
            dr = min(abs(r1 - r2), N - abs(r1 - r2))
            ds = abs((-q1 - r1) - (-q2 - r2))
            ds = min(ds % N, N - ds % N)
            d = max(dq, dr, ds)
            if d <= max_d:
                neighbors_by_dist[other] = d
        dist_from[idx] = neighbors_by_dist

    # Build cumulative sets: within[d][idx] = all cells within distance d
    for d in range(1, max_d + 1):
        level = {}
        for idx in range(N_CELLS):
            level[idx] = frozenset(
                other for other, od in dist_from[idx].items() if od <= d
            )
        table[d] = level

    return table


_NEIGHBORS_WITHIN = _build_neighbor_table()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MCTSNode:
    """MCTS node with list-based children for fast PUCT.

    Children stored as parallel Python lists for minimal per-element access
    overhead.  At root PosNodes, level2 dict maps stone_1 action indices to
    child MCTSNode objects for stone_2 selection.  At non-root PosNodes,
    actions are encoded pair indices (s1 * N_CELLS + s2).
    """
    __slots__ = ('visit_count', 'n', 'actions', 'priors', 'visits', 'values',
                 'terminals', 'term_vals', 'action_map', 'level2',
                 '_has_terminal')

    def __init__(self):
        self.visit_count: int = 0
        self.n: int = 0
        self.actions: list | None = None     # [K] int
        self.priors: list | None = None      # [K] float
        self.visits: list | None = None      # [K] int
        self.values: list | None = None      # [K] float
        self.terminals: list | None = None   # [K] bool
        self.term_vals: list | None = None   # [K] float
        self.action_map: dict | None = None  # action_idx -> local_idx
        self.level2: dict | None = None      # action_idx -> MCTSNode
        self._has_terminal: bool = False


class PosNode:
    """Expanded position in the multi-ply MCTS tree.

    Root nodes (is_root=True) use two-level decomposition:
      move_node selects stone_1, move_node.level2[s1] selects stone_2.
      children maps (s1_idx, s2_idx) -> child PosNode.

    Non-root nodes (is_root=False) use flat pair selection:
      move_node selects an encoded pair (s1*N_CELLS+s2).
      children maps pair_action_idx -> child PosNode.
    """
    __slots__ = ('move_node', 'children', '_marginal', 'player', 'value',
                 'is_root')

    def __init__(self):
        self.move_node: MCTSNode = MCTSNode()
        self.children: dict | None = None     # pair_key -> PosNode
        self._marginal: torch.Tensor | None = None  # [N_CELLS]
        self.player: Player | None = None
        self.value: float = 0.0
        self.is_root: bool = False


def _init_node_children(node: MCTSNode, actions_priors: list[tuple[int, float]]):
    """Initialize list-based children on a node from (action, prior) pairs."""
    n = len(actions_priors)
    node.n = n
    node.actions = [a for a, _ in actions_priors]
    priors = [p for _, p in actions_priors]
    total = sum(priors)
    if total > 0:
        node.priors = [p / total for p in priors]
    else:
        u = 1.0 / n
        node.priors = [u] * n
    node.visits = [0] * n
    node.values = [0.0] * n
    node.terminals = [False] * n
    node.term_vals = [0.0] * n
    node.action_map = {a: i for i, a in enumerate(node.actions)}


@dataclass
class LeafInfo:
    """Info returned by select_leaf for batched NN eval."""
    path: list[tuple[MCTSNode, int]]  # [(node, action_idx), ...]
    pair_depths: list[int] = field(default_factory=list)  # pair depth per entry
    current_player: Player | None = None
    is_terminal: bool = False
    terminal_value: float = 0.0
    # Delta from root position: cells placed as (q, r, channel)
    # channel 0 = root player's stones, channel 1 = opponent's stones
    deltas: list[tuple[int, int, int]] = field(default_factory=list)
    player_flipped: bool = False  # True if leaf's current_player != root's
    needs_expansion: bool = False
    expand_parent: PosNode | None = None
    expand_pair: object = None  # (s1, s2) tuple for root, int for non-root


@dataclass
class MCTSTree:
    root_pos: PosNode
    pair_probs: torch.Tensor | None = None   # [N, N] for root level-2 expansion
    root_planes: torch.Tensor | None = None  # [2, BOARD_SIZE, BOARD_SIZE]
    root_player: Player | None = None
    root_value: float = 0.0
    root_occupied: frozenset | None = None   # occupied cells at root
    noise_dist_scale: float = 0.0            # for distance-aware exploration noise
    max_cand_dist: int | None = None         # distance gate for candidates
    next_dist_frac: float = 0.0              # interpolation fraction for next ring
    board_width: int = BOARD_SIZE            # width of the board (for dynamic sizing)
    n_cells: int = N_CELLS                   # board_width ** 2


# ---------------------------------------------------------------------------
# Coordinate helpers (torus -- no offsets)
# ---------------------------------------------------------------------------

def _cell_to_idx(q: int, r: int, width: int = BOARD_SIZE) -> int:
    return q * width + r


def _idx_to_cell(idx: int, width: int = BOARD_SIZE) -> tuple[int, int]:
    return idx // width, idx % width


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _hex_dist_torus(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex distance on a BOARD_SIZE×BOARD_SIZE torus."""
    N = BOARD_SIZE
    dq = min(abs(q1 - q2), N - abs(q1 - q2))
    dr = min(abs(r1 - r2), N - abs(r1 - r2))
    # axial: s = -q - r
    ds = abs((-q1 - r1) - (-q2 - r2))
    ds = min(ds % N, N - ds % N)
    return max(dq, dr, ds)


def _get_candidates(game_or_board) -> set[tuple[int, int]]:
    """Return all empty cells on the torus."""
    if hasattr(game_or_board, 'get_occupied_set'):
        return set(_ALL_CELLS - game_or_board.get_occupied_set())
    if hasattr(game_or_board, 'board'):
        return set(_ALL_CELLS - game_or_board.board.keys())
    return set(_ALL_CELLS - game_or_board.keys())


def _nearby_candidates(
    occupied_indices: set[int] | frozenset[int],
    max_dist: int,
    next_dist_frac: float = 0.0,
) -> set[int]:
    """Return cell indices within *max_dist* of any occupied cell (fast lookup).

    When *next_dist_frac* > 0, each cell at exactly max_dist+1 is included
    with that probability — gradual introduction of the next distance ring.
    """
    if not occupied_indices:
        return set()
    table = _NEIGHBORS_WITHIN[max_dist]
    result: set[int] = set()
    for stone_idx in occupied_indices:
        result |= table[stone_idx]
    result -= occupied_indices

    # Gradually introduce next-distance cells
    if next_dist_frac > 0 and max_dist < BOARD_SIZE // 2:
        next_table = _NEIGHBORS_WITHIN[max_dist + 1]
        next_ring: set[int] = set()
        for stone_idx in occupied_indices:
            next_ring |= next_table[stone_idx]
        next_ring -= result
        next_ring -= occupied_indices
        if next_ring:
            for idx in next_ring:
                if np.random.random() < next_dist_frac:
                    result.add(idx)

    return result


# ---------------------------------------------------------------------------
# PUCT (vectorized)
# ---------------------------------------------------------------------------

def _puct_select_py(node: MCTSNode, c: float = PUCT_C,
                    fpu: float = 0.0) -> int:
    """Select child with highest PUCT score. Pure Python fallback.

    *fpu* is the Q-value assigned to unvisited children (First Play Urgency).
    Typically parent_value - FPU_REDUCTION so unvisited moves are pessimistic.
    """
    c_sqrt = c * math.sqrt(node.visit_count)
    best = -1e30
    best_a = -1
    actions = node.actions
    priors = node.priors
    visits = node.visits
    values = node.values
    if node._has_terminal:
        terminals = node.terminals
        term_vals = node.term_vals
        for i in range(node.n):
            vc = visits[i]
            if terminals[i]:
                q = term_vals[i]
            elif vc > 0:
                q = values[i] / vc
            else:
                q = fpu
            s = q + c_sqrt * priors[i] / (1 + vc)
            if s > best:
                best = s
                best_a = actions[i]
    else:
        for i in range(node.n):
            vc = visits[i]
            q = values[i] / vc if vc > 0 else fpu
            s = q + c_sqrt * priors[i] / (1 + vc)
            if s > best:
                best = s
                best_a = actions[i]
    return best_a


# Use Cython version if available, else fall back to Python
try:
    from mcts._puct_cy import puct_select as _puct_select
except ImportError:
    _puct_select = _puct_select_py


# ---------------------------------------------------------------------------
# Dirichlet noise (distance-aware)
# ---------------------------------------------------------------------------

def _min_dist_to_stones_torus(q: int, r: int, occupied, N: int = BOARD_SIZE) -> int:
    """Minimum hex distance from (q, r) to any occupied cell on the torus."""
    min_d = N
    for oq, or_ in occupied:
        raw_dq = q - oq
        raw_dr = r - or_
        for a in (-1, 0, 1):
            dq = raw_dq + a * N
            for b in (-1, 0, 1):
                dr = raw_dr + b * N
                d = max(abs(dq), abs(dr), abs(dq + dr))
                if d < min_d:
                    min_d = d
                    if d <= 1:
                        return d
    return min_d


def _add_exploration_noise(node: MCTSNode, alpha: float | None = None,
                           frac: float = DIRICHLET_FRAC,
                           occupied: frozenset | None = None,
                           noise_dist_scale: float = 0.0):
    """Add Dirichlet noise to priors with distance-based weighting.

    final_prior = (1 - frac) * prior + frac * Dir(alpha)

    When *alpha* is None (default), it scales as 10/n_candidates so the
    noise concentration adapts to the candidate count -- smoother for small
    sets, sparser for large ones.

    Each candidate's noise fraction is scaled by its hex distance to the
    nearest existing stone:
      d <= 2:  weight = 1.0  (full noise, matches distillation radius)
      d > 2:   weight = exp(-(d - 2) / noise_dist_scale)

    noise_dist_scale controls the exponential tail beyond dist 2:
      0.0   -> noise only on cells within dist 2
      1.0   -> dist 3 ≈ 37%, dist 4 ≈ 14%, dist 5 ≈ 5%
      3.0   -> dist 3 ≈ 72%, dist 4 ≈ 51%, dist 5 ≈ 37%
      large -> effectively uniform noise
    """
    if node.actions is None:
        return
    n = node.n
    if alpha is None:
        alpha = max(DIRICHLET_ALPHA, 10.0 / n)
    dirichlet = np.random.dirichlet([alpha] * n)
    priors = node.priors

    if occupied is None or not occupied:
        # No stones on board — apply uniform noise
        keep = 1.0 - frac
        node.priors = [keep * priors[i] + frac * dirichlet[i]
                       for i in range(n)]
        return

    # Per-candidate noise fraction based on distance to nearest stone
    new_priors = [0.0] * n
    for i in range(n):
        q, r = _idx_to_cell(node.actions[i])
        d = _min_dist_to_stones_torus(q, r, occupied)
        if d <= 2:
            w = 1.0
        elif noise_dist_scale <= 0.0:
            w = 0.0
        else:
            w = math.exp(-(d - 2) / noise_dist_scale)
        fi = frac * w
        new_priors[i] = (1.0 - fi) * priors[i] + fi * dirichlet[i]

    # Renormalize
    total = sum(new_priors)
    if total > 0:
        node.priors = [p / total for p in new_priors]
    else:
        node.priors = new_priors


# ---------------------------------------------------------------------------
# Undo helper
# ---------------------------------------------------------------------------

def _undo_all(game, states: list):
    """Undo all moves in reverse order."""
    for q, r, state in reversed(states):
        game.undo_move(q, r, state)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_tree_from_eval(
    game,
    root_value: float,
    pair_probs: torch.Tensor,
    marginal: torch.Tensor,
    root_planes: torch.Tensor,
    add_noise: bool = True,
    noise_dist_scale: float = 0.0,
    max_cand_dist: int | None = DEFAULT_MAX_CAND_DIST,
    next_dist_frac: float = 0.0,
) -> MCTSTree:
    """Build an MCTSTree from pre-computed NN outputs (no model call).

    When *max_cand_dist* is set, only empty cells within that hex distance
    of an existing stone are considered as candidates.  *next_dist_frac*
    controls gradual introduction of the next distance ring.
    """
    pos = PosNode()
    pos.value = root_value
    cp = game.current_player
    if hasattr(cp, 'value'):
        pos.player = cp
    else:
        pos.player = Player(cp)
    pos.is_root = True
    pos._marginal = marginal

    if hasattr(game, 'get_occupied_set'):
        occupied = game.get_occupied_set()
        has_stones = game.move_count > 0
    else:
        occupied = game.board.keys()
        has_stones = bool(game.board)

    occupied_frozen = frozenset(occupied)

    if has_stones:
        if max_cand_dist is not None:
            occ_idx = frozenset(_cell_to_idx(q, r) for q, r in occupied)
            cand_indices = list(_nearby_candidates(
                occ_idx, max_cand_dist, next_dist_frac))
        else:
            occ_set = set(occupied)
            cand_indices = [_cell_to_idx(q, r)
                            for q, r in _ALL_CELLS - occ_set]
    else:
        cand_indices = [_cell_to_idx(BOARD_SIZE // 2, BOARD_SIZE // 2)]

    cand_values = marginal[cand_indices].tolist()
    cand_priors = list(zip(cand_indices, cand_values))
    cand_priors.sort(key=lambda x: x[1], reverse=True)

    _init_node_children(pos.move_node, cand_priors)

    if add_noise:
        _add_exploration_noise(pos.move_node, occupied=occupied_frozen,
                               noise_dist_scale=noise_dist_scale)

    root_player = cp if hasattr(cp, 'value') else Player(cp)
    return MCTSTree(
        root_pos=pos,
        pair_probs=pair_probs,
        root_planes=root_planes,
        root_player=root_player,
        root_value=root_value,
        root_occupied=occupied_frozen,
        noise_dist_scale=noise_dist_scale,
        max_cand_dist=max_cand_dist,
        next_dist_frac=next_dist_frac,
    )


def create_tree(
    game,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
    noise_dist_scale: float = 0.0,
    max_cand_dist: int | None = DEFAULT_MAX_CAND_DIST,
    next_dist_frac: float = 0.0,
) -> MCTSTree:
    """Create a single MCTS tree with one B=1 NN forward pass."""
    from model.resnet import board_to_planes_torus

    planes = board_to_planes_torus(game.board, game.current_player)

    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        value, pair_logits, _, _ = model(x)

    root_value = value[0].item()
    pair_probs = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(
        N_CELLS, N_CELLS).cpu()
    marginal = pair_probs.sum(dim=-1)

    return _build_tree_from_eval(
        game, root_value, pair_probs, marginal, planes, add_noise,
        noise_dist_scale=noise_dist_scale,
        max_cand_dist=max_cand_dist, next_dist_frac=next_dist_frac)


@torch.no_grad()
def create_trees_batched(
    games: list,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
    noise_dist_scale: float = 0.0,
    max_cand_dist: int | None = DEFAULT_MAX_CAND_DIST,
    next_dist_frac: float = 0.0,
) -> list[MCTSTree]:
    """Create trees for multiple games in one batched forward pass."""
    from model.resnet import board_to_planes_torus

    B = len(games)
    if B == 0:
        return []

    # All boards are fixed size -- just stack
    model_dtype = next(model.parameters()).dtype
    batch = torch.zeros(B, 2, BOARD_SIZE, BOARD_SIZE, dtype=model_dtype)
    for i, game in enumerate(games):
        if hasattr(game, 'to_planes_tensor'):
            batch[i] = game.to_planes_tensor()
        else:
            batch[i] = board_to_planes_torus(game.board, game.current_player)

    batch = batch.to(device)
    values, pair_logits, _, _ = model(batch)

    trees = []
    for i, game in enumerate(games):
        root_value = values[i].item()
        pp = F.softmax(pair_logits[i].reshape(-1), dim=0).reshape(
            N_CELLS, N_CELLS).cpu()
        mg = pp.sum(dim=-1)
        tree = _build_tree_from_eval(
            game, root_value, pp, mg, batch[i].cpu(), add_noise,
            noise_dist_scale=noise_dist_scale,
            max_cand_dist=max_cand_dist, next_dist_frac=next_dist_frac)
        trees.append(tree)

    return trees


@torch.no_grad()
def create_tree_dynamic(
    game,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = False,
    min_size: int = BOARD_SIZE,
    margin: int = 6,
) -> tuple[MCTSTree, int, int]:
    """Create an MCTS tree for an infinite-grid game with dynamic board size.

    Computes a bounding box around stones, pads by *margin*, clamps to
    at least *min_size* × *min_size*, and runs the model on that grid.
    No distance gating is applied (evaluation uses full candidate set).

    Returns (tree, offset_q, offset_r) where offsets map grid coords back
    to real coords via real_q = grid_q - offset_q.
    """
    from model.resnet import board_to_planes

    planes, off_q, off_r, h, w = board_to_planes(
        game.board, game.current_player, min_size=min_size, margin=margin)

    bw = w  # board width for this tree
    nc = h * w  # n_cells

    x = planes.unsqueeze(0).to(device)
    value, pair_logits, _, _ = model(x)

    root_value = value[0].item()
    pair_probs = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(nc, nc).cpu()
    marginal = pair_probs.sum(dim=-1)

    # Build candidate set: all cells in grid that are empty
    occupied_grid = set()
    for (q, r) in game.board:
        occupied_grid.add((q + off_q, r + off_r))

    all_cells = {(gq, gr) for gq in range(h) for gr in range(w)}
    cands = all_cells - occupied_grid

    cand_indices = [gq * bw + gr for gq, gr in cands]
    cand_values = marginal[cand_indices].tolist()
    cand_priors = list(zip(cand_indices, cand_values))
    cand_priors.sort(key=lambda x: x[1], reverse=True)

    pos = PosNode()
    pos.value = root_value
    cp = game.current_player
    if hasattr(cp, 'value'):
        pos.player = cp
    else:
        pos.player = Player(cp)
    pos.is_root = True
    pos._marginal = marginal

    _init_node_children(pos.move_node, cand_priors)
    if add_noise:
        _add_exploration_noise(pos.move_node)

    root_player = cp if hasattr(cp, 'value') else Player(cp)
    occupied_frozen = frozenset(occupied_grid)

    tree = MCTSTree(
        root_pos=pos,
        pair_probs=pair_probs,
        root_planes=planes,
        root_player=root_player,
        root_value=root_value,
        root_occupied=occupied_frozen,
        board_width=bw,
        n_cells=nc,
    )
    return tree, off_q, off_r


def compute_max_cand_dist(round_num: int) -> tuple[int | None, float]:
    """Distance gate for the given self-play round.

    Returns (max_dist, next_frac):
      max_dist:   fully-included distance (None = no limit)
      next_frac:  fraction of (max_dist+1) cells to include (0.0–1.0)

    Starts at DIST_GATE_BASE (2) and ramps the next ring linearly over
    DIST_GATE_RAMP_ROUNDS, then steps up and repeats.
    """
    d = DIST_GATE_BASE + round_num // DIST_GATE_RAMP_ROUNDS
    frac = (round_num % DIST_GATE_RAMP_ROUNDS) / DIST_GATE_RAMP_ROUNDS
    # Cap at DIST_GATE_BASE (no ramp beyond it)
    if d > DIST_GATE_BASE:
        d = DIST_GATE_BASE
        frac = 0.0
    if d >= BOARD_SIZE // 2:
        return None, 0.0  # no limit — full board
    return d, frac


# ---------------------------------------------------------------------------
# Level-2 expansion (root only)
# ---------------------------------------------------------------------------

def _expand_level2(
    tree: MCTSTree,
    pos: PosNode,
    stone1_idx: int,
    game,
    add_noise: bool = True,
) -> MCTSNode | None:
    """Expand level-2 children for a stone_1 action at root.

    Uses tree.pair_probs for conditional priors.  Candidates are filtered
    by the tree's max_cand_dist setting.
    """
    cond_probs = tree.pair_probs[stone1_idx]  # [N_CELLS]

    # All empty cells except stone_1, filtered by distance
    _bw = tree.board_width
    _nc = tree.n_cells
    if tree.max_cand_dist is not None:
        occ_idx = set(
            _cell_to_idx(q, r, _bw) for q, r in tree.root_occupied
        ) | {stone1_idx}
        cand_indices = list(
            _nearby_candidates(occ_idx, tree.max_cand_dist,
                               tree.next_dist_frac))
    else:
        occ_set = set(_cell_to_idx(q, r, _bw) for q, r in tree.root_occupied)
        occ_set.add(stone1_idx)
        cand_indices = [i for i in range(_nc) if i not in occ_set]

    cand_values = cond_probs[cand_indices].tolist()
    cand_priors = list(zip(cand_indices, cand_values))
    cand_priors.sort(key=lambda x: x[1], reverse=True)

    if not cand_priors:
        return None

    l2_node = MCTSNode()
    _init_node_children(l2_node, cand_priors)

    if add_noise:
        if hasattr(game, 'get_occupied_set'):
            occ = game.get_occupied_set()
        else:
            occ = frozenset(game.board.keys())
        _add_exploration_noise(l2_node, occupied=occ,
                               noise_dist_scale=tree.noise_dist_scale)

    if pos.move_node.level2 is None:
        pos.move_node.level2 = {}
    pos.move_node.level2[stone1_idx] = l2_node
    return l2_node


# ---------------------------------------------------------------------------
# Select leaf (multi-ply)
# ---------------------------------------------------------------------------

def select_leaf(tree: MCTSTree, game) -> LeafInfo:
    """Select a leaf via PUCT, descending through child PosNodes.

    Root (two-level): PUCT stone_1 then PUCT stone_2, check child.
    Non-root (flat):  PUCT selects a complete encoded pair, check child.
    Makes temporary moves on the game, undoes all before returning.
    """
    path: list[tuple[MCTSNode, int]] = []
    pair_depths: list[int] = []
    states: list[tuple[int, int, object]] = []
    deltas: list[tuple[int, int, int]] = []
    pos = tree.root_pos
    depth = 0
    root_cp = tree.root_player
    root_fpu = tree.root_value - FPU_REDUCTION
    _bw = tree.board_width

    while depth < MAX_DEPTH:
        if pos.is_root:
            # ---- Root: two-level (stone_1 -> stone_2) ----

            # Level 1: select stone_1
            s1_idx = _puct_select(pos.move_node, fpu=root_fpu)
            s1_q, s1_r = _idx_to_cell(s1_idx, _bw)

            path.append((pos.move_node, s1_idx))
            pair_depths.append(depth)

            state = game.save_state()
            states.append((s1_q, s1_r, state))
            game.make_move(s1_q, s1_r)

            ch = depth % 2
            deltas.append((s1_q, s1_r, ch))

            # Terminal after stone_1?
            if game.game_over:
                local = pos.move_node.action_map[s1_idx]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all(game, states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Single-move turn (first move of game)?
            if game.moves_left_in_turn == 0:
                cp = game.current_player
                _undo_all(game, states)
                return LeafInfo(
                    path=path, pair_depths=pair_depths,
                    current_player=cp, deltas=deltas,
                    player_flipped=(cp != root_cp))

            # Level 2: expand lazily, select stone_2
            l2_node = (pos.move_node.level2 or {}).get(s1_idx)
            if l2_node is None:
                l2_node = _expand_level2(
                    tree, pos, s1_idx, game, add_noise=(depth == 0))

            if l2_node is None or l2_node.actions is None:
                cp = game.current_player
                _undo_all(game, states)
                return LeafInfo(
                    path=path, pair_depths=pair_depths,
                    current_player=cp, deltas=deltas,
                    player_flipped=(cp != root_cp))

            s2_idx = _puct_select(l2_node, fpu=root_fpu)
            s2_q, s2_r = _idx_to_cell(s2_idx, _bw)

            path.append((l2_node, s2_idx))
            pair_depths.append(depth)

            state = game.save_state()
            states.append((s2_q, s2_r, state))
            game.make_move(s2_q, s2_r)
            deltas.append((s2_q, s2_r, ch))

            # Terminal after stone_2?
            if game.game_over:
                local = l2_node.action_map[s2_idx]
                l2_node.terminals[local] = True
                l2_node.term_vals[local] = 1.0
                l2_node._has_terminal = True
                _undo_all(game, states)
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

            cp = game.current_player
            _undo_all(game, states)
            return LeafInfo(
                path=path, pair_depths=pair_depths,
                current_player=cp, deltas=deltas,
                player_flipped=(cp != root_cp),
                needs_expansion=needs_exp,
                expand_parent=pos, expand_pair=pair_key)

        else:
            # ---- Non-root: flat pair selection ----

            child_fpu = pos.value - FPU_REDUCTION
            pair_action = _puct_select(pos.move_node, fpu=child_fpu)
            _nc = tree.n_cells
            _bw = tree.board_width
            s1_idx = pair_action // _nc
            s2_idx = pair_action % _nc
            s1_q, s1_r = _idx_to_cell(s1_idx, _bw)
            s2_q, s2_r = _idx_to_cell(s2_idx, _bw)

            path.append((pos.move_node, pair_action))
            pair_depths.append(depth)

            ch = depth % 2

            # Make stone_1
            state = game.save_state()
            states.append((s1_q, s1_r, state))
            game.make_move(s1_q, s1_r)
            deltas.append((s1_q, s1_r, ch))

            if game.game_over:
                local = pos.move_node.action_map[pair_action]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all(game, states)
                return LeafInfo(path=path, pair_depths=pair_depths,
                                is_terminal=True, terminal_value=1.0)

            # Make stone_2
            state = game.save_state()
            states.append((s2_q, s2_r, state))
            game.make_move(s2_q, s2_r)
            deltas.append((s2_q, s2_r, ch))

            if game.game_over:
                local = pos.move_node.action_map[pair_action]
                pos.move_node.terminals[local] = True
                pos.move_node.term_vals[local] = 1.0
                pos.move_node._has_terminal = True
                _undo_all(game, states)
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

            cp = game.current_player
            _undo_all(game, states)
            return LeafInfo(
                path=path, pair_depths=pair_depths,
                current_player=cp, deltas=deltas,
                player_flipped=(cp != root_cp),
                needs_expansion=needs_exp,
                expand_parent=pos, expand_pair=pair_action)

    # MAX_DEPTH reached
    cp = game.current_player
    _undo_all(game, states)
    return LeafInfo(
        path=path, pair_depths=pair_depths,
        current_player=cp, deltas=deltas,
        player_flipped=(cp != root_cp))


# ---------------------------------------------------------------------------
# Backprop (multi-ply)
# ---------------------------------------------------------------------------

def expand_and_backprop(
    tree: MCTSTree,
    leaf: LeafInfo,
    nn_value: float,
):
    """Backpropagate a value through the multi-ply path.

    Sign alternates at pair boundaries: within a pair both entries get the
    same sign; across pair boundaries the sign flips.

    value_for_mover = the value from the perspective of whoever placed the
    last stone in the path.
      - terminal: terminal_value (1.0 = that player won)
      - non-terminal: -nn_value (nn evaluates from NEXT player's perspective)

    For path entry at pair_depth k with deepest pair_depth d:
      sign = +1 if (d-k) even, -1 if odd.
    """
    if leaf.is_terminal:
        value_for_mover = leaf.terminal_value
    else:
        value_for_mover = -nn_value

    if not leaf.path:
        return

    d = leaf.pair_depths[-1]

    for (node, action_idx), k in zip(leaf.path, leaf.pair_depths):
        sign = 1 if (d - k) % 2 == 0 else -1

        local = node.action_map[action_idx]
        node.visits[local] += 1
        node.values[local] += sign * value_for_mover
        node.visit_count += 1


# ---------------------------------------------------------------------------
# Child PosNode expansion
# ---------------------------------------------------------------------------

def maybe_expand_leaf(
    tree: MCTSTree,
    leaf: LeafInfo,
    marginal: torch.Tensor,
    top_pair_indices: torch.Tensor,
    top_pair_values: torch.Tensor,
    nn_value: float = 0.0,
):
    """Create a child PosNode at the leaf if expansion conditions are met.

    Args:
        marginal: [N_CELLS] marginalized priors for the leaf position.
        top_pair_indices: [K] indices into flattened N*N pair probs.
        top_pair_values: [K] corresponding probabilities.
        nn_value: NN value estimate from the leaf player's perspective,
                  used as FPU baseline for the child's PUCT selections.
    """
    if not leaf.needs_expansion or leaf.is_terminal:
        return
    if leaf.expand_parent is None:
        return

    parent = leaf.expand_parent
    pair_key = leaf.expand_pair

    # Guard against double-expansion
    if parent.children is not None and pair_key in parent.children:
        return

    # Occupied cells at leaf position
    _bw = tree.board_width
    occupied_idx = {_cell_to_idx(q, r, _bw) for q, r in tree.root_occupied}
    for q, r, _ch in leaf.deltas:
        occupied_idx.add(_cell_to_idx(q, r, _bw))

    # Filter top pairs: exclude occupied cells, self-pairs
    _nc = tree.n_cells
    actions_priors = []
    for idx_val, prob_val in zip(top_pair_indices.tolist(),
                                 top_pair_values.tolist()):
        s1 = idx_val // _nc
        s2 = idx_val % _nc
        if s1 == s2 or s1 in occupied_idx or s2 in occupied_idx:
            continue
        actions_priors.append((idx_val, prob_val))
        if len(actions_priors) >= NON_ROOT_TOP_K:
            break

    if not actions_priors:
        return

    child = PosNode()
    child.player = leaf.current_player
    child.is_root = False
    child._marginal = marginal
    child.value = nn_value

    _init_node_children(child.move_node, actions_priors)
    # No exploration noise at non-root

    if parent.children is None:
        parent.children = {}
    parent.children[pair_key] = child


# ---------------------------------------------------------------------------
# Visit extraction and move selection (root only)
# ---------------------------------------------------------------------------

def get_pair_visits(tree: MCTSTree) -> dict[tuple[int, int], int]:
    """Collect visit counts for (stone1_idx, stone2_idx) pairs at root."""
    visits = {}
    root = tree.root_pos.move_node
    if root.actions is None or root.level2 is None:
        return visits
    for i in range(root.n):
        s1_idx = root.actions[i]
        l2 = root.level2.get(s1_idx)
        if l2 is None or l2.actions is None:
            continue
        for j in range(l2.n):
            vc = l2.visits[j]
            if vc > 0:
                visits[(s1_idx, l2.actions[j])] = vc
    return visits


def get_single_visits(tree: MCTSTree) -> dict[tuple[int, int], int]:
    """Get visit counts for single-move case (pairs with same stone)."""
    visits = {}
    root = tree.root_pos.move_node
    if root.actions is None:
        return visits
    for i in range(root.n):
        vc = root.visits[i]
        if vc > 0:
            a = root.actions[i]
            visits[(a, a)] = vc
    return visits


def select_move_pair(
    tree: MCTSTree,
    temperature: float = 1.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select a (stone1, stone2) move pair based on visit counts.

    Returns ((q1,r1), (q2,r2)) in torus coordinates.
    """
    _bw = tree.board_width
    pair_visits = get_pair_visits(tree)
    if not pair_visits:
        # Fallback: best stone_1 by visits
        root = tree.root_pos.move_node
        best_local = max(range(root.n), key=lambda i: root.visits[i])
        best_s1 = root.actions[best_local]
        s1_cell = _idx_to_cell(best_s1, _bw)
        l2 = root.level2.get(best_s1) if root.level2 else None
        if l2 is not None and l2.actions is not None:
            best_l2 = max(range(l2.n), key=lambda i: l2.visits[i])
            s2_cell = _idx_to_cell(l2.actions[best_l2], _bw)
        else:
            s2_cell = s1_cell
        return s1_cell, s2_cell

    pairs = list(pair_visits.keys())
    counts = torch.tensor([pair_visits[p] for p in pairs], dtype=torch.float32)

    if temperature < 0.05:
        best_idx = counts.argmax().item()
    else:
        logits = counts.log() / temperature
        probs = F.softmax(logits, dim=0)
        best_idx = torch.multinomial(probs, 1).item()

    s1_idx, s2_idx = pairs[best_idx]
    s1_cell = _idx_to_cell(s1_idx, _bw)
    s2_cell = _idx_to_cell(s2_idx, _bw)
    return s1_cell, s2_cell


def select_single_move(tree: MCTSTree) -> tuple[int, int]:
    """Select a single move (for moves_left == 1) from marginalized visits."""
    root = tree.root_pos.move_node
    best_local = max(range(root.n), key=lambda i: root.visits[i])
    return _idx_to_cell(root.actions[best_local], tree.board_width)
