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
from mcts.model import BOARD_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUCT_C = 2.0            # tighter exploration for 200-sim budget
EXPAND_VISITS = 1       # expand on first visit (standard AlphaZero)
MAX_DEPTH = 50          # safety limit on pair-move depth
NON_ROOT_TOP_K = 50     # candidate pairs for non-root flat selection
DIRICHLET_ALPHA = 0.02  # ~10/N_legal, Connect-6 convention for ~600 candidates
DIRICHLET_FRAC = 0.25   # standard AlphaZero/gomoku noise weight
N_CELLS = BOARD_SIZE * BOARD_SIZE
_ALL_CELLS = frozenset((q, r) for q in range(BOARD_SIZE) for r in range(BOARD_SIZE))


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


# ---------------------------------------------------------------------------
# Coordinate helpers (torus -- no offsets)
# ---------------------------------------------------------------------------

def _cell_to_idx(q: int, r: int) -> int:
    return q * BOARD_SIZE + r


def _idx_to_cell(idx: int) -> tuple[int, int]:
    return idx // BOARD_SIZE, idx % BOARD_SIZE


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _get_candidates(game_or_board) -> set[tuple[int, int]]:
    """Return all empty cells on the torus."""
    if hasattr(game_or_board, 'get_occupied_set'):
        return set(_ALL_CELLS - game_or_board.get_occupied_set())
    if hasattr(game_or_board, 'board'):
        return set(_ALL_CELLS - game_or_board.board.keys())
    return set(_ALL_CELLS - game_or_board.keys())


# ---------------------------------------------------------------------------
# PUCT (vectorized)
# ---------------------------------------------------------------------------

def _puct_select_py(node: MCTSNode, c: float = PUCT_C) -> int:
    """Select child with highest PUCT score. Pure Python fallback."""
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
                q = 0.0
            s = q + c_sqrt * priors[i] / (1 + vc)
            if s > best:
                best = s
                best_a = actions[i]
    else:
        for i in range(node.n):
            vc = visits[i]
            q = values[i] / vc if vc > 0 else 0.0
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
# Dirichlet noise
# ---------------------------------------------------------------------------

def _add_exploration_noise(node: MCTSNode, alpha: float = DIRICHLET_ALPHA,
                           frac: float = DIRICHLET_FRAC):
    """Add Dirichlet noise to priors (standard AlphaZero).

    final_prior = (1 - frac) * prior + frac * Dir(alpha)
    No uniform noise -- it scatters stones across the board in games
    where play should be clustered (gomoku, hex, connect-6).
    """
    if node.actions is None:
        return
    n = node.n
    dirichlet = np.random.dirichlet([alpha] * n)
    keep = 1.0 - frac
    priors = node.priors
    node.priors = [keep * priors[i] + frac * dirichlet[i]
                   for i in range(n)]


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
) -> MCTSTree:
    """Build an MCTSTree from pre-computed NN outputs (no model call).

    Uses ALL empty cells as stone_1 candidates (no top-K pruning).
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

    if has_stones:
        cands = _ALL_CELLS - set(occupied)
    else:
        cands = {(BOARD_SIZE // 2, BOARD_SIZE // 2)}

    # Vectorized: one tensor index + one .tolist() instead of N .item() calls
    cand_indices = [_cell_to_idx(q, r) for q, r in cands]
    cand_values = marginal[cand_indices].tolist()
    cand_priors = list(zip(cand_indices, cand_values))
    cand_priors.sort(key=lambda x: x[1], reverse=True)

    _init_node_children(pos.move_node, cand_priors)

    if add_noise:
        _add_exploration_noise(pos.move_node)

    root_player = cp if hasattr(cp, 'value') else Player(cp)
    return MCTSTree(
        root_pos=pos,
        pair_probs=pair_probs,
        root_planes=root_planes,
        root_player=root_player,
        root_value=root_value,
        root_occupied=frozenset(occupied),
    )


def create_tree(
    game,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> MCTSTree:
    """Create a single MCTS tree with one B=1 NN forward pass."""
    from mcts.model import board_to_planes_torus

    planes = board_to_planes_torus(game.board, game.current_player)

    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        value, pair_logits, _, _ = model(x)

    root_value = value[0].item()
    pair_probs = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(
        N_CELLS, N_CELLS).cpu()
    marginal = pair_probs.sum(dim=-1)

    return _build_tree_from_eval(
        game, root_value, pair_probs, marginal, planes, add_noise)


@torch.no_grad()
def create_trees_batched(
    games: list,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> list[MCTSTree]:
    """Create trees for multiple games in one batched forward pass."""
    from mcts.model import board_to_planes_torus

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
            game, root_value, pp, mg, batch[i].cpu(), add_noise)
        trees.append(tree)

    return trees


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

    Uses tree.pair_probs for conditional priors.  ALL remaining empty cells
    are candidates.
    """
    cond_probs = tree.pair_probs[stone1_idx]  # [N_CELLS]

    # All empty cells except stone_1
    cands = _get_candidates(game)
    cands.discard(_idx_to_cell(stone1_idx))

    # Vectorized: one tensor index + one .tolist() instead of N .item() calls
    cand_indices = [_cell_to_idx(q, r) for q, r in cands]
    cand_values = cond_probs[cand_indices].tolist()
    cand_priors = list(zip(cand_indices, cand_values))
    cand_priors.sort(key=lambda x: x[1], reverse=True)

    if not cand_priors:
        return None

    l2_node = MCTSNode()
    _init_node_children(l2_node, cand_priors)

    if add_noise:
        _add_exploration_noise(l2_node)

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

    while depth < MAX_DEPTH:
        if pos.is_root:
            # ---- Root: two-level (stone_1 -> stone_2) ----

            # Level 1: select stone_1
            s1_idx = _puct_select(pos.move_node)
            s1_q, s1_r = _idx_to_cell(s1_idx)

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

            s2_idx = _puct_select(l2_node)
            s2_q, s2_r = _idx_to_cell(s2_idx)

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

            pair_action = _puct_select(pos.move_node)
            s1_idx = pair_action // N_CELLS
            s2_idx = pair_action % N_CELLS
            s1_q, s1_r = _idx_to_cell(s1_idx)
            s2_q, s2_r = _idx_to_cell(s2_idx)

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
):
    """Create a child PosNode at the leaf if expansion conditions are met.

    Args:
        marginal: [N_CELLS] marginalized priors for the leaf position.
        top_pair_indices: [K] indices into flattened N*N pair probs.
        top_pair_values: [K] corresponding probabilities.
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
    occupied_idx = {_cell_to_idx(q, r) for q, r in tree.root_occupied}
    for q, r, _ch in leaf.deltas:
        occupied_idx.add(_cell_to_idx(q, r))

    # Filter top pairs: exclude occupied cells, self-pairs
    actions_priors = []
    for idx_val, prob_val in zip(top_pair_indices.tolist(),
                                 top_pair_values.tolist()):
        s1 = idx_val // N_CELLS
        s2 = idx_val % N_CELLS
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
    pair_visits = get_pair_visits(tree)
    if not pair_visits:
        # Fallback: best stone_1 by visits
        root = tree.root_pos.move_node
        best_local = max(range(root.n), key=lambda i: root.visits[i])
        best_s1 = root.actions[best_local]
        s1_cell = _idx_to_cell(best_s1)
        l2 = root.level2.get(best_s1) if root.level2 else None
        if l2 is not None and l2.actions is not None:
            best_l2 = max(range(l2.n), key=lambda i: l2.visits[i])
            s2_cell = _idx_to_cell(l2.actions[best_l2])
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
    s1_cell = _idx_to_cell(s1_idx)
    s2_cell = _idx_to_cell(s2_idx)
    return s1_cell, s2_cell


def select_single_move(tree: MCTSTree) -> tuple[int, int]:
    """Select a single move (for moves_left == 1) from marginalized visits."""
    root = tree.root_pos.move_node
    best_local = max(range(root.n), key=lambda i: root.visits[i])
    return _idx_to_cell(root.actions[best_local])
