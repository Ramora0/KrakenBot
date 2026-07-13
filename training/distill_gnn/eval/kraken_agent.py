"""In-process KrakenBot HexResNet agent for the head-to-head referee.

Differs from mcts_bot.MCTSBot in two ways that matter for evaluation:
  * loads via HexResNet.from_checkpoint, so WDL vs scalar value heads and the
    legacy clamp pair head are handled correctly (MCTSBot's strict=False load
    silently leaves a random value head for mismatched checkpoints);
  * reads the value through model.expected_value (works for both WDL [B,3] and
    scalar [B,1] heads), and supports a wall-clock per-turn budget so both
    engines can be matched at ~equal time controls.

The MCTS itself reuses KrakenBot's own tree (mcts.tree), infinite-grid / zero-pad
path — the same search MCTSBot uses.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import HexGame, Player
from model.resnet import HexResNet, BOARD_SIZE


class _ScalarValueModel(nn.Module):
    """Adapt a HexResNet to the scalar-value interface KrakenBot's MCTS expects:
    forward returns (value[B,1], pair_logits, moves_left, chain) where value is
    the expected scalar (works for both WDL [B,3] and legacy scalar [B,1] heads)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, mask=None):
        v, pair, ml, chain = self.base(x, mask) if mask is not None else self.base(x)
        return self.base.expected_value(v).reshape(-1, 1), pair, ml, chain

    def set_padding_mode(self, mode):
        self.base.set_padding_mode(mode)


def _virtual_loss(leaf, sign_mult):
    """Apply (sign_mult=+1) or revert (sign_mult=-1) a virtual loss along
    leaf.path: pretend the playout was lost for the leaf's mover so parallel
    selections within one eval batch diverge instead of picking the same leaf."""
    if not leaf.path:
        return
    d = leaf.pair_depths[-1]
    for (node, a), k in zip(leaf.path, leaf.pair_depths):
        sign = 1 if (d - k) % 2 == 0 else -1
        local = node.action_map[a]
        node.visits[local] += sign_mult
        node.values[local] += sign_mult * sign * -1.0
        node.visit_count += sign_mult


class KrakenAgent:
    def __init__(self, model_path, n_sims=200, time_budget_ms=None, device=None,
                 max_sims=200000, name=None, eval_batch=12, log_temp=None):
        self.n_sims = n_sims
        self.time_budget_ms = time_budget_ms
        self.max_sims = max_sims
        self.eval_batch = max(1, int(eval_batch))
        self.name = name or model_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.base = HexResNet.from_checkpoint(model_path, map_location=self.device)
        if log_temp is not None:
            # Inference-time softening of the pair policy: pair logits are
            # cosine sims in [-1,1] * exp(log_temp) (trained value ~3.22 -> x25,
            # very peaked). Lower log_temp -> flatter priors -> PUCT leans more
            # on search value. Requires the normalized-QK head (not legacy).
            if bool(self.base.pair_head.legacy_clamp):
                raise ValueError("log_temp override needs the non-legacy pair head")
            with torch.no_grad():
                self.base.pair_head.log_temp.fill_(float(log_temp))
        self.value_head = self.base.value_head
        self.model = _ScalarValueModel(self.base).to(self.device).eval()
        self.last_sims = 0

    # board: {(q,r): int(1|2)}, current_player int, moves_remaining int
    def choose(self, board_int, current_player, moves_remaining):
        game = HexGame(win_length=6)
        game.board = {(q, r): Player(v) for (q, r), v in board_int.items()}
        game.current_player = Player(int(current_player))
        game.moves_left_in_turn = int(moves_remaining)
        game.move_count = len(board_int)
        game.winner = Player.NONE
        game.game_over = False
        moves = self._get_move(game)
        return [(int(q), int(r)) for (q, r) in moves]

    @torch.no_grad()
    def _get_move(self, game):
        from mcts.tree import (
            create_tree_dynamic, select_leaf, expand_and_backprop,
            maybe_expand_leaf, select_move_pair, select_single_move,
        )
        self._nodes = 0
        if not game.board:
            return [(0, 0)]

        self.model.set_padding_mode('zeros')
        tree, off_q, off_r = create_tree_dynamic(
            game, self.model, self.device, add_noise=False, min_size=BOARD_SIZE)
        proxy_game = _proxy(game, off_q, off_r)
        self._nodes = 1

        sims = 0
        if self.time_budget_ms is not None:
            deadline = time.perf_counter() + self.time_budget_ms / 1000.0
            while time.perf_counter() < deadline and sims < self.max_sims:
                sims += self._batch_sims(tree, proxy_game, select_leaf,
                                         expand_and_backprop, maybe_expand_leaf)
        else:
            while sims < self.n_sims:
                sims += self._batch_sims(tree, proxy_game, select_leaf,
                                         expand_and_backprop, maybe_expand_leaf,
                                         limit=self.n_sims - sims)
        self.last_sims = sims
        self.last_root_value = tree.root_value
        self.model.set_padding_mode('circular')

        if game.moves_left_in_turn == 1:
            gq, gr = select_single_move(tree)
            return [(gq - off_q, gr - off_r)]
        (g1q, g1r), (g2q, g2r) = select_move_pair(tree, temperature=0.1)
        return [(g1q - off_q, g1r - off_r), (g2q - off_q, g2r - off_r)]

    def _leaf_planes(self, tree, leaf):
        planes = tree.root_planes.clone()
        if leaf.player_flipped:
            planes = planes.flip(0)
        for gq, gr, ch in leaf.deltas:
            actual_ch = (1 - ch) if leaf.player_flipped else ch
            planes[actual_ch, gq, gr] = 1.0
        return planes

    def _batch_sims(self, tree, proxy_game, select_leaf, expand_and_backprop,
                    maybe_expand_leaf, limit=None):
        """Collect up to eval_batch leaves under virtual loss, evaluate them in
        ONE forward, then revert + backprop. Terminal leaves backprop
        immediately. Returns the number of sims performed."""
        budget = self.eval_batch if limit is None else min(self.eval_batch, limit)
        done = 0
        leaves = []
        for _ in range(budget):
            leaf = select_leaf(tree, proxy_game)
            if leaf.is_terminal:
                expand_and_backprop(tree, leaf, 0.0)
                done += 1
                continue
            _virtual_loss(leaf, +1)
            leaves.append(leaf)
        if not leaves:
            return max(done, 1)

        x = torch.stack([self._leaf_planes(tree, leaf) for leaf in leaves]
                        ).to(self.device)
        value, pair_logits, _, _ = self.model(x)  # scalar [B,1] via adapter
        for i, leaf in enumerate(leaves):
            _virtual_loss(leaf, -1)
            nn_val = value[i, 0].item()
            expand_and_backprop(tree, leaf, nn_val)
            if leaf.needs_expansion:
                logits = pair_logits[i]
                flat = logits.reshape(-1)
                top_raw, top_idxs = flat.topk(min(200, flat.shape[0]))
                top_vals = F.softmax(top_raw, dim=0)
                marginal_logits = logits.logsumexp(dim=-1)
                marginal = F.softmax(marginal_logits, dim=0).cpu()
                maybe_expand_leaf(tree, leaf, marginal, top_idxs.cpu(),
                                  top_vals.cpu(), nn_value=nn_val)
            done += 1
            self._nodes += 1
        return done


def _proxy(game, off_q, off_r):
    # Infinite-grid HexGame (NOT ToroidalHexGame): the dynamic tree can exceed
    # the 25-torus, and wrapping there both collides cells and invents bogus
    # seam wins. The eval agent runs the net with zero padding, so true
    # infinite-grid semantics are the correct match.
    p = HexGame(win_length=game.win_length)
    p.board = {}
    for (rq, rr), player in game.board.items():
        p.board[(rq + off_q, rr + off_r)] = player
    p.current_player = game.current_player
    p.moves_left_in_turn = game.moves_left_in_turn
    p.move_count = game.move_count
    p.winner = game.winner
    p.game_over = game.game_over
    return p
