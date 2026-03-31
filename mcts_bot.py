"""MCTS-powered bot for HexTicTacToe.

Wraps the MCTS engine into a Bot subclass. Translates between the real game's
unbounded coordinates and the 25x25 torus used internally by MCTS.
"""

import os

import torch
import torch.nn.functional as F

from bot import Bot
from game import HexGame, ToroidalHexGame, TORUS_SIZE
from mcts.tree import N_CELLS, NON_ROOT_TOP_K
from mcts.model import HexResNet

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "training", "resnet_results", "best.pt"
)

# Anchor: maps real (0,0) to torus center (9,9)
_ANCHOR_Q = TORUS_SIZE // 2
_ANCHOR_R = TORUS_SIZE // 2


class MCTSBot(Bot):
    """Bot that uses MCTS with a neural network for move selection."""

    pair_moves = True

    def __init__(self, time_limit=1.0, model_path=None, n_sims=200,
                 device=None):
        super().__init__(time_limit)
        self._nodes = 0
        self.n_sims = n_sims

        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model = HexResNet()
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _torus_to_real(self, tq, tr):
        return tq - _ANCHOR_Q, tr - _ANCHOR_R

    @torch.no_grad()
    def get_move(self, game: HexGame) -> list[tuple[int, int]]:
        from mcts.tree import (
            create_tree, select_leaf, expand_and_backprop,
            maybe_expand_leaf, select_move_pair, select_single_move,
        )

        self.last_depth = self.n_sims
        self._nodes = 0

        # Empty board: always play (0, 0)
        if not game.board:
            return [(0, 0)]

        # Translate real game to toroidal game
        torus_game = ToroidalHexGame.from_hex_game(
            game, anchor_q=_ANCHOR_Q, anchor_r=_ANCHOR_R)

        # Create tree (1 NN eval)
        tree = create_tree(torus_game, self.model, self.device, add_noise=False)
        self._nodes = 1

        # Run simulations
        for _ in range(self.n_sims):
            leaf = select_leaf(tree, torus_game)
            if leaf.is_terminal:
                expand_and_backprop(tree, leaf, 0.0)
            else:
                # Delta eval from root planes
                planes = tree.root_planes.clone()
                if leaf.player_flipped:
                    planes = planes.flip(0)
                for gq, gr, ch in leaf.deltas:
                    actual_ch = (1 - ch) if leaf.player_flipped else ch
                    planes[actual_ch, gq, gr] = 1.0
                x = planes.unsqueeze(0).to(self.device)
                value, pair_logits, _, _ = self.model(x)
                nn_val = value[0].item()
                expand_and_backprop(tree, leaf, nn_val)

                # Create child PosNode if expansion threshold reached
                if leaf.needs_expansion:
                    logits = pair_logits[0]                       # [N, N]
                    flat = logits.reshape(-1)                     # [N²]
                    top_raw, top_idxs = flat.topk(200)
                    top_vals = F.softmax(top_raw, dim=0)
                    marginal_logits = logits.logsumexp(dim=-1)    # [N]
                    marginal = F.softmax(marginal_logits, dim=0).cpu()
                    maybe_expand_leaf(
                        tree, leaf, marginal, top_idxs.cpu(), top_vals.cpu())

                self._nodes += 1

        # Select move and translate back to real coords
        if game.moves_left_in_turn == 1:
            tq, tr = select_single_move(tree)
            rq, rr = self._torus_to_real(tq, tr)
            return [(rq, rr)]
        else:
            (t1q, t1r), (t2q, t2r) = select_move_pair(tree, temperature=0.1)
            r1 = self._torus_to_real(t1q, t1r)
            r2 = self._torus_to_real(t2q, t2r)
            return [r1, r2]

    def __str__(self):
        return f"MCTSBot({self.n_sims})"
