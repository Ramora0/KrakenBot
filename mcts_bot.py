"""MCTS-powered bot for HexTicTacToe.

Wraps the MCTS engine into a Bot subclass.  For infinite-grid games, uses
zero-padded convolutions and dynamic board sizing.  For torus games (selfplay),
uses the fixed 25×25 torus path with circular padding.
"""

import os

import torch
import torch.nn.functional as F

from bot import Bot
from game import HexGame, ToroidalHexGame, TORUS_SIZE
from model.resnet import HexResNet, BOARD_SIZE

_TORUS_CENTER = TORUS_SIZE // 2

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "training", "resnet_results", "best.pt"
)


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
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
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

    @torch.no_grad()
    def get_move(self, game) -> list[tuple[int, int]]:
        from mcts.tree import (
            create_tree, create_tree_dynamic, select_leaf,
            expand_and_backprop, maybe_expand_leaf,
            select_move_pair, select_single_move,
        )

        self.last_depth = self.n_sims
        self._nodes = 0
        is_torus = isinstance(game, ToroidalHexGame)

        # Empty board: always play center
        if not game.board:
            if is_torus:
                return [(_TORUS_CENTER, _TORUS_CENTER)]
            return [(0, 0)]

        # Create tree — torus path (circular pad) vs infinite grid (zero pad)
        if is_torus:
            tree = create_tree(game, self.model, self.device, add_noise=False)
            proxy_game = game
            off_q = off_r = 0
        else:
            self.model.set_padding_mode('zeros')
            tree, off_q, off_r = create_tree_dynamic(
                game, self.model, self.device, add_noise=False,
                min_size=BOARD_SIZE)
            # Proxy game with grid-mapped coordinates for select_leaf
            proxy_game = ToroidalHexGame(win_length=game.win_length)
            proxy_game.board = {}
            for (rq, rr), player in game.board.items():
                proxy_game.board[(rq + off_q, rr + off_r)] = player
            proxy_game.current_player = game.current_player
            proxy_game.moves_left_in_turn = game.moves_left_in_turn
            proxy_game.move_count = game.move_count
            proxy_game.winner = game.winner
            proxy_game.game_over = game.game_over

        self._nodes = 1

        # Run simulations
        for _ in range(self.n_sims):
            leaf = select_leaf(tree, proxy_game)
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
                    logits = pair_logits[0]
                    flat = logits.reshape(-1)
                    top_raw, top_idxs = flat.topk(min(200, flat.shape[0]))
                    top_vals = F.softmax(top_raw, dim=0)
                    marginal_logits = logits.logsumexp(dim=-1)
                    marginal = F.softmax(marginal_logits, dim=0).cpu()
                    maybe_expand_leaf(
                        tree, leaf, marginal, top_idxs.cpu(), top_vals.cpu(),
                        nn_value=nn_val)

                self._nodes += 1

        self.last_root_value = tree.root_value

        # Restore circular padding if we switched
        if not is_torus:
            self.model.set_padding_mode('circular')

        # Select move and translate back to real coords
        if game.moves_left_in_turn == 1:
            gq, gr = select_single_move(tree)
            if is_torus:
                return [(gq, gr)]
            return [(gq - off_q, gr - off_r)]
        else:
            (g1q, g1r), (g2q, g2r) = select_move_pair(tree, temperature=0.1)
            if is_torus:
                return [(g1q, g1r), (g2q, g2r)]
            return [(g1q - off_q, g1r - off_r),
                    (g2q - off_q, g2r - off_r)]

    def __str__(self):
        return f"MCTSBot({self.n_sims})"
