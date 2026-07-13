"""Teacher labeler: run the hexo-strix (hexo-a0) GNN + Gumbel MCTS on KrakenBot
positions and return distillation targets.

MUST be run with the hexo-strix venv python (it imports hexo_rs + hexo_a0):

    C:/Users/Lee/coding/python/AI/hexo-strix/.venv/Scripts/python.exe \
        -m training.distill_gnn.teacher   # from the KrakenBot project root

Coordinate parity (validated 2026-07-12):
  * KrakenBot and hexo share the game: A opens at (0,0), then players alternate
    placing 2 stones; 6-in-a-row along a hex axis wins. Every KrakenBot position
    has (0,0)=Player.A, matching hexo's mandatory P1@(0,0) seed.
  * So the board->GameState map is a pure relabel (A->"P1", B->"P2") with NO
    translation. from_state() treats the (0,0)=P1 entry as its redundant seed.
  * Axial (q,r) is used identically on both sides; identity move mapping is
    confirmed by the winning-move parity check in check_parity.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

# --- make KrakenBot's pure-python `game` importable (for Player enum / rules) ---
_KRAKEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _KRAKEN_ROOT not in sys.path:
    sys.path.insert(0, _KRAKEN_ROOT)

# --- make the hexo_a0 workspace package importable (not pip-installed) -------
_HEXO_STRIX_ROOT = os.environ.get(
    "HEXO_STRIX_ROOT", r"C:/Users/Lee/coding/python/AI/hexo-strix")
_HEXO_A0_SRC = os.path.join(_HEXO_STRIX_ROOT, "hexo-a0", "src")
if os.path.isdir(_HEXO_A0_SRC) and _HEXO_A0_SRC not in sys.path:
    sys.path.insert(0, _HEXO_A0_SRC)

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_axis_graph, game_to_axis_graph_batch, game_to_graph, game_to_graph_batch
from hexo_a0.head_to_head import load_checkpoint

# KrakenBot player ints (game.Player): A=1, B=2
KRAKEN_A = 1
KRAKEN_B = 2


def _player_int_to_hexo(p: int) -> str:
    return "P1" if int(p) == KRAKEN_A else "P2"


@dataclass
class Labeled:
    """Teacher targets for one pre-turn KrakenBot position."""
    best_pair: list[tuple[int, int]]      # teacher's chosen two stones, axial (q,r)
    pi1: list[tuple[tuple[int, int], float]]  # improved policy over stone-1 cells
    value: float                          # teacher net value, current-player POV, [-1,1]


@dataclass
class LabeledJoint:
    """Full 2-stone policy targets for one pre-turn position.

    value : teacher net value (current-player POV).
    pi1   : top-32 first-move marginal  [((q,r), p), ...]  (== teacher pi_net).
    joint : for each of the top-k first moves a_i:
                (a_i coord, pi1(a_i), pi2_top_m)
            where pi2_top_m = [((q,r), p), ...] is pi_net over the SECOND stone
            evaluated at P+a_i (empty list if a_i already wins alone).
    """
    value: float
    pi1: list[tuple[tuple[int, int], float]]
    joint: list[tuple[tuple[int, int], float, list[tuple[tuple[int, int], float]]]]


class Teacher:
    """Loads the hexo-a0 checkpoint and labels KrakenBot positions."""

    def __init__(self, ckpt_path: str, device: str | None = None,
                 n_simulations: int = 128, m_actions: int = 16,
                 c_visit: int = 50, c_scale: float = 1.0):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        loaded = load_checkpoint(ckpt_path, self.device)
        self.model = loaded.model
        self.mc: ModelConfig = loaded.model_config
        self.train_steps = loaded.train_steps
        self.model.eval()

        gc = None
        # Rebuild a GameConfig from the checkpoint if present (win_length etc.)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gcfg = raw.get("game_config", {}) or {}
        self.game_config = hexo_rs.GameConfig(
            win_length=gcfg.get("win_length", 6),
            placement_radius=gcfg.get("placement_radius", 6),
            max_moves=gcfg.get("max_moves", 300),
        )

        self.mcts_config = hexo_rs.MCTSConfig(
            n_simulations=n_simulations, m_actions=m_actions,
            c_visit=c_visit, c_scale=c_scale,
        )

        # Variable-sized graph batches fragment the CUDA caching allocator, which
        # otherwise balloons toward full VRAM over a long labeling run. Clear it
        # on a cadence (same idea as hexo self-play's CacheClearGate).
        self._eval_calls = 0
        self._empty_cache_every = 16

        # Graph builders matched to the checkpoint's training config.
        prune = getattr(self.mc, "prune_empty_edges", False)
        threat = getattr(self.mc, "threat_features", False)
        rel = getattr(self.mc, "relative_stone_encoding", False)
        if getattr(self.mc, "graph_type", "hex") == "axis":
            self._graph_fn = lambda g: game_to_axis_graph(
                g, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
            self._graph_batch_fn = lambda gs: game_to_axis_graph_batch(
                gs, prune_empty_edges=prune, threat_features=threat, relative_stones=rel)
        else:
            self._graph_fn = lambda g: game_to_graph(g, threat_features=threat, relative_stones=rel)
            self._graph_batch_fn = lambda gs: game_to_graph_batch(gs, threat_features=threat, relative_stones=rel)

    # -- eval_fn matching the contract in self_play.py ------------------------
    def _eval_fn(self, states):
        from torch_geometric.data import Batch
        data_list = (self._graph_batch_fn(states) if len(states) > 1
                     else [self._graph_fn(states[0])])
        batch = Batch.from_data_list(data_list).to(self.device)
        with torch.inference_mode():
            policy_logits_list, values = self.model.forward_batch(batch)
        logits_list = [lg.tolist() for lg in policy_logits_list]
        values_list = [float(v.item()) for v in values]
        del batch, data_list, policy_logits_list, values
        self._eval_calls += 1
        if (self.device.type == "cuda"
                and self._eval_calls % self._empty_cache_every == 0):
            torch.cuda.empty_cache()
        return logits_list, values_list

    def _eval_tensors(self, states):
        """Like _eval_fn but returns per-graph GPU logit tensors (no .tolist()).

        Returns (list[Tensor on device], list[float values]).
        """
        from torch_geometric.data import Batch
        data_list = (self._graph_batch_fn(states) if len(states) > 1
                     else [self._graph_fn(states[0])])
        batch = Batch.from_data_list(data_list).to(self.device)
        with torch.inference_mode():
            policy_logits_list, values = self.model.forward_batch(batch)
        values_list = [float(v.item()) for v in values]
        del batch, data_list
        self._eval_calls += 1
        if (self.device.type == "cuda"
                and self._eval_calls % self._empty_cache_every == 0):
            torch.cuda.empty_cache()
        return policy_logits_list, values_list

    @staticmethod
    def _topk_coords(logits, coords, kk):
        """softmax(logits) on device, return top-kk [(coord, prob)] (kk clamped)."""
        probs = torch.softmax(logits.float(), dim=-1)
        kk = min(kk, probs.numel())
        pv, pi = torch.topk(probs, kk)
        return [(coords[i], float(p)) for p, i in zip(pv.tolist(), pi.tolist())]

    def _eval_topk_sub(self, states, kk, max_fwd):
        """Forward `states` in sub-batches of max_fwd; return per-state top-kk
        [(coord, prob)] and values. Bounds GPU memory regardless of len(states).

        Sub-batching is essential: one forward over k*chunk graphs (e.g. 4096)
        allocates several GB of attention and OOMs — cap it here.
        """
        out_topk, out_vals = [], []
        for i in range(0, len(states), max_fwd):
            sub = states[i:i + max_fwd]
            logits, vals = self._eval_tensors(sub)
            for st, lg in zip(sub, logits):
                out_topk.append(self._topk_coords(lg, sorted(st.legal_moves()), kk))
            out_vals.extend(vals)
        return out_topk, out_vals

    def label_batch_topk(self, positions, k=8, m=16, marg_k=32, max_fwd=384):
        """Distill the full 2-stone policy: pi1 marginal + pi2 over top-k first moves.

        Forwards are sub-batched at `max_fwd` graphs so peak GPU memory stays
        bounded no matter how large k or the caller's chunk is.
        """
        states = [self.make_state(b, cp, 2) for (b, cp) in positions]
        n = len(states)

        # First forward: marg_k for pi1, and reuse for the top-k first moves.
        pi1, values = self._eval_topk_sub(states, marg_k, max_fwd)
        topk_first = [row[:k] for row in pi1]   # pi1 already sorted desc

        # Build every P+a_i; track ownership so pi2 maps back to positions.
        second, owner = [], []   # owner[j] = (position_idx, a_coord, pa, second_idx|None)
        for i, moves in enumerate(topk_first):
            for (a, pa) in moves:
                st2 = states[i].clone()
                st2.apply_move(a[0], a[1])
                if st2.is_terminal() or st2.legal_move_count() == 0:
                    owner.append((i, a, pa, None))   # a wins alone -> no pi2
                else:
                    owner.append((i, a, pa, len(second)))
                    second.append(st2)

        pi2_topk = self._eval_topk_sub(second, m, max_fwd)[0] if second else []

        joint = [[] for _ in range(n)]
        for (i, a, pa, sidx) in owner:
            joint[i].append((a, pa, [] if sidx is None else pi2_topk[sidx]))

        return [LabeledJoint(value=values[i], pi1=pi1[i], joint=joint[i]) for i in range(n)]

    # -- conversion -----------------------------------------------------------
    def make_state(self, board: dict, current_player: int, moves_remaining: int):
        """KrakenBot board {(q,r): player_int} -> hexo_rs.GameState.

        `board` values are ints (1=A, 2=B) or KrakenBot Player enums.
        """
        stones = []
        for (q, r), p in board.items():
            pv = p.value if hasattr(p, "value") else int(p)
            stones.append(((int(q), int(r)), _player_int_to_hexo(pv)))
        cp = _player_int_to_hexo(
            current_player.value if hasattr(current_player, "value") else current_player)
        return hexo_rs.GameState.from_state(stones, cp, int(moves_remaining), self.game_config)

    def net_value(self, state) -> float:
        """Raw network value for the current player to move at `state`."""
        _logits, values = self._eval_fn([state])
        return values[0]

    def label_batch_raw(self, positions: list[tuple[dict, int]]) -> list[Labeled]:
        """Distill the teacher's RAW policy+value — one forward, no MCTS.

        For each pre-turn position: soft target pi1 = softmax(policy_head) over
        legal cells; value = v_net; hard pair = greedy (argmax) stone1 then a
        second batched forward for the greedy stone2. Everything is forward
        passes, so this is GPU-bound and ~100x faster than the MCTS labeler.
        """
        import math

        def _softmax(xs):
            m = max(xs)
            es = [math.exp(x - m) for x in xs]
            s = sum(es) or 1.0
            return [e / s for e in es]

        states = [self.make_state(b, cp, 2) for (b, cp) in positions]
        n = len(states)
        logits_list, values = self._eval_fn(states)

        pi1_list: list[list[tuple[tuple[int, int], float]]] = []
        stone1: list[tuple[int, int]] = []
        for st, lg in zip(states, logits_list):
            coords = sorted(st.legal_moves())
            probs = _softmax(lg)
            pi1_list.append(list(zip(coords, probs)))
            stone1.append(coords[max(range(len(probs)), key=probs.__getitem__)])

        stone2: list[tuple[int, int] | None] = [None] * n
        pend, pidx = [], []
        for i, (st, s1) in enumerate(zip(states, stone1)):
            st2 = st.clone()
            st2.apply_move(s1[0], s1[1])
            if st2.is_terminal() or st2.legal_move_count() == 0:
                continue
            pend.append(st2)
            pidx.append(i)
        if pend:
            lg2_list, _v2 = self._eval_fn(pend)
            for i, st2, lg in zip(pidx, pend, lg2_list):
                coords = sorted(st2.legal_moves())
                stone2[i] = coords[max(range(len(lg)), key=lg.__getitem__)]

        out = []
        for i in range(n):
            pair = [stone1[i]] + ([stone2[i]] if stone2[i] is not None else [])
            out.append(Labeled(best_pair=pair, pi1=pi1_list[i], value=values[i]))
        return out

    def label_batch(self, positions: list[tuple[dict, int]],
                    seed: int | None = None) -> list[Labeled]:
        """Label many pre-turn positions at once (fused GPU leaf evals).

        positions: list of (board, current_player). Each is assumed pre-turn
        with 2 stones to place. Boards already won should be filtered by the
        caller. Uses argmax(pi') for deterministic hard-pair targets.
        """
        states = [self.make_state(b, cp, 2) for (b, cp) in positions]
        n = len(states)

        # Value (raw net) for every root in one forward.
        _logits, values = self._eval_fn(states)

        # Stone 1: one batched search across all roots.
        r1 = hexo_rs.batched_gumbel_mcts(states, self._eval_fn, self.mcts_config, seed=seed)
        pi1_list: list[list[tuple[tuple[int, int], float]]] = []
        stone1: list[tuple[int, int]] = []
        for st, (_action, pol) in zip(states, r1):
            coords = sorted(st.legal_moves())
            pairs = list(zip(coords, [float(p) for p in pol]))
            pi1_list.append(pairs)
            best = max(pairs, key=lambda x: x[1])[0]
            stone1.append(best)

        # Apply stone 1; split into those needing a 2nd search vs. stone-1 wins.
        stone2: list[tuple[int, int] | None] = [None] * n
        pending_states = []
        pending_idx = []
        for i, (st, s1) in enumerate(zip(states, stone1)):
            st2 = st.clone()
            st2.apply_move(s1[0], s1[1])
            # Skip terminal OR no-legal-move states: batched_gumbel_mcts panics
            # on a node with no legal actions (stone-1 win, or radius saturation).
            if st2.is_terminal() or st2.legal_move_count() == 0:
                continue
            pending_states.append(st2)
            pending_idx.append(i)

        if pending_states:
            r2 = hexo_rs.batched_gumbel_mcts(
                pending_states, self._eval_fn, self.mcts_config, seed=seed)
            for i, st2, (_a, pol) in zip(pending_idx, pending_states, r2):
                coords = sorted(st2.legal_moves())
                best = max(zip(coords, [float(p) for p in pol]), key=lambda x: x[1])[0]
                stone2[i] = best

        out = []
        for i in range(n):
            pair = [stone1[i]] + ([stone2[i]] if stone2[i] is not None else [])
            out.append(Labeled(best_pair=pair, pi1=pi1_list[i], value=values[i]))
        return out

    def label_position(self, board: dict, current_player: int,
                       moves_remaining: int = 2, seed: int | None = None) -> Labeled:
        """Full teacher targets for a pre-turn position.

        Runs a Gumbel MCTS search per stone (2 plies) to get the chosen pair and
        the stone-1 improved policy, plus a raw net value at the root.
        """
        state = self.make_state(board, current_player, moves_remaining)
        value = self.net_value(state)

        best_pair: list[tuple[int, int]] = []
        pi1: list[tuple[tuple[int, int], float]] = []
        cur = state
        for ply in range(moves_remaining):
            if cur.is_terminal():
                break
            action, improved_policy, _visits = hexo_rs.gumbel_mcts_with_stats(
                cur, self._eval_fn, self.mcts_config, seed=seed)
            if ply == 0:
                coords = sorted(cur.legal_moves())
                pi1 = list(zip(coords, [float(p) for p in improved_policy]))
            best_pair.append((int(action[0]), int(action[1])))
            cur = cur.clone()
            cur.apply_move(action[0], action[1])

        return Labeled(best_pair=best_pair, pi1=pi1, value=value)


if __name__ == "__main__":
    # Smoke test: load teacher, label the empty-ish opening, print shapes.
    ckpt = os.environ.get("TEACHER_CKPT",
                          r"C:/Users/Lee/OneDrive/Desktop/checkpoint_00237000.pt")
    t = Teacher(ckpt, n_simulations=64, m_actions=16)
    print(f"Loaded teacher: {t.train_steps} steps, graph={t.mc.graph_type}, "
          f"device={t.device}")
    # P2 to move after A's (0,0) opening, 2 stones to place.
    board = {(0, 0): KRAKEN_A}
    lab = t.label_position(board, current_player=KRAKEN_B, moves_remaining=2, seed=0)
    print(f"opening: value={lab.value:+.3f}  best_pair={lab.best_pair}  "
          f"pi1_top={sorted(lab.pi1, key=lambda x: -x[1])[:3]}")
