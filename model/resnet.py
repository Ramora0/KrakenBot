"""ResNet + attention pair policy for HexTicTacToe.

Dual-head model predicting win rate (value) and move PAIR probabilities
(policy) from board positions. Fully convolutional — works at any board size.

Architecture:
  - Conv stem + N residual blocks (GroupNorm, size-independent)
  - Value head: masked global average pooling → FC → 3-way categorical
    distribution over {lose, draw, win}
  - Policy head: bilinear attention over cell embeddings → N×N pair logits
    Symmetrized (order of stones in a pair doesn't matter),
    diagonal masked (can't place both on same cell).
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 25


class ThreatStem(nn.Module):
    """Fixed (non-learned) strix-style threat features from the 2 stone planes.

    For each player p and each cell c: the number of OPEN win windows
    (win_length consecutive cells along one of the 3 hex axes, containing c,
    with zero opponent stones) holding >= n stones of p, for n in {2,3,4,5}.
    8 planes total (4 thresholds x 2 players), scaled by 1/6 per threshold.

    Axes in (q, r) plane coords: (1,0) = dim H, (0,1) = dim W, (1,-1) = the
    anti-diagonal. Zero padding is semantically exact here: the game board is
    infinite, so cells beyond the crop are genuinely open empties.

    Everything is two all-ones convolutions per axis (window sums, then
    window->cell coverage), so it runs fused on GPU inside the forward pass —
    no per-leaf CPU cost in the batched search.
    """

    THRESHOLDS = (2, 3, 4, 5)
    OUT_CHANNELS = 2 * len(THRESHOLDS)

    def __init__(self, win_length=6):
        super().__init__()
        L = self.win_length = win_length
        kv = torch.zeros(1, 1, L, 1); kv[0, 0, :, 0] = 1.0
        kh = torch.zeros(1, 1, 1, L); kh[0, 0, 0, :] = 1.0
        kd = torch.zeros(1, 1, L, L)
        for j in range(L):
            kd[0, 0, j, L - 1 - j] = 1.0
        self.register_buffer("kv", kv, persistent=False)
        self.register_buffer("kh", kh, persistent=False)
        self.register_buffer("kd", kd, persistent=False)

    def _axis_counts(self, P, O, k, pad_w):
        """Per-cell count of open windows containing the cell with >= n stones
        of P, along one axis. The window pass pads by win_length-1 on both ends
        of the axis dims (pad_w, (l,r,t,b)); the coverage pass is then an exact
        valid conv with the same kernel. Returns [B, 4, H, W]."""
        Wp = F.conv2d(F.pad(P, pad_w), k)
        Wo = F.conv2d(F.pad(O, pad_w), k)
        open_w = (Wo == 0).float()
        outs = []
        for n in self.THRESHOLDS:
            hit = (Wp >= n).float() * open_w
            outs.append(F.conv2d(hit, k))
        return torch.cat(outs, dim=1)

    def forward(self, x):
        L = self.win_length - 1
        feats = []
        for P, O in ((x[:, 0:1], x[:, 1:2]), (x[:, 1:2], x[:, 0:1])):
            per_axis = (
                self._axis_counts(P, O, self.kv, (0, 0, L, L)) +
                self._axis_counts(P, O, self.kh, (L, L, 0, 0)) +
                self._axis_counts(P, O, self.kd, (L, L, L, L)))
            feats.append(per_axis)
        return torch.cat(feats, dim=1) / 6.0


class ResBlock(nn.Module):
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode='circular', bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode='circular', bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + x)


class PairPolicyHead(nn.Module):
    """Bilinear attention head producing N×N pair logits.

    A(i,j) = (q_i · k_j + q_j · k_i) / 2  (symmetrized)
    Diagonal masked to -inf. Padding cells masked if mask provided.
    """

    def __init__(self, channels, head_dim=64):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, head_dim, 1)
        self.k_proj = nn.Conv2d(channels, head_dim, 1)
        self.scale = head_dim ** -0.5  # legacy dot-product scale
        # Cosine-similarity logits scaled by a learnable temperature. This
        # replaces a hard clamp(-100, 100) on unbounded dot-product logits,
        # which created a zero-gradient dead zone: training drove all logits to
        # the +100 ceiling, collapsing the policy to uniform (unlearnable). With
        # unit-norm q/k the logits live in [-temp, temp] with smooth gradients
        # everywhere, so the pair policy can actually be trained.
        self.log_temp = nn.Parameter(torch.tensor(math.log(20.0)))
        # False -> normalized-QK temperature path (fixed). True -> original
        # clamp path, set automatically when loading a pre-fix checkpoint (which
        # lacks log_temp) so old weights still evaluate as trained. See
        # HexResNet.load_state_dict below.
        self.register_buffer("legacy_clamp", torch.tensor(False))

    def forward(self, trunk_features, mask=None):
        B, C, H, W = trunk_features.shape
        N = H * W

        if bool(self.legacy_clamp):
            Q = self.q_proj(trunk_features).flatten(2)
            K = self.k_proj(trunk_features).flatten(2)
            A = torch.bmm(Q.transpose(1, 2), K) * self.scale
            A = (A + A.transpose(1, 2)) / 2
            A = A.clamp(-100, 100)
        else:
            Q = F.normalize(self.q_proj(trunk_features).flatten(2), dim=1)  # [B, d, N]
            K = F.normalize(self.k_proj(trunk_features).flatten(2), dim=1)  # [B, d, N]
            temp = self.log_temp.clamp(max=math.log(100.0)).exp()
            A = torch.bmm(Q.transpose(1, 2), K) * temp  # cosine sim in [-1,1] * temp
            A = (A + A.transpose(1, 2)) / 2  # symmetrize

        # Mask diagonal (can't place both stones on same cell)
        diag = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(diag, float("-inf"))

        # Mask padding cells
        if mask is not None:
            mask_flat = mask.reshape(B, -1)  # [B, N]
            pair_mask = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)  # [B, N, N]
            A = A.masked_fill(pair_mask == 0, float("-inf"))

        return A


class HexResNet(nn.Module):
    def __init__(self, in_channels=2, num_blocks=10, num_filters=128,
                 gn_groups=8, v_channels=32, pair_head_dim=64,
                 chain_channels=32, value_head="wdl", threat_stem=False):
        super().__init__()
        self.value_head = value_head  # "wdl" (3-way categorical) or "scalar" (legacy)

        # Optional fixed strix-style threat features derived from the stone
        # planes on the fly (adds ThreatStem.OUT_CHANNELS input channels).
        self.threat = ThreatStem() if threat_stem else None
        stem_in = in_channels + (ThreatStem.OUT_CHANNELS if threat_stem else 0)

        # Stem
        self.stem_conv = nn.Conv2d(stem_in, num_filters, 3, padding=1,
                                   padding_mode='circular', bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups, num_filters)

        # Residual trunk
        self.blocks = nn.Sequential(
            *[ResBlock(num_filters, gn_groups) for _ in range(num_blocks)]
        )

        # Value head: conv → mean+max pool → FC → 3-way categorical
        # (logits over {lose, draw, win})
        self.v_conv = nn.Conv2d(num_filters, v_channels, 1, bias=False)
        self.v_gn = nn.GroupNorm(gn_groups, v_channels)
        self.v_fc1 = nn.Linear(v_channels * 2, 256)  # mean + max = 2x channels
        self.v_fc2 = nn.Linear(256, 3 if value_head == "wdl" else 1)
        # Scalar values associated with each outcome bin, for expected value.
        self.register_buffer("value_bins", torch.tensor([-1.0, 0.0, 1.0]))

        # Moves-left head: same pooled vector as value → FC → ReLU → FC
        self.ml_fc1 = nn.Linear(v_channels * 2, 256)
        self.ml_fc2 = nn.Linear(256, 1)

        # Chain head: per-cell per-direction unblocked chain for each player
        # trunk → 1x1 conv → ReLU → 1x1 conv → 6 channels
        # [cur_d0, cur_d1, cur_d2, opp_d0, opp_d1, opp_d2]
        self.chain_conv1 = nn.Conv2d(num_filters, chain_channels, 1)
        self.chain_conv2 = nn.Conv2d(chain_channels, 6, 1)

        # Pair policy head
        self.pair_head = PairPolicyHead(num_filters, pair_head_dim)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: [B, C, H, W] board planes
            mask: [B, 1, H, W] float mask (1=valid, 0=padding). None=all valid.

        Returns:
            value_logits: [B, 3] categorical logits over {lose, draw, win}
            pair_logits: [B, N, N] raw pair logits (diagonal=-inf, padding=-inf)
            moves_left: [B] predicted remaining moves (>= 0)
            chain: [B, 6, H, W] per-cell per-direction unblocked chain length
        """
        if self.threat is not None:
            with torch.no_grad():
                x = torch.cat([x, self.threat(x[:, :2])], dim=1)
        s = F.relu(self.stem_gn(self.stem_conv(x)))
        t = self.blocks(s)

        # Pooled features (shared by value and moves-left heads)
        v_feat = F.relu(self.v_gn(self.v_conv(t)))  # [B, v_ch, H, W]
        if mask is not None:
            v_mean = (v_feat * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp(min=1)
            v_max = (v_feat + (mask - 1) * 1e9).amax(dim=[2, 3])
        else:
            v_mean = v_feat.mean(dim=[2, 3])
            v_max = v_feat.amax(dim=[2, 3])
        v_pooled = torch.cat([v_mean, v_max], dim=-1)  # [B, 2*v_ch]

        # Value head: 3-way categorical logits over {lose, draw, win}
        value_logits = self.v_fc2(F.relu(self.v_fc1(v_pooled)))  # [B, 3]

        # Moves-left head (from same pooled vector)
        moves_left = F.relu(self.ml_fc2(F.relu(self.ml_fc1(v_pooled)))).squeeze(-1)

        # Pair policy head
        pair_logits = self.pair_head(t, mask)

        # Chain head: per-cell per-direction chain for current/opponent
        chain = self.chain_conv2(F.relu(self.chain_conv1(t)))  # [B, 6, H, W]

        return value_logits, pair_logits, moves_left, chain

    def expected_value(self, value_logits):
        """Expected scalar value in [-1, 1] from categorical logits.

        E[value] = sum_k P(k) * value_bins[k], with bins {lose:-1, draw:0, win:1}.
        Returns [B].
        """
        if getattr(self, "value_head", "wdl") == "scalar":
            return value_logits.squeeze(-1).clamp(-1.0, 1.0)  # legacy regression head
        probs = F.softmax(value_logits, dim=-1)
        return probs @ self.value_bins

    @staticmethod
    def marginalize(pair_logits):
        """Marginalize pair logits to single-move logits.

        P(cell_i) = sum_j P(i,j) → use logsumexp for numerical stability.
        Returns [B, N] logits.
        """
        return pair_logits.logsumexp(dim=-1)

    def set_padding_mode(self, mode: str):
        """Switch all Conv2d layers between 'circular' and 'zeros'."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = mode

    def load_state_dict(self, state_dict, strict=True, **kw):
        """Back-compat: a checkpoint saved before the pair-head fix has no
        ``pair_head.log_temp``. Load it into the original clamp path so the old
        weights evaluate exactly as they were trained (and ignore the new head
        params, which stay at init and are unused in legacy mode)."""
        if "pair_head.log_temp" not in state_dict:
            self.pair_head.legacy_clamp = torch.tensor(True)
            return super().load_state_dict(state_dict, strict=False, **kw)
        if strict and "pair_head.legacy_clamp" not in state_dict:
            # new-style (has log_temp) but predates the legacy_clamp buffer
            state_dict = dict(state_dict)
            state_dict["pair_head.legacy_clamp"] = self.pair_head.legacy_clamp
        return super().load_state_dict(state_dict, strict=strict, **kw)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", **overrides):
        """Construct a model matching a saved checkpoint and load it, inferring
        num_blocks / num_filters / value_head / legacy-pair-head automatically.
        Works for both the new WDL/temperature checkpoints and old
        scalar-value/clamp-pair Kraken checkpoints. Returns the loaded model."""
        obj = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(obj, dict):
            sd = (obj.get("model_state_dict") or obj.get("model")
                  or obj.get("state_dict") or obj)
        else:
            sd = obj
        num_filters = sd["stem_conv.weight"].shape[0]
        num_blocks = sum(1 for k in sd
                         if k.startswith("blocks.") and k.endswith(".conv1.weight"))
        value_head = "scalar" if sd["v_fc2.weight"].shape[0] == 1 else "wdl"
        threat_stem = sd["stem_conv.weight"].shape[1] > 2
        kwargs = dict(num_blocks=num_blocks, num_filters=num_filters,
                      value_head=value_head, threat_stem=threat_stem)
        kwargs.update(overrides)
        model = cls(**kwargs)
        model.load_state_dict(sd)
        return model


def board_to_planes(board_dict, current_player, pad_to=None, min_size=None,
                    margin=6):
    """Convert {(q,r): player_int} board to planes tensor.

    Channel 0: current player's stones.
    Channel 1: opponent's stones.

    If pad_to is given, centers in a (pad_to x pad_to) grid.
    Otherwise uses tight bounding box + *margin*-cell margin on each side,
    clamped to at least *min_size* × *min_size*.

    Returns (planes, offset_q, offset_r, board_h, board_w).
    """
    cp = current_player.value if hasattr(current_player, 'value') else current_player
    if not board_dict:
        size = pad_to or min_size or 13
        return torch.zeros(2, size, size), 0, 0, size, size

    qs = [q for q, _r in board_dict]
    rs = [r for _q, r in board_dict]
    min_q, max_q = min(qs), max(qs)
    min_r, max_r = min(rs), max(rs)

    if pad_to is not None:
        h = w = pad_to
    else:
        h = max_q - min_q + 1 + 2 * margin
        w = max_r - min_r + 1 + 2 * margin
        if min_size:
            h = max(h, min_size)
            w = max(w, min_size)

    off_q = (h - (max_q - min_q + 1)) // 2 - min_q
    off_r = (w - (max_r - min_r + 1)) // 2 - min_r

    planes = torch.zeros(2, h, w)
    for (q, r), player in board_dict.items():
        gq = q + off_q
        gr = r + off_r
        pv = player.value if hasattr(player, 'value') else player
        if pv == cp:
            planes[0, gq, gr] = 1.0
        else:
            planes[1, gq, gr] = 1.0

    return planes, off_q, off_r, h, w


def board_to_planes_torus(board_dict, current_player):
    """Convert torus-coordinate board dict to fixed [2, BOARD_SIZE, BOARD_SIZE] tensor.

    board_dict: {(q, r): int_or_Player} where 0 <= q, r < BOARD_SIZE.
    current_player: int or Player enum.
    Returns: planes tensor [2, BOARD_SIZE, BOARD_SIZE].
    """
    cp = current_player.value if hasattr(current_player, 'value') else current_player
    planes = torch.zeros(2, BOARD_SIZE, BOARD_SIZE)
    for (q, r), player in board_dict.items():
        pv = player.value if hasattr(player, 'value') else player
        if pv == cp:
            planes[0, q, r] = 1.0
        else:
            planes[1, q, r] = 1.0
    return planes


def parse_board_json(board_json):
    """Parse board JSON string to {(q,r): player_int} dict."""
    return {
        tuple(int(x) for x in k.split(",")): v
        for k, v in json.loads(board_json).items()
    }


def move_to_index(q, r, off_q, off_r, width):
    return (q + off_q) * width + (r + off_r)


def index_to_move(idx, off_q, off_r, width):
    return idx // width - off_q, idx % width - off_r


if __name__ == "__main__":
    model = HexResNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    for size in [11, 19, 25]:
        x = torch.randn(4, 2, size, size)
        mask = torch.ones(4, 1, size, size)
        v, pair, ml, chain = model(x, mask)
        N = size * size
        single = HexResNet.marginalize(pair)
        ev = model.expected_value(v)
        print(f"  {size}x{size}: value={v.shape}, pair={pair.shape}, "
              f"moves_left={ml.shape}, chain={chain.shape}, "
              f"E[v]=[{ev.min().item():.3f}, {ev.max().item():.3f}]")

        # Verify symmetry and diagonal masking
        assert torch.allclose(pair, pair.transpose(1, 2)), "Not symmetric!"
        assert (pair[:, range(N), range(N)] == float("-inf")).all(), "Diagonal not masked!"
        assert v.shape == (4, 3), f"value shape wrong: {v.shape}"
        assert ((ev >= -1) & (ev <= 1)).all(), "expected value out of range"
        assert ml.shape == (4,), f"moves_left shape wrong: {ml.shape}"
        assert chain.shape == (4, 6, size, size), f"chain shape wrong: {chain.shape}"
        assert (ml >= 0).all(), "moves_left should be non-negative"
    print("All checks passed.")
