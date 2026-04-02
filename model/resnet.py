"""ResNet + attention pair policy for HexTicTacToe.

Dual-head model predicting win rate (value) and move PAIR probabilities
(policy) from board positions. Fully convolutional — works at any board size.

Architecture:
  - Conv stem + N residual blocks (GroupNorm, size-independent)
  - Value head: masked global average pooling → FC → tanh
  - Policy head: bilinear attention over cell embeddings → N×N pair logits
    Symmetrized (order of stones in a pair doesn't matter),
    diagonal masked (can't place both on same cell).
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 25


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
        self.scale = head_dim ** -0.5

    def forward(self, trunk_features, mask=None):
        B, C, H, W = trunk_features.shape
        N = H * W

        Q = self.q_proj(trunk_features).flatten(2).float()  # [B, d, N]
        K = self.k_proj(trunk_features).flatten(2).float()  # [B, d, N]

        A = torch.bmm(Q.transpose(1, 2), K) * self.scale  # [B, N, N]
        A = (A + A.transpose(1, 2)) / 2  # symmetrize
        A = A.clamp(-100, 100)  # prevent extreme logits

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
                 chain_channels=32):
        super().__init__()

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1,
                                   padding_mode='circular', bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups, num_filters)

        # Residual trunk
        self.blocks = nn.Sequential(
            *[ResBlock(num_filters, gn_groups) for _ in range(num_blocks)]
        )

        # Value head: conv → mean+max pool → FC
        self.v_conv = nn.Conv2d(num_filters, v_channels, 1, bias=False)
        self.v_gn = nn.GroupNorm(gn_groups, v_channels)
        self.v_fc1 = nn.Linear(v_channels * 2, 256)  # mean + max = 2x channels
        self.v_fc2 = nn.Linear(256, 1)

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
            value: [B] scalar in [-1, 1]
            pair_logits: [B, N, N] raw pair logits (diagonal=-inf, padding=-inf)
            moves_left: [B] predicted remaining moves (>= 0)
            chain: [B, 6, H, W] per-cell per-direction unblocked chain length
        """
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

        # Value head
        value = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v_pooled)))).squeeze(-1)

        # Moves-left head (from same pooled vector)
        moves_left = F.relu(self.ml_fc2(F.relu(self.ml_fc1(v_pooled)))).squeeze(-1)

        # Pair policy head
        pair_logits = self.pair_head(t, mask)

        # Chain head: per-cell per-direction chain for current/opponent
        chain = self.chain_conv2(F.relu(self.chain_conv1(t)))  # [B, 6, H, W]

        return value, pair_logits, moves_left, chain

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
        print(f"  {size}x{size}: value={v.shape}, pair={pair.shape}, "
              f"moves_left={ml.shape}, chain={chain.shape}, "
              f"v=[{v.min().item():.3f}, {v.max().item():.3f}]")

        # Verify symmetry and diagonal masking
        assert torch.allclose(pair, pair.transpose(1, 2)), "Not symmetric!"
        assert (pair[:, range(N), range(N)] == float("-inf")).all(), "Diagonal not masked!"
        assert ml.shape == (4,), f"moves_left shape wrong: {ml.shape}"
        assert chain.shape == (4, 6, size, size), f"chain shape wrong: {chain.shape}"
        assert (ml >= 0).all(), "moves_left should be non-negative"
    print("All checks passed.")
