"""D6 symmetry group for the hex grid on a 25x25 torus.

Precomputes the 12 permutation tables (6 rotations x 2 reflections)
for axial hex coordinates (q, r) mod 25. Used to randomly augment
training samples so the model sees each position in all orientations.
"""

import numpy as np
import torch

from model.resnet import BOARD_SIZE

N = BOARD_SIZE  # 25

# 12 symmetry transforms as linear coefficient matrices (a, b, c, d):
#   new_q = (a*q + b*r) % N
#   new_r = (c*q + d*r) % N
SYMMETRY_COEFFS = [
    # 6 rotations
    ( 1,  0,  0,  1),   # R0: identity
    ( 0, -1,  1,  1),   # R1: 60 deg
    (-1, -1,  1,  0),   # R2: 120 deg
    (-1,  0,  0, -1),   # R3: 180 deg
    ( 0,  1, -1, -1),   # R4: 240 deg
    ( 1,  1, -1,  0),   # R5: 300 deg
    # 6 reflections (apply (q,r)->(r,q) then rotate)
    ( 0,  1,  1,  0),   # S0: reflect
    (-1,  0,  1,  1),   # S1: reflect + R1
    (-1, -1,  0,  1),   # S2: reflect + R2
    ( 0, -1, -1,  0),   # S3: reflect + R3
    ( 1,  0, -1, -1),   # S4: reflect + R4
    ( 1,  1,  0, -1),   # S5: reflect + R5
]


def _build_permutations():
    """Build forward permutation tables: PERMS[k][old_flat] = new_flat."""
    perms = np.zeros((12, N * N), dtype=np.int64)
    for k, (a, b, c, d) in enumerate(SYMMETRY_COEFFS):
        for q in range(N):
            for r in range(N):
                old_idx = q * N + r
                new_q = (a * q + b * r) % N
                new_r = (c * q + d * r) % N
                new_idx = new_q * N + new_r
                perms[k, old_idx] = new_idx
    return perms


PERMS = _build_permutations()                          # [12, 625]
INV_PERMS = np.zeros_like(PERMS)                       # [12, 625]
for _k in range(12):
    for _i in range(N * N):
        INV_PERMS[_k, PERMS[_k, _i]] = _i

PERMS_TORCH = torch.from_numpy(PERMS).long()           # [12, 625]
INV_PERMS_TORCH = torch.from_numpy(INV_PERMS).long()   # [12, 625]


def apply_symmetry_planes(planes: torch.Tensor, k: int) -> torch.Tensor:
    """Apply symmetry k to board planes [C, N, N]. Returns new planes."""
    C = planes.shape[0]
    flat = planes.reshape(C, -1)        # [C, 625]
    inv = INV_PERMS_TORCH[k]           # [625]
    return flat[:, inv].reshape(C, N, N)


# ---------------------------------------------------------------------------
# Direction permutation for per-axis chain targets
# ---------------------------------------------------------------------------

# Forward direction permutation: DIR_PERMS[k][src] = dst direction index.
# Directions: 0=(1,0), 1=(0,1), 2=(1,-1).
DIR_PERMS = [
    [0, 1, 2],  # R0: identity
    [1, 2, 0],  # R1: 60°   (d0→d1, d1→d2, d2→d0)
    [2, 0, 1],  # R2: 120°  (d0→d2, d1→d0, d2→d1)
    [0, 1, 2],  # R3: 180°  (identity on undirected lines)
    [1, 2, 0],  # R4: 240°
    [2, 0, 1],  # R5: 300°
    [1, 0, 2],  # S0: reflect (swap d0↔d1)
    [2, 1, 0],  # S1: reflect + R1 (swap d0↔d2)
    [0, 2, 1],  # S2: reflect + R2 (swap d1↔d2)
    [1, 0, 2],  # S3: reflect + R3
    [2, 1, 0],  # S4: reflect + R4
    [0, 2, 1],  # S5: reflect + R5
]


def _invert_perm3(p):
    inv = [0, 0, 0]
    for i, v in enumerate(p):
        inv[v] = i
    return inv


# Indexing permutation for 6 channels: result[_CHAIN_CH_PERM[k]] reorders
# [cur_d0, cur_d1, cur_d2, opp_d0, opp_d1, opp_d2] to match symmetry k.
_CHAIN_CH_PERM = torch.tensor([
    (lambda inv: [inv[0], inv[1], inv[2], inv[0]+3, inv[1]+3, inv[2]+3])(
        _invert_perm3(dp))
    for dp in DIR_PERMS
], dtype=torch.long)


def apply_symmetry_chain(chain: torch.Tensor, k: int) -> torch.Tensor:
    """Apply symmetry k to chain targets/masks [6, N, N].

    Performs spatial permutation then reorders direction sub-channels.
    Channels 0-2 = current player directions, 3-5 = opponent directions.
    """
    result = apply_symmetry_planes(chain, k)
    if k == 0:
        return result
    return result[_CHAIN_CH_PERM[k]]


def apply_symmetry_visits_sparse(visit_entries: list, k: int) -> list:
    """Remap sparse visit entries [(flat_pair_idx, prob), ...] under symmetry k.

    Returns new list of (new_flat_pair_idx, prob).
    """
    if k == 0 or not visit_entries:
        return visit_entries
    perm = PERMS[k]  # numpy for fast scalar lookup
    NN = N * N
    result = []
    for flat_idx, prob in visit_entries:
        a = flat_idx // NN
        b = flat_idx % NN
        new_a = int(perm[a])
        new_b = int(perm[b])
        result.append((new_a * NN + new_b, prob))
    return result


def verify_symmetries():
    """Verify D6 symmetry tables are correct."""
    from game import HEX_DIRECTIONS

    NN = N * N

    for k in range(12):
        # Each permutation is a bijection
        assert len(set(PERMS[k])) == NN, f"Symmetry {k}: not a bijection"

        # Inverse is correct
        for i in range(NN):
            assert INV_PERMS[k, PERMS[k, i]] == i, \
                f"Symmetry {k}: inverse failed at {i}"

        # Hex directions are preserved (as a set, up to sign)
        a, b, c, d = SYMMETRY_COEFFS[k]
        transformed = set()
        for dq, dr in HEX_DIRECTIONS:
            new_dq = (a * dq + b * dr) % N
            new_dr = (c * dq + d * dr) % N
            # Normalize: direction and its negative are the same line
            if new_dq > N // 2:
                new_dq = N - new_dq
                new_dr = N - new_dr
            if new_dq == 0 and new_dr > N // 2:
                new_dr = N - new_dr
            transformed.add((new_dq, new_dr % N))

        original = set()
        for dq, dr in HEX_DIRECTIONS:
            ndq = dq % N
            ndr = dr % N
            if ndq > N // 2:
                ndq = N - ndq
                ndr = N - ndr
            if ndq == 0 and ndr > N // 2:
                ndr = N - ndr
            original.add((ndq, ndr % N))

        assert transformed == original, \
            f"Symmetry {k}: directions not preserved: {transformed} != {original}"

    # Group closure: composing any two symmetries gives another in the group
    for i in range(12):
        for j in range(12):
            composed = PERMS[i][PERMS[j]]
            found = False
            for k in range(12):
                if np.array_equal(composed, PERMS[k]):
                    found = True
                    break
            assert found, f"Symmetry {i} o {j} not in group"

    # Verify DIR_PERMS: each symmetry maps directions correctly (forward map)
    for k, (a, b, c, d) in enumerate(SYMMETRY_COEFFS):
        for src_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
            new_dq = (a * dq + b * dr) % N
            new_dr = (c * dq + d * dr) % N
            # Normalize to canonical direction (positive or canonical form)
            if new_dq > N // 2:
                new_dq = N - new_dq
                new_dr = (N - new_dr) % N
            if new_dq == 0 and new_dr > N // 2:
                new_dr = N - new_dr
            # Find which canonical direction this matches
            dst_idx = None
            for j, (cq, cr) in enumerate(HEX_DIRECTIONS):
                ncq = cq % N
                ncr = cr % N
                if ncq > N // 2:
                    ncq = N - ncq
                    ncr = (N - ncr) % N
                if ncq == 0 and ncr > N // 2:
                    ncr = N - ncr
                if (new_dq, new_dr) == (ncq, ncr):
                    dst_idx = j
                    break
            assert dst_idx is not None, \
                f"Symmetry {k}: direction {src_idx} mapped to unknown"
            # Forward: DIR_PERMS[k][src_idx] should equal dst_idx
            assert DIR_PERMS[k][src_idx] == dst_idx, \
                f"Symmetry {k}: DIR_PERMS[{k}][{src_idx}] = " \
                f"{DIR_PERMS[k][src_idx]}, expected {dst_idx}"

    # Verify direction permutation group closure (forward composition)
    for i in range(12):
        for j in range(12):
            # Find composed spatial symmetry: first j, then i
            composed = PERMS[i][PERMS[j]]
            comp_k = None
            for k in range(12):
                if np.array_equal(composed, PERMS[k]):
                    comp_k = k
                    break
            # Composed forward dir perm: first j, then i
            dp_i = DIR_PERMS[i]
            dp_j = DIR_PERMS[j]
            composed_dp = [dp_i[dp_j[d]] for d in range(3)]
            assert composed_dp == DIR_PERMS[comp_k], \
                f"Dir perm {i} o {j}: got {composed_dp}, " \
                f"expected {DIR_PERMS[comp_k]}"

    print("All 12 D6 symmetries verified: bijections, direction-preserving, "
          "group closure, direction permutations.")


if __name__ == "__main__":
    verify_symmetries()
