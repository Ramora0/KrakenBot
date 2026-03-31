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
    """Apply symmetry k to board planes [2, N, N]. Returns new planes."""
    flat = planes.reshape(2, -1)        # [2, 625]
    inv = INV_PERMS_TORCH[k]           # [625]
    return flat[:, inv].reshape(2, N, N)


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

    print("All 12 D6 symmetries verified: bijections, direction-preserving, group closure.")


if __name__ == "__main__":
    verify_symmetries()
