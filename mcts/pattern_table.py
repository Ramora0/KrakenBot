"""Pattern canonicalization for minimax evaluation windows.

Enumerates all base-3 patterns of a given window length, groups them into
canonical equivalence classes under reversal (and optionally piece-swap)
symmetry, and returns lookup arrays for fast pattern_int -> value mapping.
"""


def _int_to_pattern(i, wl):
    """Convert integer to base-3 pattern tuple (LSB first)."""
    pat = []
    for _ in range(wl):
        pat.append(i % 3)
        i //= 3
    return tuple(pat)


def _swap_pieces(pat):
    """Swap pieces: 1<->2, 0 unchanged."""
    return tuple(({0: 0, 1: 2, 2: 1}[c]) for c in pat)


def build_arrays(wl, enforce_piece_swap=True):
    """Build canonical pattern tables for window length `wl`.

    Returns (canon_patterns, canon_index, canon_sign, num_canon, num_patterns)
    where:
      canon_patterns: list of canonical pattern tuples
      canon_index:    pattern_int -> index into canon_patterns
      canon_sign:     pattern_int -> +1, -1, or 0 (multiplier for the value)
      num_canon:      len(canon_patterns)
      num_patterns:   3 ** wl
    """
    num_patterns = 3 ** wl
    canon_patterns = []
    canon_lookup = {}  # canon tuple -> index
    pattern_map = {}   # pattern_int -> (canon_index, sign)

    for i in range(num_patterns):
        pat = _int_to_pattern(i, wl)
        p_flip = pat[::-1]

        if enforce_piece_swap:
            p_swap = _swap_pieces(pat)
            p_swap_flip = _swap_pieces(p_flip)

            pos_variants = {pat, p_flip}
            neg_variants = {p_swap, p_swap_flip}

            if pos_variants & neg_variants:
                # Self-symmetric under piece swap => forced to 0
                canon = min(pos_variants | neg_variants)
                if canon not in canon_lookup:
                    canon_lookup[canon] = len(canon_patterns)
                    canon_patterns.append(canon)
                pattern_map[i] = (canon_lookup[canon], 0)
            else:
                all_variants = pos_variants | neg_variants
                canon = min(all_variants)
                if canon not in canon_lookup:
                    canon_lookup[canon] = len(canon_patterns)
                    canon_patterns.append(canon)
                ci = canon_lookup[canon]
                sign = 1 if canon in pos_variants else -1
                pattern_map[i] = (ci, sign)
        else:
            # Only flip (reversal) symmetry
            canon = min(pat, p_flip)
            if canon not in canon_lookup:
                canon_lookup[canon] = len(canon_patterns)
                canon_patterns.append(canon)
            pattern_map[i] = (canon_lookup[canon], 1)

    canon_index = [0] * num_patterns
    canon_sign = [0] * num_patterns
    for pi, (ci, s) in pattern_map.items():
        canon_index[pi] = ci
        canon_sign[pi] = s

    return canon_patterns, canon_index, canon_sign, len(canon_patterns), num_patterns
