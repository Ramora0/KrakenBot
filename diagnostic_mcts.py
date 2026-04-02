"""Diagnostic tests for MCTS far-stone visits.

Tests:
  1. Model prior concentration — what fraction of pair mass is within dist 2?
  2. Noise impact — does Dirichlet inflate far-cell priors?
  3. MCTS visit distribution — do far pairs accumulate visits after 200 sims?
  4. OOD conditional — is P(s2|s1=far) uniform?

Usage:
  python diagnostic_mcts.py --checkpoint training/mcts_results/round_48.pt
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from game import ToroidalHexGame, Player
from model.resnet import HexResNet, board_to_planes_torus, BOARD_SIZE
from mcts.tree import (
    create_tree, select_leaf, expand_and_backprop, maybe_expand_leaf,
    get_pair_visits, _idx_to_cell, _cell_to_idx, _add_exploration_noise,
    _build_tree_from_eval, N_CELLS, DIRICHLET_ALPHA, DIRICHLET_FRAC,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_dist(q1, r1, q2, r2):
    """Hex (axial) distance on torus."""
    N = BOARD_SIZE
    dq = min(abs(q1 - q2), N - abs(q1 - q2))
    dr = min(abs(r1 - r2), N - abs(r1 - r2))
    # s = -q - r in axial coords; hex dist = max(|dq|, |dr|, |ds|)
    # On torus we need to consider wrapping for s too
    s1, s2 = -q1 - r1, -q2 - r2
    ds = min(abs(s1 - s2) % N, N - abs(s1 - s2) % N)
    return max(dq, dr, ds)


def min_dist_to_stones(q, r, occupied):
    """Minimum hex distance from (q,r) to any occupied cell."""
    if not occupied:
        return 0
    return min(hex_dist(q, r, oq, or_) for oq, or_ in occupied)


def make_early_game():
    """Create a game with a few stones near center — typical early position."""
    game = ToroidalHexGame()
    c = BOARD_SIZE // 2  # 12
    # Player A's first move (single stone)
    game.make_move(c, c)
    # Player B places 2 stones
    game.make_move(c + 1, c)
    game.make_move(c, c + 1)
    # Player A places 2 stones
    game.make_move(c - 1, c + 1)
    game.make_move(c + 1, c - 1)
    return game


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    # Infer num_blocks from checkpoint keys
    block_ids = [int(k.split('.')[1]) for k in state if k.startswith("blocks.")]
    num_blocks = max(block_ids) + 1 if block_ids else 10
    # Infer num_filters from stem conv weight shape
    num_filters = state["stem_conv.weight"].shape[0]
    model = HexResNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 1: Model prior concentration
# ---------------------------------------------------------------------------

def test_prior_concentration(model, device):
    print("=" * 70)
    print("TEST 1: Model prior concentration")
    print("=" * 70)

    game = make_early_game()
    occupied = set(game.board.keys())

    planes = board_to_planes_torus(game.board, game.current_player)
    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        value, pair_logits, _, _ = model(x)

    pp = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(N_CELLS, N_CELLS).cpu()
    marginal = pp.sum(dim=-1)

    # Bucket marginal by distance
    dist_mass = defaultdict(float)
    dist_cells = defaultdict(int)
    far_cells = []  # (dist, idx, marginal_val)

    for idx in range(N_CELLS):
        q, r = _idx_to_cell(idx)
        if (q, r) in occupied:
            continue
        d = min_dist_to_stones(q, r, occupied)
        m = marginal[idx].item()
        dist_mass[d] += m
        dist_cells[d] += 1
        if d > 2:
            far_cells.append((d, idx, m))

    print(f"\nBoard: {len(occupied)} stones, Player {game.current_player.name}'s turn")
    print(f"Value estimate: {value[0].item():.4f}")
    print(f"\nMarginal mass by distance to nearest stone:")
    total = sum(dist_mass.values())
    cumulative = 0.0
    for d in sorted(dist_mass.keys()):
        cumulative += dist_mass[d]
        pct = 100 * dist_mass[d] / total
        cum_pct = 100 * cumulative / total
        print(f"  dist {d:2d}: {dist_mass[d]:.6f} ({pct:6.2f}%)  "
              f"[{dist_cells[d]:3d} cells]  cumulative: {cum_pct:.1f}%")

    within_2 = sum(dist_mass[d] for d in dist_mass if d <= 2)
    print(f"\nTotal mass within dist 2: {within_2/total*100:.2f}%")
    print(f"Total mass beyond dist 2: {(1 - within_2/total)*100:.2f}%")

    # Top-5 far cells
    far_cells.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop-5 marginal values for cells beyond dist 2:")
    for d, idx, m in far_cells[:5]:
        q, r = _idx_to_cell(idx)
        print(f"  ({q:2d},{r:2d}) dist={d} marginal={m:.6f}")

    return pp, marginal, occupied


# ---------------------------------------------------------------------------
# Test 2: Noise impact
# ---------------------------------------------------------------------------

def test_noise_impact(model, device):
    print("\n" + "=" * 70)
    print("TEST 2: Noise impact on far-cell priors")
    print("=" * 70)

    game = make_early_game()
    occupied = set(game.board.keys())

    planes = board_to_planes_torus(game.board, game.current_player)
    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        value, pair_logits, _, _ = model(x)

    pp = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(N_CELLS, N_CELLS).cpu()
    marginal = pp.sum(dim=-1)

    # Build tree without noise as baseline
    tree_clean = _build_tree_from_eval(
        game, value[0].item(), pp, marginal, planes, add_noise=False)

    # Classify root actions by distance
    root = tree_clean.root_pos.move_node
    far_base = {}
    for i in range(root.n):
        idx = root.actions[i]
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        if d > 2:
            far_base[idx] = root.priors[i]

    max_far_base = max(far_base.values()) if far_base else 0.0
    print(f"\nBaseline (no noise):")
    print(f"  Root level-1 candidates: {root.n}")
    print(f"  Far cells (dist>2): {len(far_base)}")
    print(f"  Max far-cell prior: {max_far_base:.6f}")

    # Run noise many times to see distribution
    n_trials = 1000
    max_far_noised = []
    for _ in range(n_trials):
        tree = _build_tree_from_eval(
            game, value[0].item(), pp, marginal.clone(), planes, add_noise=True)
        r = tree.root_pos.move_node
        max_f = 0.0
        for i in range(r.n):
            idx = r.actions[i]
            q, rc = _idx_to_cell(idx)
            d = min_dist_to_stones(q, rc, occupied)
            if d > 2:
                max_f = max(max_f, r.priors[i])
        max_far_noised.append(max_f)

    arr = np.array(max_far_noised)
    print(f"\nWith Dirichlet noise (alpha={DIRICHLET_ALPHA}, frac={DIRICHLET_FRAC}):")
    print(f"  Max far-cell prior across {n_trials} trials:")
    print(f"    mean:   {arr.mean():.6f}  (vs baseline {max_far_base:.6f})")
    print(f"    median: {np.median(arr):.6f}")
    print(f"    p95:    {np.percentile(arr, 95):.6f}")
    print(f"    p99:    {np.percentile(arr, 99):.6f}")
    print(f"    max:    {arr.max():.6f}")
    if max_far_base > 0:
        print(f"    inflation ratio (mean): {arr.mean()/max_far_base:.1f}x")
        print(f"    inflation ratio (p99):  {np.percentile(arr, 99)/max_far_base:.1f}x")

    # PUCT threshold: at what prior does a far cell get visited before
    # a cell with the median prior and 0 visits?
    near_priors = []
    for i in range(root.n):
        idx = root.actions[i]
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        if d <= 2:
            near_priors.append(root.priors[i])
    if near_priors:
        median_near = np.median(near_priors)
        # PUCT: Q + c * sqrt(N) * P / (1+n)
        # Unvisited far cell score: 0 + c*sqrt(N)*P_far/(1+0) = c*sqrt(N)*P_far
        # Unvisited near cell: c*sqrt(N)*P_near
        # They compete directly when both unvisited. Far cell wins if P_far > P_near
        print(f"\n  Median near-cell prior: {median_near:.6f}")
        print(f"  Fraction of trials where max far prior > median near prior: "
              f"{(arr > median_near).mean()*100:.1f}%")


# ---------------------------------------------------------------------------
# Test 3: MCTS visit distribution
# ---------------------------------------------------------------------------

def test_mcts_visits(model, device, n_sims=200):
    print("\n" + "=" * 70)
    print(f"TEST 3: MCTS visit distribution ({n_sims} sims)")
    print("=" * 70)

    game = make_early_game()
    occupied = set(game.board.keys())

    tree = create_tree(game, model, device, add_noise=True)

    # Track max depth
    max_depth = 0

    for sim in range(n_sims):
        leaf = select_leaf(tree, game)
        depth = len(leaf.pair_depths)
        max_depth = max(max_depth, depth)

        if leaf.is_terminal:
            expand_and_backprop(tree, leaf, 0.0)
            continue

        if leaf.needs_expansion and leaf.deltas:
            # Forward pass for expansion
            from model.resnet import board_to_planes_torus as btp
            tmp_board = dict(game.board)
            cp = game.current_player
            for q, r, ch in leaf.deltas:
                p = cp if ch == 0 else (Player.B if cp == Player.A else Player.A)
                tmp_board[(q, r)] = p
            next_cp = leaf.current_player
            planes = btp(tmp_board, next_cp)
            x = planes.unsqueeze(0).to(device)
            with torch.no_grad():
                v, pl, _, _ = model(x)
            nn_value = v[0].item()

            flat_logits = pl[0].reshape(-1)
            top_raw, top_idxs = flat_logits.topk(200)
            top_vals = F.softmax(top_raw, dim=-1).cpu()
            marginal_logits = pl[0].logsumexp(dim=-1)
            marg = F.softmax(marginal_logits.reshape(-1), dim=-1).cpu()

            expand_and_backprop(tree, leaf, nn_value)
            maybe_expand_leaf(tree, leaf, marg, top_idxs.cpu(), top_vals)
        else:
            expand_and_backprop(tree, leaf, 0.0)

    # Collect pair visits
    pair_visits = get_pair_visits(tree)
    total_pair_visits = sum(pair_visits.values())

    print(f"\nTotal pair visits: {total_pair_visits}")
    print(f"Unique pairs visited: {len(pair_visits)}")
    print(f"Max tree depth: {max_depth}")

    # Classify by distance
    far_pairs = []
    near_pairs = []
    for (s1, s2), vc in pair_visits.items():
        q1, r1 = _idx_to_cell(s1)
        q2, r2 = _idx_to_cell(s2)
        d1 = min_dist_to_stones(q1, r1, occupied)
        d2 = min_dist_to_stones(q2, r2, occupied)
        max_d = max(d1, d2)
        if max_d > 2:
            far_pairs.append(((s1, s2), vc, d1, d2))
        else:
            near_pairs.append(((s1, s2), vc, d1, d2))

    far_total = sum(vc for _, vc, _, _ in far_pairs)
    near_total = sum(vc for _, vc, _, _ in near_pairs)

    print(f"\nNear pairs (both within dist 2): {len(near_pairs)}, "
          f"total visits: {near_total}")
    print(f"Far pairs (either stone dist>2): {len(far_pairs)}, "
          f"total visits: {far_total}")
    if total_pair_visits > 0:
        print(f"Far visit fraction: {far_total/total_pair_visits*100:.1f}%")

    # Visit histogram
    if pair_visits:
        counts = sorted(pair_visits.values(), reverse=True)
        print(f"\nVisit distribution (top-10 pairs):")
        for i, c in enumerate(counts[:10]):
            s1, s2 = [k for k, v in pair_visits.items() if v == c][0]
            q1, r1 = _idx_to_cell(s1)
            q2, r2 = _idx_to_cell(s2)
            d1 = min_dist_to_stones(q1, r1, occupied)
            d2 = min_dist_to_stones(q2, r2, occupied)
            print(f"  #{i+1}: ({q1},{r1})-({q2},{r2}) visits={c} "
                  f"dist=({d1},{d2})")

    # Far pairs detail
    if far_pairs:
        far_pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"\nFar pairs detail (top-10):")
        for (s1, s2), vc, d1, d2 in far_pairs[:10]:
            q1, r1 = _idx_to_cell(s1)
            q2, r2 = _idx_to_cell(s2)
            print(f"  ({q1},{r1})-({q2},{r2}) visits={vc} dist=({d1},{d2})")

    return tree, pair_visits, occupied


# ---------------------------------------------------------------------------
# Test 4: OOD conditional for far stone_1
# ---------------------------------------------------------------------------

def test_ood_conditional(pp, occupied):
    print("\n" + "=" * 70)
    print("TEST 4: OOD conditional P(s2|s1=far)")
    print("=" * 70)

    # Find the highest-marginal far cell
    marginal = pp.sum(dim=-1)
    far_cells = []
    for idx in range(N_CELLS):
        q, r = _idx_to_cell(idx)
        if (q, r) in occupied:
            continue
        d = min_dist_to_stones(q, r, occupied)
        if d > 2:
            far_cells.append((idx, d, marginal[idx].item()))

    if not far_cells:
        print("No far cells found!")
        return

    # Take the far cell with highest marginal
    far_cells.sort(key=lambda x: x[2], reverse=True)
    far_idx, far_dist, far_marg = far_cells[0]
    fq, fr = _idx_to_cell(far_idx)

    cond = pp[far_idx].clone()
    # Zero out occupied cells and self
    for oq, or_ in occupied:
        cond[_cell_to_idx(oq, or_)] = 0.0
    cond[far_idx] = 0.0
    cond_sum = cond.sum().item()
    if cond_sum > 0:
        cond_norm = cond / cond_sum
    else:
        print("All conditional mass is zero!")
        return

    # Entropy
    log_p = torch.log(cond_norm + 1e-30)
    entropy = -(cond_norm * log_p).sum().item()
    n_valid = (cond_norm > 0).sum().item()
    max_entropy = math.log(n_valid) if n_valid > 0 else 0

    print(f"\nFar cell: ({fq},{fr}) dist={far_dist} marginal={far_marg:.6f}")
    print(f"Conditional P(s2 | s1=({fq},{fr})):")
    print(f"  Valid s2 candidates: {int(n_valid)}")
    print(f"  Entropy: {entropy:.2f}  (max uniform: {max_entropy:.2f})")
    print(f"  Entropy ratio: {entropy/max_entropy*100:.1f}% of uniform")

    # Top-5 cells in conditional
    top5_vals, top5_idxs = cond_norm.topk(5)
    print(f"\n  Top-5 s2 cells:")
    for v, i in zip(top5_vals.tolist(), top5_idxs.tolist()):
        q, r = _idx_to_cell(i)
        d = min_dist_to_stones(q, r, occupied)
        d_to_s1 = hex_dist(q, r, fq, fr)
        print(f"    ({q:2d},{r:2d}) P={v:.6f} dist_stones={d} dist_s1={d_to_s1}")

    # How much conditional mass is near vs far
    near_mass = 0.0
    far_mass = 0.0
    for idx in range(N_CELLS):
        if cond_norm[idx].item() < 1e-30:
            continue
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        if d <= 2:
            near_mass += cond_norm[idx].item()
        else:
            far_mass += cond_norm[idx].item()
    print(f"\n  Conditional mass near stones (dist<=2): {near_mass*100:.1f}%")
    print(f"  Conditional mass far from stones (dist>2): {far_mass*100:.1f}%")

    # Also check a NEAR cell for comparison
    near_cells = []
    for idx in range(N_CELLS):
        q, r = _idx_to_cell(idx)
        if (q, r) in occupied:
            continue
        d = min_dist_to_stones(q, r, occupied)
        if d <= 2:
            near_cells.append((idx, d, marginal[idx].item()))
    if near_cells:
        near_cells.sort(key=lambda x: x[2], reverse=True)
        near_idx, near_dist, near_marg = near_cells[0]
        nq, nr = _idx_to_cell(near_idx)

        cond_near = pp[near_idx].clone()
        for oq, or_ in occupied:
            cond_near[_cell_to_idx(oq, or_)] = 0.0
        cond_near[near_idx] = 0.0
        cs = cond_near.sum().item()
        if cs > 0:
            cn_norm = cond_near / cs
            log_p2 = torch.log(cn_norm + 1e-30)
            ent2 = -(cn_norm * log_p2).sum().item()
            print(f"\n  Comparison — near cell ({nq},{nr}) dist={near_dist} "
                  f"marginal={near_marg:.6f}")
            print(f"  Conditional entropy: {ent2:.2f} "
                  f"({ent2/max_entropy*100:.1f}% of uniform)")


# ---------------------------------------------------------------------------
# Test 5: Visit siphoning — tree depth profile
# ---------------------------------------------------------------------------

def test_visit_siphoning(tree, pair_visits):
    print("\n" + "=" * 70)
    print("TEST 5: Visit siphoning — tree structure")
    print("=" * 70)

    root = tree.root_pos
    # Count children at root
    n_children = len(root.children) if root.children else 0
    print(f"\nRoot children (expanded pairs): {n_children}")

    # Count total nodes at each depth
    depth_nodes = defaultdict(int)
    depth_visits = defaultdict(int)

    def walk(pos, d):
        depth_nodes[d] += 1
        depth_visits[d] += pos.move_node.visit_count
        if pos.children:
            for child in pos.children.values():
                walk(child, d + 1)

    walk(root, 0)

    print(f"\nTree depth profile:")
    for d in sorted(depth_nodes.keys()):
        print(f"  depth {d}: {depth_nodes[d]} nodes, "
              f"{depth_visits[d]} total visits")

    # Root visit concentration
    root_node = root.move_node
    total_root = root_node.visit_count
    if root_node.level2:
        l2_visits = sum(
            l2.visit_count for l2 in root_node.level2.values()
        )
    else:
        l2_visits = 0

    # Visits that went into children vs stayed at root level
    child_visits = sum(depth_visits[d] for d in depth_visits if d > 0)
    print(f"\n  Root visit_count: {total_root}")
    print(f"  Visits in subtree (depth>0): {child_visits}")
    if total_root > 0:
        print(f"  Fraction siphoned to subtree: "
              f"{child_visits/total_root*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--n-sims", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    # Test 1
    pp, marginal, occupied = test_prior_concentration(model, device)

    # Test 2
    test_noise_impact(model, device)

    # Test 3
    tree, pair_visits, _ = test_mcts_visits(model, device, args.n_sims)

    # Test 4
    test_ood_conditional(pp, occupied)

    # Test 5
    test_visit_siphoning(tree, pair_visits)


if __name__ == "__main__":
    main()
