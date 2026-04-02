"""Trace exactly WHY PUCT visits far cells.

Replays PUCT selection sim-by-sim at root level-1 and level-2,
logging the moment a far cell first gets selected and why.

Usage:
  python diagnostic_puct_trace.py --checkpoint training/distill/resnet_results/checkpoint.pt
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
    create_tree, _build_tree_from_eval, _idx_to_cell, _cell_to_idx,
    _add_exploration_noise, N_CELLS, DIRICHLET_ALPHA, DIRICHLET_FRAC,
    PUCT_C, _puct_select_py,
)


def hex_dist(q1, r1, q2, r2):
    N = BOARD_SIZE
    dq = min(abs(q1 - q2), N - abs(q1 - q2))
    dr = min(abs(r1 - r2), N - abs(r1 - r2))
    s1, s2 = -q1 - r1, -q2 - r2
    ds = min(abs(s1 - s2) % N, N - abs(s1 - s2) % N)
    return max(dq, dr, ds)


def min_dist_to_stones(q, r, occupied):
    if not occupied:
        return 0
    return min(hex_dist(q, r, oq, or_) for oq, or_ in occupied)


def make_early_game():
    game = ToroidalHexGame()
    c = BOARD_SIZE // 2
    game.make_move(c, c)
    game.make_move(c + 1, c)
    game.make_move(c, c + 1)
    game.make_move(c - 1, c + 1)
    game.make_move(c + 1, c - 1)
    return game


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    block_ids = [int(k.split('.')[1]) for k in state if k.startswith("blocks.")]
    num_blocks = max(block_ids) + 1 if block_ids else 10
    num_filters = state["stem_conv.weight"].shape[0]
    model = HexResNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def simulate_level1_puct(node, occupied, n_sims=200):
    """Simulate PUCT at level-1 only (ignoring level-2/children).

    Shows when far cells first get visited and why.
    """
    # Classify actions by distance
    action_dist = {}
    for i in range(node.n):
        idx = node.actions[i]
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        action_dist[idx] = d

    # Show prior distribution
    near_prior_total = sum(node.priors[i] for i in range(node.n)
                          if action_dist[node.actions[i]] <= 2)
    far_prior_total = sum(node.priors[i] for i in range(node.n)
                         if action_dist[node.actions[i]] > 2)

    # Top-10 priors
    sorted_by_prior = sorted(range(node.n), key=lambda i: node.priors[i], reverse=True)
    print(f"\nLevel-1 prior distribution after noise:")
    print(f"  Total near (d<=2) prior: {near_prior_total:.4f}")
    print(f"  Total far  (d>2)  prior: {far_prior_total:.4f}")
    print(f"  Candidates: {node.n} ({sum(1 for d in action_dist.values() if d <= 2)} near, "
          f"{sum(1 for d in action_dist.values() if d > 2)} far)")

    print(f"\n  Top-15 priors:")
    for rank, i in enumerate(sorted_by_prior[:15]):
        idx = node.actions[i]
        q, r = _idx_to_cell(idx)
        d = action_dist[idx]
        tag = "FAR" if d > 2 else ""
        print(f"    #{rank+1}: ({q:2d},{r:2d}) prior={node.priors[i]:.6f} dist={d} {tag}")

    # Find highest-prior far cell
    far_indices = [(i, node.priors[i]) for i in range(node.n)
                   if action_dist[node.actions[i]] > 2]
    far_indices.sort(key=lambda x: x[1], reverse=True)

    if far_indices:
        top_far_i, top_far_prior = far_indices[0]
        top_far_idx = node.actions[top_far_i]
        fq, fr = _idx_to_cell(top_far_idx)
        print(f"\n  Highest far-cell prior: ({fq},{fr}) = {top_far_prior:.6f} "
              f"dist={action_dist[top_far_idx]}")

    # Simulate PUCT selections
    # Use fake visits/values arrays to replay
    visits = [0] * node.n
    values = [0.0] * node.n
    visit_count = 0

    first_far_sim = None

    for sim in range(n_sims):
        # PUCT select
        c_sqrt = PUCT_C * math.sqrt(visit_count)
        best_score = -1e30
        best_i = -1

        for i in range(node.n):
            vc = visits[i]
            q_val = values[i] / vc if vc > 0 else 0.0
            score = q_val + c_sqrt * node.priors[i] / (1 + vc)
            if score > best_score:
                best_score = score
                best_i = i

        idx = node.actions[best_i]
        d = action_dist[idx]

        # Simulate a value return (assume neutral ~0 for simplicity)
        visits[best_i] += 1
        visit_count += 1

        if d > 2 and first_far_sim is None:
            first_far_sim = sim
            q, r = _idx_to_cell(idx)
            print(f"\n  FIRST FAR VISIT at sim {sim}:")
            print(f"    Cell: ({q},{r}) dist={d} prior={node.priors[best_i]:.6f}")
            print(f"    Score: {best_score:.6f}")
            print(f"    visit_count (N): {visit_count}")
            print(f"    c*sqrt(N)*P/(1+0) = {PUCT_C * math.sqrt(visit_count) * node.priors[best_i]:.6f}")

            # What was the runner-up?
            scores = []
            for i in range(node.n):
                vc = visits[i] - (1 if i == best_i else 0)  # before this visit
                q_val = 0.0  # simplified
                s = q_val + PUCT_C * math.sqrt(visit_count - 1) * node.priors[i] / (1 + vc)
                scores.append((i, s))
            scores.sort(key=lambda x: x[1], reverse=True)

            print(f"    Top-5 PUCT scores at that moment:")
            for rank, (i, s) in enumerate(scores[:5]):
                ci = node.actions[i]
                cq, cr = _idx_to_cell(ci)
                cd = action_dist[ci]
                tag = " <-- SELECTED" if i == best_i else ""
                print(f"      ({cq:2d},{cr:2d}) score={s:.6f} prior={node.priors[i]:.6f} "
                      f"visits={visits[i] - (1 if i == best_i else 0)} dist={cd}{tag}")

    # Final visit distribution
    near_visits = sum(visits[i] for i in range(node.n) if action_dist[node.actions[i]] <= 2)
    far_visits = sum(visits[i] for i in range(node.n) if action_dist[node.actions[i]] > 2)
    n_visited = sum(1 for v in visits if v > 0)
    n_far_visited = sum(1 for i in range(node.n) if visits[i] > 0 and action_dist[node.actions[i]] > 2)

    print(f"\n  After {n_sims} sims (level-1 only, no children):")
    print(f"    Distinct cells visited: {n_visited} ({n_far_visited} far)")
    print(f"    Near visits: {near_visits}, Far visits: {far_visits}")

    if first_far_sim is None:
        print(f"    No far cell visited in {n_sims} sims!")

    return first_far_sim


def analyze_level2_noise(tree, occupied):
    """Check how noise at level-2 affects far-cell exploration."""
    print(f"\n{'='*60}")
    print("Level-2 analysis: conditional priors + noise")
    print(f"{'='*60}")

    # Take the top stone_1 cell
    root = tree.root_pos.move_node
    sorted_by_prior = sorted(range(root.n), key=lambda i: root.priors[i], reverse=True)
    top_i = sorted_by_prior[0]
    top_s1 = root.actions[top_i]
    tq, tr = _idx_to_cell(top_s1)

    print(f"\nTop s1 cell: ({tq},{tr}) prior={root.priors[top_i]:.6f}")

    # Get conditional priors (before noise)
    cond = tree.pair_probs[top_s1].clone()
    # Zero occupied and self
    for oq, or_ in occupied:
        cond[_cell_to_idx(oq, or_)] = 0.0
    cond[top_s1] = 0.0
    cond_total = cond.sum().item()
    if cond_total > 0:
        cond = cond / cond_total

    # Classify
    n_cands = (cond > 0).sum().item()
    near_mass = 0
    far_mass = 0
    for idx in range(N_CELLS):
        if cond[idx].item() <= 0:
            continue
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        if d <= 2:
            near_mass += cond[idx].item()
        else:
            far_mass += cond[idx].item()

    print(f"  Level-2 candidates: {int(n_cands)}")
    print(f"  Conditional prior: {near_mass*100:.2f}% near, {far_mass*100:.2f}% far")

    # After noise (many trials)
    from mcts.tree import MCTSNode, _init_node_children

    n_trials = 1000
    max_far_noised_l2 = []

    for _ in range(n_trials):
        cand_priors = [(idx, cond[idx].item()) for idx in range(N_CELLS) if cond[idx].item() > 0]
        l2 = MCTSNode()
        _init_node_children(l2, cand_priors)
        _add_exploration_noise(l2)

        max_f = 0.0
        for i in range(l2.n):
            idx = l2.actions[i]
            q, r = _idx_to_cell(idx)
            d = min_dist_to_stones(q, r, occupied)
            if d > 2:
                max_f = max(max_f, l2.priors[i])
        max_far_noised_l2.append(max_f)

    arr = np.array(max_far_noised_l2)
    # Also get baseline max far
    max_far_baseline_l2 = 0.0
    for idx in range(N_CELLS):
        if cond[idx].item() <= 0:
            continue
        q, r = _idx_to_cell(idx)
        d = min_dist_to_stones(q, r, occupied)
        if d > 2:
            max_far_baseline_l2 = max(max_far_baseline_l2, cond[idx].item())

    print(f"\n  Level-2 max far prior (no noise): {max_far_baseline_l2:.6f}")
    print(f"  Level-2 max far prior (with noise):")
    print(f"    mean: {arr.mean():.6f}")
    print(f"    p99:  {np.percentile(arr, 99):.6f}")
    if max_far_baseline_l2 > 0:
        print(f"    inflation: {arr.mean()/max_far_baseline_l2:.0f}x mean, "
              f"{np.percentile(arr, 99)/max_far_baseline_l2:.0f}x p99")

    # When would PUCT visit a far s2?
    top_cond_prior = cond.max().item()
    print(f"\n  Top conditional prior (near): {top_cond_prior:.6f}")
    print(f"  Mean noised far prior: {arr.mean():.6f}")
    if top_cond_prior > 0:
        # Far cell visited when: P_far/(1+0) > P_top/(1+n)
        # n > P_top/P_far - 1
        threshold_n = top_cond_prior / arr.mean() - 1
        print(f"  PUCT visits far s2 after ~{threshold_n:.0f} visits to top s2 (Q=0 approx)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    game = make_early_game()
    occupied = set(game.board.keys())

    # Get model output
    planes = board_to_planes_torus(game.board, game.current_player)
    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        value, pair_logits, _, _ = model(x)
    pp = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(N_CELLS, N_CELLS).cpu()
    marginal = pp.sum(dim=-1)

    # Build tree WITH noise
    tree = _build_tree_from_eval(
        game, value[0].item(), pp, marginal, planes, add_noise=True)

    print("=" * 60)
    print("PUCT TRACE: When and why does a far cell first get visited?")
    print("=" * 60)

    # Simulate level-1 PUCT (no tree expansion, pure PUCT)
    simulate_level1_puct(tree.root_pos.move_node, occupied, n_sims=200)

    # Level-2 analysis
    analyze_level2_noise(tree, occupied)


if __name__ == "__main__":
    main()
