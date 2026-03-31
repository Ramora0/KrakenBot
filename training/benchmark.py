"""Benchmark MCTS self-play simulation loop on GPU.

Usage: python -m training.benchmark [--batch 256] [--sims 200]
"""

import argparse
import time

import torch
import torch.nn.functional as F

from game import ToroidalHexGame, TORUS_SIZE
from mcts.model import HexResNet, BOARD_SIZE
from mcts.tree import (
    create_trees_batched, select_leaf, expand_and_backprop,
    maybe_expand_leaf, N_CELLS, NON_ROOT_TOP_K,
)


def make_game():
    g = ToroidalHexGame()
    c = TORUS_SIZE // 2
    g.make_move(c, c)
    g.make_move(c + 1, c)
    g.make_move(c + 1, c + 1)
    g.make_move(c - 1, c)
    g.make_move(c, c + 1)
    return g


def run_benchmark(batch_size=256, n_sims=200, num_blocks=10, num_filters=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = HexResNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    model.eval()
    print(f"Model: {num_blocks} blocks, {num_filters} filters")

    games = [make_game() for _ in range(batch_size)]
    eval_buf = torch.empty(batch_size, 2, BOARD_SIZE, BOARD_SIZE)

    # Warmup
    if device.type == "cuda":
        dummy = torch.randn(4, 2, BOARD_SIZE, BOARD_SIZE, device=device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize()

    # Tree creation
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trees = create_trees_batched(games, model, device, add_noise=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_tree = time.perf_counter() - t0
    print(f"Tree creation: {t_tree * 1000:.0f}ms")

    # Simulation loop
    timers = {k: 0.0 for k in
              ["select", "delta", "fwd", "expand_gpu", "backprop", "maybe_expand"]}
    max_depth = 0
    n_expansions = 0
    depth_counts = {}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    for sim in range(n_sims):
        # Select leaves
        t0 = time.perf_counter()
        leaves = [select_leaf(trees[i], games[i]) for i in range(batch_size)]
        timers["select"] += time.perf_counter() - t0

        for lf in leaves:
            if lf.pair_depths:
                d = lf.pair_depths[-1]
                depth_counts[d] = depth_counts.get(d, 0) + 1
                if d > max_depth:
                    max_depth = d

        eval_indices = [i for i, lf in enumerate(leaves)
                        if not lf.is_terminal and lf.deltas]
        if not eval_indices:
            continue
        B = len(eval_indices)

        # Delta planes
        t0 = time.perf_counter()
        batch = eval_buf[:B]
        for j, i in enumerate(eval_indices):
            leaf, tree = leaves[i], trees[i]
            rp = tree.root_planes
            if leaf.player_flipped:
                batch[j, 0] = rp[1]
                batch[j, 1] = rp[0]
            else:
                batch[j] = rp
            for gq, gr, ch in leaf.deltas:
                actual_ch = (1 - ch) if leaf.player_flipped else ch
                batch[j, actual_ch, gq, gr] = 1.0
        timers["delta"] += time.perf_counter() - t0

        # Forward pass
        t0 = time.perf_counter()
        batch_gpu = batch[:B].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            values, pair_logits, _, _ = model(batch_gpu)
        value_list = values.float().cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timers["fwd"] += time.perf_counter() - t0

        # Expand computation (GPU)
        t0 = time.perf_counter()
        need_expand = [j for j, i in enumerate(eval_indices)
                       if leaves[i].needs_expansion]
        expand_data = {}
        if need_expand:
            ne = len(need_expand)
            exp_logits = pair_logits[need_expand].float()
            flat_logits = exp_logits.reshape(ne, -1)
            top_raw, top_idxs = flat_logits.topk(200, dim=-1)
            top_vals = F.softmax(top_raw, dim=-1)
            marginal_logits = exp_logits.logsumexp(dim=-1)
            marginals = F.softmax(marginal_logits, dim=-1)
            del exp_logits, flat_logits, top_raw, marginal_logits
            m_cpu = marginals.cpu()
            ti_cpu = top_idxs.cpu()
            tv_cpu = top_vals.cpu()
            del marginals, top_vals, top_idxs
            for jj, j in enumerate(need_expand):
                expand_data[j] = (m_cpu[jj], ti_cpu[jj], tv_cpu[jj])
            n_expansions += ne
        del pair_logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        timers["expand_gpu"] += time.perf_counter() - t0

        # Backprop
        t0 = time.perf_counter()
        for j, i in enumerate(eval_indices):
            expand_and_backprop(trees[i], leaves[i], value_list[j])
        timers["backprop"] += time.perf_counter() - t0

        # Maybe expand
        t0 = time.perf_counter()
        for j, i in enumerate(eval_indices):
            data = expand_data.get(j)
            if data is not None:
                maybe_expand_leaf(trees[i], leaves[i], *data)
        timers["maybe_expand"] += time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total_start

    t_sum = sum(timers.values())
    print(f"\n=== {n_sims} sims x {batch_size} games ===")
    print(f"Total wall time: {t_total:.1f}s")
    print()
    labels = {
        "select": "select_leaf",
        "delta": "delta planes",
        "fwd": "NN forward",
        "expand_gpu": "expand GPU",
        "backprop": "backprop",
        "maybe_expand": "maybe_expand",
    }
    for key, t in timers.items():
        print(f"  {labels[key]:>15s}: {t:7.2f}s ({100 * t / t_sum:5.1f}%)")
    print()
    print(f"Per sim: {t_total / n_sims * 1000:.1f}ms")
    print(f"Max depth: {max_depth}")
    print(f"Depth distribution: {dict(sorted(depth_counts.items()))}")
    print(f"Expansions: {n_expansions} ({n_expansions / n_sims:.0f}/sim)")
    if device.type == "cuda":
        print(f"GPU mem peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--filters", type=int, default=128)
    args = parser.parse_args()
    run_benchmark(args.batch, args.sims, args.blocks, args.filters)
