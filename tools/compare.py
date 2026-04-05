"""Compare two model checkpoints by playing them against each other.

Usage:
    python -m tools.compare checkpoint_a.pt checkpoint_b.pt
    python -m tools.compare checkpoint_a.pt checkpoint_b.pt --n-games 512 --n-sims 200
"""

import argparse
import math
import sys
import time

import torch

from model.resnet import HexResNet
from training.selfplay.train_loop import evaluate_vs_anchor


def load_model(path, device, num_blocks=10, num_filters=128):
    """Load a model from a checkpoint file."""
    model = HexResNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def elo_diff(score):
    """Elo difference from score (clamped to avoid log(0))."""
    score = max(0.001, min(0.999, score))
    return 400 * math.log10(score / (1 - score))


def score_standard_error(wins, losses, draws):
    """Standard error of the score estimate (wins + 0.5*draws) / n."""
    n = wins + losses + draws
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    # Variance of bernoulli-ish: each game is 1 (win), 0.5 (draw), or 0 (loss)
    values = [1.0] * wins + [0.0] * losses + [0.5] * draws
    mean = score
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var / n)


def elo_confidence_interval(score, se, z=1.96):
    """Elo diff with 95% CI from score and its standard error."""
    lo = max(0.001, min(0.999, score - z * se))
    hi = max(0.001, min(0.999, score + z * se))
    return elo_diff(lo), elo_diff(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two model checkpoints head-to-head")
    parser.add_argument("checkpoint_a", type=str,
                        help="Path to first checkpoint (model A)")
    parser.add_argument("checkpoint_b", type=str,
                        help="Path to second checkpoint (model B)")
    parser.add_argument("--n-games", type=int, default=512,
                        help="Number of games to play (default: 512)")
    parser.add_argument("--n-sims", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Move selection temperature (default: 0.1)")
    parser.add_argument("--num-blocks", type=int, default=10,
                        help="ResNet blocks (default: 10)")
    parser.add_argument("--num-filters", type=int, default=128,
                        help="ResNet filters (default: 128)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, mps, cpu). Default: auto-detect")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading model A: {args.checkpoint_a}")
    model_a = load_model(args.checkpoint_a, device,
                         args.num_blocks, args.num_filters)
    print(f"Loading model B: {args.checkpoint_b}")
    model_b = load_model(args.checkpoint_b, device,
                         args.num_blocks, args.num_filters)

    print(f"\nA vs B: {args.n_games} games, {args.n_sims} sims/move, "
          f"temp={args.temperature}")
    print()

    t0 = time.time()
    # evaluate_vs_anchor treats first arg as "current" and second as "anchor"
    # Results are from model A's perspective
    result = evaluate_vs_anchor(
        model_a, model_b, device,
        n_games=args.n_games,
        n_sims=args.n_sims,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0

    wins = result["wins"]
    losses = result["losses"]
    draws = result["draws"]
    score = result["score"]
    n = wins + losses + draws

    se = score_standard_error(wins, losses, draws)
    elo = elo_diff(score)
    elo_lo, elo_hi = elo_confidence_interval(score, se)

    print(f"\n{'=' * 50}")
    print(f"  Model A: {args.checkpoint_a}")
    print(f"  Model B: {args.checkpoint_b}")
    print(f"{'=' * 50}")
    print(f"  Games:  {n}")
    print(f"  A wins: {wins}  |  B wins: {losses}  |  Draws: {draws}")
    print(f"  Score:  {100 * score:.1f}% +/- {100 * se:.1f}%")
    print(f"  Elo:    {elo:+.0f}  [{elo_lo:+.0f}, {elo_hi:+.0f}] (95% CI)")
    print(f"  Time:   {elapsed:.0f}s")
    print(f"{'=' * 50}")

    if elo > 0:
        print(f"  Model A is stronger by ~{elo:.0f} Elo")
    elif elo < 0:
        print(f"  Model B is stronger by ~{-elo:.0f} Elo")
    else:
        print(f"  Models are equal")


if __name__ == "__main__":
    main()
