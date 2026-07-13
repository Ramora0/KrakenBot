"""Phase 2b: train HexResNet (WDL value) by soft distillation of the teacher.

Policy is distilled as DISTRIBUTIONS, not argmax: one soft cross-entropy of the
joint N*N pair softmax against the top-k mixture T(a_k, b) = pa(a_k)*pi2(b|a_k)
(k teacher first moves, each with its exact second-move conditional). The pi1
marginal KL is logged as a diagnostic only. For solver-proven positions the
winning-set loss (incl. the forced move) replaces the soft policy loss with an
exact hard target.

value : soft cross-entropy to a two-hot WDL target whose MEAN equals v_net
        (solver-proven -> one-hot); chain : auxiliary from (augmented) planes.

Random D6 augmentation (model/symmetry.py) is applied per batch. Data is split
three ways by game_id (train/val/test); val drives checkpoint selection, test is
a held-out generalization estimate. All metrics log to wandb.

Run in the KrakenBot venv:
  python -m training.distill_gnn.train_distill_gnn --epochs 15 --amp
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model.resnet import BOARD_SIZE, HexResNet
from model.symmetry import PERMS_TORCH, INV_PERMS_TORCH
from training.distill.train_resnet import compute_chain_targets_batch, _winning_policy_loss
from training.distill_gnn.preprocess import CACHE_VERSION, build_cache

N = BOARD_SIZE * BOARD_SIZE

# TensorDataset column order (keep in sync across load/augment/loops):
# 0 planes 1 moves 2 value 3 proven 4 pi1_idx 5 pi1_p
# 6 joint_a_idx[B,K] 7 joint_a_p[B,K] 8 joint_b_idx[B,K,M] 9 joint_b_p[B,K,M]
# 10 forced_idx 11 wsingles 12 wpairs
_ARRAYS = ["planes", "moves", "value", "proven", "pi1_idx", "pi1_p",
           "joint_a_idx", "joint_a_p", "joint_b_idx", "joint_b_p", "forced_idx",
           "winning_singles", "winning_pairs"]
_DTYPES = [torch.uint8, torch.int64, torch.float32, torch.int64, torch.int64,
           torch.float32, torch.int64, torch.float32, torch.int64, torch.float32,
           torch.int64, torch.int64, torch.int64]


def load_cache(cache_dir, shard_dir, val_fraction=0.1, test_fraction=0.1, seed=42):
    done = os.path.join(cache_dir, "DONE")
    ok = os.path.exists(done) and open(done).read().startswith(CACHE_VERSION + ":")
    if not ok:
        print("Building cache from shards...", flush=True)
        build_cache(shard_dir, cache_dir)

    arr = {n: np.load(os.path.join(cache_dir, n + ".npy")) for n in _ARRAYS}
    gids = np.load(os.path.join(cache_dir, "game_ids.npy"))

    uniq = sorted(set(gids.tolist()))
    rng = random.Random(seed); rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_fraction))
    n_test = max(1, int(len(uniq) * test_fraction))
    val_games = set(uniq[:n_val]); test_games = set(uniq[n_val:n_val + n_test])
    val_mask = np.isin(gids, list(val_games)); test_mask = np.isin(gids, list(test_games))
    tr = np.where(~val_mask & ~test_mask)[0]; va = np.where(val_mask)[0]; te = np.where(test_mask)[0]
    print(f"  {len(gids):,} positions / {len(uniq):,} games -> "
          f"train {len(tr):,} ({len(uniq)-n_val-n_test} g) / "
          f"val {len(va):,} ({n_val} g) / test {len(te):,} ({n_test} g)", flush=True)

    def ds(idx):
        return TensorDataset(*[
            torch.from_numpy(np.ascontiguousarray(arr[n][idx])).to(dt)
            for n, dt in zip(_ARRAYS, _DTYPES)])
    return ds(tr), ds(va), ds(te)


# ---------------------------------------------------------------------------
# Targets & augmentation
# ---------------------------------------------------------------------------

def wdl_soft_target(value, proven):
    """Scalar teacher value [B] -> soft [B,3] over {lose,draw,win} (bins -1,0,1)."""
    B = value.shape[0]
    tgt = value.new_zeros(B, 3)
    pos = value >= 0
    tgt[pos, 2] = value[pos]; tgt[pos, 1] = 1.0 - value[pos]
    neg = ~pos
    tgt[neg, 0] = -value[neg]; tgt[neg, 1] = 1.0 + value[neg]
    pw = proven == 1
    if pw.any():
        tgt[pw] = 0.0; tgt[pw, 2] = 1.0
    pl = proven == -1
    if pl.any():
        tgt[pl] = 0.0; tgt[pl, 0] = 1.0
    return tgt


def _remap(idx, perm):
    valid = idx >= 0
    out = idx.clone()
    out[valid] = perm[idx[valid]]
    return out


def augment_batch(batch, device):
    """Apply one random D6 symmetry to planes + all target indices in-place-ish."""
    (planes, moves, value, proven, pi1_idx, pi1_p, ja, jap, jb, jbp,
     forced_idx, ws, wp) = batch
    k = random.randrange(12)
    if k == 0:
        return batch
    inv = INV_PERMS_TORCH[k].to(device); perm = PERMS_TORCH[k].to(device)
    B, C, _, _ = planes.shape
    planes = planes.reshape(B, C, N)[:, :, inv].reshape(B, C, BOARD_SIZE, BOARD_SIZE)
    return (planes, _remap(moves, perm), value, proven, _remap(pi1_idx, perm), pi1_p,
            _remap(ja, perm), jap, _remap(jb, perm), jbp,
            _remap(forced_idx, perm), _remap(ws, perm), _remap(wp, perm))


def _dist_kd_per(logp, idx, p):
    """-sum p*log q over a sparse target (idx[-1]=pad). logp [B,N]. Returns [B].

    Padding slots (and any cell the student masked to -inf) are zeroed *before*
    the multiply so 0 * -inf never produces NaN.
    """
    valid = idx >= 0
    pn = p * valid
    pn = pn / pn.sum(dim=1, keepdim=True).clamp(min=1e-8)
    gathered = torch.gather(logp, 1, idx.clamp(min=0))
    gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    return -(pn * gathered).sum(dim=1)


def marginal_kd_per(pair_logits, pi1_idx, pi1_p):
    """CE(pi1 top-32 || softmax(marginalize(pair))), per row.

    Auxiliary loss (weight --w-marg) and diagnostic. The joint mixture CE
    already contains an exact marginal-CE component, but only over the top-8
    first moves; this term extends the supervised support to pi1's top-32,
    training the tail ordering of the very distribution root PUCT consumes
    (marginalize == logsumexp over stone-2, same as the search)."""
    logp = F.log_softmax(HexResNet.marginalize(pair_logits), dim=-1)
    return _dist_kd_per(logp, pi1_idx, pi1_p)


def joint_kd_per(pair_logits, a_idx, a_p, b_idx, b_p):
    """Soft cross-entropy of the joint pair softmax to the top-k MIXTURE
    T(a_k, b) = pa(a_k) * pi2(b | a_k), one exact conditional per first move.

    This is the head's native objective (one softmax over the flattened N*N pair
    logits, same shape as the hard-CE the model was designed for), so it trains
    well — unlike a separate marginal+conditional decomposition, which optimizes
    poorly on the symmetric masked head. The target's a-marginal recovers the
    (renormalized top-k of) pi1 and each conditional is the teacher's true
    pi2|a_k, realizing "distribution over first moves, full distribution over
    seconds". Unordered pairs a==b are dropped. a_idx/a_p [B,K],
    b_idx/b_p [B,K,M] (idx -1 = pad). Returns [B].
    """
    B = pair_logits.shape[0]
    flat = pair_logits.reshape(B, -1).float()
    logZ = flat.logsumexp(dim=-1)                                    # [B]
    a_v = (a_idx >= 0).float(); b_v = (b_idx >= 0).float()
    T = (a_p * a_v).unsqueeze(2) * (b_p * b_v)                       # [B,K,M]
    T = T * (a_idx.unsqueeze(2) != b_idx).float()                    # drop a==b
    mass = T.reshape(B, -1).sum(dim=1)
    T = T / mass.clamp(min=1e-8).reshape(B, 1, 1)
    pf = a_idx.clamp(min=0).unsqueeze(2) * N + b_idx.clamp(min=0)
    A_at = torch.gather(flat, 1, pf.reshape(B, -1)).reshape(T.shape)
    A_at = torch.where(T > 0, A_at, torch.zeros_like(A_at))          # avoid 0 * -inf
    per = logZ - (T * A_at).sum(dim=(1, 2))                          # = -sum T log q
    return torch.where(mass > 0, per, torch.zeros_like(per))         # no target -> 0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(vlogits, pair_logits, chain_pred, planes, batch, weights):
    (_planes, moves, value, proven, pi1_idx, pi1_p, ja, jap, jb, jbp,
     forced_idx, wsingles, wpairs) = batch
    w_value, w_pol, w_chain, w_marg = weights
    B = vlogits.shape[0]

    vtgt = wdl_soft_target(value, proven)
    value_loss = -(vtgt * F.log_softmax(vlogits, dim=-1)).sum(dim=-1).mean()

    joint_per = joint_kd_per(pair_logits, ja, jap, jb, jbp)

    # proven positions: exact winning-set (incl. forced move) instead of soft KD
    ws = wsingles.clone()
    has_forced = forced_idx >= 0
    if has_forced.any():
        free = (ws < 0).float().argmax(dim=1)
        rows = torch.arange(B, device=ws.device)[has_forced]
        ws[rows, free[has_forced]] = forced_idx[has_forced]
    win_per, has_win = _winning_policy_loss(pair_logits, ws, wpairs)

    policy_per = torch.where(has_win, win_per, joint_per)
    policy_loss = policy_per.mean()

    with torch.no_grad():
        ctgt, cmask = compute_chain_targets_batch(planes)
    chain_loss = (((chain_pred - ctgt) ** 2) * cmask).sum() / cmask.sum().clamp(min=1)

    # marginal CE over pi1 top-32; skipped on winning-set rows (the overlay
    # one-hots the policy there and the raw teacher marginal may disagree)
    marg_per = marginal_kd_per(pair_logits, pi1_idx, pi1_p)
    marg_per = torch.where(has_win, torch.zeros_like(marg_per), marg_per)
    marg = marg_per.mean()
    total = (w_value * value_loss + w_pol * policy_loss + w_chain * chain_loss
             + w_marg * marg)
    return total, value_loss, policy_loss, marg, chain_loss


# ---------------------------------------------------------------------------
# Evaluation (shared by val + test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, weights, device):
    model.eval()
    agg = np.zeros(4); seen = 0; v_sign = 0; pair_hit = 0
    for batch in loader:
        batch = [t.to(device) for t in batch]
        batch[0] = batch[0].float()
        vlogits, pair_logits, _ml, chain = model(batch[0])
        _, vl, pol, marg, cl = compute_loss(vlogits, pair_logits, chain, batch[0], batch, weights)
        bs = len(batch[2]); seen += bs
        agg += np.array([vl.item(), pol.item(), marg.item(), cl.item()]) * bs
        ev = model.expected_value(vlogits)
        v_sign += ((ev > 0) == (batch[2] > 0)).sum().item()
        m1, m2 = batch[1][:, 0], batch[1][:, 1]
        both = (m1 >= 0) & (m2 >= 0)
        if both.any():
            top = pair_logits.reshape(bs, -1).argmax(dim=-1)
            pa, pb = top // N, top % N
            hit = ((pa == m1) & (pb == m2)) | ((pa == m2) & (pb == m1))
            pair_hit += (hit & both).sum().item()
    m = agg / max(seen, 1)
    return {"value": m[0], "policy": m[1], "marg": m[2], "chain": m[3],
            "v_sign": v_sign / max(seen, 1), "pair_acc": pair_hit / max(seen, 1)}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    train_ds, val_ds, test_ds = load_cache(
        args.cache_dir, args.shard_dir, args.val_fraction, args.test_fraction, args.seed)
    weights = (args.w_value, args.w_policy, args.w_chain, args.w_marg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HexResNet(num_blocks=args.num_blocks, num_filters=args.num_filters).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"warm start from {args.init_from} (epoch {ckpt.get('epoch')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_blocks}x{args.num_filters}, {n_params:,} params (WDL)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(total_steps, 1))))
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project,
                       name=args.run_name or f"distill_b{args.num_blocks}f{args.num_filters}",
                       config={**vars(args), "n_params": n_params,
                               "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)})
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}); continuing without", flush=True)
            use_wandb = False

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "distill_gnn.pt")
    best_val = float("inf"); t0 = time.time(); step = 0

    for epoch in range(args.epochs):
        model.train()
        agg = np.zeros(4); seen = 0
        for batch in train_loader:
            batch = [t.to(device, non_blocking=True) for t in batch]
            batch[0] = batch[0].float()
            batch = augment_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                vlogits, pair_logits, _ml, chain = model(batch[0])
                loss, vl, pol, marg, cl = compute_loss(
                    vlogits, pair_logits, chain, batch[0], batch, weights)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0); opt.step()
            sched.step(); step += 1
            bs = len(batch[2]); seen += bs
            agg += np.array([vl.item(), pol.item(), marg.item(), cl.item()]) * bs
            if use_wandb and step % args.log_every == 0:
                wandb.log({"step": step, "lr": opt.param_groups[0]["lr"],
                           "train_step/value": vl.item(), "train_step/policy": pol.item(),
                           "train_step/marg": marg.item(), "train_step/total": loss.item(),
                           "temp": model.pair_head.log_temp.exp().item()}, step=step)

        tr = agg / max(seen, 1)
        val = evaluate(model, val_loader, weights, device)
        test = evaluate(model, test_loader, weights, device)
        print(f"ep{epoch+1}/{args.epochs} | train v={tr[0]:.3f} pol={tr[1]:.3f} marg={tr[2]:.3f} | "
              f"val v={val['value']:.3f} pol={val['policy']:.3f} marg={val['marg']:.3f} "
              f"vsign={val['v_sign']:.3f} pacc={val['pair_acc']:.3f} | "
              f"test v={test['value']:.3f} vsign={test['v_sign']:.3f} pacc={test['pair_acc']:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)

        if use_wandb:
            wandb.log({"epoch": epoch + 1,
                       "train/value": tr[0], "train/policy": tr[1], "train/marg": tr[2], "train/chain": tr[3],
                       **{f"val/{k}": v for k, v in val.items()},
                       **{f"test/{k}": v for k, v in test.items()},
                       "gap/value": val["value"] - tr[0], "gap/policy": val["policy"] - tr[1]}, step=step)

        val_total = val["value"] + val["policy"] + args.w_marg * val["marg"]
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "args": vars(args), "val": val, "test": test}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        if val_total < best_val:
            best_val = val_total
            bp = os.path.join(args.output_dir, "distill_gnn_best.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "args": vars(args), "val": val, "test": test}, bp + ".tmp")
            os.replace(bp + ".tmp", bp)
            print(f"  -> new best (val_total={val_total:.3f})", flush=True)
            if use_wandb:
                wandb.run.summary["best_val_total"] = val_total
                wandb.run.summary["best_epoch"] = epoch + 1
                for k, v in test.items():
                    wandb.run.summary[f"best_test_{k}"] = v

    print(f"Done in {time.time()-t0:.0f}s. Best val_total={best_val:.3f} -> {ckpt_path}", flush=True)
    if use_wandb:
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=os.path.join(os.path.dirname(__file__), "data", "labeled_joint"))
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "data", "cache_joint"))
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--num-filters", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-chain", type=float, default=0.1)
    ap.add_argument("--w-marg", type=float, default=0.5,
                    help="weight of the top-32 marginal CE aux loss (root PUCT "
                         "consumes the marginal; joint CE only covers top-8)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--wandb-project", default="krakenbot-distill")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to warm-start model weights from (fresh "
                         "optimizer/schedule; use after an interrupted run)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
