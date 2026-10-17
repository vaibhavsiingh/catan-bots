"""Distil the search into the policy: fine-tune on the search's chosen moves (+ outcome value loss).

usage: python train_exit.py PREFIX_w*.npz --init ppo2_it20.pt --out exit1.pt
Searched decisions get weight 1, unsearched (policy-confident, kept as an anchor) get --w_unsearched.
"""

import argparse
import glob as globmod
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import CatanTransformer
from train_bc import DEV, KEYS, batch

KEYS_X = KEYS + ["searched"]


def load(paths):
    parts = [dict(np.load(p)) for p in paths]
    A = max(p["a_int"].shape[1] for p in parts)
    for p in parts:
        pad = A - p["a_int"].shape[1]
        if pad:
            p["a_int"] = np.pad(p["a_int"], ((0, 0), (0, pad), (0, 0)), constant_values=-1)
            p["a_res"] = np.pad(p["a_res"], ((0, 0), (0, pad), (0, 0)))
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS_X}


def evaluate(model, d, idx, bs=1024):
    model.eval()
    acc = ce = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, y, _ = batch(d, idx[i:i + bs])
            logits, _ = model(*x)
            ce += F.cross_entropy(logits, y, reduction="sum").item()
            acc += (logits.argmax(1) == y).sum().item()
    model.train()
    return ce / max(1, len(idx)), acc / max(1, len(idx))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_unsearched", type=float, default=0.3)
    args = ap.parse_args()
    torch.manual_seed(0)
    paths = [p for pat in args.data for p in sorted(globmod.glob(pat))]
    d = load(paths)
    N = len(d["y"])
    perm = np.random.default_rng(0).permutation(N)
    n_val = min(10000, N // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    val_s = val_idx[d["searched"][val_idx]]
    ck = torch.load(args.init, map_location="cpu")
    model = CatanTransformer(d=ck["d"], layers=ck["layers"])
    model.load_state_dict(ck["model"])
    model.to(DEV)
    print(f"{N} samples ({d['searched'].mean():.0%} searched) from {len(paths)} files, device {DEV}", flush=True)
    ce0, acc0 = evaluate(model, d, val_s)
    print(f"before: searched-move val_ce {ce0:.3f} acc {acc0:.3f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr_idx) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        np.random.shuffle(tr_idx)
        for i in range(0, len(tr_idx) - args.bs + 1, args.bs):
            idx = tr_idx[i:i + args.bs]
            x, y, v = batch(d, idx)
            w = torch.from_numpy(np.where(d["searched"][idx], 1.0, args.w_unsearched).astype(np.float32)).to(DEV)
            logits, val = model(*x)
            loss = ((F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum()
                    + 0.5 * F.binary_cross_entropy_with_logits(val, v))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        ce, acc = evaluate(model, d, val_s)
        print(f"epoch {ep}: searched-move val_ce {ce:.3f} acc {acc:.3f}", flush=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "d": ck["d"], "layers": ck["layers"]}, args.out)
