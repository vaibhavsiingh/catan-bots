"""Behavior cloning: policy cross-entropy on teacher action + BCE on game outcome.

usage: python train_bc.py DATA.npz [DATA2.npz ...] --out ckpt.pt --epochs 4
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from model import CatanTransformer

KEYS = ["tile", "node", "edge", "player", "glob", "a_int", "a_res", "n_act", "y", "v"]


def load(paths):
    parts = [dict(np.load(p)) for p in paths]
    A = max(p["a_int"].shape[1] for p in parts)
    for p in parts:
        pad = A - p["a_int"].shape[1]
        if pad:
            p["a_int"] = np.pad(p["a_int"], ((0, 0), (0, pad), (0, 0)), constant_values=-1)
            p["a_res"] = np.pad(p["a_res"], ((0, 0), (0, pad), (0, 0)))
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch(d, idx, dev=None):
    dev = dev or DEV
    t = lambda k: torch.from_numpy(d[k][idx].astype(np.float32)).to(dev)
    a_int = torch.from_numpy(d["a_int"][idx].astype(np.int64)).to(dev)
    n = torch.from_numpy(d["n_act"][idx].astype(np.int64)).to(dev)
    mask = torch.arange(a_int.shape[1], device=dev)[None] < n[:, None]
    y = torch.from_numpy(d["y"][idx].astype(np.int64)).to(dev)
    return (t("tile"), t("node"), t("edge"), t("player"), t("glob"), a_int, t("a_res"), mask), y, t("v")


def evaluate(model, d, idx, bs=1024):
    model.eval()
    correct = loss = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, y, v = batch(d, idx[i:i + bs])
            logits, val = model(*x)
            loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
    model.train()
    return loss / len(idx), correct / len(idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--init", default=None)
    args = ap.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(0)

    d = load(args.data)
    N = len(d["y"])
    perm = np.random.default_rng(0).permutation(N)
    val_idx, tr_idx = perm[:5000], perm[5000:]
    model = CatanTransformer(d=args.d, layers=args.layers)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model.to(DEV)
    print("device", DEV)
    print(f"{N} samples, {sum(p.numel() for p in model.parameters())/1e3:.0f}k params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr_idx) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        np.random.shuffle(tr_idx)
        for i in range(0, len(tr_idx) - args.bs + 1, args.bs):
            x, y, v = batch(d, tr_idx[i:i + args.bs])
            logits, val = model(*x)
            loss = F.cross_entropy(logits, y) + 0.5 * F.binary_cross_entropy_with_logits(val, v)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        vl, va = evaluate(model, d, val_idx)
        print(f"epoch {ep} val_ce {vl:.3f} val_acc {va:.3f}", flush=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "d": args.d, "layers": args.layers}, args.out)
