"""Train ValueNet: BCE(win) + MSE(dvp/2) + MSE(margin).

usage: python train_value.py DATA.npz [...] --out val.pt --epochs 4
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from value_model import ValueNet

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["tile", "node", "edge", "player", "glob", "win", "dvp", "margin"]


def batch(d, idx):
    t = lambda k: torch.from_numpy(d[k][idx].astype(np.float32)).to(DEV)
    return [t(k) for k in KEYS[:5]], t("win"), t("dvp"), t("margin")


def evaluate(model, d, idx, bs=2048):
    model.eval()
    s = np.zeros(4)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, w, dv, m = batch(d, idx[i:i + bs])
            wl, pdv, pm = model(*x)
            s += [F.binary_cross_entropy_with_logits(wl, w, reduction="sum").item(),
                  ((torch.sigmoid(wl) > 0.5) == (w > 0.5)).float().sum().item(),
                  F.mse_loss(pdv, dv, reduction="sum").item(),
                  F.mse_loss(pm, m, reduction="sum").item()]
    model.train()
    s /= len(idx)
    return dict(bce=s[0], acc=s[1], dvp_mse=s[2], dvp_var=float(d["dvp"][idx].astype(np.float32).var()),
                margin_mse=s[3])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="val.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--init", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    parts = [dict(np.load(p)) for p in args.data]
    d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    N = len(d["win"])
    # split by contiguous block (samples within a game are correlated; blocks approximate a game split)
    nv = min(20000, N // 10)
    val_idx, tr_idx = np.arange(N - nv, N), np.arange(N - nv)
    model = ValueNet(d=args.d, layers=args.layers)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model.to(DEV)
    print(f"{N} samples, {sum(p.numel() for p in model.parameters())/1e3:.0f}k params, device {DEV}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr_idx) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    t0, step, best = time.time(), 0, float("inf")
    for ep in range(args.epochs):
        np.random.shuffle(tr_idx)
        for i in range(0, len(tr_idx) - args.bs + 1, args.bs):
            x, w, dv, m = batch(d, tr_idx[i:i + args.bs])
            wl, pdv, pm = model(*x)
            loss = (F.binary_cross_entropy_with_logits(wl, w) + 0.25 * F.mse_loss(pdv, dv)
                    + F.mse_loss(pm, m))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 1000 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        ev = evaluate(model, d, val_idx)
        print(f"epoch {ep} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
        if ev["bce"] >= best:
            continue
        best = ev["bce"]
        print(f"  saved (best bce {best:.4f})", flush=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "d": args.d, "layers": args.layers}, args.out)
