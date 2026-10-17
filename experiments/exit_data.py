"""Training data for the lookahead network's trainer (not included) from a tools/ab.py run with --log.

Per decision of our agent: its position with the move it played as the policy target, and, when that move
ended our turn, the position the opponent faced (no policy target). Both are labelled with the result from
the point of view of the player to move, as the value head is trained. A game's log is used only if it holds
exactly as many decisions as the harness's record of that game (the alignment check).

usage: python experiments/exit_data.py AB_DIR/ARM LOG_DIR OUT.npz
"""

import glob
import json
import os
import pickle
import sys

import numpy as np

KEYS = ["tile", "node", "edge", "me", "opp", "global"]
N_MOVES = 294


def main(arm_dir, log_dir, out):
    cols = {k: [] for k in KEYS}
    mask, target, won, game = [], [], [], []
    used = skipped = 0
    for f in sorted(glob.glob(os.path.join(arm_dir, "*_*_*.json"))):
        seed0, lo, _ = map(int, os.path.basename(f)[:-5].split("_"))
        for k, r in enumerate(sorted(json.load(open(f)), key=lambda r: r["game"])):
            path = os.path.join(log_dir, f"{seed0}_{lo}_g{k:03d}.pkl")
            recs = []
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    while True:
                        try:
                            recs.append(pickle.load(fh))
                        except EOFError:
                            break
            if len(recs) != r["a"]["decisions"]:
                skipped += 1
                continue
            used += 1
            w = {"a": 1, "b": 0}.get(r["winner"], -1)
            for rec in recs:
                rows = [(rec["x"], rec["mask"], rec["move"], w)]
                if rec["after"] is not None:
                    rows.append((rec["after"], np.ones(N_MOVES, bool), None, w if w < 0 else 1 - w))
                for x, m, mv, label in rows:
                    for key in KEYS:
                        cols[key].append(x[key])
                    t = np.zeros(N_MOVES, np.float16)
                    if mv is not None:
                        t[mv] = 1
                    mask.append(m)
                    target.append(t)
                    won.append(label)
                    game.append(seed0 * 10000 + r["game"])
    data = {k: np.stack(v) for k, v in cols.items()}
    data.update(mask=np.stack(mask), target=np.stack(target), won=np.array(won, np.int8),
                game=np.array(game, np.int64), vp=np.full((len(won), 2), -1, np.int8))
    np.savez(out, **data)
    has = data["target"].sum(1) > 0
    print(f"{used} games used, {skipped} skipped (log/record mismatch); {len(won):,} positions: "
          f"{has.sum():,} with a move target, {(~has).sum():,} after our turn; "
          f"win label mean {np.mean(data['won'][data['won'] >= 0]):.3f} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
