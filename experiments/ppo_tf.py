"""PPO fine-tuning of the BC transformer (RL post-training).

Learner samples from its policy vs opponents (F, or a frozen past snapshot of itself).
Reward: +-1 terminal, plus 0.1 * change in (my VP - opp VP) between my decisions.
GAE over the learner's own decision sequence.

usage: python ppo_tf.py INIT.pt OUT.pt [iters=60] [games_per_iter=90]
"""

import copy
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catanatron import Game, Color
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import get_actual_victory_points
from encode import encode_state, encode_actions
from model import CatanTransformer
from gen_data import pack
from train_bc import batch, DEV

GAMMA, LAM, CLIP, VF_COEF, ENT_COEF = 0.999, 0.95, 0.15, 0.5, 0.003
W = int(os.environ.get("WORKERS", 15))
SHAPE = float(os.environ.get("SHAPE", 0.1))
KL_COEF = float(os.environ.get("KL_COEF", 0.0))
# opponent mix, e.g. "F:0.35,AB:0.15,SELF:0.5"
OPP_MIX = [(k, float(v)) for k, v in (x.split(":") for x in os.environ.get("OPP_MIX", "F:0.7,SELF:0.3").split(","))]


def to_v(logit):
    return 2 * torch.sigmoid(logit) - 1  # value head in [-1, 1]


class Sampler(Player):
    def __init__(self, color, model, record):
        super().__init__(color)
        self.model, self.record, self.traj = model, record, []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        acts = list(playable_actions)
        s = encode_state(game, self.color)
        a = encode_actions(game, self.color, acts)
        t = lambda x: torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.model(*[t(x) for x in s], t(a[0]), t(a[1]),
                                   torch.ones(1, len(acts), dtype=torch.bool))
        dist = torch.distributions.Categorical(logits=logits[0])
        i = dist.sample().item()
        if self.record:
            st = game.state
            opp = [c for c in st.colors if c != self.color][0]
            vpd = get_actual_victory_points(st, self.color) - get_actual_victory_points(st, opp)
            self.traj.append([s, a, i, dist.log_prob(torch.tensor(i)).item(), to_v(v).item(), vpd])
        return acts[i]


def collect(args):
    sd, opp_sd, seeds, cfg = args
    torch.set_num_threads(1)
    sd = {k: torch.from_numpy(v) for k, v in sd.items()}
    if opp_sd is not None:
        opp_sd = {k: torch.from_numpy(v) for k, v in opp_sd.items()}
    model = CatanTransformer(d=cfg["d"], layers=cfg["layers"]); model.load_state_dict(sd); model.eval()
    opp_model = None
    if opp_sd is not None:
        opp_model = CatanTransformer(d=cfg["d"], layers=cfg["layers"]); opp_model.load_state_dict(opp_sd); opp_model.eval()
    stats = {f"{k}_{x}": 0 for k, _ in OPP_MIX for x in ("n", "w")}
    stats.update(games=0, wins=0)
    samples = []
    for s in seeds:
        rng = random.Random(s)
        kind = rng.choices([k for k, _ in OPP_MIX], [p for _, p in OPP_MIX])[0]
        if kind == "SELF" and opp_model is None:
            kind = "F"
        colors = [Color.RED, Color.BLUE]
        rng.shuffle(colors)  # random seat
        me = Sampler(colors[0], model, True)
        opp = {"F": lambda c: ValueFunctionPlayer(c), "AB": lambda c: AlphaBetaPlayer(c),
               "SELF": lambda c: Sampler(c, opp_model, False)}[kind](colors[1])
        players = [me, opp] if colors[0] == Color.RED else [opp, me]
        g = Game(players, seed=s)
        g.play()
        w = g.winning_color()
        final = 0.0 if w is None else (1.0 if w == me.color else -1.0)
        stats["games"] += 1; stats["wins"] += final > 0
        stats[kind + "_n"] += 1; stats[kind + "_w"] += final > 0
        tr = me.traj
        if not tr:
            continue
        # rewards between consecutive decisions: shaped VP-diff delta; terminal on the last
        rews = [SHAPE * (tr[k + 1][5] - tr[k][5]) for k in range(len(tr) - 1)] + [final]
        vals = [x[4] for x in tr] + [0.0]
        adv, gae = [0.0] * len(tr), 0.0
        for k in reversed(range(len(tr))):
            delta = rews[k] + GAMMA * vals[k + 1] - vals[k]
            gae = delta + GAMMA * LAM * gae
            adv[k] = gae
        for k, x in enumerate(tr):
            # reuse pack(): (state, actions, label=chosen idx, value-target slot=return)
            samples.append(((x[0], x[1], x[2], adv[k] + vals[k]), x[3], adv[k]))
    return samples, stats


if __name__ == "__main__":
    init, out = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    gpi = int(sys.argv[4]) if len(sys.argv) > 4 else 90
    torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", 8)))
    ck = torch.load(init, map_location="cpu")
    cfg = {"d": ck["d"], "layers": ck["layers"]}
    model = CatanTransformer(**cfg); model.load_state_dict(ck["model"]); model.to(DEV)
    ref = CatanTransformer(**cfg); ref.load_state_dict(ck["model"]); ref.to(DEV).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LR", 2e-5)))
    cpu_sd = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    snapshots = [cpu_sd()]
    seed0 = random.randrange(10**8)
    t0 = time.time()
    for it in range(iters):
        sd = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        opp_sds = [{k: v.numpy() for k, v in random.choice(snapshots).items()} for _ in range(W)]
        seeds = [seed0 + it * 10000 + j for j in range(gpi)]
        with Pool(W) as pool:
            res = pool.map(collect, [(sd, opp_sds[i], seeds[i::W], cfg) for i in range(W)])
        samples = [s for r, _ in res for s in r]
        st = {k: sum(x[k] for _, x in res) for k in res[0][1]}
        d = pack([s[0] for s in samples])
        old_lp = torch.tensor([s[1] for s in samples], device=DEV)
        adv = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=DEV)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = len(samples)
        model.train()
        for ep in range(3):
            perm = np.random.permutation(N)
            for i in range(0, N, 256):
                idx = perm[i:i + 256]
                x, y, ret = batch(d, idx)
                logits, v = model(*x)
                logp = torch.log_softmax(logits, -1)
                lp = logp.gather(1, y[:, None]).squeeze(1)
                ratio = torch.exp(lp - old_lp[idx])
                a = adv[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
                ent = -(logp.exp() * logp.masked_fill(~x[-1], 0)).sum(-1).mean()
                vl = F.mse_loss(to_v(v), ret)
                loss = pg + VF_COEF * vl - ENT_COEF * ent
                if KL_COEF:
                    with torch.no_grad():
                        ref_logp = torch.log_softmax(ref(*x)[0], -1)
                    kl = (ref_logp.exp() * (ref_logp - logp)).masked_fill(~x[-1], 0).sum(-1).mean()
                    loss = loss + KL_COEF * kl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        opp_str = " ".join(f"{k} {st[k+'_w']}/{st[k+'_n']}" for k, _ in OPP_MIX)
        print(f"iter {it} samples {N} winrate(all) {st['wins']/st['games']:.3f} "
              f"{opp_str} pg {pg.item():.3f} vl {vl.item():.3f} ent {ent.item():.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if it % 5 == 4:
            snapshots.append(cpu_sd())
            torch.save({"model": cpu_sd(), **cfg}, out)
        if it % 10 == 9:
            torch.save({"model": cpu_sd(), **cfg}, out.replace(".pt", f"_it{it+1}.pt"))
    torch.save({"model": cpu_sd(), **cfg}, out)
