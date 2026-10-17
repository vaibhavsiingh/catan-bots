"""Combined agent (value nets distilled from a rollout teacher, override margin 0.02): agent2 (a2.py) plays every decision; in the main turn, the
value-net ensemble (v4net.py + value_w*.npz) scores the same actions and overrides agent2's pick
when its best action is estimated more than MARGIN higher in P(win).

Measured in-process vs F, 800 fresh paired games: 80.9% vs agent2 alone 76.9% (+4.0 +-3.1).
CPU guard: the net is consulted only while plenty of the game's allowance is left.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json      # noqa: E402

import a2        # noqa: E402
import v4net     # noqa: E402

MARGIN = 0.02        # P(win) the net must gain over agent2's pick to override it
NET_MIN_CPU = 0.25   # seconds of allowance that must remain to consult the net

# Ablation switches (environment; unset = the agent as shipped):
#   DISTILL_MARGIN=inf          never override: agent2 alone
#   DISTILL_CONFIG='{"planner": false}'      a2.CONFIG overrides (planner, win_check, knight_rule, ...)
#   DISTILL_A2_WEIGHTS='{"dev_owned": 0}'    a2.WEIGHTS overrides (lr_always, port2, port3, dev_owned)
#   DISTILL_OPENING=0           hand-written opening score instead of the learned opening model
#   DISTILL_JUDGE=outcome       the judge before distillation (trained on game outcomes): ablation/outcome_value_w*.npz
#   AGENT_WEIGHTS=a.npz,b.npz   any other judge networks (v4net)
MARGIN = float(os.environ.get("DISTILL_MARGIN", MARGIN))
a2.CONFIG.update(json.loads(os.environ.get("DISTILL_CONFIG", "{}")))
a2.WEIGHTS.update(json.loads(os.environ.get("DISTILL_A2_WEIGHTS", "{}")))
if os.environ.get("DISTILL_OPENING", "1") == "0":
    a2._W = None


class Agent(a2.Agent):
    def __init__(self):
        super().__init__()
        judge = os.environ.get("DISTILL_JUDGE")
        here = os.path.dirname(os.path.abspath(__file__))
        self.net = v4net.Agent(weights=None if not judge else sorted(
            os.path.join(here, "ablation", f) for f in os.listdir(os.path.join(here, "ablation"))
            if f.startswith(judge + "_value_w")))
        self._obs = None

    def decide(self, obs, actions):
        self._obs = obs
        return super().decide(obs, actions)

    def _main(self, world, me, actions, deadline):
        i = super()._main(world, me, actions, deadline)
        obs = self._obs
        try:
            if obs is None or (obs.cpu_left is not None and obs.cpu_left < NET_MIN_CPU):
                return i
            vals = self.net.scores(obs, actions, world=world)
            j = int(vals.argmax())
            if j != i and vals[j] - vals[i] > MARGIN:
                return j
        except Exception:
            pass
        return i
