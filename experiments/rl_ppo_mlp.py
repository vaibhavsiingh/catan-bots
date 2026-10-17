"""Unrestricted RL baseline: MaskablePPO + MLP on catanatron's gym env (flat feature vector, 290-ish discrete actions).

usage: python rl_ppo_mlp.py OUT_DIR [timesteps=2_000_000] [n_envs=6]
"""

import os
import sys
import random

import gymnasium
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import catanatron.gym  # noqa: F401  (registers env)
from catanatron import Color, RandomPlayer
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points

ENEMY = Color.RED
ME = Color.BLUE


class MixPlayer(Player):
    """Picks R / W / F per game (curriculum: weak opponents give early wins to learn from)."""

    def __init__(self, color, probs=(0.3, 0.3, 0.4)):
        super().__init__(color)
        self.pool = [RandomPlayer(color), WeightedRandomPlayer(color), ValueFunctionPlayer(color)]
        self.probs, self.game_id, self.cur = probs, None, None

    def decide(self, game, playable_actions):
        if game.id != self.game_id:
            self.game_id = game.id
            self.cur = random.choices(self.pool, self.probs)[0]
        return self.cur.decide(game, playable_actions)


class Shaped:
    """+-1 for win/loss, plus 0.1 * change in (my VP - opp VP) each step."""

    def __init__(self):
        self.prev = 0

    def __call__(self, action, game, p0_color):
        w = game.winning_color()
        diff = get_actual_victory_points(game.state, p0_color) - get_actual_victory_points(game.state, ENEMY)
        r = 0.1 * (diff - self.prev)
        self.prev = diff
        if w is not None:
            self.prev = 0
            r += 1.0 if w == p0_color else -1.0
        return r


def make_env(rank):
    def _f():
        torch.set_num_threads(1)
        random.seed(rank)
        env = gymnasium.make("catanatron/Catanatron-v0",
                             config={"enemies": [MixPlayer(ENEMY)], "reward_function": Shaped()})
        env = Monitor(env)
        return ActionMasker(env, lambda e: np.array(e.unwrapped.action_masks(), dtype=bool))
    return _f


if __name__ == "__main__":
    out = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(2)
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    model = MaskablePPO("MlpPolicy", env, n_steps=1024, batch_size=512, n_epochs=4, gamma=0.995,
                        learning_rate=3e-4, ent_coef=0.01, verbose=1,
                        policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=None)
    model.learn(steps, callback=CheckpointCallback(100_000 // n_envs, out, name_prefix="ppo"))
    model.save(os.path.join(out, "final"))
