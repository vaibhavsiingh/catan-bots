# Experiments (archived)

Earlier research code, kept for reference. It is not maintained: the checkpoints and data it reads are not
in this repository, and a few scripts import external bots that are not included either (`gtblend`: a graph transformer blended
with the V2-Tuned heuristic; `bot`: a transformer net with opponent-aware search), looked up in `external/` or
`$EXTERNAL_BOTS`.

- `encode.py`, `model.py`, `tf_player.py`: entity-token transformer and its catanatron player (as in `agents/ppo`)
- `gen_data.py`, `train_bc.py`: behaviour cloning from F
- `ppo_tf.py`, `rl_ppo_mlp.py`: PPO with an opponent pool; PPO with an MLP policy (sb3)
- `gen_value.py`, `value_model.py`, `train_value.py`, `learned_value.py`: separate value networks
- `search_player.py`, `counter_player.py`, `ows_player.py`: one-ply lookahead, opponent-model rollout search,
  strategy-conditioned search
- `gen_exit.py`, `train_exit.py`, `exit_data.py`: expert iteration
- `heur.py`, `my_value.py`, `tune.py`, `cma_tune.py`, `fw_agent.py`: hand-written evaluations and weight tuning
- `openings.py`, `strat_stats.py`, `strat_report.py`, `judge.py`, `tsrch_probe.py`: opening and strategy studies
- `arena.py`: seat-swapped matches between catanatron players (no masking, no clock)
