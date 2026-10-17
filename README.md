# Catan bots

Agents for 1v1 Settlers of Catan on [catanatron](https://github.com/bcollazo/catanatron), a fair match harness
to rate them, and a web app to play against the strongest one.

**The `distill` agent beats catanatron's ValueFunctionPlayer (F) 84.7% of the time, and its AlphaBeta search
82–84%, at depth 2 or 3 and even when AlphaBeta sees the hidden cards.** It needs about 0.3 s of CPU per game.

## Results

Win rates in 1v1, standard rules, first to 10 VP. Every agent sees only what a player sees (the opponent's
development cards are hidden). Each board is played from both seats with the same dice, so two agents can be
compared board by board. Records are in `results/gauntlet/`; `python tools/summarize.py` rebuilds this table.

| Opponent | Games | distill | F (reference) | Paired gap |
|---|---|---|---|---|
| F (catanatron ValueFunctionPlayer) | 300 | **84.7% ± 4.1** | 51.3% | +33 ± 5 |
| AlphaBeta, depth 2 | 300 | **84.0% ± 4.1** | 50.7% | +33 ± 7 |
| AlphaBeta, depth 2, sees hidden cards | 300 | **82.7% ± 4.3** | 51.3% | |
| AlphaBeta, depth 3, sees hidden cards | 100 | **82.0% ± 7.5** | 46.0% | |
| MCTS (100 simulations), Random, WeightedRandom, VictoryPoint | 100–300 | 100% | 100% | |
| HexMachina E2-G, E7-B, E5-B-G (LLM-evolved players, [repro](https://github.com/OguroHayato/hexmachina-repro)) | 100 each | 89%, 91%, 92% | 57%, 70%, 62% | |
| `ppo` (this repo) | 300 | 76.7% | 36.7% | +40 ± 7 |
| `lookahead` (this repo) | 300 | 74.7% | 32.7% | +42 ± 7 |
| `ppo_search` (this repo) | 300 | 71.7% | 35.0% | +37 ± 7 |
| `heuristic` (this repo) | 300 | 69.0% | 42.7% | +26 ± 7 |

± is a 95% interval. The paired gap is distill's win rate minus F's against the same opponent on the same boards.
"Sees hidden cards" runs use `tools/native_match.py`, where the opponent plays on the true game state, as it
does in catanatron's own matches. Catanatron's README rates AlphaBeta at 80% against F over 25 games; with
hidden cards masked, or over 300 games, we measure it level with F.

## Agents

All agents implement one interface, `Agent().decide(obs, playable_actions) -> index`, and run in the harness.

| Agent | What it is |
|---|---|
| `agents/distill` | F's value function extended with a turn planner (trade chains, road → settlement), win checks, knight and robber rules and a learned opening model, plus an ensemble of three small MLPs (102 features) that override a move when they rate another one clearly better. The strongest agent. |
| `agents/lookahead` | A transformer policy/value network over board entities, with a value-guided lookahead over its top moves (setup placements and two-step turns) under the CPU clock. |
| `agents/ppo` | An entity-token transformer (its own encoder) trained by behaviour cloning from F, then PPO against an opponent pool. One forward pass per decision. |
| `agents/ppo_search` | `ppo` plus an opponent-model search: rollouts of our turn and the opponent's reply, leaves scored by a value net. |
| `agents/heuristic` | A linear evaluation over hand-designed features with a one-move lookahead that averages over chance. |
| `agents/f` | catanatron's ValueFunctionPlayer as an agent: the reference opponent. |
| `agents/random` | Uniformly random legal moves. |

## Quick start

Python 3.11. The engine is pinned to one catanatron commit: PyPI's catanatron has a different action space.

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # Linux: add --extra-index-url https://download.pytorch.org/whl/cpu for CPU torch
```

Play against the bot in the browser:
```
python web/server.py                     # http://localhost:8000/
```

Rate two agents (300 games, both seats of 150 boards, with the CPU clock):
```
python -m harness.evaluate --a agents/distill/agent.py --b agents/f/agent.py --games 300
```

Paired A/B of several variants in parallel, and the full benchmark:
```
python tools/ab.py runs/test distill=agents/distill/agent.py F=agents/f/agent.py --games 300 --noclock
tools/gauntlet.sh            # every opponent above, 300 games each
tools/gauntlet_native.sh     # MCTS and AlphaBeta depth 3 natively (too slow for the clock), HexMachina optionally
python tools/summarize.py    # the results table
```

## The harness

`harness/` plays games between agents in separate processes:

- **Hidden information.** An agent receives a copy of the game in which the opponent's unplayed development
  cards (and the deck) are redrawn at random. Resource hands stay visible: in 1v1 they follow exactly from
  the bank. `obs.sample_world(rng)` gives a full catanatron `Game` consistent with what the agent sees.
- **CPU clock.** 300 ms per game plus 12.5 ms per decision, at most 1.5 s; after that the harness plays random
  moves for the agent. A decision silent for 3 s is killed. `--noclock` turns both off for strength tests.
- **Fair comparisons.** Game *i* is played on board `seed0 + i // 2`, seats alternate, and both seats of a
  board get the same dice.
- **Determinism.** catanatron orders bank-trade offers by memory address on Python ≤ 3.11; the harness sorts
  them so every game replays exactly.

`tools/native_match.py` instead lets a catanatron bot play on the true state, for upper-bound tests.

## Web app

`web/server.py` (standard library only) serves a 1v1 game against `agents/distill` (choose another with
`CATAN_AGENT=path/to/agent.py`). Dice and steals come from the server, not from either player.
`web/Dockerfile` builds an image of the public site.

`python web/server.py --advisor` additionally serves, on localhost only, a tool that mirrors a game played
elsewhere (enter the board and the opponent's moves by clicking) and shows the bot's moves. `tools/advise.py`
is the same in the terminal.

## What did not help

Measured with paired A/B tests, each within noise or worse: averaging the value over chance outcomes, wider
opening search, confidence gating, value ensembles for the lookahead agent, opening search with F's replies,
expert-iteration fine-tuning, scaling single weights of F's value function, CMA-ES over F's weights,
re-distilling the distill agent into a smaller one, and extra heuristics on top of the distill agent
(opponent denial, Longest Road and development-card bonuses).

## Layout

```
agents/        the agents (each folder is self-contained: agent.py plus its weights)
harness/       match harness: processes, clock, masking, paired seats and dice
tools/         ab.py (paired A/B runs), native_match.py, cat_agent.py (catanatron bots as agents),
               gauntlet scripts, summarize.py, advise.py
web/           the web app
results/       game records behind the tables above
experiments/   earlier research code, kept for reference (see experiments/README.md)
```

## License

GPL-3.0 (`LICENSE`). The harness is adapted from GPL-3.0 code; see `NOTICE.md`.
