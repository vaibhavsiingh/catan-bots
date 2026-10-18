## Headline

| match | games | win % | notes |
|---|---|---|---|
| distill vs F (clock on) | 1000 | 84.8 ± 2.2 | 7 ms/decision, 0 games over the clock |
| distill vs AlphaBeta-2 | 1000 | 83.2 ± 2.3 | paired gap to F: +34.7 ± 3.7 pts |
| F vs AlphaBeta-2 | 1000 | 48.5 ± 3.1 |  |

## Round robin

| agent | rating | 95% interval | games |
|---|---|---|---|
| ens_distill | 1765 | 1746 – 1782 | 2800 |
| distill | 1752 | 1730 – 1773 | 2800 |
| lookahead | 1600 | 1581 – 1619 | 2800 |
| ppo_search | 1596 | 1580 – 1615 | 2800 |
| heuristic | 1551 | 1534 – 1568 | 2800 |
| ppo | 1548 | 1529 – 1569 | 2800 |
| F | 1500 | 1500 – 1500 | 2800 |
| AB2 | 1485 | 1464 – 1503 | 2800 |

Bradley-Terry, Elo scale, F = 1500; every pair 400 games on the same boards, clock off.

## Round robin: win % of the row agent

|  | ens_distill | distill | lookahead | ppo_search | heuristic | ppo | F | AB2 |
|---|---|---|---|---|---|---|---|---|
| ens_distill | — | 53 | 68 | 73 | 77 | 78 | 83 | 85 |
| distill | 47 | — | 72 | 71 | 78 | 76 | 79 | 82 |
| lookahead | 32 | 28 | — | 50 | 56 | 59 | 66 | 62 |
| ppo_search | 27 | 29 | 50 | — | 57 | 55 | 65 | 65 |
| heuristic | 23 | 22 | 44 | 43 | — | 49 | 57 | 62 |
| ppo | 22 | 24 | 41 | 45 | 51 | — | 55 | 58 |
| F | 17 | 21 | 34 | 35 | 43 | 45 | — | 54 |
| AB2 | 15 | 18 | 38 | 35 | 38 | 42 | 46 | — |

## Head to head

| match | games | win % |
|---|---|---|
| ens_distill vs distill | 2000 | 52.3 ± 2.2 |

Both seats of every board, same dice, clock off.

## Ablations

| arm | vs F | vs AB2 | vs lookahead | gap to full, mean (pts) |
|---|---|---|---|---|
| full agent | 84.3 ± 2.9 | 81.5 ± 3.1 | 74.0 ± 3.5 |  |
| all additions off | 63.7 ± 3.8 | 63.8 ± 3.8 | 49.2 ± 4.0 | -21.1 ± 2.5 |
| no value-net judge | 77.7 ± 3.3 | 75.5 ± 3.4 | 63.0 ± 3.9 | -7.9 ± 2.3 |
| judge trained on outcomes only | 81.5 ± 3.1 | 78.3 ± 3.3 | 66.5 ± 3.8 | -4.5 ± 2.2 |
| no knight rule | 80.5 ± 3.2 | 79.3 ± 3.2 | 70.3 ± 3.7 | -3.2 ± 1.8 |
| no opening model | 82.2 ± 3.1 | 82.2 ± 3.1 | 70.5 ± 3.6 | -1.7 ± 2.2 |
| no development-card term | 84.0 ± 2.9 | 81.5 ± 3.1 | 69.3 ± 3.7 | -1.7 ± 1.8 |
| no robber rule | 85.3 ± 2.8 | 82.0 ± 3.1 | 71.7 ± 3.6 | -0.3 ± 1.6 |
| no planner (also turns its chains off) | 84.2 ± 2.9 | 82.0 ± 3.1 | 74.7 ± 3.5 | +0.3 ± 1.4 |
| no trade/road chains | 84.3 ± 2.9 | 82.0 ± 3.1 | 73.5 ± 3.5 | +0.0 ± 1.0 |
| no win check | 84.3 ± 2.9 | 81.5 ± 3.1 | 74.0 ± 3.5 | +0.0 ± 0.0 |

Win % against each opponent (600 games each); the last column averages the paired per-board gaps to the full agent over the opponents. Turning the planner and the win check off together (no_turn_search) plays exactly the games of no planner.

## Margin sweep

| judge margin | vs F | vs AB2 | vs lookahead | gap to default (pts) |
|---|---|---|---|---|
| 0 | 83.3 ± 3.0 | 82.7 ± 3.0 | 70.8 ± 3.6 | -1.0 ± 2.0 |
| 0.01 | 84.3 ± 2.9 | 81.0 ± 3.1 | 71.5 ± 3.6 | -1.0 ± 1.7 |
| 0.03 (default) | 84.3 ± 2.9 | 81.5 ± 3.1 | 74.0 ± 3.5 | 0 |
| 0.05 | 80.2 ± 3.2 | 79.3 ± 3.2 | 70.5 ± 3.6 | -3.3 ± 2.1 |
| 0.1 | 80.7 ± 3.2 | 79.7 ± 3.2 | 64.5 ± 3.8 | -5.0 ± 2.2 |
| ∞ (no judge) | 77.7 ± 3.3 | 75.5 ± 3.4 | 63.0 ± 3.9 | -7.9 ± 2.3 |

The judge replaces the rule-based move when the value net rates its own best move this much higher in P(win); 0 lets it choose every main-turn move.

## Stronger baselines

| opponent | games | distill win % | F win % |
|---|---|---|---|
| AlphaBeta, depth 3 | 300 | 83.3 ± 4.2 | 45.0 ± 5.6 |
| HexMachina e2g | 300 | 89.7 ± 3.4 | 57.0 ± 5.6 |
| HexMachina e7b | 300 | 95.3 ± 2.4 | 73.0 ± 5.0 |
| HexMachina e5bg | 300 | 90.7 ± 3.3 | 59.7 ± 5.6 |
| GreedyPlayouts | 100 | 97.0 ± 3.3 | 88.0 ± 6.4 |
| MCTS, 1000 playouts | 100 | 100.0 ± 0.0 | 98.0 ± 2.7 |
| AlphaBeta, depth 2 | 300 | 82.7 ± 4.3 | 51.3 ± 5.7 |
| MCTS, 100 playouts | 100 | 100.0 ± 0.0 | 100.0 ± 0.0 |

The opponent plays on the true game (tools/native_match.py), clock off.
