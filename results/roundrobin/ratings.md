Ratings (Bradley-Terry, Elo scale, F = 1500; 95% bootstrap interval)

| agent | rating | 95% interval | games |
|---|---|---|---|
| ens_distill | 1765 | 1745 – 1783 | 2800 |
| distill | 1752 | 1735 – 1772 | 2800 |
| lookahead | 1600 | 1582 – 1616 | 2800 |
| ppo_search | 1596 | 1579 – 1615 | 2800 |
| heuristic | 1551 | 1530 – 1569 | 2800 |
| ppo | 1548 | 1530 – 1566 | 2800 |
| F | 1500 | 1500 – 1500 | 2800 |
| AB2 | 1485 | 1464 – 1501 | 2800 |

Win rate of the row agent against the column agent (decisive games):

| | ens_distill | distill | lookahead | ppo_search | heuristic | ppo | F | AB2 |
|---|---|---|---|---|---|---|---|---|
| ens_distill | — | 53% | 68% | 73% | 77% | 78% | 83% | 85% |
| distill | 47% | — | 72% | 71% | 78% | 76% | 79% | 82% |
| lookahead | 32% | 28% | — | 50% | 56% | 59% | 66% | 62% |
| ppo_search | 27% | 29% | 50% | — | 57% | 55% | 65% | 65% |
| heuristic | 23% | 22% | 44% | 43% | — | 49% | 57% | 62% |
| ppo | 22% | 24% | 41% | 45% | 51% | — | 55% | 58% |
| F | 17% | 21% | 34% | 35% | 43% | 45% | — | 54% |
| AB2 | 15% | 18% | 38% | 35% | 38% | 42% | 46% | — |
