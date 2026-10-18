"""Team agent: F-structured evaluation with targeted fixes + a learned opening planner.

* Opening: a logistic model of P(win | placement features), fitted on simulated games, scores
  candidate settlements after simulating the rest of the snake draft (opponent = F-style greedy).
* Mid-game: one-ply search over the legal moves with an F-style evaluation that also counts hidden
  VP cards (so dev cards are bought for their expected VP), a longest-road race term and the
  expectation over the dev deck for purchases. Bank trades, Year of Plenty, Monopoly and roads are
  additionally credited with the best settlement/city they make buildable this turn (up to two
  trades deep), a narrow lookahead that does not let the search spend cards on dev-card lotteries.
  Bank trades that enable no build are only made when holding more than 7 cards, and once we hold
  longest road we keep a two-road margin over the opponent.
* Road saving (sub4): while we already have a legal settlement spot and settlements left, a paid
  road is penalised unless it completes a settlement this turn, so wood and brick are kept for the
  settlement instead of being spent on road length. Claiming longest road still outweighs it.
* Robber targeting (sub4): the opponent's production is weighted 1.3 on wheat and ore and 0.8 on
  the rest, so the robber and knights go after the city resources first.
"""
import json
import os

import vsk_core as core

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLACE = json.load(open(os.path.join(_HERE, "place_model.json")))

WEIGHTS = dict(hidden_vp=3e14, lr_progress=1e14, lr_defense=5e13,
               eprw=(0.8, 0.8, 0.8, 1.3, 1.3))   # robber: weight the opponent's wheat and ore higher
SEARCH = dict(depth=0, combo=2, trade_only_combo=True, road_save=3e13)


class Agent(core.SearchAgent):
    W = dict(core.F_WEIGHTS, **WEIGHTS)
    S = dict(core.DEFAULT_SEARCH, **SEARCH)
    PLACER = core.Placer(_PLACE)
