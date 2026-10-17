"""F with an ore-wheat-sheep bias: per-resource production weights + extra dev-card / army weight.

spec: OWS:wheat,ore,sheep,wood,brick:dev_mult   e.g. OWS:1.5,1.5,1,0.6,0.6:10
  production weights multiply each resource's effective production (F uses 1 for all);
  dev_mult multiplies F's hand_devs and army_size weights.
"""

from catanatron.features import build_production_features
from catanatron.players.value import ValueFunctionPlayer, base_fn, DEFAULT_WEIGHTS, TRANSLATE_VARIETY

RES = ["WHEAT", "ORE", "SHEEP", "WOOD", "BRICK"]


def weighted_production(sample, name, w, variety):
    prod = sum(w[r] * sample[f"EFFECTIVE_{name}_{r}_PRODUCTION"] for r in RES)
    if variety:
        prod += sum(sample[f"EFFECTIVE_{name}_{r}_PRODUCTION"] != 0 for r in RES) * TRANSLATE_VARIETY * 2.778 / 100
    return prod


def ows_fn(w, dev_mult):
    params = dict(DEFAULT_WEIGHTS, hand_devs=DEFAULT_WEIGHTS["hand_devs"] * dev_mult,
                  army_size=DEFAULT_WEIGHTS["army_size"] * dev_mult)
    base = base_fn(params)
    pf = build_production_features(True)
    ones = {r: 1.0 for r in RES}

    def fn(game, color):
        v = base(game, color)
        s = pf(game, color)
        # swap F's uniform production term for the weighted one (ours only; enemy term unchanged)
        return v + params["production"] * (weighted_production(s, "P0", w, True)
                                           - weighted_production(s, "P0", ones, True))
    return fn


class OWSPlayer(ValueFunctionPlayer):
    def __init__(self, color, w, dev_mult=1.0):
        super().__init__(color)
        self.fn = ows_fn(w, dev_mult)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        best, best_a = float("-inf"), None
        for a in playable_actions:
            g = game.copy()
            g.execute(a)
            v = self.fn(g, self.color)
            if v > best:
                best, best_a = v, a
        return best_a


def from_spec(spec, color):
    _, ws, dm = spec.split(":")
    return OWSPlayer(color, dict(zip(RES, map(float, ws.split(",")))), float(dm))
