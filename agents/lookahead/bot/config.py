"""Every knob of our bot in one place. The defaults ARE the frozen final bot (tag final-v1).

Where values come from (later wins):
    1. the defaults below
    2. environment variables: CATAN_MODEL, CATAN_K, CATAN_DEPTH2, CATAN_ROAD, ... (see ENV)
    3. a variant string, from the CATAN_VARIANT environment variable
    4. a variant string passed in code: Config.load("depth2=0,road=0")

A variant string is "key=value" pairs separated by commas. Keys are the field names below, the
aliases in ALIASES (budget, opp, value), or the name of a nudge in bot/nudges.py (its weight):
    "depth2=0"                     the final bot without the 2-step search
    "road=0,army=2,budget=600"     no Longest Road nudge, Largest Army nudge 2, 600 ms search budget
"""
import dataclasses
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass
class Config:
    model: str = os.path.join(ROOT, "checkpoints", "final.pt")   # policy (and value) checkpoint
    value_model: str = ""          # separate checkpoint for the value head; "" = same as model
    opp_model: str = "value"       # the opponent inside simulations: value | alphabeta | self
    k: int = 4                     # candidate moves searched (the policy's top K)
    m: int = 4                     # simulations per candidate (or per pair with depth2)
    beta: float = 30.0             # score = logp / beta + log(p_win)
    conf: float = 0.9              # skip the search when the policy's top move is at least this likely
    budget_ms: float = 450.0       # hard time budget per searched decision (shipped; 600 tested, not better on 5001)
    depth2: bool = True            # also branch on our policy's top-2 follow-ups within the turn
    force: bool = False            # always search tactical moves (win now, take Longest Road / Army)
    force_cap_ms: float = 50.0     # time cap for finding those tactical moves
    bias: str = ""                 # logit bonus per action type, e.g. "BUY_DEVELOPMENT_CARD:2"
    # ideas from Vaibhav's bot (VAIB_NOTES.md); all off by default
    judge: str = "own"             # leaf evaluator: own (our value head) | vaib (his value net) | avg (mean)
    prior: str = "own"             # candidates + follow-ups from: own (our policy) | vaib (his PPO policy)
    rollout: str = "own"           # our moves inside simulations: own | vaib
    setup_horizon: str = "turn"    # setup simulations stop: turn (our turn again) | full (setup finished)
    search_prompts: str = "core"   # search at: core (main turn, robber, setup) | all (every 2+ move choice)
    opp_mix: float = 0.0           # share of each candidate's simulations with opp_alt as the opponent
    opp_alt: str = "self"          # the alternative opponent model: self (our policy) | vaib (his policy)
    fast_copy: bool = False        # exact, faster copies of catanatron's board/state (bot/fastcopy.py)
    budget_mode: str = "fixed"     # fixed | adaptive: critical decisions get budget_hi_ms instead of budget_ms
    budget_hi_ms: float = 650.0    # adaptive: opening placements, robber moves, and when either side has >= 8 VP
    partial: bool = False          # out of time before every candidate had a simulation: compare the ones that did
    cpu_clock: bool = False        # measure the budget in this process's CPU time (the harness clock), not wall time
    nudges: dict = dataclasses.field(default_factory=lambda: {"road": 3.0})   # name -> weight

    # ---- loading -----------------------------------------------------------------------------
    @classmethod
    def load(cls, variant=""):
        """Defaults, then environment variables, then CATAN_VARIANT, then `variant`."""
        cfg = cls()
        for name, env in ENV.items():
            if env in os.environ:
                cfg.set(name, os.environ[env])
        from bot.nudges import REGISTRY
        for name in REGISTRY:
            env = "CATAN_" + name.upper()
            if env in os.environ:
                cfg.set(name, os.environ[env])
        return cfg.with_variant(os.environ.get("CATAN_VARIANT", "")).with_variant(variant)

    def with_variant(self, variant):
        """A copy with the variant string's settings applied."""
        cfg = dataclasses.replace(self, nudges=dict(self.nudges))
        for part in filter(None, (p.strip() for p in (variant or "").split(","))):
            key, sep, value = part.partition("=")
            if not sep:
                raise ValueError(f"variant part {part!r} is not key=value")
            cfg.set(key.strip(), value.strip())
        return cfg

    def set(self, key, value):
        key = ALIASES.get(key, key)
        fields = {f.name: f for f in dataclasses.fields(self)}
        if key in fields and key != "nudges":
            setattr(self, key, _parse(fields[key].type, value))
            return
        from bot.nudges import REGISTRY
        if key not in REGISTRY:
            known = sorted(set(fields) - {"nudges"} | set(ALIASES) | set(REGISTRY))
            raise ValueError(f"unknown setting {key!r}; known: {', '.join(known)}")
        self.nudges[key] = float(value)

    def describe(self):
        """The settings that differ from the final bot, as a variant string ('' = the final bot)."""
        base = Config()
        parts = [f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self)
                 if f.name != "nudges" and getattr(self, f.name) != getattr(base, f.name)]
        for name in sorted(set(self.nudges) | set(base.nudges)):
            if self.nudges.get(name, 0.0) != base.nudges.get(name, 0.0):
                parts.append(f"{name}={self.nudges.get(name, 0.0):g}")
        return ",".join(parts)


ENV = {"model": "CATAN_MODEL", "value_model": "CATAN_VALUE_MODEL", "opp_model": "CATAN_OPP_MODEL",
       "k": "CATAN_K", "m": "CATAN_M", "beta": "CATAN_BETA", "conf": "CATAN_CONF",
       "budget_ms": "CATAN_BUDGET_MS", "depth2": "CATAN_DEPTH2", "force": "CATAN_FORCE",
       "force_cap_ms": "CATAN_FORCE_CAP_MS", "bias": "CATAN_BIAS", "judge": "CATAN_JUDGE",
       "prior": "CATAN_PRIOR", "rollout": "CATAN_ROLLOUT", "setup_horizon": "CATAN_SETUP_HORIZON",
       "search_prompts": "CATAN_SEARCH_PROMPTS", "opp_mix": "CATAN_OPP_MIX", "opp_alt": "CATAN_OPP_ALT",
       "fast_copy": "CATAN_FAST_COPY", "budget_mode": "CATAN_BUDGET_MODE", "budget_hi_ms": "CATAN_BUDGET_HI_MS"}
ALIASES = {"budget": "budget_ms", "opp": "opp_model", "value": "value_model"}


def _parse(kind, value):
    kind = kind if isinstance(kind, str) else kind.__name__
    if kind == "bool":
        if str(value).lower() in ("1", "true", "yes", "on"):
            return True
        if str(value).lower() in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return {"int": int, "float": float, "str": str}[kind](value)
