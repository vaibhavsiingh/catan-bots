"""Make catanatron deterministic.

THE BUG. `catanatron.models.actions.inner_maritime_trade_possibilities` collects bank and port
trade offers in a Python `set`. Those offers contain `None` as padding, for example

    ('WHEAT', 'WHEAT', 'WHEAT', None, 'WOOD')

On Python 3.11 and older, `hash(None)` derives from `id(None)`. Address space layout
randomization moves that address on every process start, so the iteration order of the set
changes from run to run. Setting PYTHONHASHSEED does not control it.

A bot that takes an argmax over the action list then picks a different move whenever two
moves score equally, and the game diverges from that point.

MEASURED, before this fix:
  * one fixed game played in 4 fresh processes produced 4 different results
  * 1.75% of games differed between two runs of the same seed set

After this fix, 400 games replayed on the same seeds give identical results.

THE FIX. Return the offers sorted by their contents, so the order depends on the data and not
on memory layout. Python 3.12 made `hash(None)` a constant for the same reason.

We apply this at import time instead of editing the installed package. Nothing on disk changes
and `pip install --force-reinstall` cannot silently undo it.
"""
import os
import sys

import catanatron.models.actions as _actions

_APPLIED = False
_REEXEC_FLAG = "_CATAN_HARNESS_HASHSEED_REEXEC"


def ensure_hash_seed():
    """Restart this process with PYTHONHASHSEED=0 if it is not already set.

    THE SECOND HALF OF DETERMINISM. Sorting the trade offers removes one source of run-to-run
    variation. It does not remove them all: the engine also iterates sets of strings, and
    Python randomizes string hashes per process unless PYTHONHASHSEED is fixed.

    MEASURED. The same 60-game run, three times, with the trade fix applied but the hash seed
    left free: 3953, 3785 and 3822 agent decisions. With PYTHONHASHSEED=0: 3901, 3901, 3901.

    PYTHONHASHSEED only takes effect at interpreter start, so it cannot be set from inside a
    running program. The only reliable fix is to start again. Call this FIRST, before any other
    import does work. Returns False when nothing was needed.

    If you write your own driver script instead of using harness.evaluate, call this at the
    top of it, or run your script with PYTHONHASHSEED=0 yourself.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return False
    if os.environ.get(_REEXEC_FLAG) == "1":
        print("harness: could not set PYTHONHASHSEED=0; results will not be reproducible.",
              file=sys.stderr)
        return False
    env_argv = list(sys.argv)
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None:                      # started with "python -m package.module"
        argv = [sys.executable, "-m", spec.name] + env_argv[1:]
    else:                                     # started with "python script.py"
        argv = [sys.executable] + env_argv
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ[_REEXEC_FLAG] = "1"
    if os.name == "nt":
        # Windows has no exec: os.execv starts a new process and returns to the prompt at once, and
        # it mis-quotes a path with spaces. Run the restarted copy as a child and pass on its exit.
        import subprocess
        sys.exit(subprocess.call(argv))
    os.execv(sys.executable, argv)


def _sort_key(offer):
    # None sorts before any resource name.
    return tuple("" if x is None else x for x in offer)


def apply():
    """Idempotent. Returns True if this call installed the patch."""
    global _APPLIED
    if _APPLIED:
        return False
    original = _actions.inner_maritime_trade_possibilities

    def deterministic(hand_freqdeck, bank_freqdeck, port_resources):
        return sorted(original(hand_freqdeck, bank_freqdeck, port_resources), key=_sort_key)

    deterministic.__doc__ = original.__doc__
    _actions.inner_maritime_trade_possibilities = deterministic
    _protect_feature_ordering()
    _APPLIED = True
    return True


def _protect_feature_ordering():
    """SECOND FIX: `get_feature_ordering` must not touch the global random generator.

    On its first call it builds a throwaway Game with no seed, and an unseeded Game reseeds Python's
    global `random`. That first call happens inside whichever game first asks for a feature
    vector, so that game's later random events (dice, robber steals in an agent's look-ahead) come
    from a different stream than the same seed gives once the cache is warm. Measured: one seed
    played three times in one process gave 52, 69 and 69 decisions. With this wrapper it gives the
    same game every time, in any order. The feature ORDER was never affected: it is sorted.
    """
    import functools
    import random

    from catanatron import features as _features

    inner = _features.get_feature_ordering

    @functools.wraps(inner)
    def ordering(*args, **kwargs):
        state = random.getstate()
        try:
            return inner(*args, **kwargs)
        finally:
            random.setstate(state)

    _features.get_feature_ordering = ordering


def is_applied():
    return _APPLIED
