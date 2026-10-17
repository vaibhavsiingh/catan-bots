"""Match harness for 1v1 Catan agents on catanatron: isolated agent processes, a per-game CPU clock,
hidden information masked, paired seats and dice.

Importing this package applies the determinism fix described in `harness._patch`. Import it
before you create any Game.
"""
from harness import _patch

_patch.apply()

__all__ = ["_patch"]
__version__ = "1.0.0"
