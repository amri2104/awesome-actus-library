"""Stage-specific simulators for the pension extension.

- Stage 1: ClosedFundSimulator (closed fund, no new entrants)
- Stage 2: OpenFundSimulator   (open fund, deterministic new entrants)
"""

from .Stage1_closed import ClosedFundSimulator
from .Stage2_open import OpenFundSimulator

__all__ = ["ClosedFundSimulator", "OpenFundSimulator"]
