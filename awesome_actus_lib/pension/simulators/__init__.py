"""Stage-specific simulators for the pension extension.

- Stage 1: ClosedFundSimulator (closed fund, no new entrants)
- Stage 2: OpenFundSimulator   (open fund, deterministic new entrants)
- Stage 4: DynamicFundSimulator (open fund + Stage-3 mortality
           + deterministic salary growth + falling UWS path)
"""

from .closed_fund import ClosedFundSimulator
from .open_fund import OpenFundSimulator
from .dynamic_fund import DynamicFundSimulator, Stage4Dynamics
from .liability_paths import LiabilityPath, simulate_liability_paths

__all__ = [
    "ClosedFundSimulator",
    "OpenFundSimulator",
    "DynamicFundSimulator",
    "Stage4Dynamics",
    "simulate_liability_paths",
    "LiabilityPath",
]
