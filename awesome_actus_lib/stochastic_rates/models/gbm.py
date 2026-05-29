from __future__ import annotations
from typing import Optional
import numpy as np

from ..simulation import SimulationResult

Array = np.ndarray

class GBMModel:
    """
    S(t + dt) = S(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0, 1).
    """
    
    name = "geometric_brownian_motion"

    def __init__(
        self,
        *,
        S0: float,
        mu: float,
        sigma: float,
        seed: Optional[int] = None,
    ):
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self._sim: Optional[SimulationResult] = None

    def reset_rng(self) -> None:
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

    def simulate(
        self,
        *,
        T: float,
        M: int,
        I: int,
        rng: Optional[np.random.Generator] = None,
    ) -> SimulationResult:
        if rng is None:
            if self._rng is not None:
                rng = self._rng
            else:
                rng = np.random.default_rng()

        dt = float(T / M)
        times = np.linspace(0, T, M + 1)
        
        # rates here represent simulated prices S
        prices = np.empty((M + 1, I), dtype=float)
        prices[0, :] = self.S0

        # Exact transition step-by-step
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol_sqrt_dt = self.sigma * np.sqrt(dt)

        for t in range(1, M + 1):
            Z = rng.normal(0.0, 1.0, size=I)
            prices[t, :] = prices[t - 1, :] * np.exp(drift + vol_sqrt_dt * Z)

        sim = SimulationResult(times=times, rates=prices)
        self._sim = sim
        return sim

    @property
    def simulation(self) -> Optional[SimulationResult]:
        return self._sim

    def require_simulation(self) -> SimulationResult:
        if self._sim is None:
            raise RuntimeError("No simulation available. Call simulate(...) first.")
        return self._sim

    def explain(self) -> str:
        doc = self.__class__.__doc__
        return doc.strip() if doc else ""

    def __repr__(self) -> str:
        return f"GBMModel(S0={self.S0}, mu={self.mu}, sigma={self.sigma})"
