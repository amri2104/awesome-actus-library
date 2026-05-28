from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

from ..fund import PensionFund
from ..mortality import MortalityTable
from .deckungsgrad import DeckungsgradAnalysis
from .stochastic_alm import StochasticALMAnalysis

def _find_index_for_date(sim, base_date: str, target_date: str, freq_key: str = "M") -> int:
    base = pd.to_datetime(base_date)
    target = pd.to_datetime(target_date)
    
    _PANDAS_FREQ_MAP = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS", "A": "YS"}
    pandas_freq = _PANDAS_FREQ_MAP.get(freq_key, freq_key)
    
    dates = pd.date_range(start=base, periods=sim.times.size, freq=pandas_freq).strftime("%Y-%m-%d").tolist()
    target_str = target.strftime("%Y-%m-%d")
    
    if target_str in dates:
        return dates.index(target_str)
    
    # If not exact match, find closest date
    date_ts = pd.to_datetime(dates)
    idx = np.argmin(np.abs(date_ts - target))
    return int(idx)


class StochasticDeckungsgradAnalysis:
    """Stochastic solvency analysis (Art. 44 BVV2) using book-valued bonds/cash and stochastically simulated equities."""

    def __init__(
        self,
        salm: StochasticALMAnalysis,
        fund: PensionFund,
        mortality: MortalityTable,
        *,
        bond_book: float,
        cash_book: float,
        equity_sleeve_0: float,
    ):
        """
        Parameters
        ----------
        salm : StochasticALMAnalysis
            The run stochastics manager.
        fund : PensionFund
            The PensionFund representing the liability demographics at t0.
        mortality : MortalityTable
            MortalityTable used to value retired cohorts' liability Barwert.
        bond_book : float
            Total book value of bond portfolio at t0.
        cash_book : float
            Total book value of cash/liquid portfolio at t0.
        equity_sleeve_0 : float
            Initial equity allocation at t0 (represented by equity_sleeve_0).
        """
        self.salm = salm
        self.fund = fund
        self.mortality = mortality
        self.bond_book = float(bond_book)
        self.cash_book = float(cash_book)
        self.equity_sleeve_0 = float(equity_sleeve_0)

        # Compute deterministic Vorsorgekapital (VK) at t0
        dga = DeckungsgradAnalysis(
            self.fund,
            self.mortality,
            self.bond_book + self.cash_book + self.equity_sleeve_0,
        )
        self._vk_t0 = dga.vorsorgekapital_t0()

    @property
    def vorsorgekapital_t0(self) -> float:
        return self._vk_t0

    def deckungsgrad_distribution(self, as_of: str) -> np.ndarray:
        """Solvency Deckungsgrad (V_i(d) / VK) across all scenarios at as_of."""
        self.salm._check_ran()
        sim_equity = self.salm.equity_simulation
        n_paths = self.salm.n_paths

        # If equity is not simulated or degenerate (e.g. sigma=0), equity value is static
        if sim_equity is None or sim_equity.rates.std() < 1e-12:
            equity_val = self.equity_sleeve_0
            V_i = self.bond_book + self.cash_book + equity_val
            return np.full(n_paths, V_i / self._vk_t0)

        freq = "M" if self.salm.steps_per_year == 12 else "Y"
        idx = _find_index_for_date(sim_equity, self.salm.base_date, as_of, freq)

        # S_i(d) price for all paths at index idx
        S_i = sim_equity.rates[idx, :]
        S0 = sim_equity.rates[0, 0]

        # equity_value_i(d) = equity_sleeve_0 * S_i(d) / S0
        equity_value = self.equity_sleeve_0 * (S_i / S0)

        # V_i(d) = bond_book + cash_book + equity_value_i(d)
        V_i = self.bond_book + self.cash_book + equity_value

        # DG_i(d) = V_i(d) / VK_t0
        return V_i / self._vk_t0

    def summary(self, as_of: str) -> dict:
        """Distributional + tail-risk summary of the solvency Deckungsgrad at as_of."""
        dist = self.deckungsgrad_distribution(as_of)
        p5 = float(np.percentile(dist, 5))
        return {
            "as_of": as_of,
            "mean": float(np.mean(dist)),
            "std": float(np.std(dist, ddof=1)),
            "p5": p5,
            "p50": float(np.percentile(dist, 50)),
            "p95": float(np.percentile(dist, 95)),
            "min": float(np.min(dist)),
            "max": float(np.max(dist)),
            "dg_es5": float(np.mean(dist[dist <= p5])),
            "p_underfunded": float(np.mean(dist < 1.0)),
        }

    def fan_chart_data(
        self,
        dates: Sequence[str],
        quantiles: Sequence[int] = (5, 25, 50, 75, 95),
    ) -> pd.DataFrame:
        """Quantiles of the stochastically simulated Deckungsgrad over dates."""
        rows = {}
        for d in dates:
            dist = self.deckungsgrad_distribution(d)
            rows[pd.to_datetime(d)] = {f"p{q}": float(np.percentile(dist, q)) for q in quantiles}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()
