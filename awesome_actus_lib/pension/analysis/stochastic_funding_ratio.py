from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

from ..fund import PensionFund
from ..mortality import MortalityTable
from ..simulators.liability_paths import LiabilityPath
from .conversion import annuity_due
from .funding_ratio import FundingRatioAnalysis
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


class StochasticFundingRatioAnalysis:


    def __init__(
        self,
        salm: StochasticALMAnalysis,
        fund: PensionFund,
        mortality: MortalityTable,
        *,
        bond_book: float,
        cash_book: float,
        equity_sleeve_0: float,
        liab_paths: Optional[List[LiabilityPath]] = None,
    ):
        
        self.salm = salm
        self.fund = fund
        self.mortality = mortality
        self.bond_book = float(bond_book)
        self.cash_book = float(cash_book)
        self.equity_sleeve_0 = float(equity_sleeve_0)
        self.liab_paths = liab_paths

        if liab_paths is not None and len(liab_paths) != salm.n_paths:
            raise ValueError(
                f"liab_paths length ({len(liab_paths)}) must match "
                f"salm.n_paths ({salm.n_paths}) for position-wise pairing."
            )

        # Deterministic t0 reference (kept for back-compat and t0 collapse).
        fra = FundingRatioAnalysis(
            self.fund,
            self.mortality,
            self.bond_book + self.cash_book + self.equity_sleeve_0,
        )
        self._pc_t0 = fra.pension_capital_t0()
        self._fr_t0 = fra.funding_ratio_t0()

    @property
    def pension_capital_t0(self) -> float:
        return self._pc_t0

    def _legacy_distribution(self, as_of: str) -> np.ndarray:
        """Stage 5a Phase 2 behaviour: V_i(d) / PC_t0 with frozen denominator."""
        sim_equity = self.salm.equity_simulation
        n_paths = self.salm.n_paths

        if sim_equity is None or sim_equity.rates.std() < 1e-12:
            equity_val = self.equity_sleeve_0
            V_i = self.bond_book + self.cash_book + equity_val
            return np.full(n_paths, V_i / self._pc_t0)

        freq = "M" if self.salm.steps_per_year == 12 else "Y"
        idx = _find_index_for_date(sim_equity, self.salm.base_date, as_of, freq)
        S_i = sim_equity.rates[idx, :]
        S0 = sim_equity.rates[0, 0]
        equity_value = self.equity_sleeve_0 * (S_i / S0)

        V_i = self.bond_book + self.cash_book + equity_value
        return V_i / self._pc_t0


    def _equity_value(self, as_of: str, path_i: int) -> float:
        sim_equity = self.salm.equity_simulation
        if sim_equity is None or sim_equity.rates.std() < 1e-12:
            return self.equity_sleeve_0
        freq = "M" if self.salm.steps_per_year == 12 else "Y"
        idx = _find_index_for_date(sim_equity, self.salm.base_date, as_of, freq)
        S_i = float(sim_equity.rates[idx, path_i])
        S0 = float(sim_equity.rates[0, 0])
        return self.equity_sleeve_0 * (S_i / S0)

    def _path_V(self, as_of: str, path_i: int) -> float:

        as_of_ts = pd.to_datetime(as_of)
        base_ts = pd.to_datetime(self.salm.base_date)

        asset_df = self.salm._asset_cfs[path_i].events_df
        if asset_df is None or asset_df.empty:
            bond_income = 0.0
        else:
            atimes = pd.to_datetime(asset_df["time"])
            mask = (asset_df["type"] == "IP") & (atimes > base_ts) & (atimes <= as_of_ts)
            bond_income = float(asset_df.loc[mask, "payoff"].sum())

        liab_df = self.liab_paths[path_i].cashflows.events_df
        if liab_df is None or liab_df.empty:
            net_liab = 0.0
        else:
            ltimes = pd.to_datetime(liab_df["time"])
            mask_l = (
                (ltimes > base_ts)
                & (ltimes <= as_of_ts)
                & (liab_df["type"] != "RISK_CONTRIB")
            )
            net_liab = float(liab_df.loc[mask_l, "payoff"].sum())

        eq = self._equity_value(as_of, path_i)
        return self.bond_book + self.cash_book + bond_income + net_liab + eq

    def _path_PC(self, as_of: str, path_i: int) -> float:
        as_of_year = pd.to_datetime(as_of).year
        start_year = self.fund.start_date.year
        if as_of_year <= start_year:
            return self._pc_t0

        log_year = as_of_year - 1
        trace = self.liab_paths[path_i].headcount_trace
        policy = self.fund.policy
        technical_rate = policy.technical_rate
        terminal_age = policy.terminal_age

        pc = 0.0
        for row in trace:
            if row["year"] != log_year:
                continue
            age_at_t = as_of_year - row["birth_year"]
            if age_at_t >= terminal_age:
                continue
            n = row["headcount"]
            if n <= 0:
                continue
            if row["status"] == "active":
                pc += row["agh_per_capita"] * n
            else:
                ax = annuity_due(
                    age_at_t, row["gender"], technical_rate,
                    self.mortality, terminal_age,
                    cal_year=as_of_year,
                )
                pc += row["annual_pension_per_capita"] * n * ax
        return pc

    
    def funding_ratio_distribution(self, as_of: str) -> np.ndarray:
        """Solvency funding-ratio distribution at ``as_of`` across all paths.

        Legacy mode (``liab_paths=None``): ``V_i(d) / PC_t0``.
        Forward mode: ``V_t,i / PC_t,i`` with coupled survivors.
        """
        self.salm._check_ran()
        if self.liab_paths is None:
            return self._legacy_distribution(as_of)

        n = len(self.liab_paths)
        out = np.empty(n, dtype=float)
        for i in range(n):
            pc = self._path_PC(as_of, i)
            if pc == 0.0:
                out[i] = np.nan
            else:
                out[i] = self._path_V(as_of, i) / pc
        return out

    def summary(self, as_of: str) -> dict:
        """Distributional + tail-risk summary of the solvency funding ratio at as_of."""
        dist = self.funding_ratio_distribution(as_of)
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
            "fr_es5": float(np.mean(dist[dist <= p5])),
            "p_underfunded": float(np.mean(dist < 1.0)),
        }

    def fan_chart_data(
        self,
        dates: Sequence[str],
        quantiles: Sequence[int] = (5, 25, 50, 75, 95),
    ) -> pd.DataFrame:
        """Quantiles of the stochastically simulated funding ratio over dates."""
        rows = {}
        for d in dates:
            dist = self.funding_ratio_distribution(d)
            rows[pd.to_datetime(d)] = {f"p{q}": float(np.percentile(dist, q)) for q in quantiles}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()
