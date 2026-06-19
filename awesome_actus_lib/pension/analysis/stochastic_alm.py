from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from awesome_actus_lib.models.cashFlowStream import CashFlowStream

from awesome_actus_lib import PublicActusService
from awesome_actus_lib.stochastic_rates import (
    CurveCalibrator,
    create_model,
    simulation_to_reference_index,
)

from .alm import ALMAnalysis

_CURVE_FITTING_MODELS = ("hull_white", "ho_lee")


class StochasticALMAnalysis:

    def __init__(
        self,
        asset_portfolio,
        liabilities_cf: Union[CashFlowStream, List[CashFlowStream]],
        calibrator: CurveCalibrator,
        *,
        technical_rate: float,
        model: str = "hull_white",
        model_params: Optional[dict] = None,
        equity_params: Optional[dict] = None,
        n_paths: int = 500,
        horizon_years: float = 40.0,
        steps_per_year: int = 12,
        base_date: str = "2025-01-01",
        market_code: str = "IR_SCENARIO",
        seed: Optional[int] = 42,
        service: Optional[PublicActusService] = None,
        verbose: bool = True,
    ):
        self.asset_portfolio = asset_portfolio
        if isinstance(liabilities_cf, list):
            if len(liabilities_cf) != int(n_paths):
                raise ValueError(
                    f"liabilities_cf list length ({len(liabilities_cf)}) "
                    f"must match n_paths ({n_paths})"
                )
            self.liabilities_cf = list(liabilities_cf)
            self._liab_is_path_list = True
        else:
            self.liabilities_cf = liabilities_cf
            self._liab_is_path_list = False
        self.calibrator = calibrator
        self.technical_rate = float(technical_rate)
        self.model = model
        self.model_params = dict(model_params or {})
        self.equity_params = dict(equity_params or {}) if equity_params is not None else None
        self.n_paths = int(n_paths)
        self.horizon_years = float(horizon_years)
        self.steps_per_year = int(steps_per_year)
        self.base_date = base_date
        self.market_code = market_code
        self.seed = seed
        self.service = service or PublicActusService()
        self.verbose = verbose

        self._sim = None
        self._sim_equity = None
        self._asset_cfs: List = []
        self._ran = False


    def run(self) -> "StochasticALMAnalysis":

        T = self.horizon_years
        M = int(round(T * self.steps_per_year))

        params = dict(self.model_params)
        if "r0" not in params:
            raise ValueError("model_params must include 'r0' (initial short rate).")
        if self.model in _CURVE_FITTING_MODELS:
            params.setdefault("calibrator", self.calibrator)

        model = create_model(self.model, seed=self.seed, **params)
        self._sim = model.simulate(T=T, M=M, I=self.n_paths)

        if self.equity_params:
            from awesome_actus_lib.stochastic_rates.models.gbm import GBMModel
            S0 = self.equity_params.get("S0", 100.0)
            mu = self.equity_params.get("mu", 0.05)
            sigma = self.equity_params.get("sigma", 0.15)
            
            gbm_seed = self.seed + 1000 if self.seed is not None else None
            gbm_model = GBMModel(S0=S0, mu=mu, sigma=sigma, seed=gbm_seed)
            self._sim_equity = gbm_model.simulate(T=T, M=M, I=self.n_paths)

        freq = "M" if self.steps_per_year == 12 else "Y"

        if self.verbose:
            print(f"  [StochasticALM] model={self.model}  paths={self.n_paths}  "
                  f"T={T:g}y  M={M}  freq={freq}")
            if self._sim_equity is not None:
                print(f"  [StochasticALM] equity_model=gbm  S0={self.equity_params.get('S0')}  "
                      f"mu={self.equity_params.get('mu')}  sigma={self.equity_params.get('sigma')}")
            print(f"  [StochasticALM] generating {self.n_paths} ACTUS scenarios "
                  f"(one service call each)...")

        self._asset_cfs = []
        for i in range(self.n_paths):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"    scenario {i + 1}/{self.n_paths}")
                
            rf_rates = simulation_to_reference_index(
                self._sim,
                base_date=self.base_date,
                market_code=self.market_code,
                path=i,
                freq=freq,
            )
            
            risk_factors = [rf_rates]
            
            if self._sim_equity is not None:
                rf_equity = simulation_to_reference_index(
                    self._sim_equity,
                    base_date=self.base_date,
                    market_code="EQ_INDEX",
                    path=i,
                    freq=freq,
                )
                risk_factors.append(rf_equity)
                
            asset_cfs_i = self.service.generateEvents(self.asset_portfolio, riskFactors=risk_factors)
            self._asset_cfs.append(asset_cfs_i)

        self._ran = True
        return self

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("Call run() before requesting results.")

    def funding_ratio_distribution(self, as_of: str) -> np.ndarray:

        self._check_ran()
        ratios = np.empty(self.n_paths, dtype=float)
        for i, asset_cfs_i in enumerate(self._asset_cfs):
            liab_i = (
                self.liabilities_cf[i] if self._liab_is_path_list
                else self.liabilities_cf
            )
            alm = ALMAnalysis(
                assets_cf=asset_cfs_i,
                liabilities_cf=liab_i,
                flat_rate=self.technical_rate,
            )
            ratios[i] = alm.funding_ratio(as_of=as_of)
        return ratios

    def summary(self, as_of: str) -> dict:
        dist = self.funding_ratio_distribution(as_of)
        p5 = float(np.percentile(dist, 5))
        return {
            "as_of": as_of,
            "n_paths": self.n_paths,
            "mean": float(np.mean(dist)),
            "std": float(np.std(dist, ddof=1)),
            "p5": p5,
            "p50": float(np.percentile(dist, 50)),
            "p95": float(np.percentile(dist, 95)),
            "min": float(np.min(dist)),
            "max": float(np.max(dist)),
            "fr_p5": p5,                                      # 95% VaR-level floor
            "fr_es5": float(np.mean(dist[dist <= p5])),       # expected shortfall below p5
            "p_underfunded": float(np.mean(dist < 1.0)),      # P(funding ratio < 100%)
        }

    def fan_chart_data(
        self,
        dates: Sequence[str],
        quantiles: Sequence[int] = (5, 25, 50, 75, 95),
    ) -> pd.DataFrame:
        self._check_ran()
        rows = {}
        for d in dates:
            dist = self.funding_ratio_distribution(d)
            rows[pd.to_datetime(d)] = {f"p{q}": float(np.percentile(dist, q)) for q in quantiles}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()

    @property
    def simulation(self):
        self._check_ran()
        return self._sim

    @property
    def equity_simulation(self):
        self._check_ran()
        return self._sim_equity
