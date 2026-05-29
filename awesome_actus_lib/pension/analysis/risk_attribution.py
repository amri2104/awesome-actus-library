from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from awesome_actus_lib import PublicActusService
from awesome_actus_lib.models.RiskFactor import ReferenceIndex
from awesome_actus_lib.models.cashFlowStream import CashFlowStream
from awesome_actus_lib.stochastic_rates import CurveCalibrator

from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from ..simulators import DynamicFundSimulator, Stage4Dynamics
from ..simulators.liability_paths import simulate_liability_paths
from .alm import ALMAnalysis
from .stochastic_alm import StochasticALMAnalysis


class RiskAttribution:

    def __init__(
        self,
        *,
        asset_portfolio,
        fund: PensionFund,
        entry_policy: EntryPolicy,
        dynamics: Stage4Dynamics,
        mortality: MortalityTable,
        calibrator: CurveCalibrator,
        technical_rate: float,
        model: str = "hull_white",
        model_params: Optional[dict] = None,
        n_paths: int = 200,
        horizon_years: int = 40,
        steps_per_year: int = 12,
        base_date: str = "2025-01-01",
        market_code: str = "IR_SCENARIO",
        seed_assets: int = 42,
        seed_liabilities: int = 4242,
        service: Optional[PublicActusService] = None,
        verbose: bool = True,
    ):
        self.asset_portfolio = asset_portfolio
        self.fund = fund
        self.entry_policy = entry_policy
        self.dynamics = dynamics
        self.mortality = mortality
        self.calibrator = calibrator
        self.technical_rate = float(technical_rate)
        self.model = model
        self.model_params = dict(model_params or {})
        self.n_paths = int(n_paths)
        self.horizon_years = int(horizon_years)
        self.steps_per_year = int(steps_per_year)
        self.base_date = base_date
        self.market_code = market_code
        self.seed_assets = int(seed_assets)
        self.seed_liabilities = int(seed_liabilities)
        self.service = service or PublicActusService()
        self.verbose = verbose

        self._det_asset_cfs: Optional[CashFlowStream] = None
        self._det_liab_cfs: Optional[CashFlowStream] = None
        self._stoch_asset_cfs: Optional[List[CashFlowStream]] = None
        self._liab_paths_retirees: Optional[List[CashFlowStream]] = None
        self._liab_paths_all: Optional[List[CashFlowStream]] = None

    def _ensure_det_assets(self) -> None:
        if self._det_asset_cfs is not None:
            return
        if self.verbose:
            print(f"  [RiskAttribution] generating deterministic asset CFS "
                  f"(flat IR_SCENARIO @ {self.technical_rate:.2%})")
        start_year = int(self.base_date[:4])
        flat_df = pd.DataFrame({
            "date": [f"{y}-01-01" for y in range(start_year, start_year + self.horizon_years + 1)],
            "value": [self.technical_rate] * (self.horizon_years + 1),
        })
        rf = ReferenceIndex(marketObjectCode=self.market_code, source=flat_df)
        self._det_asset_cfs = self.service.generateEvents(self.asset_portfolio, riskFactors=[rf])

    def _ensure_det_liabilities(self) -> None:
        if self._det_liab_cfs is not None:
            return
        if self.verbose:
            print("  [RiskAttribution] generating deterministic liability CFS")
        sim = DynamicFundSimulator(
            fund=self.fund,
            entry_policy=self.entry_policy,
            dynamics=self.dynamics,
            mortality=self.mortality,
        )
        self._det_liab_cfs = sim.run(horizon_years=self.horizon_years)

    def _ensure_stoch_assets(self) -> None:
        if self._stoch_asset_cfs is not None:
            return
        self._ensure_det_liabilities()
        if self.verbose:
            print(f"  [RiskAttribution] generating {self.n_paths} stochastic asset CFS "
                  f"(model={self.model})")
        salm = StochasticALMAnalysis(
            asset_portfolio=self.asset_portfolio,
            liabilities_cf=self._det_liab_cfs,  
            calibrator=self.calibrator,
            technical_rate=self.technical_rate,
            model=self.model,
            model_params=self.model_params,
            n_paths=self.n_paths,
            horizon_years=float(self.horizon_years),
            steps_per_year=self.steps_per_year,
            base_date=self.base_date,
            market_code=self.market_code,
            seed=self.seed_assets,
            service=self.service,
            verbose=self.verbose,
        )
        salm.run()
        self._stoch_asset_cfs = list(salm._asset_cfs)

    def _ensure_liab_paths(self, filt: str) -> List[CashFlowStream]:
        if filt == "retirees":
            cached = self._liab_paths_retirees
        elif filt == "all":
            cached = self._liab_paths_all
        else:
            raise ValueError(f"filt must be 'retirees' or 'all', got {filt!r}")
        if cached is not None:
            return cached
        if self.verbose:
            print(f"  [RiskAttribution] generating {self.n_paths} liability paths "
                  f"(filter={filt})")
        paths = simulate_liability_paths(
            fund=self.fund,
            entry_policy=self.entry_policy,
            dynamics=self.dynamics,
            mortality=self.mortality,
            horizon_years=self.horizon_years,
            n_paths=self.n_paths,
            seed=self.seed_liabilities,
            mortality_filter=filt,
        )
        if filt == "retirees":
            self._liab_paths_retirees = paths
        else:
            self._liab_paths_all = paths
        return paths

    def _pair_fr(
        self,
        asset: Union[CashFlowStream, List[CashFlowStream]],
        liab: Union[CashFlowStream, List[CashFlowStream]],
        as_of: str,
    ) -> np.ndarray:
        ratios = np.empty(self.n_paths, dtype=float)
        for i in range(self.n_paths):
            a = asset[i] if isinstance(asset, list) else asset
            l = liab[i] if isinstance(liab, list) else liab
            alm = ALMAnalysis(
                assets_cf=a,
                liabilities_cf=l,
                flat_rate=self.technical_rate,
            )
            ratios[i] = alm.funding_ratio(as_of=as_of)
        return ratios

    def run_baseline(self, as_of: Optional[str] = None) -> np.ndarray:
        as_of = as_of or self.base_date
        self._ensure_det_assets()
        self._ensure_det_liabilities()
        alm = ALMAnalysis(
            assets_cf=self._det_asset_cfs,
            liabilities_cf=self._det_liab_cfs,
            flat_rate=self.technical_rate,
        )
        fr = alm.funding_ratio(as_of=as_of)
        return np.full(self.n_paths, fr, dtype=float)

    def run_mortality_retirees_only(self, as_of: Optional[str] = None) -> np.ndarray:
        as_of = as_of or self.base_date
        self._ensure_det_assets()
        liab = self._ensure_liab_paths("retirees")
        return self._pair_fr(self._det_asset_cfs, liab, as_of)

    def run_mortality_all_cohorts(self, as_of: Optional[str] = None) -> np.ndarray:
        as_of = as_of or self.base_date
        self._ensure_det_assets()
        liab = self._ensure_liab_paths("all")
        return self._pair_fr(self._det_asset_cfs, liab, as_of)

    def run_assets_only(self, as_of: Optional[str] = None) -> np.ndarray:
        as_of = as_of or self.base_date
        self._ensure_stoch_assets()
        self._ensure_det_liabilities()
        return self._pair_fr(self._stoch_asset_cfs, self._det_liab_cfs, as_of)

    def run_full(self, as_of: Optional[str] = None) -> np.ndarray:
        as_of = as_of or self.base_date
        self._ensure_stoch_assets()
        liab = self._ensure_liab_paths("all")
        return self._pair_fr(self._stoch_asset_cfs, liab, as_of)

    def attribution(self, as_of: Optional[str] = None) -> Dict[str, Any]:
        as_of = as_of or self.base_date
        dists = {
            "baseline": self.run_baseline(as_of),
            "mortality_retirees_only": self.run_mortality_retirees_only(as_of),
            "mortality_all_cohorts": self.run_mortality_all_cohorts(as_of),
            "assets_only": self.run_assets_only(as_of),
            "full": self.run_full(as_of),
        }
        var_full = float(np.var(dists["full"], ddof=1))
        var_assets = float(np.var(dists["assets_only"], ddof=1))
        var_mort_all = float(np.var(dists["mortality_all_cohorts"], ddof=1))
        var_mort_ret = float(np.var(dists["mortality_retirees_only"], ddof=1))
        var_active = var_mort_all - var_mort_ret

        additivity_residual = (
            abs(var_full - (var_assets + var_mort_all)) / var_full
            if var_full > 0 else float("nan")
        )
        active_share = var_active / var_mort_all if var_mort_all > 0 else float("nan")
        return {
            "as_of": as_of,
            "n_paths": self.n_paths,
            "var_full": var_full,
            "var_assets_only": var_assets,
            "var_mortality_all": var_mort_all,
            "var_mortality_retirees": var_mort_ret,
            "var_active_mortality_empirical": var_active,
            "active_mortality_share_of_total_mortality": active_share,
            "additivity_residual": additivity_residual,
            "dists": dists,
        }

    def summary(self, as_of: Optional[str] = None) -> pd.DataFrame:
        as_of = as_of or self.base_date
        attr = self.attribution(as_of)
        rows = []
        for name, dist in attr["dists"].items():
            rows.append({
                "config": name,
                "mean": float(np.mean(dist)),
                "std": float(np.std(dist, ddof=1)),
                "p5": float(np.percentile(dist, 5)),
                "p50": float(np.percentile(dist, 50)),
                "p95": float(np.percentile(dist, 95)),
                "var": float(np.var(dist, ddof=1)),
            })
        return pd.DataFrame(rows)
