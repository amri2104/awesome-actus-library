"""Stage 5a — Monte-Carlo ALM under stochastic short rates.

Reuses awesome_actus_lib.stochastic_rates (HullWhite, CurveCalibrator,
SimulationResult) for path generation. Pension cashflows are produced by
a single deterministic DynamicFundSimulator run (Stage 4); per-path
NPV variation comes from path-specific discounting of those cashflows.

Two analytics:
  - t0 NPV distributions for assets and liabilities, using the
    path-integrated short rate exp(-Σ r_path(t_j) Δt).
  - DG-Pfad at multiple valuation dates t*:
      Vorsorgevermögen(t*, path) = CSH_cash(t*)
                                       [par, flat, path-independent,
                                        = Σ pre-t* fund cashflows
                                          (asset + liability, native signs)]
                                 + Σ_{t_k >= t*} asset_CF_k · P_HW(t*, t_k; r_t*^path)
      Vorsorgekapital(t*)        = Σ active AGH(t*)×hc(t*)
                                 + Σ retired pension(t*)×hc(t*) × ä_age(t*)
                                  [flat policy.technical_rate — Option α]
    DG(t*, path) = Vorsorgevermögen / Vorsorgekapital

    Flat cash = deliberate no-reinvestment assumption: pre-t* fund
    cashflows accumulate at par with zero growth. Future stage could
    layer a deposit rate on the cash account.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...models.CSH_gen import CSH
from ...models.cashFlowStream import CashFlowStream
from ...stochastic_rates.models.base import ShortRateModel
from ...stochastic_rates.simulation import SimulationResult
from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from ..policy import PensionPolicy
from .conversion import annuity_due


_ENTRY_RE = re.compile(r"^ENTRY_(\d{4})_AGE(\d+)$")


@dataclass
class _CohortMeta:
    cohort_id: str
    birth_year: int
    gender: str
    initial_status: str
    initial_pension: Optional[float]
    initial_headcount: float
    initial_agh: float


@dataclass
class FundProjection:
    """Deterministic per-(year, cohort) state extracted from a
    DynamicFundSimulator after run().

    Reads:
      - simulator.cohort_headcount_log: list[{year, cohort_id, headcount, status}]
      - liability_cf.events_df: INTEREST_CREDIT.agh_per_capita per (year, cohort),
                                RETIREMENT_CONV.annual_pension_per_capita per cohort
                                + gender (Stage-4 enriched).
      - fund.cohorts + optional entry_policy for cohort metadata
        (birth_year, gender).
    """

    base_year: int
    horizon_years: int
    cohort_meta: Dict[str, _CohortMeta]
    # (year, cohort_id) -> headcount, status
    hc_status: Dict[Tuple[int, str], Tuple[float, str]] = field(default_factory=dict)
    # (year, cohort_id) -> agh_per_capita (only for active cohorts in active year)
    agh_per_capita: Dict[Tuple[int, str], float] = field(default_factory=dict)
    # cohort_id -> annual_pension_per_capita (set in retirement year, valid afterward)
    pension_per_capita: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_simulator(
        cls,
        simulator,
        liability_cf: CashFlowStream,
        entry_policy: Optional[EntryPolicy] = None,
    ) -> "FundProjection":
        fund: PensionFund = simulator.fund
        base_year = fund.start_date.year
        horizon = int(max((r["year"] for r in simulator.cohort_headcount_log),
                          default=base_year)) - base_year + 1

        cohort_meta: Dict[str, _CohortMeta] = {}
        for c in fund.cohorts:
            cohort_meta[c.cohort_id] = _CohortMeta(
                cohort_id=c.cohort_id,
                birth_year=c.birth_year,
                gender=c.gender,
                initial_status=c.status,
                initial_pension=c.annual_pension,
                initial_headcount=float(c.headcount),
                initial_agh=float(c.accrued_savings),
            )
        for row in simulator.cohort_headcount_log:
            cid = row["cohort_id"]
            if cid in cohort_meta:
                continue
            m = _ENTRY_RE.match(cid)
            if m is None:
                continue
            year = int(m.group(1))
            age = int(m.group(2))
            gender = entry_policy.gender if entry_policy is not None else "unisex"
            cohort_meta[cid] = _CohortMeta(
                cohort_id=cid,
                birth_year=year - age,
                gender=gender,
                initial_status="active",
                initial_pension=None,
                initial_headcount=float(entry_policy.headcount
                                        if entry_policy is not None else 0.0),
                initial_agh=0.0,
            )

        hc_status: Dict[Tuple[int, str], Tuple[float, str]] = {}
        for row in simulator.cohort_headcount_log:
            hc_status[(int(row["year"]), row["cohort_id"])] = (
                float(row["headcount"]),
                row["status"],
            )

        df = liability_cf.events_df
        ic = df[df["type"] == "INTEREST_CREDIT"]
        agh: Dict[Tuple[int, str], float] = {}
        for _, r in ic.iterrows():
            year = int(str(r["time"])[:4])
            agh[(year, r["contractId"])] = float(r["agh_per_capita"])

        pension: Dict[str, float] = {}
        rc = df[df["type"] == "RETIREMENT_CONV"]
        for _, r in rc.iterrows():
            pension[r["contractId"]] = float(r["annual_pension_per_capita"])

        return cls(
            base_year=base_year,
            horizon_years=horizon,
            cohort_meta=cohort_meta,
            hc_status=hc_status,
            agh_per_capita=agh,
            pension_per_capita=pension,
        )

    def state_at(self, year: int, cohort_id: str) -> Optional[dict]:
        meta = self.cohort_meta.get(cohort_id)
        if meta is None:
            return None
        # Cohorts that exist only from a future entry year have no state earlier.
        if year < meta.birth_year:
            return None
        key = (year, cohort_id)
        if key in self.hc_status:
            hc, status = self.hc_status[key]
        else:
            # Initial-retired cohorts (e.g. RETIRED_1955) are stepped by Stage 4
            # and therefore present in the log for every sim year up to terminal_age.
            # If absent, treat as outside simulation range -> None.
            return None

        age = year - meta.birth_year
        pension_pc = self.pension_per_capita.get(cohort_id, meta.initial_pension or 0.0)
        if status == "active":
            agh_pc = self.agh_per_capita.get(key, meta.initial_agh)
        else:
            agh_pc = 0.0
        return {
            "cohort_id": cohort_id,
            "year": year,
            "status": status,
            "headcount": hc,
            "agh_per_capita": agh_pc,
            "pension_per_capita": pension_pc,
            "age": age,
            "gender": meta.gender,
            "birth_year": meta.birth_year,
        }

    def cohorts_at(self, year: int) -> List[dict]:
        states = []
        for cid in self.cohort_meta:
            s = self.state_at(year, cid)
            if s is not None:
                states.append(s)
        return states


@dataclass
class StochasticALMAnalysis:
    """Monte-Carlo ALM analysis under stochastic short rates."""

    asset_cf: CashFlowStream
    liability_cf: CashFlowStream
    projection: FundProjection
    policy: PensionPolicy
    mortality: MortalityTable
    model: ShortRateModel
    sim: SimulationResult
    base_date: pd.Timestamp

    def __post_init__(self):
        self.base_date = pd.to_datetime(self.base_date)
        # Precompute cumulative integral of short-rate paths (left Riemann sum)
        # cum_int_left[k, i] = Σ_{j=0}^{k-1} rates[j, i] * dt
        rates = self.sim.rates
        dt = float(self.sim.dt)
        M_plus_1 = rates.shape[0]
        cum = np.zeros_like(rates)
        if M_plus_1 > 1:
            cum[1:, :] = np.cumsum(rates[:-1, :], axis=0) * dt
        self._cum_int_left = cum  # (M+1, n_paths)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _event_times_years(self, df: pd.DataFrame) -> np.ndarray:
        return ((pd.to_datetime(df["time"]) - self.base_date).dt.days.values
                / 365.25)

    def _disc_factors(self, event_times_years: np.ndarray) -> np.ndarray:
        """(n_events, n_paths) discount factors via path-integrated short rate."""
        times = self.sim.times
        idx = np.searchsorted(times, event_times_years, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        return np.exp(-self._cum_int_left[idx, :])

    # ------------------------------------------------------------------
    # t0 NPV distributions
    # ------------------------------------------------------------------
    def asset_npv_t0_distribution(self) -> np.ndarray:
        df = self.asset_cf.events_df
        df = df[df["payoff"] != 0.0]
        if df.empty:
            return np.zeros(self.sim.n_paths)
        et = self._event_times_years(df)
        po = df["payoff"].astype(float).values
        disc = self._disc_factors(et)
        return np.sum(po[:, None] * disc, axis=0)

    def liability_npv_t0_distribution(self) -> np.ndarray:
        df = self.liability_cf.events_df
        df = df[df["payoff"] != 0.0]
        if df.empty:
            return np.zeros(self.sim.n_paths)
        et = self._event_times_years(df)
        po = df["payoff"].astype(float).values
        disc = self._disc_factors(et)
        return np.sum(po[:, None] * disc, axis=0)

    def funding_ratio_t0_distribution(self) -> np.ndarray:
        npv_a = self.asset_npv_t0_distribution()
        npv_l = self.liability_npv_t0_distribution()
        denom = np.where(npv_l == 0, np.nan, np.abs(npv_l))
        return npv_a / denom

    # ------------------------------------------------------------------
    # DG path (Vorsorgevermögen / Vorsorgekapital per stichtag, per path)
    # ------------------------------------------------------------------
    def _vorsorgekapital_at(self, year: int) -> float:
        vk = 0.0
        for state in self.projection.cohorts_at(year):
            if state["status"] == "active":
                vk += state["agh_per_capita"] * state["headcount"]
            else:
                age = state["age"]
                if age >= self.policy.terminal_age:
                    continue
                a_x = annuity_due(age, state["gender"], self.policy.technical_rate,
                                  self.mortality, self.policy.terminal_age)
                vk += state["pension_per_capita"] * state["headcount"] * a_x
        return vk

    def _vorsorgevermoegen_at(self, year: int, r_t_paths: np.ndarray) -> np.ndarray:
        """Vorsorgevermögen(t*, path) = CSH_cash(t*) + MV_remaining_bonds(t*, path).

        CSH_cash(t*) — par, flat, path-independent. Sum of ALL fund cashflows
        that occurred strictly BEFORE t*, native payoff signs:
          + asset CF (received bond coupons + matured principals)
          + liability CF (contributions positive, pensions/admin negative)
        RISK_CONTRIB is included as real cash received; the model-wide
        asymmetry (no offsetting risk-benefit outflow) is inherited uniformly
        here and is documented elsewhere. Represented as an ACTUS CSH
        contract for portfolio completeness — its notionalPrincipal IS the
        par cash term used in VV.

        MV_remaining_bonds(t*, path) — path-dependent. Cashflows with
        event_time_years >= t* priced via HullWhite.zcb_price(t*, t_k, r_t*^path).
        A cashflow exactly at t* counts as remaining (zcb_price(t*,t*)=1=par),
        so the partition is clean.

        At t* = base_year cash = 0 (no pre-t0 flows), so VV(t0) is unchanged
        relative to the original implementation.
        """
        base_year = self.base_date.year
        t_star = float(year - base_year)

        # --- 1) CSH cash account (par, flat, path-independent) -----------
        a_df = self.asset_cf.events_df
        a_df = a_df[a_df["payoff"] != 0.0]
        a_et = self._event_times_years(a_df) if not a_df.empty else np.array([])
        a_po = a_df["payoff"].astype(float).values if not a_df.empty else np.array([])

        l_df = self.liability_cf.events_df
        l_df = l_df[l_df["payoff"] != 0.0]
        l_et = self._event_times_years(l_df) if not l_df.empty else np.array([])
        l_po = l_df["payoff"].astype(float).values if not l_df.empty else np.array([])

        cash = float(np.sum(a_po[a_et < t_star])) + float(np.sum(l_po[l_et < t_star]))

        cash_csh = CSH(
            contractID=f"CASH_{year}",
            contractRole="RPA",
            creatorID="PK_ALM",
            currency="CHF",
            notionalPrincipal=cash,
            statusDate=f"{year}-01-01T00:00:00",
        )
        cash_par = float(cash_csh.terms["notionalPrincipal"].value)

        # --- 2) MV of remaining bonds (path-dependent) -------------------
        mv = np.zeros(len(r_t_paths))
        if a_et.size:
            mask = a_et >= t_star
            et_r = a_et[mask]
            po_r = a_po[mask]
            for k, t_k in enumerate(et_r):
                zcb_paths = np.array([
                    self.model.zcb_price(t_star, float(t_k), r_t=float(r))
                    for r in r_t_paths
                ])
                mv += po_r[k] * zcb_paths

        return cash_par + mv

    def deckungsgrad_path(self, valuation_years: Sequence[int]) -> pd.DataFrame:
        base_year = self.base_date.year
        times = self.sim.times
        rows = []
        for vyear in valuation_years:
            t_star = float(vyear - base_year)
            t_idx = int(np.clip(np.searchsorted(times, t_star, side="right") - 1,
                                0, len(times) - 1))
            r_t_paths = self.sim.rates[t_idx, :]
            vk = self._vorsorgekapital_at(vyear)
            vv = self._vorsorgevermoegen_at(vyear, r_t_paths)
            dg = (vv / vk) if vk != 0 else np.full_like(vv, np.nan)
            for i in range(len(vv)):
                rows.append({
                    "year": vyear,
                    "path": i,
                    "vv": float(vv[i]),
                    "vk": float(vk),
                    "dg": float(dg[i]),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------
    @staticmethod
    def var(dist: np.ndarray, alpha: float = 0.95) -> float:
        """One-sided VaR at level alpha on a loss distribution (positive = loss)."""
        return float(np.percentile(dist, 100.0 * alpha))

    @staticmethod
    def es(dist: np.ndarray, alpha: float = 0.95) -> float:
        thr = np.percentile(dist, 100.0 * alpha)
        tail = dist[dist >= thr]
        return float(np.mean(tail)) if tail.size else float("nan")

    @staticmethod
    def quantiles(dist: np.ndarray,
                  qs=(0.05, 0.5, 0.95)) -> Dict[float, float]:
        return {float(q): float(np.percentile(dist, 100.0 * q)) for q in qs}
