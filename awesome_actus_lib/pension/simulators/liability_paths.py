from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...models.cashFlowStream import CashFlowStream
from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from .Stage4_dynamic import DynamicFundSimulator, Stage4Dynamics


@dataclass(frozen=True)
class LiabilityPath:

    cashflows: CashFlowStream
    headcount_trace: List[Dict]


def _build_headcount_trace(
    simulator: DynamicFundSimulator,
    cashflows: CashFlowStream,
    fund: PensionFund,
    entry_policy: EntryPolicy,
) -> List[Dict]:
    cohort_meta: Dict[str, Tuple[int, str, float, float]] = {}
    for c in fund.cohorts:
        cohort_meta[c.cohort_id] = (
            c.birth_year,
            c.gender,
            float(c.accrued_savings),
            float(c.annual_pension or 0.0),
        )

    df = cashflows.events_df
    agh_lookup: Dict[Tuple[int, str], float] = {}
    pension_lookup: Dict[Tuple[int, str], float] = {}
    if not df.empty:
        ic = df[df["type"] == "INTEREST_CREDIT"]
        if not ic.empty and "agh_per_capita" in ic.columns:
            for _, r in ic.iterrows():
                y = int(pd.to_datetime(r["time"]).year)
                agh_lookup[(y, r["contractId"])] = float(r["agh_per_capita"])
        rc = df[df["type"] == "RETIREMENT_CONV"]
        if not rc.empty and "annual_pension_per_capita" in rc.columns:
            for _, r in rc.iterrows():
                y = int(pd.to_datetime(r["time"]).year)
                pension_lookup[(y, r["contractId"])] = float(r["annual_pension_per_capita"])

    agh_running: Dict[str, float] = {cid: meta[2] for cid, meta in cohort_meta.items()}
    pension_running: Dict[str, float] = {cid: meta[3] for cid, meta in cohort_meta.items()}

    trace: List[Dict] = []
    for entry in simulator.cohort_headcount_log:
        cid = entry["cohort_id"]
        year = int(entry["year"])
        if cid not in cohort_meta:
            # New entrant
            try:
                _, entry_year_str, age_token = cid.split("_")
                entry_year = int(entry_year_str)
                entry_age = int(age_token.replace("AGE", ""))
            except (ValueError, AttributeError):
                # Fallback: assume entry at this year with policy.entry_age.
                entry_year = year
                entry_age = entry_policy.entry_age
            cohort_meta[cid] = (
                entry_year - entry_age,
                entry_policy.gender,
                0.0,
                0.0,
            )
            agh_running[cid] = 0.0
            pension_running[cid] = 0.0

        if (year, cid) in agh_lookup:
            agh_running[cid] = agh_lookup[(year, cid)]
        if (year, cid) in pension_lookup:
            pension_running[cid] = pension_lookup[(year, cid)]

        birth_year, gender, _, _ = cohort_meta[cid]
        trace.append({
            "year": year,
            "cohort_id": cid,
            "birth_year": birth_year,
            "gender": gender,
            "status": entry["status"],
            "headcount": float(entry["headcount"]),
            "agh_per_capita": agh_running[cid],
            "annual_pension_per_capita": pension_running[cid],
        })
    return trace


def simulate_liability_paths(
    fund: PensionFund,
    entry_policy: EntryPolicy,
    dynamics: Stage4Dynamics,
    mortality: MortalityTable,
    *,
    horizon_years: int,
    n_paths: int,
    seed: Optional[int] = None,
    mortality_filter: str = "all",
    return_headcount_log: bool = False,
):
    if n_paths <= 0:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    seed_seq = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(int(n_paths))]
    if not return_headcount_log:
        out: List[CashFlowStream] = []
        for rng in rngs:
            sim = DynamicFundSimulator(
                fund=fund,
                entry_policy=entry_policy,
                dynamics=dynamics,
                mortality=mortality,
                stochastic_mortality=True,
                mortality_filter=mortality_filter,
                rng=rng,
            )
            out.append(sim.run(horizon_years=horizon_years))
        return out

    out_lp: List[LiabilityPath] = []
    for rng in rngs:
        sim = DynamicFundSimulator(
            fund=fund,
            entry_policy=entry_policy,
            dynamics=dynamics,
            mortality=mortality,
            stochastic_mortality=True,
            mortality_filter=mortality_filter,
            rng=rng,
        )
        cfs = sim.run(horizon_years=horizon_years)
        trace = _build_headcount_trace(sim, cfs, fund, entry_policy)
        out_lp.append(LiabilityPath(cashflows=cfs, headcount_trace=trace))
    return out_lp
