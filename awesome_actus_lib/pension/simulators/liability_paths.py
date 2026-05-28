"""Stage 5b — Monte-Carlo liability cashflow paths.

Generates ``n_paths`` independent liability ``CashFlowStream`` realisations
under binomial cohort transitions (Maffra/Armstrong/Pennanen 2021 Eq. 1)
applied uniformly across cohorts. Path independence is guaranteed via
``numpy.random.SeedSequence(seed).spawn(n_paths)``: each path gets its own
spawned ``Generator``, and inside each simulator the per-cohort RNGs are
further spawned from that path's master rng (see
``ClosedFundSimulator._get_cohort_rng``).

This is a pure-Python loop: no network calls, no ACTUS service round-trips.
The bottleneck is the per-path simulator pass over all cohorts and years.

Stage 5c addition: when ``return_headcount_log=True`` the function returns
``List[LiabilityPath]`` instead of ``List[CashFlowStream]``. Each
``LiabilityPath`` bundles the path's cashflows with a ``headcount_trace``
giving per-year, per-cohort survivor counts plus the cohort metadata
required to value VK_t,i forward (age via birth_year, gender, accrued AGH
per capita for actives, annual pension per capita for retirees). The trace
is built from ``DynamicFundSimulator.cohort_headcount_log`` and from the
existing per-capita fields on the liability CashFlowStream events — no new
state is added to the simulator itself, so Stage 1-5b behaviour is
unaffected.
"""

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
    """One stochastic-mortality liability path.

    Attributes
    ----------
    cashflows:
        The path's liability ``CashFlowStream`` (same object the Stage 5b
        list-return mode produces).
    headcount_trace:
        ``List[dict]`` with one entry per (sim_year, cohort_id). Each dict
        carries:
          - ``year``: simulation year (end-of-year snapshot)
          - ``cohort_id``: cohort identifier
          - ``birth_year``: cohort's birth year (for age-at-t lookup)
          - ``gender``: ``"m"`` / ``"f"`` / ``"unisex"``
          - ``status``: ``"active"`` or ``"retired"`` at end of that year
          - ``headcount``: surviving headcount at end of that year (post
            mortality decrement)
          - ``agh_per_capita``: per-capita accrued AGH at end of that year
            (active cohorts; for retirees it freezes at conversion-time
            value)
          - ``annual_pension_per_capita``: per-capita annual pension once
            retired, ``0.0`` while active.
    """

    cashflows: CashFlowStream
    headcount_trace: List[Dict]


def _build_headcount_trace(
    simulator: DynamicFundSimulator,
    cashflows: CashFlowStream,
    fund: PensionFund,
    entry_policy: EntryPolicy,
) -> List[Dict]:
    """Enrich ``simulator.cohort_headcount_log`` with cohort metadata.

    Pulls per-capita AGH from INTEREST_CREDIT events and per-capita
    annual pension from RETIREMENT_CONV events — both already emitted
    by the DynamicFundSimulator. New-entrant cohort metadata (birth_year,
    gender) is reconstructed from the ``ENTRY_{year}_AGE{age}`` id pattern
    plus ``entry_policy``. Original cohorts are taken from ``fund.cohorts``.
    """
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
            # New entrant -- reconstruct meta from id + entry_policy.
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
    """Return ``n_paths`` independent stochastic-mortality liability streams.

    Parameters
    ----------
    fund, entry_policy, dynamics, mortality
        Same building blocks as a standard ``DynamicFundSimulator`` run.
    horizon_years
        Simulation horizon in years.
    n_paths
        Number of Monte-Carlo paths to generate.
    seed
        Master seed. ``None`` uses non-reproducible entropy.
    mortality_filter
        ``"all"`` (default): binomial decrement applied to active AND
        retired cohorts each year. ``"retirees"``: binomial decrement
        only on retired cohorts (matches the Stage-3 deterministic scope
        but with stochastic draws); actives remain at fixed headcount.
    return_headcount_log
        Stage 5c switch. ``False`` (default) keeps the byte-identical
        Stage-5b return type (``List[CashFlowStream]``). ``True`` returns
        ``List[LiabilityPath]`` with the per-path survivor trace bundled
        in — required by the forward funding-ratio roll in
        ``StochasticFundingRatioAnalysis``.

    Returns
    -------
    List[CashFlowStream] or List[LiabilityPath]
        Length ``n_paths``. Element ``i`` is the liability state of path
        ``i`` and is paired position-wise with stochastic asset streams in
        ``StochasticALMAnalysis`` / ``RiskAttribution`` /
        ``StochasticFundingRatioAnalysis``.
    """
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
