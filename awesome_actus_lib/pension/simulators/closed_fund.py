"""Closed fund simulator (no new entrants)."""

from copy import deepcopy
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from ...models.cashFlowStream import CashFlowStream
from .. import events as ev
from ..cohort import Cohort
from ..fund import PensionFund
from ..mortality import MortalityTable


class ClosedFundSimulator:

    def __init__(
        self,
        fund: PensionFund,
        mortality: Optional[MortalityTable] = None,
        *,
        stochastic_mortality: bool = False,
        mortality_filter: str = "all",
        rng: Optional[np.random.Generator] = None,
    ):
        if mortality_filter not in ("all", "retirees"):
            raise ValueError(
                f"mortality_filter must be 'all' or 'retirees', got {mortality_filter!r}"
            )
        self.fund = fund
        self.mortality = mortality
        self.stochastic_mortality = bool(stochastic_mortality)
        self.mortality_filter = mortality_filter
        self._rng_master = rng
        self._cohort_rngs: Dict[str, np.random.Generator] = {}
        self.cohort_headcount_log: List[dict] = []

    def run(self, horizon_years: int) -> CashFlowStream:
        per_contract_events = {c.cohort_id: [] for c in self.fund.cohorts}
        # Deep copy to keep input fund immutable across runs.
        cohorts = [deepcopy(c) for c in self.fund.cohorts]
        policy = self.fund.policy
        start = self.fund.start_date
        self.cohort_headcount_log = []
        self._cohort_rngs = {}

        for year_offset in range(horizon_years):
            sim_year = start.year + year_offset
            event_date = date(sim_year, 12, 31).isoformat() + "T00:00:00"

            for cohort in cohorts:
                age = sim_year - cohort.birth_year
                if age >= policy.terminal_age:
                    continue
                self._step_cohort(
                    cohort=cohort,
                    age=age,
                    event_date=event_date,
                    sink=per_contract_events[cohort.cohort_id],
                    policy=policy,
                )
                self._apply_mortality(cohort, age, sim_year)
                self.cohort_headcount_log.append({
                    "year": sim_year,
                    "cohort_id": cohort.cohort_id,
                    "headcount": cohort.headcount,
                    "status": cohort.status,
                })

        raw_response = [
            {"contractID": cid, "events": evts, "children": []}
            for cid, evts in per_contract_events.items()
        ]
        return CashFlowStream(
            portfolio=self.fund.to_dummy_portfolio(),
            riskFactors=None,
            raw_response=raw_response,
        )

    def _get_cohort_rng(self, cohort_id: str) -> np.random.Generator:
        
        existing = self._cohort_rngs.get(cohort_id)
        if existing is not None:
            return existing
        if self._rng_master is None:
            self._rng_master = np.random.default_rng()
        child = self._rng_master.spawn(1)[0]
        self._cohort_rngs[cohort_id] = child
        return child

    def _apply_mortality(self, cohort: Cohort, age: int, cal_year: Optional[int] = None) -> None:
        if self.mortality is None:
            return
        p = self.mortality.survival_probability(age, cohort.gender, cal_year)
        if not self.stochastic_mortality:
            # Deterministic decrement: retirees only, fractional survival.
            if cohort.status != "retired":
                return
            cohort.headcount = cohort.headcount * p
            return
        # Stochastic decrement: binomial cohort transition.
        if self.mortality_filter == "retirees" and cohort.status != "retired":
            return
        hc = int(round(cohort.headcount))
        if hc <= 0:
            cohort.headcount = 0
            return
        rng = self._get_cohort_rng(cohort.cohort_id)
        cohort.headcount = int(rng.binomial(hc, p))

    def _step_cohort(self, cohort: Cohort, age: int, event_date: str, sink: list, policy):
        if cohort.status == "active":
            if age < policy.retirement_age:
                self._emit_active(cohort, age, event_date, sink, policy)
            else:
                self._emit_retirement(cohort, event_date, sink, policy)
        else:
            self._emit_pension(cohort, event_date, sink)

    def _emit_active(self, cohort: Cohort, age: int, event_date: str, sink: list, policy):
        insured = policy.insured_salary(cohort.gross_salary)
        sav_rate = policy.savings_rate(age)
        sav_amt = sav_rate * insured * cohort.headcount
        risk_amt = policy.risk_contribution_rate * insured * cohort.headcount
        admin = policy.admin_cost_per_member * cohort.headcount

        interest_pc = cohort.accrued_savings * policy.credited_interest_rate()
        cohort.accrued_savings = cohort.accrued_savings + interest_pc + sav_rate * insured

        sink.append({"time": event_date, "type": ev.SAV_CONTRIB, "payoff": sav_amt})
        sink.append({"time": event_date, "type": ev.RISK_CONTRIB, "payoff": risk_amt})
        sink.append({"time": event_date, "type": ev.INTEREST_CREDIT,
                     "payoff": 0.0, "interest_per_capita": interest_pc,
                     "agh_per_capita": cohort.accrued_savings})
        sink.append({"time": event_date, "type": ev.ADMIN_COST, "payoff": -admin})

    def _emit_retirement(self, cohort: Cohort, event_date: str, sink: list, policy):
        cohort.annual_pension = cohort.accrued_savings * policy.conversion_rate
        cohort.status = "retired"
        sink.append({
            "time": event_date,
            "type": ev.RETIREMENT_CONV,
            "payoff": 0.0,
            "agh_per_capita_at_conversion": cohort.accrued_savings,
            "annual_pension_per_capita": cohort.annual_pension,
        })
        self._emit_pension(cohort, event_date, sink)

    def _emit_pension(self, cohort: Cohort, event_date: str, sink: list):
        amount = -(cohort.annual_pension or 0.0) * cohort.headcount
        sink.append({"time": event_date, "type": ev.PENSION_PAYMENT, "payoff": amount})
