from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .. import events as ev
from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from .open_fund import OpenFundSimulator


@dataclass(frozen=True)
class Stage4Dynamics:
    salary_growth: float = 0.0
    conversion_rate_path: Optional[Dict[int, float]] = None
    threshold_index_period: int = 5


class DynamicFundSimulator(OpenFundSimulator):

    def __init__(
        self,
        fund: PensionFund,
        entry_policy: EntryPolicy,
        dynamics: Stage4Dynamics,
        mortality: Optional[MortalityTable] = None,
        *,
        stochastic_mortality: bool = False,
        mortality_filter: str = "all",
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(
            fund,
            entry_policy,
            mortality=mortality,
            stochastic_mortality=stochastic_mortality,
            mortality_filter=mortality_filter,
            rng=rng,
        )
        if dynamics.threshold_index_period < 1:
            raise ValueError(
                f"Stage4Dynamics.threshold_index_period must be >= 1, "
                f"got {dynamics.threshold_index_period}"
            )
        self.dynamics = dynamics

    def _emit_active(self, cohort, age, event_date, sink, policy):
        sim_year = cohort.birth_year + age
        t = sim_year - self.fund.start_date.year
        g = self.dynamics.salary_growth
        k = self.dynamics.threshold_index_period

        salary_factor = (1.0 + g) ** t
        threshold_factor = (1.0 + g) ** (k * (t // k))

        grown_salary = cohort.gross_salary * salary_factor

        # Inline insured-salary computation against indexed thresholds.
        # policy.insured_salary would use the fixed thresholds and is therefore
        # not reusable here.
        min_sal = policy.bvg_min_salary * threshold_factor
        max_sal = policy.bvg_max_salary * threshold_factor
        coord = policy.coordination_deduction * threshold_factor
        if grown_salary < min_sal:
            insured = 0.0
        else:
            insured = max(0.0, min(grown_salary, max_sal) - coord)

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

    def _applied_uws(self, sim_year, policy):
        path = self.dynamics.conversion_rate_path
        if not path:
            return policy.conversion_rate
        keys = [y for y in path if y <= sim_year]
        return path[max(keys)] if keys else policy.conversion_rate

    def _emit_retirement(self, cohort, event_date, sink, policy):
        sim_year = int(event_date[:4])
        age = sim_year - cohort.birth_year
        applied_uws = self._applied_uws(sim_year, policy)

        cohort.annual_pension = cohort.accrued_savings * applied_uws
        cohort.status = "retired"
        sink.append({
            "time": event_date,
            "type": ev.RETIREMENT_CONV,
            "payoff": 0.0,
            "agh_per_capita_at_conversion": cohort.accrued_savings,
            "annual_pension_per_capita": cohort.annual_pension,
            "age_at_conversion": age,
            "gender": cohort.gender,
            "headcount_at_conversion": cohort.headcount,
        })
        self._emit_pension(cohort, event_date, sink)
