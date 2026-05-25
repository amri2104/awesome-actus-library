"""Stage 4 — deterministic time-varying parameters.

Extends Stage 2's OpenFundSimulator with:

- annual salary growth (geometric, factor (1+g)^t)
- piecewise BVG threshold indexation (steps every threshold_index_period years)
- a falling applied UWS path (conversion_rate_path mapping retirement year -> rate)
- enriched RETIREMENT_CONV events carrying age, gender, headcount_at_conversion
  so PensionierungsverlustAnalysis can run without external metadata.

Defaults reproduce Stage 3 numerically (payoff/time/type values match
exactly): salary_growth=0.0 plus conversion_rate_path=None yields the
same numerical output as Stage 3. The RETIREMENT_CONV event dict is
not byte-identical — Stage 4 always carries three additive fields
(age_at_conversion, gender, headcount_at_conversion) — but downstream
analyses (ALMAnalysis/LiquidityAnalysis/ValueAnalysis) route by
type/payoff/time only, so the additive fields are inert.

Only _emit_active and _emit_retirement are overridden — run() is inherited
from Stage 1/2 unchanged. sim_year is reconstructed from arguments already
in the parent signatures (cohort.birth_year + age, or int(event_date[:4]))
so no Stage 1/2 source file is touched.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .. import events as ev
from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from .Stage2_open import OpenFundSimulator


@dataclass(frozen=True)
class Stage4Dynamics:
    """Configuration for Stage-4 deterministic time-varying parameters.

    salary_growth:
        annual geometric growth rate of gross_salary; 0.0 disables growth.
    conversion_rate_path:
        optional mapping {retirement_year -> applied_uws} with
        step-held-forward lookup: at retirement year y the applied UWS is
        path[max(k in path with k <= y)]; before the first key the
        fallback is policy.conversion_rate. None disables the path
        entirely (always policy.conversion_rate).
    threshold_index_period:
        BVG thresholds (min/max salary, coordination deduction) step
        upward every k years by (1+salary_growth)**k. k=1 ⇒ smooth
        annual indexation, large k ⇒ effectively frozen thresholds.
    """
    salary_growth: float = 0.0
    conversion_rate_path: Optional[Dict[int, float]] = None
    threshold_index_period: int = 5


class DynamicFundSimulator(OpenFundSimulator):
    """Stage 4 simulator: open fund + Stage-3 mortality + deterministic
    time-varying salary growth and UWS path.

    Inherits run(), _step_cohort, _emit_pension, _apply_mortality and
    the cohort_headcount_log/headcount_log from Stage 1/2. Overrides only
    _emit_active (salary growth + indexed thresholds) and _emit_retirement
    (UWS path + event enrichment).
    """

    def __init__(self, fund: PensionFund, entry_policy: EntryPolicy,
                 dynamics: Stage4Dynamics,
                 mortality: Optional[MortalityTable] = None):
        super().__init__(fund, entry_policy, mortality=mortality)
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

        interest_pc = cohort.accrued_savings * policy.applied_interest_rate
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

        # Event enrichment: extra Stage-4 fields for PensionierungsverlustAnalysis.
        # Stage 1/2/3 consumers route by 'type'/'payoff'/'time' only, so unknown
        # fields are inert. headcount is read pre-mortality (mortality decrement
        # happens after _step_cohort returns).
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
