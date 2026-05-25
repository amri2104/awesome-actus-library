"""Stage 2 — open fund simulator with deterministic new entrants."""

from copy import deepcopy
from datetime import date
from typing import List, Optional

from ...models.cashFlowStream import CashFlowStream
from ..cohort import Cohort
from ..entry import EntryPolicy
from ..fund import _DummyPortfolio, PensionFund
from ..mortality import MortalityTable
from .Stage1_closed import ClosedFundSimulator


class OpenFundSimulator(ClosedFundSimulator):
    """Stage 2 simulator: open fund with deterministic annual new entrants.

    At the start of each simulation year, one new Cohort (accrued_savings=0)
    is created and added to the working cohort list. Inherited _step_cohort
    machinery handles contributions, retirement, and pensions identically for
    both original and new-entrant cohorts.

    After run(), headcount_log contains per-year active/retired headcounts
    (aggregated), and cohort_headcount_log contains per-cohort per-year
    headcount (inherited from ClosedFundSimulator).

    Optional Stage 3 mortality: see ClosedFundSimulator.
    """

    def __init__(self, fund: PensionFund, entry_policy: EntryPolicy,
                 mortality: Optional[MortalityTable] = None):
        super().__init__(fund, mortality=mortality)
        self.entry_policy = entry_policy
        self.headcount_log: List[dict] = []

    def run(self, horizon_years: int) -> CashFlowStream:
        per_contract_events = {c.cohort_id: [] for c in self.fund.cohorts}
        cohorts = [deepcopy(c) for c in self.fund.cohorts]
        policy = self.fund.policy
        start = self.fund.start_date
        self.headcount_log = []
        self.cohort_headcount_log = []

        for year_offset in range(horizon_years):
            sim_year = start.year + year_offset
            event_date = date(sim_year, 12, 31).isoformat() + "T00:00:00"

            ep = self.entry_policy
            new_cohort = Cohort(
                cohort_id=f"ENTRY_{sim_year}_AGE{ep.entry_age}",
                birth_year=sim_year - ep.entry_age,
                headcount=ep.headcount,
                gross_salary=ep.gross_salary,
                accrued_savings=0.0,
                gender=ep.gender,
            )
            cohorts.append(new_cohort)
            per_contract_events[new_cohort.cohort_id] = []

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
                self._apply_mortality(cohort, age)
                self.cohort_headcount_log.append({
                    "year": sim_year,
                    "cohort_id": cohort.cohort_id,
                    "headcount": cohort.headcount,
                    "status": cohort.status,
                })

            active_hc = sum(
                c.headcount for c in cohorts
                if c.status == "active" and (sim_year - c.birth_year) < policy.terminal_age
            )
            retired_hc = sum(
                c.headcount for c in cohorts
                if c.status == "retired" and (sim_year - c.birth_year) < policy.terminal_age
            )
            self.headcount_log.append({"year": sim_year, "active": active_hc, "retired": retired_hc})

        raw_response = [
            {"contractID": cid, "events": evts, "children": []}
            for cid, evts in per_contract_events.items()
        ]
        return CashFlowStream(
            portfolio=_DummyPortfolio(cohorts),
            riskFactors=None,
            raw_response=raw_response,
        )
