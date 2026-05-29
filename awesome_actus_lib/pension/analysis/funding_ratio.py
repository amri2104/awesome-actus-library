import pandas as pd

from ..fund import PensionFund
from ..mortality import MortalityTable
from .conversion import annuity_due


class FundingRatioAnalysis:
    
    def __init__(self, fund: PensionFund, mortality: MortalityTable,
                 pension_assets: float):
        has_retired = any(c.status == "retired" for c in fund.cohorts)
        if has_retired and mortality is None:
            raise ValueError(
                "FundingRatioAnalysis: mortality must not be None when the "
                "fund contains retired cohorts (annuity_due is undefined)."
            )
        if pension_assets < 0:
            raise ValueError(
                f"pension_assets must be >= 0, got {pension_assets}"
            )
        self.fund = fund
        self.mortality = mortality
        self.pension_assets = float(pension_assets)

    def _age_at_t0(self, cohort) -> int:
        return self.fund.start_date.year - cohort.birth_year

    def pension_capital_t0(self) -> float:
        policy = self.fund.policy
        pc = 0.0
        for c in self.fund.cohorts:
            if c.status == "active":
                pc += c.accrued_savings * c.headcount
            else:
                age0 = self._age_at_t0(c)
                a_x = annuity_due(age0, c.gender, policy.technical_rate,
                                  self.mortality, policy.terminal_age,
                                  cal_year=self.fund.start_date.year)
                pc += (c.annual_pension or 0.0) * c.headcount * a_x
        return pc

    def funding_ratio_t0(self) -> float:
        pc = self.pension_capital_t0()
        if pc == 0:
            raise ValueError(
                "pension_capital_t0 == 0; cannot compute funding ratio."
            )
        return self.pension_assets / pc

    def breakdown(self) -> pd.DataFrame:
        policy = self.fund.policy
        rows = []
        for c in self.fund.cohorts:
            if c.status == "active":
                a_x = None
                contribution = c.accrued_savings * c.headcount
                basis = "AGH"
            else:
                age0 = self._age_at_t0(c)
                a_x = annuity_due(age0, c.gender, policy.technical_rate,
                                  self.mortality, policy.terminal_age,
                                  cal_year=self.fund.start_date.year)
                contribution = (c.annual_pension or 0.0) * c.headcount * a_x
                basis = "pension"
            rows.append({
                "cohort_id": c.cohort_id,
                "status": c.status,
                "basis": basis,
                "headcount": c.headcount,
                "annuity_due": a_x,
                "pc_contribution": contribution,
            })
        return pd.DataFrame(rows)
