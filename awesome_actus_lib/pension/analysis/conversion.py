from typing import Optional

import pandas as pd

from ...models.cashFlowStream import CashFlowStream
from ..mortality import MortalityTable
from ..policy import PensionPolicy


def annuity_due(age: int, gender: str, technical_rate: float,
                mortality: MortalityTable, terminal_age: int,
                cal_year: Optional[int] = None) -> float:
    if terminal_age < age:
        return 0.0
    v = 1.0 / (1.0 + technical_rate)
    total = 0.0
    kpx = 1.0
    K = terminal_age - age
    for k in range(K + 1):
        total += (v ** k) * kpx
        year_k = None if cal_year is None else cal_year + k
        kpx *= mortality.survival_probability(age + k, gender, year_k)
    return total


def technical_uws(age: int, gender: str, technical_rate: float,
                  mortality: MortalityTable, terminal_age: int,
                  cal_year: Optional[int] = None) -> float:
    return 1.0 / annuity_due(age, gender, technical_rate, mortality,
                             terminal_age, cal_year)


class RetirementLossAnalysis:

    REQUIRED_FIELDS = (
        "age_at_conversion",
        "gender",
        "headcount_at_conversion",
        "agh_per_capita_at_conversion",
        "annual_pension_per_capita",
    )

    def __init__(self, liability_cf: CashFlowStream,
                 policy: PensionPolicy, mortality: MortalityTable):
        if mortality is None:
            raise ValueError(
            )
        self.liability_cf = liability_cf
        self.policy = policy
        self.mortality = mortality

    def summary(self) -> pd.DataFrame:
        df = self.liability_cf.events_df
        conv = df[df["type"] == "RETIREMENT_CONV"]
        missing = [f for f in self.REQUIRED_FIELDS if f not in conv.columns]
        if missing:
            raise ValueError(
                f"RetirementLossAnalysis: RETIREMENT_CONV events are "
                f"missing required Stage-4 fields {missing}. The liability "
                f"CashFlowStream must come from DynamicFundSimulator (Stage 4)."
            )

        rows = []
        for _, c in conv.iterrows():
            cid = c["contractId"]
            agh = float(c["agh_per_capita_at_conversion"])
            pension = float(c["annual_pension_per_capita"])
            age = int(c["age_at_conversion"])
            gender = str(c["gender"])
            hc = float(c["headcount_at_conversion"])
            year = int(str(c["time"])[:4])

            applied = pension / agh if agh > 0 else 0.0
            a_x = annuity_due(age, gender, self.policy.technical_rate,
                              self.mortality, self.policy.terminal_age,
                              cal_year=year)
            tech = 1.0 / a_x if a_x > 0 else float("nan")
            v_pc = agh * (applied * a_x - 1.0)

            rows.append({
                "cohort_id": cid,
                "year": year,
                "age": age,
                "gender": gender,
                "AGH_per_capita": agh,
                "applied_uws": applied,
                "technical_uws": tech,
                "annuity_due": a_x,
                "loss_per_capita": v_pc,
                "headcount": hc,
                "loss_total": v_pc * hc,
            })
        return pd.DataFrame(rows)
