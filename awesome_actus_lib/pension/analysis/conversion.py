"""Stage 4 — annuity-due, technical UWS, pension-conversion-loss analysis.

annuity_due(age, gender, technical_rate, mortality, terminal_age) computes
the actuarial present value of an annuity-due of 1 per period starting at
``age``, discounted at ``technical_rate``, with survival from ``mortality``.

technical_uws is 1/annuity_due — the conversion rate that would make the
applied pension exactly cover the discounted expected liability.

PensionierungsverlustAnalysis consumes a Stage-4 liability CashFlowStream
(produced by DynamicFundSimulator). It reads enriched RETIREMENT_CONV
events (age_at_conversion, gender, headcount_at_conversion) directly and
needs no external cohort metadata.
"""

import pandas as pd

from ...models.cashFlowStream import CashFlowStream
from ..mortality import MortalityTable
from ..policy import PensionPolicy


def annuity_due(age: int, gender: str, technical_rate: float,
                mortality: MortalityTable, terminal_age: int) -> float:
    """ä_x = sum_{k=0}^{K} v^k * kpx, K = terminal_age - age, v = 1/(1+i).

    kpx is built recursively: 0px = 1, (k+1)px = kpx * survival_probability(age+k).
    """
    if terminal_age < age:
        return 0.0
    v = 1.0 / (1.0 + technical_rate)
    total = 0.0
    kpx = 1.0
    K = terminal_age - age
    for k in range(K + 1):
        total += (v ** k) * kpx
        kpx *= mortality.survival_probability(age + k, gender)
    return total


def technical_uws(age: int, gender: str, technical_rate: float,
                  mortality: MortalityTable, terminal_age: int) -> float:
    """Technically correct UWS = 1 / ä_x for the given (age, gender)."""
    return 1.0 / annuity_due(age, gender, technical_rate, mortality, terminal_age)


class PensionierungsverlustAnalysis:
    """Compare applied UWS against technical UWS = 1/ä_x per retirement event.

    Per-capita loss:  verlust_pc    = AGH * (applied_uws * ä_x - 1)
    Total loss:       verlust_total = verlust_pc * headcount_at_conversion

    Reads from a Stage-4 liability CashFlowStream. The RETIREMENT_CONV
    events must carry the enriched fields (age_at_conversion, gender,
    headcount_at_conversion) — i.e. the stream must come from
    DynamicFundSimulator. Streams from Stage 1/2/3 simulators raise a
    clear error.
    """

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
                "PensionierungsverlustAnalysis requires a MortalityTable "
                "(annuity_due is undefined without survival probabilities)."
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
                f"PensionierungsverlustAnalysis: RETIREMENT_CONV events are "
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
                              self.mortality, self.policy.terminal_age)
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
                "verlust_per_capita": v_pc,
                "headcount": hc,
                "verlust_total": v_pc * hc,
            })
        return pd.DataFrame(rows)
