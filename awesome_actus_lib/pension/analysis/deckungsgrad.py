"""Stage 4 — Art-44-BVV2-naher Deckungsgrad at the valuation date t0.

The existing ``ALMAnalysis.funding_ratio`` discounts a **net** liability
cashflow stream (contributions minus pension payments minus admin). That
ratio is mix-sensitive and is **not** a regulatory Deckungsgrad in the
sense of Art. 44 BVV2.

The correct Deckungsgrad uses

    Deckungsgrad_t0 = Vorsorgevermoegen / Vorsorgekapital_t0

where the Vorsorgekapital is built from the **fund state at t0** (not
from projected events):

    active cohort : vk_contribution = accrued_savings * headcount       (AGH-based)
    retired cohort: vk_contribution = annual_pension * headcount * ä_x  (PV of future pensions)

with ä_x = annuity_due(age_at_t0, gender, technical_rate, mortality,
terminal_age) reused from ``conversion.py``.

This is a **t0-only** view. A time-varying Deckungsgrad-path would
require projecting AGH and headcount forward (Tier 2 — not implemented).
"""

import pandas as pd

from ..fund import PensionFund
from ..mortality import MortalityTable
from .conversion import annuity_due


class DeckungsgradAnalysis:
    """Compute the BVV2-Art-44-style Deckungsgrad at t0 from a PensionFund.

    Parameters
    ----------
    fund:
        the PensionFund whose cohorts define the Vorsorgekapital at
        fund.start_date.
    mortality:
        MortalityTable used for ä_x on retired cohorts. Must not be None
        if any cohort has status == "retired".
    vorsorgevermoegen:
        market value of assets at t0 (e.g. Σ Bond-Notionals at par).
    """

    def __init__(self, fund: PensionFund, mortality: MortalityTable,
                 vorsorgevermoegen: float):
        has_retired = any(c.status == "retired" for c in fund.cohorts)
        if has_retired and mortality is None:
            raise ValueError(
                "DeckungsgradAnalysis: mortality must not be None when the "
                "fund contains retired cohorts (annuity_due is undefined)."
            )
        if vorsorgevermoegen < 0:
            raise ValueError(
                f"vorsorgevermoegen must be >= 0, got {vorsorgevermoegen}"
            )
        self.fund = fund
        self.mortality = mortality
        self.vorsorgevermoegen = float(vorsorgevermoegen)

    def _age_at_t0(self, cohort) -> int:
        return self.fund.start_date.year - cohort.birth_year

    def vorsorgekapital_t0(self) -> float:
        policy = self.fund.policy
        vk = 0.0
        for c in self.fund.cohorts:
            if c.status == "active":
                vk += c.accrued_savings * c.headcount
            else:
                age0 = self._age_at_t0(c)
                a_x = annuity_due(age0, c.gender, policy.technical_rate,
                                  self.mortality, policy.terminal_age)
                vk += (c.annual_pension or 0.0) * c.headcount * a_x
        return vk

    def deckungsgrad_t0(self) -> float:
        vk = self.vorsorgekapital_t0()
        if vk == 0:
            raise ValueError(
                "Vorsorgekapital_t0 == 0; cannot compute Deckungsgrad."
            )
        return self.vorsorgevermoegen / vk

    def breakdown(self) -> pd.DataFrame:
        policy = self.fund.policy
        rows = []
        for c in self.fund.cohorts:
            if c.status == "active":
                a_x = None
                beitrag = c.accrued_savings * c.headcount
                basis = "AGH"
            else:
                age0 = self._age_at_t0(c)
                a_x = annuity_due(age0, c.gender, policy.technical_rate,
                                  self.mortality, policy.terminal_age)
                beitrag = (c.annual_pension or 0.0) * c.headcount * a_x
                basis = "pension"
            rows.append({
                "cohort_id": c.cohort_id,
                "status": c.status,
                "basis": basis,
                "headcount": c.headcount,
                "annuity_due": a_x,
                "vk_beitrag": beitrag,
            })
        return pd.DataFrame(rows)
