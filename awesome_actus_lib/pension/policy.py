from dataclasses import dataclass, field
from typing import Dict, Tuple


def default_bvg_scale() -> Dict[Tuple[int, int], float]:
    return {
        (25, 34): 0.07,
        (35, 44): 0.10,
        (45, 54): 0.15,
        (55, 64): 0.18,
    }


@dataclass(frozen=True)
class PensionPolicy:
    bvg_min_interest_rate: float = 0.0125
    applied_interest_rate: float = 0.02
    technical_rate: float = 0.0176
    conversion_rate: float = 0.0523
    retirement_age: int = 65
    terminal_age: int = 100
    savings_contribution_by_age: Dict[Tuple[int, int], float] = field(
        default_factory=default_bvg_scale
    )
    risk_contribution_rate: float = 0.015
    admin_cost_per_member: float = 400.0
    coordination_deduction: float = 26460.0
    bvg_min_salary: float = 22680.0
    bvg_max_salary: float = 90720.0

    def insured_salary(self, gross_salary: float) -> float:
        if gross_salary < self.bvg_min_salary:
            return 0.0
        capped = min(gross_salary, self.bvg_max_salary)
        return max(0.0, capped - self.coordination_deduction)

    def savings_rate(self, age: int) -> float:
        for (lo, hi), rate in self.savings_contribution_by_age.items():
            if lo <= age <= hi:
                return rate
        return 0.0

    def credited_interest_rate(self) -> float:
        """Effective rate credited to AGH each year = applied rate, floored
        at the statutory BVG minimum (Art. 15 BVG).

        In a real fund the minimum binds only on the *mandatory* part of the
        retirement assets. This single-AGH-pot model has no mandatory/extra-
        mandatory split, so the floor is applied to the whole AGH — the
        faithful simplification given the model's scope. With the baseline
        (applied 2.0% > min 1.25%) the floor is inactive and this equals
        ``applied_interest_rate``; it only bites if a scenario sets the
        applied rate below the statutory minimum.
        """
        return max(self.applied_interest_rate, self.bvg_min_interest_rate)
