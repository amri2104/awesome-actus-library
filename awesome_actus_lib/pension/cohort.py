from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

CohortStatus = Literal["active", "retired"]
Gender = Literal["m", "f", "unisex"]

_VALID_GENDERS = ("m", "f", "unisex")


@dataclass
class Cohort:
    cohort_id: str
    birth_year: int
    headcount: float
    gross_salary: float
    accrued_savings: float
    status: CohortStatus = "active"
    annual_pension: Optional[float] = None
    gender: Gender = "unisex"

    def __post_init__(self) -> None:
        # Context-free checks. Runs on construction only — not on deepcopy
        # and not on the simulator's internal status flip.
        if self.headcount < 0:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: headcount must be >= 0, got {self.headcount}"
            )
        if self.gross_salary < 0:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: gross_salary must be >= 0, got {self.gross_salary}"
            )
        if self.accrued_savings < 0:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: accrued_savings must be >= 0, got {self.accrued_savings}"
            )
        if self.annual_pension is not None and self.annual_pension < 0:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: annual_pension must be >= 0, got {self.annual_pension}"
            )
        if self.status == "retired" and self.annual_pension is None:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: status='retired' requires annual_pension to be set "
                f"(otherwise pension payments would silently be 0)"
            )
        if self.gender not in _VALID_GENDERS:
            raise ValueError(
                f"Cohort {self.cohort_id!r}: gender must be one of {_VALID_GENDERS}, "
                f"got {self.gender!r}"
            )

    def current_age(self, as_of: date) -> int:
        return as_of.year - self.birth_year
