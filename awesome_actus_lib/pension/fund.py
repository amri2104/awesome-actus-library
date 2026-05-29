import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import List, Union

from .cohort import Cohort
from .policy import PensionPolicy


class _DummyPortfolio:

    def __init__(self, cohorts: List[Cohort]):
        self.contracts = list(cohorts)

    def __len__(self) -> int:
        return len(self.contracts)


@dataclass
class PensionFund:
    policy: PensionPolicy
    start_date: Union[date, str]
    cohorts: List[Cohort] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.start_date, str):
            self.start_date = date.fromisoformat(self.start_date.split("T")[0])
        for cohort in self.cohorts:
            self._validate_cohort(cohort)

    def add_cohort(self, cohort: Cohort) -> None:
        self._validate_cohort(cohort)
        self.cohorts.append(cohort)

    def _validate_cohort(self, cohort: Cohort) -> None:
        age = self.start_date.year - cohort.birth_year
        if age < 0:
            raise ValueError(
                f"Cohort {cohort.cohort_id!r}: birth_year {cohort.birth_year} is after "
                f"start_date year {self.start_date.year} (age = {age})"
            )
        if age >= self.policy.terminal_age:
            raise ValueError(
                f"Cohort {cohort.cohort_id!r}: age {age} at start_date is >= "
                f"terminal_age {self.policy.terminal_age}; cohort would emit no events"
            )
        # Active cohorts with age >= retirement_age are the legitimate
        # auto-conversion path (e.g. ACTIVE_1955 at age 70). Do NOT warn.
        if cohort.status == "retired" and age < self.policy.retirement_age:
            warnings.warn(
                f"Cohort {cohort.cohort_id!r}: status='retired' at age {age} is below "
                f"retirement_age {self.policy.retirement_age} (early retirement)",
                stacklevel=2,
            )

    def to_dummy_portfolio(self) -> _DummyPortfolio:
        return _DummyPortfolio(self.cohorts)
