from dataclasses import dataclass, field
from datetime import date
from typing import List, Union

from .cohort import Cohort
from .policy import PensionPolicy


class _DummyPortfolio:
    """Duck-typed Portfolio replacement for CashFlowStream consumption.

    CashFlowStream uses __len__ in __str__ and accesses .contracts in plot().
    Analysis classes pass it through but do not inspect it.
    """

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

    def add_cohort(self, cohort: Cohort) -> None:
        self.cohorts.append(cohort)

    def to_dummy_portfolio(self) -> _DummyPortfolio:
        return _DummyPortfolio(self.cohorts)
