"""Stage 2 — deterministic new-entrant policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntryPolicy:
    """Parameters for deterministic annual new entrants.

    entry_age < 25 is valid: savings_rate() returns 0.0 until age 25,
    so SAV_CONTRIB and RISK_CONTRIB will be zero for those years.
    """
    entry_age: int
    headcount: int
    gross_salary: float
