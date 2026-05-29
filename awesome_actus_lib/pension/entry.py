from dataclasses import dataclass
from typing import Literal

Gender = Literal["m", "f", "unisex"]


@dataclass(frozen=True)
class EntryPolicy:
    entry_age: int
    headcount: int
    gross_salary: float
    gender: Gender = "unisex"
