---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 2 — Open Fund Pension Simulation

Stage 2 extends Stage 1 by allowing **deterministic annual new
entrants**. Everything else from Stage 1 still applies — read
[stage1.md](stage1.md) first for the mental model, the three core
building blocks (`PensionPolicy`, `Cohort`, `PensionFund`), event types,
analysis layer, and merging asset/liability cashflows.

This document covers only the deltas. Sections cross-reference Stage 1
where the behaviour is unchanged.

------------------------------------------------------------------------

## 1. What Changes vs Stage 1

| Aspect                      | Stage 1                            | Stage 2                                            |
|----------------|---------------------|-----------------------------------|
| Fund openness               | closed (population fixed at t=0)   | open (new cohort added each year)                  |
| New entrants                | none                               | one new `Cohort` per simulation year               |
| Simulator class             | `ClosedFundSimulator`              | `OpenFundSimulator` (subclass)                     |
| Extra config object         | —                                  | `EntryPolicy`                                      |
| Extra runtime state         | —                                  | `simulator.headcount_log` (per-year active/retired) |
| Output type                 | `CashFlowStream`                   | `CashFlowStream` (same)                            |
| Event types emitted         | unchanged (see [stage1.md §8](stage1.md)) | unchanged                                          |
| `PensionPolicy`, `Cohort`, `PensionFund` | unchanged              | unchanged                                          |
| Analysis / merging API      | unchanged (`ALMAnalysis`)          | unchanged                                          |

------------------------------------------------------------------------

## 2. New Building Block — `EntryPolicy`

A frozen dataclass describing the single cohort that joins the fund
each simulation year.

| Field          | Type    | Meaning                                                            |
|----------------|---------|--------------------------------------------------------------------|
| `entry_age`    | `int`   | Age of new members at entry (`< 25` is valid; savings rate = 0 until 25). |
| `headcount`    | `int`   | Number of new members per year.                                    |
| `gross_salary` | `float` | Gross annual salary per member.                                    |

`EntryPolicy` is **frozen** — build a new one to vary the entrant
profile.

``` python
from awesome_actus_lib.pension import EntryPolicy

entry_policy = EntryPolicy(
    entry_age=25,
    headcount=20,
    gross_salary=80_000.0,
)
```

`accrued_savings` is always zero for new entrants — they enter without
prior savings.

------------------------------------------------------------------------

## 3. The Simulator — `OpenFundSimulator`

Subclass of `ClosedFundSimulator`. Same constructor pattern + the
`EntryPolicy`:

``` python
from awesome_actus_lib.pension import OpenFundSimulator

simulator = OpenFundSimulator(liability_fund, entry_policy)
liability_cfs = simulator.run(horizon_years=40)
```

### Extra step inside `run()`

For each simulation year, **before** stepping existing cohorts:

1.  Construct a new `Cohort` with
    -   `cohort_id = f"ENTRY_{sim_year}_AGE{entry_age}"`
    -   `birth_year = sim_year - entry_age`
    -   `headcount`, `gross_salary` from `EntryPolicy`
    -   `accrued_savings = 0.0`
2.  Append it to the working cohort list (deep-copied from the input
    fund — input cohorts stay immutable, see [stage1.md §3](stage1.md)).
3.  Add an empty event sink for it under its `cohort_id`.

Then the inherited `_step_cohort` machinery handles all cohorts
identically — the new entrant gets `SAV_CONTRIB`, `RISK_CONTRIB`,
`INTEREST_CREDIT`, `ADMIN_COST` events from year 1 (or zero-valued
contribution events if `entry_age < 25`, since `policy.savings_rate(age)`
returns 0 below the BVG floor).

### New attribute — `headcount_log`

After `run()`, `simulator.headcount_log` is a list of dicts:

``` python
[
    {"year": 2025, "active": 450, "retired": 0},
    {"year": 2026, "active": 470, "retired": 0},
    ...
]
```

Active and retired counts both filter out cohorts that have reached
`terminal_age`. Convert directly to a DataFrame for plotting:

``` python
import pandas as pd
headcount_df = pd.DataFrame(simulator.headcount_log)
```

------------------------------------------------------------------------

## 4. Required Order of Definition

Same as Stage 1, plus `EntryPolicy` before the simulator:

```
PensionPolicy   →   PensionFund(policy=..., start_date=...)
                          ↓
                    add_cohort(Cohort(...))   (one or more)
                          ↓
                    EntryPolicy(...)                       ← new in Stage 2
                          ↓
                    OpenFundSimulator(fund, entry_policy).run(horizon_years=...)
                          ↓
                    CashFlowStream (liability)
```

------------------------------------------------------------------------

## 5. End-to-End Stage 2 Example

``` python
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund,
    OpenFundSimulator, EntryPolicy,
)

# 1. Rules
policy = PensionPolicy()  # BVG defaults

# 2. Fund + starting cohorts
fund = PensionFund(policy=policy, start_date="2025-01-01T00:00:00")

fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0,
))
fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970", birth_year=1970, headcount=150,
    gross_salary=110_000.0, accrued_savings=350_000.0,
))
fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0,
    status="retired", annual_pension=35_000.0,
))

# 3. Entry policy — annual new entrants
entry_policy = EntryPolicy(entry_age=25, headcount=20, gross_salary=80_000.0)

# 4. Simulate
simulator = OpenFundSimulator(fund, entry_policy)
liability_cfs = simulator.run(horizon_years=40)

# 5. Stage 2 specific: inspect headcount evolution
import pandas as pd
headcount_df = pd.DataFrame(simulator.headcount_log)
print(headcount_df.head())

# 6. Downstream: identical to Stage 1 — see stage1.md §7 and §10
```

For merging with assets, ALM analysis, and plots, refer to
[stage1.md §10](stage1.md) — the API is identical because the output
type (`CashFlowStream`) is identical.

------------------------------------------------------------------------

## 6. What Stage 2 Still Does Not Do

Same limitations as Stage 1, minus the closed-fund restriction:

-   No mortality decrement — cohorts (original + new entrants) stay
    full until `terminal_age`.
-   No salary growth, no inflation indexing of pensions.
-   No stochastic returns — interest is deterministic from
    `applied_interest_rate`.
-   No biometric refinement (disability, surviving dependents are not
    modeled separately).
-   `EntryPolicy` is **single-profile** — only one entrant cohort
    archetype per year. Heterogeneous entrants (multiple ages, salary
    bands) require multiple `OpenFundSimulator` runs or a future
    extension.
-   No leavers / no fluctuation modelling (terminations, transfers
    out).

Extending any of these is the natural path for further stages.
