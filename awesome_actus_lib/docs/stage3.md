---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 3 — Deterministic Retiree Mortality (EK 2001–2005)

Stage 3 extends Stages 1 and 2 by introducing **deterministic
expected-survival mortality** for **retired** cohorts. The
one-year death probabilities `q_x` come from the Swiss
EK 2001–2005 period table (separate male and female series).
Everything else from Stages 1 and 2 still applies — read
[stage1.md](stage1.md) first for the mental model, the three core
building blocks (`PensionPolicy`, `Cohort`, `PensionFund`), event
types, analysis layer, and merging asset/liability cashflows, and
[stage2.md](stage2.md) for the open-fund extension.

This document covers only the deltas.

------------------------------------------------------------------------

## 1. What Changes vs Stage 1/2

| Aspect                          | Stage 1 / 2                                   | Stage 3                                                                    |
|----------------|----------------------|---------------------------------|
| Retiree headcount over time     | flat until `terminal_age`                     | shrinks each year by expected deaths (`headcount *= 1 - q_x`)              |
| Active headcount over time      | flat until retirement                         | unchanged — flat until retirement (active mortality is out of scope)       |
| Headcount type                  | `int`                                         | `float` (fractional survivors allowed)                                     |
| Cohort field                    | —                                             | new optional `gender: "m" | "f" | "unisex"` (default `"unisex"`)           |
| `EntryPolicy` field             | —                                             | new optional `gender` forwarded to each entrant cohort                     |
| Simulator constructor           | `ClosedFundSimulator(fund)`                   | `ClosedFundSimulator(fund, mortality=...)` (opt-in)                        |
| Per-cohort diagnostic           | —                                             | `simulator.cohort_headcount_log` populated regardless of mortality setting |
| Aggregated `headcount_log` (S2) | unchanged                                     | unchanged                                                                  |
| Output type                     | `CashFlowStream`                              | `CashFlowStream` (same)                                                    |
| Event types emitted             | unchanged                                     | unchanged — mortality is a state change, not a cashflow                    |
| Analysis / merging API          | unchanged (`ALMAnalysis`)                     | unchanged                                                                  |
| `PensionPolicy`, `PensionFund`  | unchanged                                     | unchanged                                                                  |

**Backward compatibility.** Default `mortality=None` keeps Stage 1 and
Stage 2 quickstarts byte-identical. The new `gender` field defaults to
`"unisex"`, so existing `Cohort(...)` and `EntryPolicy(...)` call
sites continue to work without change.

------------------------------------------------------------------------

## 2. New Building Block — `MortalityTable` + `ek2001_2005()`

### `MortalityTable`

Frozen dataclass storing `q_x` by age and gender.

| Field      | Type                | Meaning                                       |
|------------|---------------------|-----------------------------------------------|
| `q_male`   | `Dict[int, float]`  | one-year death probability for males, by age   |
| `q_female` | `Dict[int, float]`  | one-year death probability for females, by age |

Method `survival_probability(age, gender) -> float` returns
`1 - q_x`:

- `gender="m"` reads `q_male[age]`
- `gender="f"` reads `q_female[age]`
- `gender="unisex"` returns `1 - 0.5 * (q_male[age] + q_female[age])`
- ages outside the table return `0.0` (treated as already dead, in
  line with the existing `terminal_age = 100` boundary).

### `ek2001_2005()`

Factory returning the **EK 2001–2005 period table** with hardcoded
`q_x` extracted from
`awesome_actus_lib/pension/mortality_tables_EK.xlsx`,
sheets `EKM 0105` (men) and `EKF 0105` (women), columns `x` and
`qx` only. Everything else in that workbook (commutation columns,
the older EKM 95 / EKF 95 sheets, the `qx-Grafiken` sheet, the
`Zinssatz i` cell) is ignored. The Excel file stays in the repo for
traceability — there is **no runtime Excel dependency**.

**Period vs generation.** EK 2001–2005 is a *period* table, evaluated
on the calendar years 2001–2005. It does not encode calendar-year
mortality improvement (e.g. that a 65-year-old in 2050 will likely
have lower one-year death probability than a 65-year-old today). A
future stage may layer an improvement factor on top.

``` python
from awesome_actus_lib.pension import MortalityTable, ek2001_2005

mortality = ek2001_2005()
print(mortality.survival_probability(70, "f"))   # 1 - q_female[70]
print(mortality.survival_probability(70, "m"))   # 1 - q_male[70]
print(mortality.survival_probability(70, "unisex"))
```

------------------------------------------------------------------------

## 3. New Field — `Cohort.gender`

```python
@dataclass
class Cohort:
    ...
    gender: Literal["m", "f", "unisex"] = "unisex"
```

`"unisex"` blends male and female `q_x` 50/50, which is sensible
when a cohort represents a mixed population whose gender split is
unknown or irrelevant for the analysis. Use `"m"` or `"f"` for
populations with a known dominant gender (typical for separate
male/female pensioner blocks).

`EntryPolicy` gains the same field (default `"unisex"`); each
entrant cohort in Stage 2 carries it forward.

------------------------------------------------------------------------

## 4. The Simulator — Where Mortality Is Applied

Both `ClosedFundSimulator` and `OpenFundSimulator` accept an
optional `mortality: MortalityTable | None = None`. When set, the
simulator decrements each **retired** cohort's headcount after the
year's events are emitted:

```
for each simulation year y:
    for each cohort:
        _step_cohort(...)             # emit SAV_CONTRIB / PENSION_PAYMENT / ... events
        _apply_mortality(cohort, age) # headcount *= survival_probability(age, gender)
                                      # only when status == "retired" and mortality is set
        cohort_headcount_log.append({year, cohort_id, headcount, status})
```

Key invariants:

- **Emit-then-decrement.** The `PENSION_PAYMENT` for year `y` is
  computed with the **start-of-year** headcount; the headcount is
  reduced **after** the payment is recorded. This matches actuarial
  convention: pay the living, then count survivors at year end.
- **Retirees only.** Active cohorts are skipped by
  `_apply_mortality`. Pre-retirement mortality is intentionally out
  of scope — it raises a "what happens to the forfeited AGH on the
  death of an active member" question the current model does not
  answer.
- **Fractional headcount is allowed.** `PENSION_PAYMENT.payoff =
  -annual_pension * headcount` already works with floats; downstream
  `LiquidityAnalysis`, `ValueAnalysis`, `ALMAnalysis` are agnostic.
- **No new event type.** Mortality is a population-state change, not
  a cashflow. The effect appears implicitly in shrinking
  `PENSION_PAYMENT` payoffs year on year. Use
  `simulator.cohort_headcount_log` for the explicit per-cohort
  trajectory.

For Stage 2's `OpenFundSimulator`, the same `_apply_mortality`
helper is invoked inside its own `run()` loop (the open-fund `run()`
fully overrides the closed-fund `run()`, so a shared helper is
called from both). The aggregated `headcount_log` (active vs
retired totals) is preserved alongside the new
`cohort_headcount_log`.

------------------------------------------------------------------------

## 5. Numerical Illustration — Hand Reconciliation

Take a single retired cohort with `headcount(0) = 80`,
`birth_year = 1955`, `gender = "f"`, simulated from
`start_date = 2025-01-01`. At simulation year `y` the cohort age is
`a = y - 1955`. Expected survivors at the **end** of year `y` are

```
survivors(y) = 80 * Π_{a' = 70}^{a' = a} (1 - q_female[a'])
```

Reading the female `q_x` from EK 2001–2005 at the relevant ages and
multiplying gives a sequence that must match
`simulator.cohort_headcount_log` filtered by
`cohort_id == "RETIRED_1955"` exactly within float tolerance. The
Stage 3 quickstart's `[4] Plots` section saves this trajectory as
`stage3_v5_cohort_headcount_decline.png`.

------------------------------------------------------------------------

## 6. End-to-End Stage 3 Example

```python
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund,
    ClosedFundSimulator,
    ek2001_2005,
)

# 1. Rules
policy = PensionPolicy()

# 2. Fund + cohorts (note: gender on each cohort)
fund = PensionFund(policy=policy, start_date="2025-01-01T00:00:00")

fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0,
    gender="unisex",
))
fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970", birth_year=1970, headcount=150,
    gross_salary=110_000.0, accrued_savings=350_000.0,
    gender="unisex",
))
fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0,
    status="retired", annual_pension=31_380.0,
    gender="f",
))

# 3. Simulate with mortality
simulator = ClosedFundSimulator(fund, mortality=ek2001_2005())
liability_cfs = simulator.run(horizon_years=40)

# 4. Per-cohort headcount over time
import pandas as pd
print(pd.DataFrame(simulator.cohort_headcount_log).head())

# 5. Downstream merging / analysis: identical to Stage 1/2
#    See stage1.md §10 for the three merging variants.
```

For asset/liability merging, `ALMAnalysis`, and plots, refer to
[stage1.md §10](stage1.md) — the API is identical because the
output type (`CashFlowStream`) is identical.

------------------------------------------------------------------------

## 7. Funding-Ratio Caveat

Adding mortality reduces pension outflows. In isolation that pushes
the **absolute** liability NPV down (less is being paid out over
time), which **raises** `ALMAnalysis.funding_ratio` because the
ratio is `NPV(assets) / |NPV(liabilities)|`.

Two confounders make the magnitude hard to predict without running:

1. The current `funding_ratio` discounts a **net** liability
   cashflow stream — contributions (`SAV_CONTRIB`, `RISK_CONTRIB`)
   minus pension payments minus admin. Mortality changes only the
   pension-payment leg of that net stream. The denominator effect
   depends on which leg dominates in PV terms, and that depends on
   the cohort mix, discount rate, and horizon.
2. A truly BVG-aufsichtsrechtlicher funding ratio would compare
   pension assets against **pension capital** (pure obligation PV,
   not a net cashflow PV). Recasting `funding_ratio` along those
   lines is a separate concern, earmarked for a later stage.

Therefore: **observe** the Stage 3 funding-ratio numbers and
**compare** them against the Stage 1/2 baseline rather than
predicting the direction analytically.

------------------------------------------------------------------------

## 8. What Stage 3 Still Does Not Do

- **Stochastic (Binomial) mortality** — Stage 3 uses expected
  survival only; no Monte-Carlo replication, no confidence bands on
  headcount.
- **Active pre-retirement mortality** — would require deciding what
  happens to the forfeited AGH on death of an active member
  (employer reserve? distribution to survivors?). Left for a later
  stage.
- **Calendar-year mortality improvement** — EK 2001–2005 is a period
  table evaluated on calendar years 2001–2005. Real Swiss mortality
  has improved since; a generation table or an explicit improvement
  factor would address this.
- **Disability decrements** — risk contributions are recorded as a
  cashflow but the corresponding outflows are not modeled.
- **Surviving-dependent payments** (Witwen-/Witwerrente,
  Waisenrente).
- **Lump-sum withdrawals at retirement** (Kapitalbezug statt
  Rente).
- **BVG-aufsichtsrechtlicher funding ratio and
  Wertschwankungsreserve.**
- **Inflation indexation of pensions.**
- **Redefining `ALMAnalysis.funding_ratio`** to use pure obligation
  NPV rather than net-liability-cashflow NPV.

Each of these is a candidate scope for a future stage.
