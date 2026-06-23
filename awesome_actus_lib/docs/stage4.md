---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 4 — Deterministic Time-Varying Parameters

Stage 4 extends Stages 1–3 by introducing three **deterministic
time-varying** parameters and one new actuarial KPI:

1. **Salary growth** — annual geometric growth factor `(1 + g)^t`
   applied to each active cohort's `gross_salary` before contributions
   and AGH updates.
2. **Falling applied UWS path** — per-retirement-year override of the
   applied conversion rate, allowing scenarios such as `5.23% → 4.60%`
   over 15 years.
3. **Stepwise indexation of BVG thresholds** — `bvg_min_salary`,
   `bvg_max_salary`, `coordination_deduction` grow every
   `threshold_index_period` years (default 5), mirroring the real
   Bundesrat adjustment cycle.
4. **retirement-loss KPI** — compares the applied UWS against the
   technically correct UWS = `1 / ä_x` (annuity-due) for every
   retirement event in the stream.

No stochastic elements (those are reserved for Stage 5). No asset
rebalancing. Everything else from Stages 1–3 still applies — read
[stage1.md](stage1.md), [stage2.md](stage2.md), and
[stage3.md](stage3.md) first for mental model, building blocks, event
types, mortality, and ALM analysis.

This document covers only the deltas.

------------------------------------------------------------------------

## 1. What Changes vs Stage 1–3

| Aspect                              | Stage 1 / 2 / 3                       | Stage 4                                                                  |
|-------------------------------------|---------------------------------------|--------------------------------------------------------------------------|
| Salary over time                    | flat (`cohort.gross_salary` constant) | grows by `(1 + g)^t` each simulation year                                |
| BVG thresholds                      | constant from `PensionPolicy`         | step up every `threshold_index_period` years                             |
| Conversion rate                     | constant `policy.conversion_rate`     | overrideable per year via `conversion_rate_path`                         |
| Simulator class                     | `OpenFundSimulator`                   | `DynamicFundSimulator` (subclass of `OpenFundSimulator`)                 |
| Extra config object                 | `EntryPolicy`                         | `EntryPolicy` + **new** `Stage4Dynamics`                                 |
| `RETIREMENT_CONV` event fields      | `agh_per_capita_at_conversion`, `annual_pension_per_capita` | + **`age_at_conversion`**, **`gender`**, **`headcount_at_conversion`**   |
| Other event types                   | unchanged                             | unchanged                                                                |
| Output type                         | `CashFlowStream`                      | `CashFlowStream` (same)                                                  |
| Analysis layer                      | `ALMAnalysis`                         | `ALMAnalysis` + **new** `RetirementLossAnalysis`                  |
| `PensionPolicy`, `Cohort`, `PensionFund`, `MortalityTable` | unchanged    | unchanged                                                                |

**Backward compatibility.** Defaults reproduce Stage 3 **numerically
identical** (payoff/time/type values match exactly):
`Stage4Dynamics(salary_growth=0.0, conversion_rate_path=None,
threshold_index_period=1)` plus the same fund, entry policy, and
mortality yields the same numbers as `OpenFundSimulator` with the
same mortality. The `RETIREMENT_CONV` event dict is **not** byte-identical
— Stage 4 always carries three additive fields (`age_at_conversion`,
`gender`, `headcount_at_conversion`) that Stage 3 does not. Downstream
analyses (`ALMAnalysis`, `LiquidityAnalysis`, `ValueAnalysis`) route by
`type`/`payoff`/`time` only, so the additive fields are inert for
existing pipelines. The example file's assert `(b)` verifies the
numerical identity explicitly against an `OpenFundSimulator` run
(`Σ payoff` and `funding_ratio`).

------------------------------------------------------------------------

## 2. New Building Block — `Stage4Dynamics`

Frozen dataclass that packages the three time-varying knobs. Lives
beside `PensionPolicy` rather than inside it: `PensionPolicy` is the
fund's **rulebook** (immutable regulatory parameters), `Stage4Dynamics`
is the **scenario** (deterministic projection assumptions).

| Field                    | Type                       | Default | Meaning                                                                                       |
|--------------------------|----------------------------|---------|-----------------------------------------------------------------------------------------------|
| `salary_growth`          | `float`                    | `0.0`   | Annual geometric growth rate of `gross_salary`. `0.0` disables.                              |
| `conversion_rate_path`   | `Optional[Dict[int, float]]` | `None`  | Maps retirement year → applied UWS, **held-forward** (see below). `None` disables.          |
| `threshold_index_period` | `int`                      | `5`     | BVG thresholds step every k years. `k=1` ⇒ smooth annual indexation; very large ⇒ frozen.   |

The growth factors used inside `_emit_active` are

```
salary_factor    = (1 + g)^t
threshold_factor = (1 + g)^(k * (t // k))
```

where `t = sim_year - fund.start_date.year` and `k =
threshold_index_period`. At `g = 0` both factors equal `1.0` exactly
(IEEE-754 `1.0^t == 1.0` for any integer `t`), which is why the
Stage-3-identity assert passes without floating-point drift.

### `conversion_rate_path` — held-forward semantics

For retirement year `y` the applied UWS is `path[max(k in path with k
<= y)]`. Before the first key the fallback is `policy.conversion_rate`.
Concretely: `{2025: 0.0523, 2029: 0.0510}` means **5.23 % from 2025**,
**5.10 % from 2029 onwards** — years between are held at the prior
value, not snapped back to `policy.conversion_rate`. This is the
correct semantics for a step-down path (e.g. modelling a phased
reduction of the umhüllender UWS over a few years).

### UWS context — umhüllend (~5.23 %) vs gesetzlicher Mindest-UWS (6.8 %)

The Swiss BVG sets the **gesetzlicher Mindest-UWS** at **6.8 %** on
the **obligatorisch** (mandatory) portion of the AGH only. Most
pension funds also hold **überobligatorisch** (voluntary) capital and
publish a single **umhüllender** (enveloping) UWS that, via the
**Anrechnungsprinzip**, dilutes the 6.8 % minimum across the total
AGH. The Complementa Risiko Check-up 2024 reports an average
umhüllender UWS @65 of **5.23 %**, projected to **5.10 %** by 2029.

The model applies a single `conversion_rate` to the **total** AGH.
The example file uses two scenarios:

- **BASELINE**: `{2025: 0.0523, 2029: 0.0510}` — calibrated against
  the Complementa average umhüllender path.
- **STRESS**: `{START_YEAR: 0.068}` — the legal **obligatorisch**
  minimum applied flat to the **total** AGH, i.e. **without** the
  Anrechnungsprinzip. This is **not** a realistic enveloping fund
  but an illustration of the study's core claim that an
  over-prescribed UWS produces retirement losses. The example's
  asserts `(c)` / `(c2)` / `(c3)` use this scenario to exercise the
  positive-loss branch of the KPI.

### Limitation: ä_x is single-life, no survivor benefits

The Stage-4 `annuity_due` and `technical_uws` compute a **single-life**
annuity-due. Real BVG pensions include a Hinterlassenenrente
(spousal/orphan annuities), which increases the present value of the
liability and **lowers** the technically correct UWS. Ignoring
survivor benefits **biases `technical_uws` upward**, which in turn
makes the BASELINE UWS (5.23 % / 5.10 %) look **conservative** in
the KPI (technical UWS ≈ 5.50 % at age 65 with the EK 2001–2005 unisex
table and `technical_rate = 1.76 %`). A future stage can refine ä_x
to a joint-life annuity or layer a survivor-benefit factor on top.

`Stage4Dynamics` is **frozen** — construct a new one to vary the
scenario.

``` python
from awesome_actus_lib.pension import Stage4Dynamics

dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},   # held-forward
    threshold_index_period=5,
)
```

------------------------------------------------------------------------

## 3. The Simulator — `DynamicFundSimulator`

Subclass of `OpenFundSimulator`. Inherits `run()`, `_step_cohort`,
`_emit_pension`, `_apply_mortality`, `cohort_headcount_log`, and
`headcount_log` unchanged. Overrides only `_emit_active` and
`_emit_retirement`.

``` python
from awesome_actus_lib.pension import (
    DynamicFundSimulator, Stage4Dynamics, ek2001_2005,
)

simulator = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=dynamics,
    mortality=ek2001_2005(),
)
liability_cfs = simulator.run(horizon_years=40)
```

### What `_emit_active` does differently

For each active-cohort step in simulation year `sim_year`:

1. Reconstruct `t = sim_year - fund.start_date.year` from
   `cohort.birth_year + age`. No Stage-1 signature change needed
   because `age` is already a parent-method argument.
2. Compute `salary_factor` and `threshold_factor`.
3. Grow the salary: `grown_salary = cohort.gross_salary *
   salary_factor`.
4. Apply the **indexed** thresholds inline (do **not** call
   `policy.insured_salary` — it would use the fixed thresholds):
   ```
   min_sal = policy.bvg_min_salary       * threshold_factor
   max_sal = policy.bvg_max_salary       * threshold_factor
   coord   = policy.coordination_deduction * threshold_factor
   ```
5. The rest (savings rate lookup, contributions, admin cost, interest,
   AGH update) is identical to Stage 1's `_emit_active`, with
   `insured` reflecting the indexed thresholds and `grown_salary`.

### What `_emit_retirement` does differently

For each retirement event at year `sim_year`:

1. Recover `sim_year` from `event_date` (format
   `"YYYY-12-31T00:00:00"`, set by Stage 1's `run()`):
   `sim_year = int(event_date[:4])`. No Stage-1 signature change.
2. Look up the applied UWS via the **held-forward** helper
   `_applied_uws(sim_year, policy)`: the rate at the largest key
   `<= sim_year`, falling back to `policy.conversion_rate` before
   the first key or when `conversion_rate_path is None`. See
   §2's "conversion_rate_path — held-forward semantics" for the
   precise rule.
3. Emit a `RETIREMENT_CONV` event with three **extra** fields:
   - `age_at_conversion`     — needed for ä_x
   - `gender`                 — needed for ä_x
   - `headcount_at_conversion` — pre-mortality headcount (mortality
     decrement runs after `_step_cohort` returns)

These extra fields are **additive**. Stage 1/2/3 consumers
(`LiquidityAnalysis`, `ValueAnalysis`, `ALMAnalysis`) route by
`type`/`payoff`/`time` only, so the new columns are inert for them.

### `sim_year` reconstruction without touching Stage 1/2

Stage 4 never changes any parent-method signature. The two pieces of
information it needs — current simulation year and (for retirement)
age — are derivable from arguments already passed:

- `_emit_active(cohort, age, event_date, sink, policy)` →
  `sim_year = cohort.birth_year + age`.
- `_emit_retirement(cohort, event_date, sink, policy)` →
  `sim_year = int(event_date[:4])`,  `age = sim_year - cohort.birth_year`.

This keeps `simulators/Stage1_closed.py` and
`simulators/Stage2_open.py` numerically unchanged (no source-file
modification), which is verified at the end of the implementation via
`git diff --stat`.

------------------------------------------------------------------------

## 4. New Analysis Block — `annuity_due`, `technical_uws`, `RetirementLossAnalysis`

Lives in `awesome_actus_lib.pension.analysis.conversion`.

### `annuity_due(age, gender, technical_rate, mortality, terminal_age)`

Standard actuarial annuity-due:

```
ä_x = Σ_{k=0}^{K} v^k * kpx,    K = terminal_age - age,
                                v = 1 / (1 + technical_rate),
                                kpx = product_{j=0}^{k-1} (1 - q_{x+j}).
```

`gender ∈ {"m", "f", "unisex"}` selects the survival series from the
provided `MortalityTable`.

### `technical_uws(age, gender, technical_rate, mortality, terminal_age)`

`1 / annuity_due(...)`. The UWS that would make the applied pension
exactly cover the technically discounted expected liability.

### `RetirementLossAnalysis(liability_cf, policy, mortality)`

Reads enriched `RETIREMENT_CONV` events from a Stage-4 liability
`CashFlowStream`. Needs **no external metadata**: birth year derives
from `age_at_conversion + year`, gender and headcount come straight
from the event.

Per-capita loss:
```
loss_pc    = AGH * (applied_uws * ä_x - 1)
```
Total loss:
```
loss_total = loss_pc * headcount_at_conversion
```

`.summary()` returns a `pandas.DataFrame` with one row per
`RETIREMENT_CONV` event:

| Column               | Meaning                                            |
|----------------------|----------------------------------------------------|
| `cohort_id`          | contract ID                                        |
| `year`               | retirement (calendar) year                         |
| `age`                | age at conversion                                  |
| `gender`             | gender at conversion (drives ä_x)                  |
| `AGH_per_capita`     | accrued savings at the moment of conversion        |
| `applied_uws`        | `annual_pension_per_capita / AGH_per_capita`       |
| `technical_uws`      | `1 / ä_x`                                          |
| `annuity_due`        | `ä_x`                                              |
| `loss_per_capita` | AGH × (applied × ä_x − 1)                          |
| `headcount`          | headcount at the moment of conversion              |
| `loss_total`      | `loss_per_capita * headcount`                   |

**Hard requirement.** `mortality` must not be `None` — ä_x is
undefined without survival probabilities. The constructor raises a
`ValueError` otherwise.

**Hard requirement.** The liability stream must come from
`DynamicFundSimulator`. Streams from Stage 1/2/3 simulators do not
have the enriched fields, and `.summary()` raises a clear error
listing what is missing.

------------------------------------------------------------------------

## 5. End-to-End Stage 4 Example

```python
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund,
    DynamicFundSimulator, Stage4Dynamics,
    EntryPolicy, RetirementLossAnalysis,
    ek2001_2005,
)

# 1. Rules
policy = PensionPolicy()

# 2. Fund + cohorts
fund = PensionFund(policy=policy, start_date="2025-01-01T00:00:00")
fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0, gender="unisex",
))
fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970", birth_year=1970, headcount=150,
    gross_salary=110_000.0, accrued_savings=350_000.0, gender="unisex",
))
fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0,
    status="retired", annual_pension=31_380.0, gender="f",
))

# 3. EntryPolicy + Stage4Dynamics
entry_policy = EntryPolicy(entry_age=25, headcount=20,
                           gross_salary=80_000.0, gender="unisex")
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},   # held-forward, BASELINE
    threshold_index_period=5,
)

# 4. Simulate
simulator = DynamicFundSimulator(
    fund=fund, entry_policy=entry_policy,
    dynamics=dynamics, mortality=ek2001_2005(),
)
liability_cfs = simulator.run(horizon_years=40)

# 5. KPI
pva = RetirementLossAnalysis(
    liability_cf=liability_cfs,
    policy=policy,
    mortality=ek2001_2005(),
)
print(pva.summary())

# 6. Downstream merging / ALM is identical to Stage 1 — see stage1.md §10.
```

------------------------------------------------------------------------

## 6. What Stage 4 Still Does Not Do

- **Stochastic returns / mortality / salary** — Stage 4 is fully
  deterministic. Monte-Carlo over rate paths and binomial mortality are
  reserved for Stage 5.
- **Asset rebalancing** — the asset portfolio is a static bond
  ladder. No buy/sell logic.
- **Active pre-retirement mortality** — same scope decision as Stage 3.
- **Calendar-year mortality improvement** — EK 2001–2005 remains a
  period table. Generation tables / improvement factors not modelled.
- **Inflation indexation of pensions in payment** — `PENSION_PAYMENT`
  amounts are flat over time (mortality aside).
- **Surviving-dependent payments** (Witwen-/Witwerrente, Waisenrente).
- **Lump-sum withdrawals at retirement** (Kapitalbezug statt Rente).
- **Disability decrements** — risk contributions remain bookkeeping
  inflows without matching outflows.
- **Redefining `ALMAnalysis.funding_ratio` to BVV2 Art. 44** — still a
  net-cashflow PV ratio. See [stage3.md §7](stage3.md) for the
  caveat.

------------------------------------------------------------------------

## 7. Backward Compatibility

Stage 4's defaults give a true Stage-3 fallback:

```python
neutral = Stage4Dynamics(
    salary_growth=0.0,
    conversion_rate_path=None,
    threshold_index_period=1,
)
# DynamicFundSimulator(fund, entry_policy, neutral, mortality=...)  →
#   numerically identical to OpenFundSimulator(fund, entry_policy, mortality=...)
```

The example file's assert `(b)` verifies this against the
`OpenFundSimulator` reference: `Σ liability_cf.events_df["payoff"]`
and `ALMAnalysis.funding_ratio("2025-01-01")` must agree to within
relative tolerance `1e-9`.

`PensionPolicy`, `Cohort`, `EntryPolicy`, `MortalityTable`,
`PensionFund`, all Stage 1/2/3 simulator classes, and `ALMAnalysis`
are **not modified** by Stage 4. The change set is purely additive:
new files plus three additive edits to `__init__.py` re-exports.

------------------------------------------------------------------------

## 8. funding ratio — Netto-CF-Ratio vs Art-44-BVV2

`ALMAnalysis.funding_ratio` (from Stage 1) discounts a **net** liability
cashflow stream (`SAV_CONTRIB` + `RISK_CONTRIB` − `PENSION_PAYMENT` −
`ADMIN_COST`) and compares it to the asset NPV. That ratio is

```
funding_ratio = NPV(assets) / |NPV(net liability CF)|
```

It is **mix-sensitive** — contributions in the numerator of the net CF
shrink |NPV(net)| and inflate the ratio. It is **not** the regulatory
funding ratio in the sense of Art. 44 BVV2. It stays in the codebase as
a **diagnostic** (cashflow-coverage view), but should not be reported
as the fund's coverage.

### `FundingRatioAnalysis` — the Art-44-BVV2-near view

```
funding_ratio_t0 = pension assets / pension capital_t0
```

`pension capital_t0` is built **from the fund state at t0**, not from
projected events:

| Cohort status | Beitrag zum pension capital                                       |
|---------------|-------------------------------------------------------------------|
| `active`      | `accrued_savings * headcount`  (AGH-Buchwert, Art. 16 FZG)        |
| `retired`     | `annual_pension * headcount * ä_x`  (PV der laufenden Renten)     |

with `ä_x = annuity_due(age_at_t0, gender, technical_rate, mortality,
terminal_age)` reused from `analysis/conversion.py`.

`pension assets` is supplied by the caller — in the example it is
`Σ Bond-Notionals` (par values at t0).

``` python
from awesome_actus_lib.pension import FundingRatioAnalysis, ek2001_2005

dga = FundingRatioAnalysis(
    fund=liability_fund,
    mortality=ek2001_2005(),
    pension_assets=sum(c.terms["notionalPrincipal"].value
                          for c in asset_portfolio.contracts),
)
print(f"VK_t0       = {dga.pension_capital_t0():,.2f}")
print(f"DG_t0       = {dga.funding_ratio_t0():.2%}")
print(dga.breakdown())   # per-cohort decomposition
```

### Why two different numbers

Both views answer different questions:

- `funding_ratio` (Stage 1 ALM) asks "do future contribution + asset
  inflows cover future pension outflows in PV terms?".
- `funding_ratio_t0` asks "does today's asset value cover the booked
  obligation?" — the regulatory question.

The example file prints both side-by-side so the divergence is
visible. The example uses a hardcoded bond ladder (160M / 140M /
100M = 400M total) that produces `funding_ratio_t0 ≈ 107 %` against
the example's `pension capital_t0 ≈ 375 M` — within the Complementa
Risiko Check-up 2024 median region. Assert `(e)` is a loose sanity
bound (`0 < DG < 10`); to enforce a tighter Complementa-region
window, tighten the bound in the example.

### Limitation — no DG-path over time

`FundingRatioAnalysis` is **t0-only**. A time-varying funding ratio-path
would require projecting `accrued_savings × headcount` and pensioner
survival forward (and consistently rolling the asset side). That is a
Tier-2 extension — not implemented in Stage 4.
