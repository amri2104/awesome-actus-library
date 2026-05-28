---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 1 — Closed Fund Pension Simulation

Documentation for the Stage 1 liability-side simulator of the pension
extension in `awesome_actus_lib.pension`. Stage 1 = **closed fund**: a
fixed member population is projected over a multi-year horizon. No new
entrants, no mortality, no salary/inflation dynamics. Each step is one
calendar year.

The output is an
`awesome_actus_lib.models.cashFlowStream.CashFlowStream` — the same type
ACTUS returns for financial contracts — so the liability side plugs
directly into the existing `LiquidityAnalysis`, `ValueAnalysis`,
`IncomeAnalysis` tooling.

------------------------------------------------------------------------

## 1. Mental Model — Asset Side vs Liability Side

The asset side and the liability side look different in code but produce
the same type of object at the end.

**Asset side (already in AAL):** - You write financial contracts (e.g.
`PAM` for bonds). - The remote ACTUS engine (`PublicActusService`)
generates the events. - Output: `CashFlowStream`.

**Liability side (this Stage 1 extension):** - You **do not write
contracts.** BVG pension rules are not in the ACTUS standard, so the
ACTUS engine cannot produce them. - Instead, you describe **rules +
member population**: rules → `PensionPolicy`, population → list of
`Cohort`, container → `PensionFund`. - A local Python simulator
(`ClosedFundSimulator`) reads the fund and emits events year by year. -
Output: `CashFlowStream` — same type as asset side.

This symmetry is intentional: downstream analysis code does not need to
know which side it is looking at.

------------------------------------------------------------------------

## 2. The Three Building Blocks

### 2.1 `PensionPolicy` — the fund's rulebook

Defines the regulatory/actuarial parameters of the pension fund. Created
once and reused for the whole simulation.

Fields (defaults shown reflect Swiss BVG minima as of writing):

| Field | Default | Meaning |
|----|----|----|
| `bvg_min_interest_rate` | `0.0125` | Statutory BVG minimum interest (Art. 15 BVG). Acts as a **floor** on the crediting rate: the fund credits `max(applied_interest_rate, bvg_min_interest_rate)` via `PensionPolicy.credited_interest_rate()`. With the baseline (applied 2.0 % > 1.25 %) the floor is inactive; it binds only if a scenario sets the applied rate below the minimum. |
| `applied_interest_rate` | `0.02` | Actual interest rate the fund credits on accrued savings (AGH) each year. |
| `technical_rate` | `0.0176` | Discount rate used for actuarial liability valuation. |
| `conversion_rate` | `0.0523` | Umwandlungssatz — AGH at retirement × this = annual pension. |
| `retirement_age` | `65` | Age at which an active cohort retires. |
| `terminal_age` | `100` | Age at which a cohort stops emitting events (effective end of life). |
| `savings_contribution_by_age` | BVG scale 7/10/15/18 % | Map `(low, high) -> rate` of insured salary saved per year. |
| `risk_contribution_rate` | `0.015` | Disability/death insurance contribution, % of insured salary. |
| `admin_cost_per_member` | `400.0` | Fixed administrative cost per active member per year. |
| `coordination_deduction` | `26_460` | BVG Koordinationsabzug. |
| `bvg_min_salary` | `22_680` | Lower BVG salary threshold (below: no insured salary). |
| `bvg_max_salary` | `90_720` | Upper BVG salary cap. |

Helper methods: - `insured_salary(gross_salary)` — applies floor, cap,
coordination deduction. - `savings_rate(age)` — looks up the per-age
savings rate from the scale.

**`PensionPolicy` is frozen (immutable).** Build a new one to change a
parameter.

``` python
from awesome_actus_lib.pension import PensionPolicy

# Default BVG minima
policy = PensionPolicy()

# Custom example — higher applied interest, custom savings scale
policy = PensionPolicy(
    applied_interest_rate=0.025,
    conversion_rate=0.0500,
    savings_contribution_by_age={
        (25, 34): 0.09,
        (35, 44): 0.12,
        (45, 54): 0.16,
        (55, 64): 0.20,
    },
)
```

### 2.2 `Cohort` — a group of members

A `Cohort` is a group of insured persons with identical attributes. Use
as many cohorts as needed to express the population's heterogeneity (age
bands, salary bands, retirees, etc.).

| Field | Type | Meaning |
|----|----|----|
| `cohort_id` | `str` | Unique identifier (used as contract-ID-equivalent downstream). |
| `birth_year` | `int` | Year of birth — drives age at each simulation step. |
| `headcount` | `int` | Number of members in this cohort. |
| `gross_salary` | `float` | Gross annual salary per member. |
| `accrued_savings` | `float` | AGH per capita at t=0. |
| `status` | `"active"` \| `"retired"` | Default `"active"`. |
| `annual_pension` | `Optional[float]` | Annual pension per capita. Required for retirees, computed by the simulator at retirement otherwise. |

### 2.3 `PensionFund` — the container

Glues `PensionPolicy`, the simulation `start_date`, and the list of
cohorts together.

``` python
from awesome_actus_lib.pension import PensionFund, Cohort

# start_date accepts either an ISO 8601 string (matches PAM-style dates)
# or a datetime.date object.
fund = PensionFund(
    policy=policy,
    start_date="2025-01-01T00:00:00",
)

fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990",
    birth_year=1990,
    headcount=200,
    gross_salary=90_000.0,
    accrued_savings=80_000.0,
))
```

------------------------------------------------------------------------

## 3. The Simulator — `ClosedFundSimulator`

Stage 1's simulator. Annual time-stepped, deterministic, closed fund (no
new entrants), no mortality.

``` python
from awesome_actus_lib.pension import ClosedFundSimulator

liability_cfs = ClosedFundSimulator(fund).run(horizon_years=40)
```

### What `run()` does

For each simulation year
`y = start_date.year, ..., start_date.year + horizon - 1`, and for each
cohort:

1.  Compute `age = y - cohort.birth_year`.
2.  If `age >= terminal_age`: skip (cohort effectively gone).
3.  Dispatch by status and age:
    -   **Active and `age < retirement_age`**: emit 4 events
        -   `SAV_CONTRIB` — savings contribution inflow (payoff \> 0)
        -   `RISK_CONTRIB` — risk contribution inflow (payoff \> 0)
        -   `INTEREST_CREDIT` — bookkeeping event (payoff = 0); carries
            `interest_per_capita` and updated `agh_per_capita`
        -   `ADMIN_COST` — admin cost outflow (payoff \< 0)
        -   Internal state update:
            `accrued_savings += interest + new savings`
    -   **Active and `age >= retirement_age`**: emit
        -   `RETIREMENT_CONV` once — converts AGH to
            `annual_pension =   accrued_savings * conversion_rate`;
            status flips to `"retired"`
        -   Then a `PENSION_PAYMENT` for the same year.
    -   **Retired**: emit `PENSION_PAYMENT` (payoff =
        `-annual_pension *   headcount`).

Event timestamps are placed at year-end (`YYYY-12-31T00:00:00`).

### Output

`run()` returns a `CashFlowStream` whose `raw_response` follows the same
structure as ACTUS service output (`[{contractID, events, children}]`).
The `portfolio` attribute is a duck-typed wrapper around the cohort list
so that `__len__` and `.contracts` work; downstream analysis classes
that only use these are not affected.

### Input immutability

The simulator deep-copies the input cohorts before mutating their state,
so re-running the same fund yields identical results and the original
`Cohort` objects are not modified.

------------------------------------------------------------------------

## 4. Defining a Retiree — Variants

The Cohort schema covers active members and retirees with the same
fields, distinguished by `status` and the presence of `annual_pension`.
The combinatorics yield three practical patterns.

### Variant A — Existing retiree (typical case)

The member is already retired at t=0 and the pension is fixed.

``` python
Cohort(
    cohort_id="RETIRED_1955",
    birth_year=1955,
    headcount=80,
    gross_salary=0.0,
    accrued_savings=0.0,
    status="retired",
    annual_pension=35_000.0,
)
```

Simulator behavior: every year until age 100, one `PENSION_PAYMENT`
event of `-35_000 * 80 = -2_800_000` is emitted.

### Variant B — Just-retired, auto-converted

The cohort is past `retirement_age` but still flagged `"active"`. The
simulator triggers `RETIREMENT_CONV` automatically on the first step
using the cohort's `accrued_savings` and the policy's `conversion_rate`.

``` python
Cohort(
    cohort_id="ACTIVE_1955",  # age 70 at start = 2025
    birth_year=1955,
    headcount=80,
    gross_salary=0.0,
    accrued_savings=600_000.0,
    # status defaults to "active", annual_pension is None
)
# Year 1 (2025): RETIREMENT_CONV → annual_pension = 600_000 * 0.0523 = 31_380
# Years 2..N: PENSION_PAYMENT until age 100.
```

Use this when you want the conversion to be governed by the policy
rather than hardcoded.

### Variant C — Heterogeneous retiree groups

When pensions are not uniform across a birth-year cohort (e.g.
obligatorisch vs überobligatorisch, different past employers), split
into multiple cohorts.

``` python
Cohort("RETIRED_1955_LOW",  1955, 50, 0.0, 0.0, "retired", annual_pension=28_000.0)
Cohort("RETIRED_1955_HIGH", 1955, 30, 0.0, 0.0, "retired", annual_pension=55_000.0)
```

The simulator treats them independently. This is the only way to model
within-age-band pension dispersion in Stage 1.

------------------------------------------------------------------------

## 5. Required Order of Definition

```         
PensionPolicy   →   PensionFund(policy=..., start_date=...)
                          ↓
                    add_cohort(Cohort(...))   (one or more)
                          ↓
                    ClosedFundSimulator(fund).run(horizon_years=...)
                          ↓
                    CashFlowStream (liability)
```

`PensionPolicy` must exist before `PensionFund` because the constructor
takes it. Cohorts can be added before or after construction
(`PensionFund(..., cohorts=[...])` or `fund.add_cohort(...)`).

------------------------------------------------------------------------

## 6. Aligning Asset and Liability Timelines

For meaningful ALM analysis, the asset and liability sides must share
the same t=0.

| Side | Time field | Example |
|----|----|----|
| Asset PAM | `statusDate`, `contractDealDate`, `initialExchangeDate` | `"2025-01-01T00:00:00"` |
| Liability fund | `start_date` | `"2025-01-01T00:00:00"` |

`PensionFund.start_date` accepts the same ISO 8601 string format used by
PAM contracts (it is parsed via `date.fromisoformat`) so the example
file stays visually consistent.

If the timelines diverge, downstream comparisons (funding ratio,
cashflow matching, duration gap) compare events from different points in
time and become meaningless.

------------------------------------------------------------------------

## 7. End-to-End Stage 1 Example

``` python
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund, ClosedFundSimulator,
)

# 1. Rules
policy = PensionPolicy()  # BVG defaults

# 2. Fund + cohorts
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

# 3. Simulate
liability_cfs = ClosedFundSimulator(fund).run(horizon_years=40)

# 4. Downstream analysis (same API as for asset CashFlowStreams)
# from awesome_actus_lib.analysis.liquidity import LiquidityAnalysis
# LiquidityAnalysis(liability_cfs).plot()
```

------------------------------------------------------------------------

## 8. Event Types Reference

Defined in `awesome_actus_lib.pension.events`:

| Event | When | Payoff sign | Extra fields |
|----|----|----|----|
| `SAV_CONTRIB` | active, age \< retirement | `+` (inflow) | — |
| `RISK_CONTRIB` | active, age \< retirement | `+` (inflow) | — |
| `INTEREST_CREDIT` | active, age \< retirement | `0` (bookkeeping) | `interest_per_capita`, `agh_per_capita` |
| `ADMIN_COST` | active, age \< retirement | `−` (outflow) | — |
| `RETIREMENT_CONV` | active, age \>= retirement (once) | `0` (bookkeeping) | `agh_per_capita_at_conversion`, `annual_pension_per_capita` |
| `PENSION_PAYMENT` | retired | `−` (outflow) | — |

------------------------------------------------------------------------

## 9. What Stage 1 Does Not Do

-   No new entrants (use Stage 2 / `OpenFundSimulator` + `EntryPolicy`).
-   No mortality decrement — cohorts stay full until `terminal_age`.
-   No salary growth, no inflation indexing of pensions.
-   No stochastic returns — interest is deterministic from
    `applied_interest_rate`.
-   No biometric refinement (disability, surviving dependents are not
    modeled separately; risk contributions are recorded as a cashflow
    but the corresponding outflows are not modeled).

Extending any of these is the natural path for future stages.

------------------------------------------------------------------------

## 10. Merging Asset and Liability Cashflows

Once you have both `asset_cfs` (from
`PublicActusService.generateEvents(asset_portfolio)`) and `liability_cfs`
(from `ClosedFundSimulator(fund).run(...)`), there are three practical
ways to bring them together. Each answers a different question.

### 10.1 Variant 1 — Net Liquidity per Period

**Question:** How much cash does the fund have left each year after
paying pensions?

Aggregates each side separately per period, then takes the algebraic
sum. Liability payoffs are already negative (outflows), so the column
`net` is asset inflows + liability outflows = net liquidity.

Built-in helper: `ALMAnalysis.net_liquidity(freq=...)`.

``` python
from awesome_actus_lib.pension import ALMAnalysis

alm = ALMAnalysis(assets_cf=asset_cfs, liabilities_cf=liability_cfs)
net_table = alm.net_liquidity(freq="YE")  # year-end buckets
print(net_table)
```

Output: `pandas.DataFrame` with columns `netLiquidity_assets`,
`netLiquidity_liabilities`, `net`.

Frequency options (passed through to `LiquidityAnalysis`):

| Code  | Meaning      |
|-------|--------------|
| `M`   | monthly      |
| `Q`   | quarterly    |
| `2Q`  | half-yearly  |
| `Y` / `YE` | yearly  |

**Use when:** liquidity steering, cashflow matching, dimensioning a
liquidity buffer.

### 10.2 Variant 2 — Funding Ratio (PV Comparison)

**Question:** At a given valuation date, do the assets cover the
liabilities?

Computes `NPV(assets) / |NPV(liabilities)|`. Side-by-side comparison
of present values, not a true aggregation. The same discount source
must be applied to both sides for the ratio to be meaningful.

Two discount sources are supported:

-   **Flat rate** — pass `flat_rate=...` (e.g. the policy's
    `technical_rate`).
-   **Yield curve** — pass `discount_source=YieldCurve(...)` (or a
    `ReferenceIndex`), optionally selecting a specific curve via
    `discount_curve_code`.

``` python
# Flat-rate variant
alm = ALMAnalysis(
    assets_cf=asset_cfs,
    liabilities_cf=liability_cfs,
    flat_rate=0.0176,  # policy.technical_rate
)
fr = alm.funding_ratio(as_of="2025-01-01")
print(f"Funding Ratio: {fr:.2%}")

# Path over multiple valuation dates
fr_path = alm.funding_ratio_path([
    "2025-01-01", "2030-01-01", "2035-01-01", "2040-01-01",
])
print(fr_path)
```

`funding_ratio` returns a scalar; `funding_ratio_path` returns a
`pandas.Series` indexed by date.

**Use when:** supervisory reporting, regulatory funding-ratio
monitoring, scenario analysis at discrete reporting dates.

### 10.3 Variant 3 — Combined CashFlowStream

**Question:** What does the full picture (assets + liabilities) look
like in one object I can analyse or plot uniformly?

Merges both `raw_response` payloads into a single `CashFlowStream`.
After merging, every existing `CashFlowStream`-consuming analysis class
(`LiquidityAnalysis`, `ValueAnalysis`, `IncomeAnalysis`) operates over
both sides at once, and any portfolio-level plot includes both.

No built-in helper exists. The construction inverts
`CashFlowStream._parse_events` to rebuild `raw_response` from the
events DataFrame, concatenates, and re-wraps:

``` python
from awesome_actus_lib.models.cashFlowStream import CashFlowStream
from awesome_actus_lib.pension.fund import _DummyPortfolio


def _df_to_raw_response(events_df):
    raw = []
    for cid, grp in events_df.groupby("contractId"):
        events = grp.drop(columns=["contractId"]).to_dict(orient="records")
        raw.append({"contractID": cid, "events": events, "children": []})
    return raw


def merge_cashflow_streams(asset_cfs, liability_cfs):
    combined_raw = (
        _df_to_raw_response(asset_cfs.events_df)
        + _df_to_raw_response(liability_cfs.events_df)
    )
    combined_contracts = (
        list(asset_cfs.portfolio.contracts)
        + list(liability_cfs.portfolio.contracts)
    )
    return CashFlowStream(
        portfolio=_DummyPortfolio(combined_contracts),
        riskFactors=asset_cfs.riskFactors,
        raw_response=combined_raw,
    )


combined_cfs = merge_cashflow_streams(asset_cfs, liability_cfs)

from awesome_actus_lib.analysis.liquidity import LiquidityAnalysis
LiquidityAnalysis(combined_cfs, freq="Y").plot()
```

**Caveats:**

-   `contractID` and `cohort_id` must be **unique across both sides** —
    duplicates collapse into the same contract row.
-   `riskFactors` from the asset side wins; if you discount the
    combined stream, the asset-side curve is used by default.
-   `portfolio` is wrapped in `_DummyPortfolio` so it stays
    duck-compatible with the analysis layer, but it is not a full AAL
    `Portfolio` — methods beyond `__len__` and `.contracts` will not
    work.

**Use when:** unified plots, exporting one combined event log,
applying the same filter or aggregation to both sides.

### 10.4 Which Variant for Which Question?

| Variant                | Output                      | Typical question                                       |
|--------------|--------------|--------------------------------------------|
| 1 Net Liquidity        | `DataFrame` per period      | Will cash cover pension payments next year?            |
| 2 Funding Ratio        | scalar or `Series`          | Is the fund covered at this valuation date?            |
| 3 Combined Stream      | `CashFlowStream`            | What does the whole fund look like as one object?      |

### 10.5 Preconditions

-   Both `CashFlowStream`s must share the same t=0 (see Section 6).
-   Contract IDs across asset and liability sides must be disjoint
    (Variant 3) or at least not collide on intended distinct entities
    (Variants 1 & 2 are robust to ID collisions because the two streams
    are kept separate).
-   For PV-based variants, the same discount source should be applied
    to both sides.

