---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 5a — Stochastic Short Rates for Pension-ALM

Stage 5a layers a **Monte-Carlo short-rate model** on top of the
deterministic Stage-4 pension simulator. Pension cashflows still
come from a single `DynamicFundSimulator` run (deterministic
demography, deterministic AGH path); per-path variation enters via
**path-specific discounting** of those cashflows.

Stochastic mortality is **out of scope** here — it is reserved for
Stage 5b. Salary growth stays deterministic (Stage 4).

Everything else from Stages 1–4 still applies — read
[stage1.md](stage1.md), [stage2.md](stage2.md), [stage3.md](stage3.md),
and [stage4.md](stage4.md) first for mental model, building blocks,
event types, mortality table, dynamic parameters, conversion-loss KPI,
and Deckungsgrad analysis.

This document covers only the deltas.

------------------------------------------------------------------------

## 1. What Changes vs Stage 4

| Aspect                          | Stage 4                                         | Stage 5a                                                                  |
|---------------------------------|-------------------------------------------------|---------------------------------------------------------------------------|
| Discount rate (assets, liab)    | flat `policy.technical_rate`                    | per-path integrated short rate from Hull-White                            |
| Asset NPV                       | scalar                                          | distribution over `I` paths (mean, std, P5/P50/P95, VaR/ES)               |
| Liability NPV                   | scalar                                          | distribution over `I` paths                                               |
| Deckungsgrad                    | scalar `DG_t0`                                  | DG-Pfad: per Stichtag (5/50/95 quantiles + `Pr(DG<1)`)                    |
| Pension cashflows               | unchanged                                       | unchanged (single deterministic run)                                      |
| Mortality, demography           | deterministic (EK 2001–2005 expectation)        | unchanged                                                                 |
| Module reuse                    | `pension/*` only                                | + `stochastic_rates/*` (read-only)                                        |
| Output objects                  | `pd.DataFrame` per analysis                     | `np.ndarray` distributions + `pd.DataFrame` long-format DG-Pfad           |
| Random seed                     | n/a                                             | reproducible via `seed=` on `create_model(...)`                           |

**Backward compatibility.** Stage 1–4 quickstarts and source files
are untouched. The Stage 5a quickstart re-uses Stage 4 setup verbatim
plus adds the stochastic section.

------------------------------------------------------------------------

## 2. Reuse from `awesome_actus_lib/stochastic_rates/`

Stage 5a re-uses the existing short-rate framework wholesale. **No
file in `stochastic_rates/` is modified.**

| Component (in `stochastic_rates/`)            | Used for                                        |
|-----------------------------------------------|-------------------------------------------------|
| `CurveCalibrator.from_market(...)`            | Fit the initial CHF spot curve                  |
| `create_model("hull_white", ...)`             | Instantiate the short-rate model                |
| `ShortRateModel.simulate(T, M, I)`            | Generate `I` short-rate paths over `T` years    |
| `SimulationResult` (`times`, `rates[step,path]`) | Container with mean/percentile helpers       |
| `HullWhite.zcb_price(t, T, r_t)`              | Analytical ZCB pricing at any future stichtag   |

Default model: **Hull-White (1-factor)** — recommended by the module
README for ALM/pension work (no-arbitrage curve fit + mean reversion).
Other models (Vasicek, CIR, Ho-Lee) are equally pluggable via the
same `create_model` factory.

Path generation in the Stage 5a quickstart:

```python
from awesome_actus_lib.stochastic_rates import CurveCalibrator, create_model

cal = CurveCalibrator.from_market(
    times=[1, 2, 5, 10],
    spot_rates=[0.012, 0.014, 0.018, 0.022],
    extrapolation_mode="flat",
)
model = create_model(
    "hull_white", r0=0.012, a=0.10, sigma=0.008,
    calibrator=cal, seed=42,
)
sim = model.simulate(T=40.0, M=40 * 12, I=500)
```

------------------------------------------------------------------------

## 3. New Building Block — `FundProjection`

Lives in `awesome_actus_lib/pension/analysis/stochastic_alm.py`.
Helper that materialises a deterministic per-`(year, cohort_id)` state
view from a `DynamicFundSimulator` after `run()`.

Reads:
- `simulator.cohort_headcount_log` → headcount + status by year
- `liability_cf.events_df` filtered by `INTEREST_CREDIT` → AGH per
  capita per active (year, cohort_id)
- `liability_cf.events_df` filtered by `RETIREMENT_CONV` →
  `annual_pension_per_capita` per cohort
- `fund.cohorts` + `entry_policy` for cohort metadata
  (birth year, gender — needed for ä_x on retired cohorts)

`ENTRY_*` cohort metadata is reconstructed by parsing
`ENTRY_{year}_AGE{age}` IDs.

API:

```python
from awesome_actus_lib.pension import FundProjection

proj = FundProjection.from_simulator(
    simulator=dynamic_simulator,
    liability_cf=liability_cfs,
    entry_policy=entry_policy,
)

state = proj.state_at(year=2030, cohort_id="ACTIVE_1970")
# -> {"status": "active", "headcount": 150, "agh_per_capita": ...,
#     "pension_per_capita": 0, "age": 60, "gender": "unisex", ...}

states_2030 = proj.cohorts_at(year=2030)
# -> list of state dicts for every cohort present at 2030
```

The projection is fully deterministic — it carries no path index.
Per-path variation only enters in the asset-side valuation via the
Hull-White ZCB step (see §4).

------------------------------------------------------------------------

## 4. New Analysis — `StochasticALMAnalysis`

### 4.1 t0 NPV distributions (path-integrated discounting)

For each event (asset and liability cashflow) at time `t_k` we
discount with the path-integrated short rate of every Monte-Carlo
path:

```
disc_i(t_k) = exp(-Σ_{j: t_j < t_k} r_path_i(t_j) Δt)
```

This left-Riemann sum mirrors the reference implementation in
`stochastic_rates/models/base.py::validate_against_analytical`.

```python
salm = StochasticALMAnalysis(
    asset_cf=asset_cfs, liability_cf=liability_cfs,
    projection=projection, policy=policy, mortality=mortality,
    model=model, sim=sim,
    base_date=pd.to_datetime("2025-01-01"),
)
dist_a = salm.asset_npv_t0_distribution()       # ndarray (I,)
dist_l = salm.liability_npv_t0_distribution()   # ndarray (I,)
dist_fr = salm.funding_ratio_t0_distribution()  # ndarray (I,)
```

Bookkeeping events (`INTEREST_CREDIT`, `RETIREMENT_CONV` — payoff 0)
are skipped so they do not pollute the sum.

### 4.2 DG-Pfad at multiple Stichtage

For each valuation year `t*` and each path `i`:

```
Vorsorgevermögen(t*, i)  = Σ asset_CF_k × P_HW(t*, t_k; r_t*^{(i)})  for t_k ≥ t*
Vorsorgekapital(t*)      = Σ active AGH(t*) × hc(t*)
                         + Σ retired pension(t*) × hc(t*) × ä_age(t*)
Deckungsgrad(t*, i)      = Vorsorgevermögen / Vorsorgekapital
```

`r_t*^{(i)}` is read directly from `sim.rates[idx, i]` for the
sim-step closest to `t*`. The Hull-White ZCB price
`model.zcb_price(t*, t_k, r_t=r_t*^{(i)})` is the analytical no-arbitrage
discount factor consistent with the calibrated initial curve.

`Vorsorgekapital` is **path-invariant** by design (Option α — see
§4.3). DG variation across paths comes entirely from
`Vorsorgevermögen` — i.e. the market value of the bond portfolio
reacting to the path-realised rate `r_t*`. This mirrors the real BVV2
situation where assets are mark-to-market and liabilities are valued
on a regulatory technical rate.

```python
dg = salm.deckungsgrad_path([2025, 2030, 2035, 2040, 2045])
# long DataFrame: columns [year, path, vv, vk, dg]

agg = dg.groupby("year")["dg"].agg(
    ["mean", "std",
     lambda s: np.percentile(s, 5),
     lambda s: np.percentile(s, 50),
     lambda s: np.percentile(s, 95),
     lambda s: float((s < 1.0).mean())]
)
```

### 4.3 Option α — flat `policy.technical_rate` for ä_x

For the retired-cohort contribution to `Vorsorgekapital(t*)` we use
`annuity_due(age, gender, policy.technical_rate, mortality, terminal_age)`
with the **fixed regulatory technical rate**, not the path-realised
short rate at `t*`. Three reasons:

1. **Aufsichtsrechtlich korrekt.** The Vorsorgekapital under
   Art. 44 BVV2 is computed on the actuarial `technical_rate` —
   a regulatory parameter set by the fund's expert, not a market rate.
2. **Realistic asymmetry.** In practice the asset side is
   mark-to-market while the liability side is at technical-rate book
   value. That asymmetry **is** the source of regulatory DG
   volatility. Cancelling it would obscure the very risk the analysis
   is meant to show.
3. **Simpler interpretation.** Single source of randomness ⇒ DG
   variation maps one-to-one to interest-rate movements.

A pure-market alternative (call it Option γ: path-specific ä_x via
Hull-White ZCB forward strip) would be a Tier-2 refinement,
intentionally **not** in Stage 5a.

### 4.4 Risk metrics

`StochasticALMAnalysis` provides standalone helpers:

```python
StochasticALMAnalysis.var(dist, alpha=0.95)
StochasticALMAnalysis.es(dist,  alpha=0.95)
StochasticALMAnalysis.quantiles(dist, qs=(0.05, 0.5, 0.95))
```

`var` / `es` treat the input as a **loss distribution** (positive =
loss). The Stage 5a quickstart applies them to `mean(NPV) - NPV_path`.

------------------------------------------------------------------------

## 5. End-to-End Stage 5a Example

```python
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund,
    DynamicFundSimulator, Stage4Dynamics, EntryPolicy,
    FundProjection, StochasticALMAnalysis,
    ek2001_2005,
)
from awesome_actus_lib.stochastic_rates import CurveCalibrator, create_model
import pandas as pd

# 1. Stage 4 setup (fund, cohorts, dynamics) — see stage4.md §5
fund = ...
entry_policy = EntryPolicy(entry_age=25, headcount=20,
                           gross_salary=80_000.0, gender="unisex")
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)
mortality = ek2001_2005()

simulator = DynamicFundSimulator(
    fund=fund, entry_policy=entry_policy,
    dynamics=dynamics, mortality=mortality,
)
liability_cfs = simulator.run(horizon_years=40)
asset_cfs = ...  # bond portfolio via PublicActusService

# 2. Hull-White on CHF plateau curve
cal = CurveCalibrator.from_market(
    times=[1, 2, 5, 10], spot_rates=[0.012, 0.014, 0.018, 0.022],
    extrapolation_mode="flat",
)
model = create_model("hull_white", r0=0.012, a=0.10, sigma=0.008,
                     calibrator=cal, seed=42)
sim = model.simulate(T=40.0, M=40 * 12, I=500)

# 3. Stochastic ALM
projection = FundProjection.from_simulator(
    simulator=simulator, liability_cf=liability_cfs,
    entry_policy=entry_policy,
)
salm = StochasticALMAnalysis(
    asset_cf=asset_cfs, liability_cf=liability_cfs,
    projection=projection, policy=fund.policy, mortality=mortality,
    model=model, sim=sim,
    base_date=pd.to_datetime("2025-01-01"),
)

dist_a = salm.asset_npv_t0_distribution()
dist_l = salm.liability_npv_t0_distribution()
dg = salm.deckungsgrad_path([2025, 2030, 2035, 2040, 2045])
print(dg.groupby("year")["dg"].agg(["mean", "std",
                                    lambda s: float((s < 1.0).mean())]))
```

------------------------------------------------------------------------

## 6. Risk Metrics

| Metric                              | Computed via                                                                  |
|-------------------------------------|-------------------------------------------------------------------------------|
| Asset NPV distribution              | `salm.asset_npv_t0_distribution()`                                            |
| Liability NPV distribution          | `salm.liability_npv_t0_distribution()`                                        |
| Funding-ratio (Netto-CF) distrib.   | `salm.funding_ratio_t0_distribution()`                                        |
| Per-Stichtag DG quantiles           | `salm.deckungsgrad_path(years).groupby("year")["dg"].agg([...])`              |
| `Pr(DG < 1.0)` per Stichtag         | `(dg.query("year == y").dg < 1.0).mean()`                                     |
| VaR / ES on loss distribution       | `StochasticALMAnalysis.var(losses, 0.95)` / `.es(losses, 0.95)`               |

------------------------------------------------------------------------

## 7. What Stage 5a Still Does Not Do

- **Stochastic mortality** — reserved for Stage 5b (binomial
  Aro-Pennanen on retiree headcounts).
- **Stochastic salary growth** — Stage 4 stays deterministic.
- **Path-specific ä_x** (Option γ) — Vorsorgekapital remains on flat
  `policy.technical_rate`. See §4.3.
- **Asset rebalancing** — bond portfolio is the same VK-scaled
  static ladder from Stage 4.
- **Real market data ingestion** — the example uses a hardcoded CHF
  plateau curve. Replacing with SNB/Bloomberg feeds is left to the
  user.
- **DG-Pfad with projected fund evolution beyond Stage-4 trajectory** —
  `FundProjection` reads exactly what the Stage-4 simulator emits;
  no further demographic stochasticity is layered on.

------------------------------------------------------------------------

## 8. Backward Compatibility

Stage 5a adds three files (`analysis/stochastic_alm.py`,
`examples/Stage5a_pension_quickstart.py`, `docs/stage5a.md`) and three
additive re-export edits to `pension/__init__.py` and
`pension/analysis/__init__.py`.

The following files are **not modified** by Stage 5a:

```
awesome_actus_lib/pension/policy.py
awesome_actus_lib/pension/cohort.py
awesome_actus_lib/pension/events.py
awesome_actus_lib/pension/entry.py
awesome_actus_lib/pension/fund.py
awesome_actus_lib/pension/mortality.py
awesome_actus_lib/pension/simulators/Stage1_closed.py
awesome_actus_lib/pension/simulators/Stage2_open.py
awesome_actus_lib/pension/simulators/Stage4_dynamic.py
awesome_actus_lib/pension/analysis/alm.py
awesome_actus_lib/pension/analysis/conversion.py
awesome_actus_lib/pension/analysis/deckungsgrad.py
awesome_actus_lib/stochastic_rates/**
```

The Stage 5a quickstart's assert `(b)` continues to verify that the
Stage-4 simulator with neutral dynamics reproduces the Stage-3
`OpenFundSimulator` cashflows numerically (`Σ payoff` rel. tolerance
`1e-9`). Asserts `(g)` / `(h)` / `(i)` / `(j)` validate the
Stage-5a-specific MC machinery.
