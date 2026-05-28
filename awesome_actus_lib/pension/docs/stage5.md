---
editor_options:
  markdown:
    wrap: 72
---

# Stage 5 — Stochastic Extensions (Assets, Liabilities, Risk Attribution)

Documentation for the Stage 5 extension of `awesome_actus_lib.pension`.
Stage 5 introduces **stochastic risk channels** on top of the Stage 1–4
deterministic pipeline and is delivered in three increments:

* **Stage 5a Phase 1** — stochastic asset-side ALM via short-rate paths
  driving floating-rate bond resets, with sticky flat-rate discounting
  (`StochasticALMAnalysis`).
* **Stage 5a Phase 2** — mark-to-market **equity sleeve** driven by a
  Geometric Brownian Motion (GBM) and a forward-projected solvency
  Deckungsgrad (`StochasticDeckungsgradAnalysis`).
* **Stage 5b** — stochastic **liability** cashflow paths via binomial
  cohort transitions (`simulate_liability_paths`) and a **variance
  attribution** protocol across the asset and mortality channels
  (`RiskAttribution`).

Read [stage1.md](stage1.md) – [stage4.md](stage4.md) first for the
mental model, the building blocks (`PensionPolicy`, `Cohort`,
`PensionFund`, `EntryPolicy`, `MortalityTable`, `Stage4Dynamics`), the
event types, and the deterministic ALM toolkit. This document covers
only the Stage 5 deltas.

---

## 0. What Changes vs Stage 1–4

| Aspect                | Stage 1 / 2 / 3 / 4                      | Stage 5a Phase 1            | Stage 5a Phase 2            | Stage 5b                                |
| :-------------------- | :--------------------------------------- | :-------------------------- | :-------------------------- | :-------------------------------------- |
| **Asset side**        | Static fixed-coupon bond ladder.         | Mixed fixed/floater PAMs.   | + optional GBM equity sleeve. | Same asset side as 5a (Phase 1 or 2).   |
| **Rate path**         | Flat / deterministic.                    | Stochastic short-rate (HW). | Stochastic short-rate (HW). | Stochastic short-rate (HW).             |
| **Equity sleeve**     | --                                       | --                          | GBM mark-to-market.         | GBM optional.                           |
| **Discounting**       | Flat `technical_rate`.                   | Sticky flat `i_tech`.       | Sticky flat `i_tech`.       | Sticky flat `i_tech`.                   |
| **Liabilities**       | Deterministic (Stage 4).                 | Deterministic (Stage 4).    | Deterministic (Stage 4).    | **Stochastic** (binomial cohort decrement). |
| **Solvency lens**     | `DeckungsgradAnalysis` (t₀ snapshot).    | not extended.               | `StochasticDeckungsgradAnalysis` (path-dependent V_i(d)). | reuses Phase 2 if equity given. |
| **Reporting**         | Single deterministic numbers.            | Distribution of funding ratio per as_of. | Distribution of solvency DG per as_of. | Variance decomposition across channels. |

All Stage 5 features are **additive**. Defaults preserve Stage 4
behaviour numerically.

---

## 1. Stage 5a Phase 1 — Stochastic Asset-Side ALM

### 1.1 Mental Model — Resets vs Discounting

* **Floating-rate resets are stochastic.** PAM contracts with
  `marketObjectCodeOfRateReset == "IR_SCENARIO"` reprice on every path:
  the coupon rate at every reset date is read off the simulated
  short-rate path.
* **Discounting is sticky.** Both asset and liability NPVs use the flat
  scalar `technical_rate` regardless of which path is being evaluated.
  This is the regulatory-stickiness convention motivated qualitatively
  in FRP 4. Stochastic discounting (path-specific discount factors) is
  deliberately out of scope and recorded as a future-stage extension.
* **Liabilities are deterministic in Phase 1.** The liability cashflow
  stream is projected once via `DynamicFundSimulator` and reused across
  every asset-side path.

### 1.2 `StochasticALMAnalysis` — Parameters

Lives in `awesome_actus_lib.pension.analysis.stochastic_alm`.

| Parameter           | Type                                      | Default          | Meaning                                                            |
| :------------------ | :---------------------------------------- | :--------------- | :----------------------------------------------------------------- |
| `asset_portfolio`   | `Portfolio`                               | *required*       | Mixed portfolio with floaters bound to `market_code`.              |
| `liabilities_cf`    | `CashFlowStream` \| `List[CashFlowStream]` | *required*       | Either single deterministic stream (broadcast), or list of length `n_paths` paired position-wise with asset paths (used in Stage 5b). |
| `calibrator`        | `CurveCalibrator`                         | *required*       | Calibrated initial term structure.                                 |
| `technical_rate`    | `float`                                   | *required*       | Sticky flat discount rate $i_{\mathrm{tech}}$.                    |
| `model`             | `str`                                     | `"hull_white"`   | One of `vasicek`, `cir`, `ho_lee`, `hull_white`.                  |
| `model_params`      | `dict`                                    | `None`           | Must include `"r0"`. Curve-fitting models receive the calibrator automatically. |
| `equity_params`     | `dict`                                    | `None`           | **Phase 2 hook.** When given, also simulates a GBM equity index. Keys: `S0`, `mu`, `sigma`. |
| `n_paths`           | `int`                                     | `500`            | Number of Monte-Carlo paths.                                       |
| `horizon_years`     | `float`                                   | `40.0`           | Projection horizon.                                                |
| `steps_per_year`    | `int`                                     | `12`             | Simulation frequency per year.                                     |
| `base_date`         | `str`                                     | `"2025-01-01"`   | $t_0$ of the simulation.                                          |
| `market_code`       | `str`                                     | `"IR_SCENARIO"`  | Reference-index key wired to ACTUS floaters.                       |
| `seed`              | `int`                                     | `42`             | Master seed for the short-rate model. GBM uses `seed + 1000`.      |
| `service`           | `PublicActusService`                      | `None`           | ACTUS service handle; defaults to a fresh `PublicActusService()`.  |
| `verbose`           | `bool`                                    | `True`           | Progress messages during `run()`.                                  |

### 1.3 `run()` — Algorithm

1. Build the chosen short-rate model and simulate the rate matrix of
   shape `(M+1, n_paths)` with `M = horizon_years * steps_per_year`.
2. If `equity_params` is set, also simulate a GBM equity path matrix
   with seed `seed + 1000` (Stage 5a Phase 2).
3. For each path $i$:
   * convert the short-rate path to a `ReferenceIndex` keyed
     `market_code`;
   * if equity is on, also convert the equity path to a
     `ReferenceIndex` keyed `EQ_INDEX`;
   * call `service.generateEvents(asset_portfolio, riskFactors=...)`
     and store the asset `CashFlowStream`.

**Cost:** one ACTUS round-trip per path. Keep `n_paths` modest
(typically 200–500) for local runs.

### 1.4 Reporting

* `funding_ratio_distribution(as_of)` returns a length-`n_paths` NumPy
  array of liquidity-lens funding-ratio realisations at `as_of`. Pair
  mode (Stage 5b) uses `liabilities_cf[i]` per path; broadcast mode
  uses the single deterministic stream for every path.
* `summary(as_of)` returns a dict with:
  * `mean`, `std`, percentiles (`p5`, `p50`, `p95`), `min`, `max`;
  * `dg_p5` = 5th percentile (the 95% VaR-level floor of the funding
    ratio, since the downside is a *low* value);
  * `dg_es5` = mean of the worst-5% tail (expected shortfall);
  * `p_underfunded` = $P(\text{Funding Ratio} < 100\%)$.
* `fan_chart_data(dates, quantiles=(5, 25, 50, 75, 95))` returns a
  DataFrame of quantile paths over a list of valuation dates.

---

## 2. Stage 5a Phase 2 — Equity Sleeve and Stochastic Deckungsgrad

Phase 2 adds an **equity allocation** modelled as a Geometric Brownian
Motion (GBM) and a **path-dependent solvency Deckungsgrad** that values
that allocation at market.

### 2.1 GBM Equity Driver

Driven by `awesome_actus_lib.stochastic_rates.GBMModel` under the
physical measure:

$$
dS_t = \mu\,S_t\,dt + \sigma_S\,S_t\,dW^S_t,
\qquad
S_t = S_0 \exp\!\Bigl[\bigl(\mu - \tfrac{1}{2}\sigma_S^2\bigr) t + \sigma_S W^S_t\Bigr].
$$

Activated by passing `equity_params={"S0": ..., "mu": ..., "sigma": ...}`
to `StochasticALMAnalysis`. Independence from the short-rate driver is
imposed by seeding GBM with `seed + 1000`; the equity path matrix is
exposed as `salm.equity_simulation`.

### 2.2 `StochasticDeckungsgradAnalysis` — Parameters

Lives in `awesome_actus_lib.pension.analysis.stochastic_deckungsgrad`.

| Parameter            | Type                          | Meaning                                                            |
| :------------------- | :---------------------------- | :----------------------------------------------------------------- |
| `salm`               | `StochasticALMAnalysis`       | A run instance providing rate and (optional) equity paths.          |
| `fund`               | `PensionFund`                 | Liability demographics at $t_0$.                                   |
| `mortality`          | `MortalityTable`              | Used by the underlying `DeckungsgradAnalysis` for $\ddot{a}_x$.    |
| `bond_book`          | `float`                       | Book value of the bond portfolio at $t_0$.                         |
| `cash_book`          | `float`                       | Book value of cash and equivalents at $t_0$.                       |
| `equity_sleeve_0`    | `float`                       | Initial equity allocation at $t_0$ (mark-to-market basis).         |

### 2.3 Solvency Construction

`Vorsorgekapital_t0` is the deterministic Stage-4 quantity reused
verbatim from `DeckungsgradAnalysis` (AGH-Buchwert for actives,
$P\cdot N\cdot \ddot{a}_x$ for retirees).

The asset stock per path is

$$
V_i(d) \;=\; \texttt{bond\_book} + \texttt{cash\_book}
+ \texttt{equity\_sleeve\_0} \cdot \frac{S_i(d)}{S_0},
$$

so bonds and cash stay at book value (sticky), and only the equity
sleeve is marked to market through the GBM path. The path-dependent
Deckungsgrad is

$$
\mathrm{DG}_i(d) \;=\; \frac{V_i(d)}{\mathrm{VK}_{t_0}}.
$$

When `equity_simulation is None` (or `sigma = 0`) the asset stock is
constant across paths, and the distribution collapses to a single
scalar.

### 2.4 Reporting

* `deckungsgrad_distribution(as_of)` — NumPy array of $\mathrm{DG}_i(d)$
  across all paths.
* `summary(as_of)` — `mean`, `std`, percentiles, `dg_es5`,
  `p_underfunded`.
* `fan_chart_data(dates, quantiles=(5, 25, 50, 75, 95))` — quantile
  paths over time.

### 2.5 Scope Caveats

* Bonds and cash at **book value** by design (sticky, in line with Swiss
  accounting practice for pension funds). A fully economic balance
  sheet (mark-to-market bonds) is an obvious next extension.
* `VK_{t_0}` is held **constant over time** in this Phase. Forward
  projection of VK (active AGH roll-forward, retiree decrement, new
  retirements, $i_{\mathrm{tech}}$ re-evaluation) is identified as a
  future stage (sometimes referenced as Stage 5c).

---

## 3. Stage 5b — Stochastic Liabilities and Risk Attribution

Phase 5b lifts the liabilities from "deterministic, generated once" to
"stochastic, one independent realisation per path", and provides a
variance-attribution protocol across the asset and mortality channels.

### 3.1 Binomial Cohort Transitions

Each cohort is treated as a closed group of independent lives. Let
$N_c(t)$ be the survivor count of cohort $c$ at the start of year $t$
and $q_{x_c(t)}$ the one-year death probability at the cohort's
then-current age and gender. The yearly survivor draw is

$$
N_c(t+1) \;=\; \operatorname{Binomial}\!\bigl(N_c(t),\;
1 - q_{x_c(t)}\bigr),
$$

i.e. the post-emit-then-decrement survivor count is sampled rather than
multiplied by the expected survival probability. This is the Maffra /
Armstrong / Pennanen (2021) Eq. 1 convention. The deterministic Stage 3
behaviour is recovered in expectation: $\mathbb{E}[N_c(t+1)] =
N_c(t)\,(1-q_{x_c(t)})$.

Behavioural switches inside `ClosedFundSimulator` (inherited by
`OpenFundSimulator` and `DynamicFundSimulator`):

* `stochastic_mortality=True` activates the binomial draw.
* `mortality_filter="all"` applies the binomial decrement to **active
  and retired** cohorts; `"retirees"` restricts it to retired cohorts
  (matching the Stage 3 deterministic scope but with stochastic noise).
* `rng` is a `numpy.random.Generator`; per-cohort sub-streams are
  spawned lazily so that toggling `mortality_filter` does not perturb
  the draws of cohorts already participating.

### 3.2 `simulate_liability_paths(...)`

Lives in `awesome_actus_lib.pension.simulators.liability_paths`.

```python
def simulate_liability_paths(
    fund: PensionFund,
    entry_policy: EntryPolicy,
    dynamics: Stage4Dynamics,
    mortality: MortalityTable,
    *,
    horizon_years: int,
    n_paths: int,
    seed: Optional[int] = None,
    mortality_filter: str = "all",
) -> List[CashFlowStream]:
```

Generates `n_paths` independent liability streams. Path independence
is enforced via `numpy.random.SeedSequence(seed).spawn(n_paths)`: each
path gets its own master `Generator`, and inside each simulator the
per-cohort RNGs are further spawned from that path's master rng. The
function is a pure-Python loop — no ACTUS service round-trips — so it
is significantly cheaper than the asset-side Monte Carlo.

### 3.3 Pairing Stochastic Assets with Stochastic Liabilities

`StochasticALMAnalysis` accepts `liabilities_cf` as either a single
`CashFlowStream` (broadcast across all paths, Phase 1 convention) or a
**list of length `n_paths`** paired position-wise with the asset paths
(Stage 5b convention). The pair construction is identical for funding
ratio and for fan-chart reporting; only the `liabilities_cf[i]`
selection differs.

Typical pairing pattern:

```python
liab_paths = simulate_liability_paths(
    fund=fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality, horizon_years=40, n_paths=200, seed=4242,
)

salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths,            # list, paired with asset paths
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    n_paths=200,
    horizon_years=40,
    seed=42,
)
salm.run()
```

The asset and liability RNG chains use **distinct master seeds** so
that the two stochastic axes are independent by construction — the
property the variance-attribution identity of §3.5 relies on.

### 3.4 `RiskAttribution` — Five Configurations

Lives in `awesome_actus_lib.pension.analysis.risk_attribution`.
Empirical variance decomposition of the liquidity-lens funding ratio
across the asset and mortality channels.

| Method                            | Asset side                | Liability side                                  |
| :-------------------------------- | :------------------------ | :---------------------------------------------- |
| `run_baseline`                    | deterministic (flat IR)   | deterministic                                   |
| `run_mortality_retirees_only`     | deterministic (flat IR)   | binomial decrement, retirees only               |
| `run_mortality_all_cohorts`       | deterministic (flat IR)   | binomial decrement, actives **and** retirees    |
| `run_assets_only`                 | stochastic (Hull-White)   | deterministic                                   |
| `run_full`                        | stochastic (Hull-White)   | binomial decrement, actives **and** retirees    |

Each method returns a length-`n_paths` array of funding-ratio
realisations at a chosen `as_of`. Per-channel caches are populated
lazily (`_ensure_det_assets`, `_ensure_stoch_assets`,
`_ensure_liab_paths`), so re-querying does not regenerate the
expensive asset-side ACTUS roundtrips.

### 3.5 Variance Decomposition

`attribution(as_of)` returns a dict with:

* `var_full`, `var_assets_only`, `var_mortality_all`,
  `var_mortality_retirees`;
* `var_active_mortality_empirical = var_mortality_all -
  var_mortality_retirees`;
* `active_mortality_share_of_total_mortality`;
* `additivity_residual = |var_full - (var_assets_only +
  var_mortality_all)| / var_full`.

The decomposition is justified by the independence of the asset and
mortality RNG chains: under independence, the first-order Taylor
expansion of $\operatorname{FR}(A, L)$ gives

$$
\operatorname{Var}\!\bigl[\operatorname{FR}\bigr] \;\approx\;
\operatorname{Var}_A + \operatorname{Var}_L,
$$

so the `additivity_residual` doubles as a sanity check: a large
residual signals either a violated independence assumption or a
materially nonlinear funding-ratio response in the regime sampled.

### 3.6 Scope Caveats Specific to 5b

* **Stage 5b uses the liquidity-lens funding ratio.** The Art. 44 BVV 2
  solvency Deckungsgrad of Phase 2 is, in the current implementation,
  **invariant under stochastic mortality**: `VK_{t_0}` depends only on
  $t_0$ cohort sizes and the deterministic $\ddot{a}_x$, not on the
  realised survival path. Path-dependent solvency variance from
  mortality would require forward projection of `VK_t` (Stage 5c).
* **No survivor benefits.** When an active cohort loses members to the
  binomial draw, the released AGH leaves the balance sheet without a
  corresponding Hinterlassenenrente or death-benefit payout. This is a
  small upward bias on funded ratios with high active-cohort mortality;
  it is recorded once at the module level and revisited in the written
  report.
* **Idiosyncratic mortality only.** Systemic longevity risk —
  stochastic evolution of the survival-probability risk factors in the
  Aro/Pennanen latent-factor sense — is not modelled here.

---

## 4. End-to-End Example (All Channels)

Example combining Phase 1, Phase 2 (equity), and Stage 5b
(stochastic liabilities) with paired pairing.

```python
import numpy as np
from awesome_actus_lib import PAM, Portfolio
from awesome_actus_lib.stochastic_rates import CurveCalibrator
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund,
    DynamicFundSimulator, Stage4Dynamics, EntryPolicy,
    StochasticALMAnalysis, StochasticDeckungsgradAnalysis,
    RiskAttribution, simulate_liability_paths,
    ek2001_2005,
)

# 1. Deterministic Stage-4 building blocks
policy = PensionPolicy()
fund = PensionFund(policy=policy, start_date="2025-01-01T00:00:00")
fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0, gender="unisex",
))
fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0, status="retired",
    annual_pension=31_380.0, gender="f",
))
entry_policy = EntryPolicy(entry_age=25, headcount=20, gross_salary=80_000.0)
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)

# 2. Asset portfolio (fixed + floater)
asset_portfolio = Portfolio([
    PAM(
        contractID="FIX_2035", statusDate="2025-01-01T00:00:00",
        currency="CHF", notionalPrincipal=200_000_000, nominalInterestRate=0.02,
        maturityDate="2035-12-31T00:00:00", dayCountConvention="30E360",
    ),
    PAM(
        contractID="FLT_2035", statusDate="2025-01-01T00:00:00",
        currency="CHF", notionalPrincipal=200_000_000, nominalInterestRate=0.0,
        maturityDate="2035-12-31T00:00:00", dayCountConvention="30E360",
        cycleAnchorDateOfRateReset="2025-01-01T00:00:00", cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset="IR_SCENARIO", rateSpread=0.0060,
    ),
])

# 3. Calibrate stochastic rates
calibrator = CurveCalibrator.from_market(
    times=np.array([1, 2, 5, 10], dtype=float),
    spot_rates=np.array([0.014, 0.015, 0.0176, 0.021], dtype=float),
)

# 4a. Stage 5b — stochastic liability paths (cheap, pure Python)
liab_paths = simulate_liability_paths(
    fund=fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=ek2001_2005(),
    horizon_years=40, n_paths=200, seed=4242,
)

# 4b. Stage 5a Phase 1 + Phase 2 — stochastic assets with equity sleeve,
#     paired with stochastic liabilities
salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths,                  # list of length n_paths
    calibrator=calibrator,
    technical_rate=policy.technical_rate,       # sticky i_tech
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": 0.05, "sigma": 0.15},
    n_paths=200, horizon_years=40, base_date="2025-01-01",
    seed=42,
)
salm.run()

# 5. Liquidity-lens distribution at t = 10y
print(salm.summary(as_of="2035-01-01"))

# 6. Solvency lens (Phase 2)
sdga = StochasticDeckungsgradAnalysis(
    salm=salm, fund=fund, mortality=ek2001_2005(),
    bond_book=400_000_000, cash_book=50_000_000,
    equity_sleeve_0=150_000_000,
)
print(sdga.summary(as_of="2035-01-01"))

# 7. Variance attribution (Stage 5b)
ra = RiskAttribution(
    asset_portfolio=asset_portfolio,
    fund=fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=ek2001_2005(),
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    n_paths=200, horizon_years=40, base_date="2025-01-01",
    seed_assets=42, seed_liabilities=4242,
)
print(ra.summary(as_of="2035-01-01"))
print(ra.attribution(as_of="2035-01-01"))
```

---

## 5. Stochastic-Rates Layer (Recap)

Stage 5 leans on `awesome_actus_lib.stochastic_rates`, developed in
the predecessor PA. Reused without modification:

* `CurveCalibrator.from_market(times, spot_rates)` — cubic-spline term
  structure used as initial yield curve.
* `create_model(name, seed, **params)` — short-rate model factory;
  curve-fitting models (`hull_white`, `ho_lee`) receive the calibrator
  automatically and reproduce the initial term structure.
* `GBMModel(S0, mu, sigma, seed)` — geometric Brownian motion (added
  for Stage 5a Phase 2; used as the equity driver).
* `simulation_to_reference_index(sim, base_date, market_code, path,
  freq)` — adapter that wraps a simulated path as an AAL
  `ReferenceIndex` consumable by ACTUS contracts.

---

## 6. What Stage 5 Still Does Not Do

* **Stochastic discounting (Stage 5c).** Both sides are discounted at
  the flat sticky `technical_rate`. A consistent mark-to-market
  valuation would propagate each path's short rate into the discount
  factor on both sides.
* **VK_t projection (Stage 5c).** `StochasticDeckungsgradAnalysis`
  holds `VK_{t_0}` constant; a forward-projected solvency DG would
  evolve `VK_t` (AGH roll-forward, retiree decrement, fresh
  retirements, $i_{\mathrm{tech}}$ revaluation).
* **Mark-to-market bonds on the solvency lens.** Phase 2 holds bonds
  and cash at book value; only equity is MTM. A consistent SST-style
  economic balance sheet would MTM bonds as well.
* **Asset rebalancing.** Static buy-and-hold portfolio. No
  reinvestment, no roll-forward of matured bonds, no allocation
  rebalancing.
* **Active pre-retirement mortality with benefits.** Stage 5b applies
  binomial decrement to active cohorts but does not pay
  Hinterlassenenrente / death-benefit lump sums on the released
  liability.
* **Systemic longevity risk.** Only idiosyncratic mortality is
  modelled; calendar-year mortality improvement and latent-factor
  longevity shocks are out of scope.
* **Correlation between rate and equity drivers.** GBM and Hull–White
  share no Brownian increment; empirical regime-dependent correlation
  is acknowledged as a future extension.
