---
editor_options:
  markdown:
    wrap: 72
---

# Stage 5 — Stochastic Pension ALM Extensions

Stage 5 extends the deterministic Stage 1-4 pension-fund pipeline with
stochastic asset, liability and going-concern analyses. The implementation is
split into four increments:

- **Stage 5a**: stochastic asset-side ALM with Hull-White/Vasicek/CIR/Ho-Lee
  rate paths, ACTUS cashflows and an optional GBM equity sleeve.
- **Stage 5b**: stochastic liability paths through binomial cohort mortality
  and risk-attribution experiments.
- **Stage 5c**: forward stochastic solvency funding ratio where both assets
  and pension capital `VK_t,i` are evaluated per path over time.
- **Stage 6**: going-concern funding-ratio analysis with reinvestment,
  fixed-mix rebalancing, allocation shifts and asset-only recovery measures.

Read [stage1.md](stage1.md) to [stage4.md](stage4.md) first for the
deterministic pension engine: `PensionPolicy`, `Cohort`, `PensionFund`,
`Stage4Dynamics`, `FundingRatioAnalysis` and `RetirementLossAnalysis`.

## 1. Architecture Boundary

ACTUS/AAL is used natively on the asset side: bond and cash contracts generate
ACTUS event streams, and stochastic risk factors can drive those contracts.
The Swiss BVG liability side is not represented as fully ACTUS-native contract
types. It is implemented as a separate Pension/BVG domain engine and then
attached to the shared cashflow and ALM analysis layer.

This is a deliberate thesis boundary: ACTUS carries the asset-contract and
cashflow infrastructure; the pension-specific liability rules live in a
documented external domain engine.

## 2. Stage 5a — Stochastic Asset-Side ALM

Stage 5a introduces `StochasticALMAnalysis` in
`awesome_actus_lib.pension.analysis.stochastic_alm`.

Main behaviour:

- simulates short-rate paths from `awesome_actus_lib.stochastic_rates`;
- converts each path into ACTUS risk factors;
- generates one asset `CashFlowStream` per path;
- optionally adds a GBM equity sleeve for multi-asset solvency experiments;
- keeps liability cashflows deterministic unless later stages provide
  path-specific liabilities.

Representative examples:

- `Stage5a_floaters_quickstart.py`
- `Stage5a_Phase2_multi_asset_quickstart.py`

## 3. Stage 5b — Stochastic Liabilities and Risk Attribution

Stage 5b adds stochastic liability paths through
`simulate_liability_paths(...)`. Each path has its own binomial mortality
realisation and can include a headcount trace. This enables mortality-risk
experiments without changing the deterministic Stage 4 API.

`RiskAttribution` compares configurations such as deterministic/stochastic
assets and deterministic/stochastic mortality. The goal is not a full economic
capital model; it is an interpretable decomposition of the risk channels used
by the thesis examples.

Representative example:

- `Stage5b_mortality_quickstart.py`

## 4. Stage 5c — Forward Stochastic Solvency Funding Ratio

Stage 5c closes the gap where earlier stochastic runs changed assets or
cashflows, but the regulatory pension capital was still frozen at `t0`.

`StochasticFundingRatioAnalysis` can consume `LiabilityPath` objects and
evaluate a path-dependent surface:

```text
DG_t,i = V_t,i / VK_t,i
```

The same survivor path drives both pension outflows in the asset roll and
survivor counts in `VK_t,i`. This coupling is the critical correctness point:
a high-mortality path must reduce pension outflows and pension capital in the
same path.

Representative example:

- `Stage5c_forward_funding_ratio_quickstart.py`

Detailed design notes live in [stage5c.md](stage5c.md).

## 5. Stage 6 — Going-Concern Funding Ratio Analysis

Stage 6 implements Goal 2: allocation changes between asset classes can be
simulated in the risk analysis. The implementation lives in
`awesome_actus_lib.pension.analysis.going_concern`.

Public classes:

- `GoingConcernFundingRatioAnalysis`
- `RebalancingPolicy`
- `SanierungsPolicy`

The extension is opt-in. With `rebalancing=None` and `sanierung=None`, the
analysis delegates to Stage 5c and is checked as a regression reference.

### 5.1 Scenario Set

The current quickstart compares four strategy scenarios:

| Scenario | Meaning |
|---|---|
| A' | Buy-and-hold inside the same going-concern engine; no rebalancing date occurs within the horizon. |
| B | Fixed-mix 40% BONDS / 60% EQUITY with annual rebalancing. |
| C | Same as B, then allocation shift to 60% BONDS / 40% EQUITY from year 5. |
| D | Same as B plus `SanierungsPolicy(trigger_dg=1.00, sb_factor=0.5)`. |

Legacy Stage 5c is not used as a strategy scenario. It is kept only as a
regression reference: `rebalancing=None, sanierung=None` must remain identical
to Stage 5c.

### 5.2 Mechanics

The going-concern roll uses two book-value buckets:

- `BONDS`: fixed income, including the cash book;
- `EQUITY`: equity sleeve with deterministic book drift and optional
  mean-preserving lognormal annual shocks via `equity_sigma` and
  `equity_seed`.

Original bond/cash contracts still contribute ACTUS `IP` events. Additional
money allocated into the BONDS bucket does not generate new ACTUS contracts;
`bond_yield` is a documented book-yield approximation on the incremental BONDS
balance above the initial book value.

Net liability cashflows are reinvested or withdrawn proportionally to current
bucket weights. `RISK_CONTRIB` remains excluded from the asset roll.

### 5.3 Recovery Measures

`SanierungsPolicy` represents a minimal asset-only recovery measure:

```text
SB_t = sb_factor * SAV_CONTRIB_t
```

It fires only when `DG_{t-1} < trigger_dg`, with optional hysteresis and a
`max_years` cap. Sanierungsbeiträge are booked as pure asset inflows and
logged separately as `SANIERUNG_SB` through `gc_events(...)`. They are not
written into the liability cashflow stream.

Measures that change accrued savings or liability-event generation, such as
Minder-/Nullverzinsung, remain out of scope because they would require
regenerating liability paths conditional on realised funding ratios.

Representative example:

- `Stage6_going_concern_quickstart.py`

Detailed implementation notes live in
[stage6_going_concern_spec.md](stage6_going_concern_spec.md).

## 6. Verification

The project currently uses example scripts with hard inline assertions plus a
small offline pytest smoke for Stage 6.

Useful commands from the repository root:

```bash
PYTHONPATH=. MPLBACKEND=Agg python3 awesome_actus_lib/pension/examples/Stage5c_forward_funding_ratio_quickstart.py
PYTHONPATH=. MPLBACKEND=Agg python3 awesome_actus_lib/pension/examples/Stage6_going_concern_quickstart.py
PYTHONPATH=. python3 -m pytest awesome_actus_lib/pension/tests/test_stage6_going_concern.py
```

The Stage 6 quickstart asserts:

- `rebalancing=None, sanierung=None` is identical to Stage 5c;
- `DG_t0` is calibrated to 107.6%;
- A' never rebalances;
- B rebalances back to target weights;
- C shifts to 60/40 from year 5;
- D records Sanierungsbeiträge only when `DG_{t-1} < 100%`;
- A'/B/C have no Sanierungsbeiträge;
- D is at least B at the horizon under the same pathwise shocks.

## 7. Remaining Limitations

- No full BVG obli/ueberobli split.
- No survivor benefits.
- No systemic longevity trend shock.
- No market-consistent stochastic discounting.
- No calibrated transaction-cost model.
- No liability-feedback measures that change accrued savings; these require a
  different generate-and-regenerate architecture.
