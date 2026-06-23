---
editor_options: 
  markdown: 
    wrap: 72
---

# Stage 5c — Forward Stochastic Solvency funding ratio (Design Spec)

Stage 5c closes the structural gap left open by Stages 5a/5b: the
regulatory Art-44-BVV2 solvency funding ratio is currently **invariant**
under stochastic mortality, because `StochasticFundingRatioAnalysis`
freezes the pension capital (VK) at `t0`. Stage 5c makes both the
obligation (VK) and the asset stock **path-dependent over time**, so the
regulatory funding ratio finally responds to the Stage-5b binomial
mortality and the Stage-5a asset-side stochastics.

This is a **design spec written before implementation** (Bauplan
zuerst). Read [stage5.md](stage5.md) first for the stochastic asset/
liability engines (5a/5b), and [stage4.md](stage4.md) for
`FundingRatioAnalysis` and `annuity_due`.

This document covers only the deltas.

------------------------------------------------------------------------

## 1. The gap Stage 5c closes

Stage 5b builds stochastic mortality: `simulate_liability_paths`
produces `n_paths` independent liability streams, each with its own
binomial survival realisation. Yet the **regulatory** funding ratio does
not move across those paths.

The reason is in `StochasticFundingRatioAnalysis`: VK is computed exactly
once in `__init__` (`self._vk_t0 = FundingRatioAnalysis(...)
.pension_capital_t0()`) and used as a constant denominator for every
`as_of` date and every path. `VK_t0` depends only on t0 cohort sizes and
the **deterministic** annuity `ä_x` — no survivor-path information enters.
The numerator `V_i(d) = bond_book + cash_book + equity_sleeve_0 ·
S_i(d)/S0` varies only through the equity sleeve. Mortality therefore has
**zero** effect on the solvency funding ratio.

Two lenses, restated:

| Lens | Where | Numerator | Denominator | Reacts to 5b mortality? |
|------|-------|-----------|-------------|--------------------------|
| Liquidity FR | `ALMAnalysis.funding_ratio` | NPV(asset CF) @ flat `i_tech` | \|NPV(liab net CF)\| @ flat `i_tech` | **Yes** (payouts scale with survivors) — but diagnostic only |
| Solvency DG | `StochasticFundingRatioAnalysis` (5a/5b) | book assets + equity | **frozen** `VK_t0` | **No** — the gap |

Stage 5c turns the solvency lens into a genuine `(time × paths)` surface
so the regulatory headline KPI shows longevity risk.

------------------------------------------------------------------------

## 2. Scope

**In scope (5c core):**
- Forward per-path `VK_t,i` (obligation rolled forward, survivors per path).
- Forward book-value asset roll `V_t,i`.
- Coupling: the **same** survivor path drives both the pension-outflow
  leg of the assets and the survivor counts in VK.
- Output: forward solvency funding ratio distribution / fan chart over the
  horizon.

**Out of scope (explicit, parked):**
- **Dynamic technical rate / revaluation shock** (FRP-4-floor stepped
  `i_tech` reacting to the Hull-White paths). Regulatorily realistic, no
  P/Q issue — candidate for a later extension.
- **Recovery measures.** Stage 5c itself does not implement DG-feedback.
  Stage 6 adds the architecture-compatible subset: asset-only measures
  such as Sanierungsbeiträge that do not create or modify liability
  events. Measures that change accrued savings or liability-event
  generation, such as Minder-/Nullverzinsung, remain out of scope because
  the liability path would need to be regenerated conditional on the
  realised asset-driven DG.
- **Market-consistent stochastic discounting** of cashflows at the
  path-specific short rate. Different valuation object, introduces a P/Q
  measure-consistency problem; not the regulatory question.
- Survivor benefits (Witwen-/Witwer-/Waisenrente); systemic longevity
  (Aro/Pennanen `v_t`).

------------------------------------------------------------------------

## 3. Locked design decisions

1. **Asset roll = book value.** Held-to-maturity bonds and cash at book
   value — Art-44-conformant, cheap, **no per-year ACTUS round-trips**.
   (Equity sleeve is the only mark-to-market component, via GBM, already
   present in the class.)
2. **`VK_t` mortality = per-path** from the Stage-5b survivor trace, not
   the deterministic expected survival curve. This is the entire point of
   the stage — a deterministic VK roll would re-open the gap.
3. **`ä_x` discounting stays sticky, deterministic `i_tech`.** Only the
   survivor headcounts are stochastic. This keeps Stage 5c fully inside
   the regulatory frame and avoids any P/Q measure question.

------------------------------------------------------------------------

## 4. Enabling change — expose the survivor trace

`simulate_liability_paths` currently returns only the `CashFlowStream`
per path. The per-year, per-cohort headcount trajectory is computed by
`DynamicFundSimulator` (`cohort_headcount_log`) but discarded.

Introduce a lightweight `LiabilityPath` object bundling:
- `cashflows`: the existing `CashFlowStream`,
- `headcount_trace`: per-year, per-cohort surviving headcounts (active
  and retired), plus the cohort metadata needed to value VK (age, gender,
  accrued AGH, annual pension).

Surface it via a backward-compatible flag:

```python
liab_paths = simulate_liability_paths(
    ..., return_headcount_log=True,   # default False -> unchanged behaviour
)   # -> List[LiabilityPath]
```

Default `False` keeps every existing Stage-5b call site byte-identical.

------------------------------------------------------------------------

## 5. The math

Per path `i`, per valuation year `t`:

```
VK_t,i =  Σ_active  AGH_t,i(c) · n_active_i(c, t)
        + Σ_retired pension(c) · n_retired_i(c, t) · ä_{x(c,t)}

V_t,i  =  bond_book + cash_book                               # constant t0 book
        + Σ (bond interest income up to t,  path i)           # IP events only
        + Σ (contributions − pension payouts − admin up to t, path i)
        + equity_value_i(t)                                   # GBM, existing

DG_t,i =  V_t,i / VK_t,i
```

where the survivor counts `n_*_i(c, t)` come from
`LiabilityPath[i].headcount_trace`, `AGH_t,i(c)` is the per-cohort
end-of-year accrued book value carried by the same trace, and
`ä_{x(c,t)} = annuity_due(age(c,t), gender(c), technical_rate, mortality,
terminal_age)` at the sticky `i_tech`.

Implementation note on the asset roll: at book value the bond principal
sits inside `bond_book` from t0 onwards and converts to cash at maturity
without changing the total. The roll therefore accumulates **only**
interest income (ACTUS event type `IP`) — principal redemption (`MD`)
and initial principal exchange (`IED`) are excluded to avoid
double-counting against the constant `bond_book + cash_book`.

------------------------------------------------------------------------

## 6. Coupling invariant (the critical correctness point)

The pension-payout leg in `V_t,i` and the survivor counts in `VK_t,i`
must read the **same** `LiabilityPath[i]`. A path where pensioners die
faster shows **simultaneously** a smaller obligation (denominator) and
fewer pension outflows leaving the assets (numerator).

The rate path `i` (asset side, `seed_assets`) and the survivor path `i`
(liability side, `seed_liabilities`) stay **independent risk sources**,
paired only by index `i` — interest rates and idiosyncratic mortality are
genuinely independent here (systemic longevity coupling is explicitly out
of scope). **Never merge the seeds.** Two independent survivor draws for
numerator vs denominator is a bug, not a modelling choice.

------------------------------------------------------------------------

## 7. Where it docks in the code

- **Extend** `StochasticFundingRatioAnalysis` (analysis/
  stochastic_funding_ratio.py): constructor gains an optional
  `liab_paths: List[LiabilityPath]` keyword. When supplied, the frozen
  scalar `self._vk_t0` is replaced by a per-path, per-date `VK_t,i`
  computation; when omitted, the legacy frozen-denominator path
  (Stage 5a Phase 2 behaviour) is preserved verbatim.
- `funding_ratio_distribution(as_of)`, `summary(as_of)`,
  `fan_chart_data(...)`: now evaluate over a `(time × paths)` surface
  instead of dividing by the frozen denominator. APIs stay the same.
- **Untouched:** `StochasticALMAnalysis` Hull-White/GBM stack;
  `simulate_liability_paths` binomial mechanic (only an extended return);
  `ALMAnalysis` (liquidity lens); `FundingRatioAnalysis` (t0 snapshot
  remains as the deterministic reference); Stage 1-4 simulator files
  (`Stage1_closed.py`, `Stage2_open.py`, `Stage4_dynamic.py`) — the
  headcount-trace enrichment reconstructs cohort metadata externally in
  `simulate_liability_paths` rather than modifying the simulators.

A `t0`-only mode (current behaviour) remains available for backward
compatibility (`liab_paths=None`); the forward path is additive. The new
`LiabilityPath` symbol is exported from `awesome_actus_lib.pension`.

------------------------------------------------------------------------

## 8. Verification (example script + inline asserts)

Following the project convention (example scripts with inline asserts,
no pytest):

- **(a) t0 collapse.** Forward DG at `as_of = t0` equals the
  deterministic `FundingRatioAnalysis.funding_ratio_t0()` within float
  tolerance — at t0 the forward surface must reduce to the snapshot.
- **(b) Gap closed (the proof).** With **deterministic** assets and
  **stochastic** mortality, the forward DG distribution at `t > t0` must
  have **variance > 0**. Before Stage 5c this variance is exactly `0`.
  This assert is the formal evidence that the regulatory DG now responds
  to longevity risk.
- **(c) Coupling.** On a single high-mortality path, VK is lower **and**
  cumulative pension outflows are smaller than on a low-mortality path —
  cross-checked against the shared `headcount_trace`.

------------------------------------------------------------------------

## 9. What Stage 5c still does not do

- **Dynamic technical rate / revaluation shock** (stepped `i_tech` via
  FRP-4 floor on low-rate paths) — future extension.
- **Recovery measures.** Asset-only Sanierungsbeiträge are implemented in
  Stage 6 as a going-concern extension. Liability-feedback measures
  such as Minder-/Nullverzinsung still require a generate-and-regenerate
  architecture and remain Outlook.
- **Market-consistent stochastic discounting** — separate valuation
  lens; P/Q measure-consistency issue.
- **Survivor benefits** (Witwen-/Witwer-/Waisenrente).
- **Systemic longevity risk** (stochastic Aro/Pennanen `v_t`); Stage 5b
  models idiosyncratic mortality only.
