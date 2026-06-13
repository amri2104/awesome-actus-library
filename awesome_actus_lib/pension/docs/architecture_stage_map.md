# Architecture Stage Map

Canonical mapping between the **chronological thesis stages** (Stage 1–6) and
the **functional code modules** of the pension extension. This document is the
bridge between the written thesis narrative and the source tree: every stage
states what was built, which public classes/functions implement it, and in
which module they live.

It also records the move from chronological simulator file names
(`Stage1_closed.py`, …) to functional ones (`closed_fund.py`, …), completed on
2026-06-13; see [§5](#5-functional-renaming-of-the-simulator-files-completed).
The public facade in `pension/__init__.py` was unaffected by that move, so all
stage→class mappings below are valid regardless of the file names.

---

## 1. Stage → Module overview

| Stage | What was built | Public classes / functions | Module (current) |
|---|---|---|---|
| **1** | Closed fund, deterministic crediting, annual events | `PensionPolicy`, `Cohort`, `PensionFund`, `ClosedFundSimulator` | `policy.py`, `cohort.py`, `fund.py`, `simulators/closed_fund.py`; event types in `events.py` |
| **2** | Open fund with deterministic new entrants | `OpenFundSimulator`, `EntryPolicy` | `simulators/open_fund.py`, `entry.py` |
| **3** | Deterministic mortality (EK 2001–2005, per sex) | `MortalityTable`, `ek2001_2005()` | `mortality.py` — **no simulator file** (see [§2](#2-why-there-is-no-stage-3-simulator)) |
| **4** | Dynamics: salary growth, falling UWS path (held-forward), threshold indexation; t0 funding ratio and retirement loss | `DynamicFundSimulator`, `Stage4Dynamics`, `FundingRatioAnalysis`, `RetirementLossAnalysis`, `annuity_due`, `technical_uws`, `ALMAnalysis` | `simulators/dynamic_fund.py`, `analysis/funding_ratio.py`, `analysis/conversion.py`, `analysis/alm.py` |
| **5a** | Stochastic asset side: Hull-White / Vasicek / CIR / Ho-Lee rate paths drive ACTUS contracts; GBM equity sleeve | `StochasticALMAnalysis`, `CurveCalibrator` | `analysis/stochastic_alm.py`, `stochastic_rates/` |
| **5b** | Stochastic liabilities: binomial cohort mortality per path; variance decomposition | `simulate_liability_paths`, `LiabilityPath`, `RiskAttribution` | `simulators/liability_paths.py`, `analysis/risk_attribution.py` |
| **5c** | Forward solvency funding ratio as a (time × paths) surface: `DG_t,i = V_t,i / VK_t,i`, both sides path-dependent | `StochasticFundingRatioAnalysis` | `analysis/stochastic_funding_ratio.py` |
| **6** | Going-concern funding ratio: reinvestment, fixed-mix rebalancing, allocation shift, asset-only recovery measure | `GoingConcernFundingRatioAnalysis`, `RebalancingPolicy`, `SanierungsPolicy` | `analysis/going_concern.py` |

> Per-stage design notes live in the sibling docs `stage1.md` … `stage5c.md`
> and `stage6_going_concern_spec.md`. Those documents still use the old
> `StageN_*.py` simulator file names (left untouched by the code-only rename
> scope); the class names they cite are unchanged.

---

## 2. Why there is no "Stage 3" simulator

The simulator files exist for stages **1, 2, 4** only; stage 3 has none. This
is intentional and worth stating explicitly in the thesis:

> Stage 3 does not appear as its own simulator because mortality is
> architecturally a **risk factor** (`MortalityTable`) that drives event
> generation — not a separate fund simulator. Only the stages that introduce a
> distinct simulation step (closed → open → dynamic) get a simulator file.

This mirrors the ACTUS separation of **contract terms** from **risk factors**:
non-market contingencies (behaviour, biometrics) belong in the risk-factor
layer, not in contract terms. Mortality is therefore a `MortalityTable` risk
factor — deterministic in Stage 3, binomial-per-path in Stage 5b — consumed by
the simulators, never pressed into contract terms.

---

## 3. Layered architecture

```
   ASSET SIDE (ACTUS-native)            LIABILITY SIDE (BVG domain engine)
   PAM bonds, CSH cash                  Cohort · PensionPolicy · simulators 1–4
   → ACTUS IP / events                  → rule-based events (UWS, min. rate, …)
            \                                    /
             \    both emit into the same       /
              →→→   CashFlowStream schema   ←←←
                  (time · type · payoff · contractId)
                              |
              SHARED ANALYSIS LAYER (AAL analysis reused as-is)
       Liquidity · Value · Income · ALM · FundingRatio · RiskAttribution
                              |
              STOCHASTICS (generate-then-pair, separate seeds)
        asset paths (seed 42)  ×  liability paths (seed 4242) → index i
                              |
                  Stage 6: sequential annual roll
            (DG_{t-1} in-loop → rebalance / Sanierung as feedback)
```

Deliberate thesis boundary: the asset side is ACTUS-native; the Swiss BVG
liability side is a **documented domain engine** attached to the shared
cashflow and ALM layer. The AAL core stayed functionally unchanged and the
extension is additive — both sides converge on the `CashFlowStream` object
(`cohort_id` acts as `contractID`), which the existing analysis layer consumes
as-is. The only edits to AAL core are three backward-compatible robustness
fixes (explicit `format="ISO8601"` date parsing in `analysis/liquidity.py`,
`analysis/value.py` and `models/portfolioPlot.py`, plus one cosmetic `print`
change) — no public signature or numerical behaviour changed.

---

## 4. Simulator inheritance chain

The three simulators form a linear inheritance chain, each extending the
previous one; `liability_paths` builds on the dynamic simulator:

```
ClosedFundSimulator                 (Stage 1)
    └─ OpenFundSimulator            (Stage 2)  extends Closed
          └─ DynamicFundSimulator   (Stage 4)  extends Open
                 ⇧
   simulate_liability_paths / LiabilityPath   (Stage 5b)  builds on Dynamic
```

The chain is the **only** place where simulator modules import each other by
file path. Internal coupling (current file names):

- `open_fund.py`        → `from .closed_fund import ClosedFundSimulator`
- `dynamic_fund.py`     → `from .open_fund import OpenFundSimulator`
- `liability_paths.py`  → `from .dynamic_fund import DynamicFundSimulator, Stage4Dynamics`
- `simulators/__init__.py` re-exports all of the above by file path.

---

## 5. Functional renaming of the simulator files (completed)

The old chronological file names carried a confusing gap (1, 2, 4 — no 3) and
did not read functionally. On 2026-06-13 the three simulator files were renamed
to functional names via `git mv` (history preserved); **mathematics and
behaviour are byte-identical** — only file names and internal import paths
changed, verified by the offline regression check (16 asserts) and a facade
smoke import.

| Stage | Class | File (before) | File (now) |
|---|---|---|---|
| 1 | `ClosedFundSimulator` | `simulators/Stage1_closed.py` | `simulators/closed_fund.py` |
| 2 | `OpenFundSimulator` | `simulators/Stage2_open.py` | `simulators/open_fund.py` |
| 4 | `DynamicFundSimulator`, `Stage4Dynamics` | `simulators/Stage4_dynamic.py` | `simulators/dynamic_fund.py` |
| 5b | `simulate_liability_paths`, `LiabilityPath` | `simulators/liability_paths.py` | *(unchanged — already functional)* |

Change surface: **3 file renames + 4 import sites** (`simulators/__init__.py`,
`open_fund.py`, `dynamic_fund.py`, `liability_paths.py`). The facade
`pension/__init__.py` imports from the `.simulators` **package**, not from file
paths, so it did **not** change — which is why all external consumers
(examples, tests) keep working unchanged.

The `_fund` suffix was chosen over bare `closed.py` / `open.py` / `dynamic.py`
because it matches the `*FundSimulator` class names one-to-one and avoids
reusing the `open` builtin name as a module name.

---

## 6. The facade (`pension/__init__.py`)

`pension/__init__.py` re-exports every public name via `__all__` (23 symbols:
the policy/cohort/fund primitives, the four simulators + `Stage4Dynamics`, the
mortality helpers, and the full analysis layer through Stage 6). All example
quickstarts and the Stage 6 test import via this facade
(`from awesome_actus_lib.pension import …`); none import a simulator by its
file path. The facade is therefore the **stable contract** that made the
file-level rename in [§5](#5-functional-renaming-of-the-simulator-files-completed)
transparent to every consumer.
