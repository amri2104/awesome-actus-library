"""Stage5b_mortality_quickstart.py

Stage 5b — stochastic LIABILITY side (binomial cohort transitions) on top of Stage 5a:
  - Stage-4 deterministic liability backbone + Stage 5a mixed asset portfolio
  - RiskAttribution sweep over five configs:
      baseline / mortality retirees / mortality all / assets only / full
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.stochastic_rates import CurveCalibrator
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    DynamicFundSimulator,
    Stage4Dynamics,
    EntryPolicy,
    FundingRatioAnalysis,
    RiskAttribution,
    ek2001_2005,
    simulate_liability_paths,
)


HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("STAGE5B_N_PATHS", "200"))
SEED_ASSETS = 42
SEED_LIAB = 4242


print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 5b — stochastic liability side (mortality)")
print("=" * 78)

# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side) — same fund as Stage 4/5a
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities (same fund as Stage 4/5a)")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1960",
    birth_year=1960,
    headcount=90,
    gross_salary=120_000.0,
    accrued_savings=600_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1963",
    birth_year=1963,
    headcount=100,
    gross_salary=115_000.0,
    accrued_savings=520_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1966",
    birth_year=1966,
    headcount=110,
    gross_salary=115_000.0,
    accrued_savings=450_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970",
    birth_year=1970,
    headcount=150,
    gross_salary=110_000.0,
    accrued_savings=350_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1972",
    birth_year=1972,
    headcount=130,
    gross_salary=108_000.0,
    accrued_savings=320_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1975",
    birth_year=1975,
    headcount=140,
    gross_salary=105_000.0,
    accrued_savings=270_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1978",
    birth_year=1978,
    headcount=140,
    gross_salary=100_000.0,
    accrued_savings=220_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990",
    birth_year=1990,
    headcount=200,
    gross_salary=90_000.0,
    accrued_savings=80_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955",
    birth_year=1955,
    headcount=80,
    gross_salary=0.0,
    accrued_savings=0.0,
    status="retired",
    annual_pension=31_380.0,
    gender="f",
))

entry_policy = EntryPolicy(
    entry_age=25,
    headcount=20,
    gross_salary=80_000.0,
    gender="unisex",
)
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)
# Generational EK 2001-2005 (1.25%/yr improvement from 2003): lifts ä_x to a
# realistic level for solvency/funding-ratio valuation (PC values retirees via
# ä_x). Harmless for liquidity-lens analyses — the headcount decrement is
# unaffected (no cal_year is passed there).
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)

print(f"  Active cohorts: 8 birth-year buckets, retirement age 65")
print(f"  Retired cohort: RETIRED_1955, 80 members, pension CHF 31,380/year")
print(f"  EntryPolicy: 20 entrants/year, age 25, salary CHF 80,000")
print(f"  Stage4Dynamics: salary_growth={dynamics.salary_growth:.2%}, "
      f"CR path={dynamics.conversion_rate_path}, "
      f"threshold_index_period={dynamics.threshold_index_period}y")
print(f"  Mortality:    EK 2001-2005 generational (improvement 1.25%/yr from 2003)")
print(f"  Horizon:      {HORIZON_YEARS} years")


# =============================================================================
# 1B. PORTFOLIO DEFINITION (asset side) — mixed fixed + floating ladder
# =============================================================================
print("\n[1B] Portfolio Definition — assets (mixed fixed + floating ladder)")
print("-" * 40)

_INTEREST_ANCHOR = "2026-01-01T00:00:00"

asset_portfolio = Portfolio([
    PAM(
        contractID="FIX_2035",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=160_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2035-12-31T00:00:00",
        nominalInterestRate=0.020,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="FIX_2045",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=100_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2045-12-31T00:00:00",
        nominalInterestRate=0.025,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="FLT_2035",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=100_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2035-12-31T00:00:00",
        nominalInterestRate=0.0,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE,
        cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset=MARKET_CODE,
        rateSpread=0.0060,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0",
        counterpartyID="Bank-01",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="FLT_2050",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=40_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2050-12-31T00:00:00",
        nominalInterestRate=0.0,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE,
        cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset=MARKET_CODE,
        rateSpread=0.0080,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0",
        counterpartyID="Bank-01",
        creatorID="PK_ALM",
    ),
])

print(f"  Asset 1: CHF 160,000,000 fixed PAM, 2.0% coupon, matures 2035")
print(f"  Asset 2: CHF 100,000,000 fixed PAM, 2.5% coupon, matures 2045")
print(f"  Asset 3: CHF 100,000,000 floating PAM, +60bp spread, matures 2035")
print(f"  Asset 4: CHF  40,000,000 floating PAM, +80bp spread, matures 2050")
print(f"  Total notional: CHF 400,000,000  (65% fixed / 35% floating)")


# =============================================================================
# 2. CURVE CALIBRATION (CHF-like base curve, same as Stage 5a)
# =============================================================================
print("\n[2] Curve calibration")
print("-" * 40)

curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)
print(f"  Base curve: {curve_rates[0]:.2%} (1Y) -> {curve_rates[-1]:.2%} (10Y), "
      f"flat-forward beyond 10Y")


# =============================================================================
# 3. SOLVENCY FUNDING RATIO @ t0 (Art. 44 BVV2) — INVARIANT under stoch. mort.
# =============================================================================
print("\n[3] Solvency funding ratio @ t0 (Art. 44 BVV2 — invariant under stoch. mortality)")
print("-" * 40)

vv = sum(c.terms["notionalPrincipal"].value for c in asset_portfolio.contracts)
fra = FundingRatioAnalysis(
    fund=liability_fund,
    mortality=mortality,
    pension_assets=vv,
)
print(f"  Pension assets:       CHF {vv:>15,.2f}")
print(f"  Pension capital_t0:   CHF {fra.pension_capital_t0():>15,.2f}")
print(f"  Funding ratio_t0:                 {fra.funding_ratio_t0():>15.2%}")
print("  (Same number under deterministic and stochastic mortality —")
print("   PC_t0 uses only t0 cohort sizes + deterministic annuity ä_x.)")


# =============================================================================
# 4. RISK ATTRIBUTION — five-config sweep
# =============================================================================
print("\n[4] Risk Attribution — five-config sweep")
print("-" * 40)

ra = RiskAttribution(
    asset_portfolio=asset_portfolio,
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=dynamics,
    mortality=mortality,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=BASE_DATE,
    market_code=MARKET_CODE,
    seed_assets=SEED_ASSETS,
    seed_liabilities=SEED_LIAB,
)

attr = ra.attribution(as_of=BASE_DATE)
summary_df = ra.summary(as_of=BASE_DATE)
print("\n  Funding ratio @ t0 across configurations:")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

print("\n  Variance decomposition @ t0:")
print(f"    Var(full)                       = {attr['var_full']:.6e}")
print(f"    Var(assets_only)                = {attr['var_assets_only']:.6e}")
print(f"    Var(mortality_all_cohorts)      = {attr['var_mortality_all']:.6e}")
print(f"    Var(mortality_retirees_only)    = {attr['var_mortality_retirees']:.6e}")
print(f"    Var(active_mortality) empirical = {attr['var_active_mortality_empirical']:.6e}")
print(f"    Active-mortality share of total mortality var: "
      f"{attr['active_mortality_share_of_total_mortality']:.4%}")
print(f"    |Var(full) - [Var(assets) + Var(mort_all)]| / Var(full): "
      f"{attr['additivity_residual']:.4%}")


# =============================================================================
# VERIFY — inline asserts (a)-(e)
# =============================================================================
print("\n[VERIFY] Inline asserts")
print("-" * 40)

# (a) Determinism collapse: stochastic_mortality=False → two runs identical
sim_det_1 = DynamicFundSimulator(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
)
sim_det_2 = DynamicFundSimulator(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
)
cf_det_1 = sim_det_1.run(horizon_years=HORIZON_YEARS)
cf_det_2 = sim_det_2.run(horizon_years=HORIZON_YEARS)
events_a = cf_det_1.events_df.sort_values(["contractId", "time", "type"]).reset_index(drop=True)
events_b = cf_det_2.events_df.sort_values(["contractId", "time", "type"]).reset_index(drop=True)
pd.testing.assert_frame_equal(events_a, events_b)
print(f"  (a) determinism collapse OK ({len(events_a)} events identical).")

# (b) Falsification of mortality wiring: σ_IR=0, stochastic_mortality=True →
#     liquidity-DG distribution has std > 0. If not, mortality is not wired in.
liab_paths_b = simulate_liability_paths(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS, n_paths=min(N_PATHS, 50),
    seed=SEED_LIAB, mortality_filter="all",
)
from awesome_actus_lib.pension.analysis.alm import ALMAnalysis
ra._ensure_det_assets()
det_assets = ra._det_asset_cfs
fr_dist_b = np.array([
    ALMAnalysis(
        assets_cf=det_assets, liabilities_cf=lp, flat_rate=policy.technical_rate,
    ).funding_ratio(as_of=BASE_DATE)
    for lp in liab_paths_b
])
std_b = float(np.std(fr_dist_b, ddof=1))
assert std_b > 1e-6, \
    f"(b) σ_IR=0 + stoch mort produced zero std: {std_b:.2e} — mortality not wired"
print(f"  (b) mortality wired in (σ_IR=0 → std(FR) = {std_b:.4%}) OK.")

# (c) LLN diversification (Maffra Fig. 11): scale cohorts ×10, pensions ÷10
#     (PC invariant) → mortality variance must shrink by ~factor 10.
scaled_fund = PensionFund(policy=policy, start_date=START_DATE)
for c in liability_fund.cohorts:
    scaled_fund.add_cohort(Cohort(
        cohort_id=c.cohort_id,
        birth_year=c.birth_year,
        headcount=c.headcount * 10,
        gross_salary=c.gross_salary,
        accrued_savings=c.accrued_savings / 10.0,
        status=c.status,
        annual_pension=(c.annual_pension / 10.0) if c.annual_pension is not None else None,
        gender=c.gender,
    ))
scaled_entry = EntryPolicy(
    entry_age=entry_policy.entry_age,
    headcount=entry_policy.headcount * 10,
    gross_salary=entry_policy.gross_salary,
    gender=entry_policy.gender,
)
n_lln = min(N_PATHS, 100)
liab_paths_lln = simulate_liability_paths(
    fund=scaled_fund, entry_policy=scaled_entry, dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS, n_paths=n_lln,
    seed=SEED_LIAB, mortality_filter="all",
)
# Build deterministic asset CFS at scaled notionals (×10) to keep FR invariant.
scaled_assets = Portfolio([
    PAM(
        contractID=c.terms["contractID"].value,
        statusDate=c.terms["statusDate"].value,
        contractDealDate=c.terms["contractDealDate"].value,
        currency=c.terms["currency"].value,
        notionalPrincipal=c.terms["notionalPrincipal"].value * 10,
        initialExchangeDate=c.terms["initialExchangeDate"].value,
        maturityDate=c.terms["maturityDate"].value,
        nominalInterestRate=c.terms["nominalInterestRate"].value,
        dayCountConvention=c.terms["dayCountConvention"].value,
        contractRole=c.terms["contractRole"].value,
        **(
            {"cycleOfInterestPayment": c.terms["cycleOfInterestPayment"].value}
            if "cycleOfInterestPayment" in c.terms else {}
        ),
        **(
            {
                "cycleAnchorDateOfRateReset": c.terms["cycleAnchorDateOfRateReset"].value,
                "cycleOfRateReset": c.terms["cycleOfRateReset"].value,
                "marketObjectCodeOfRateReset": c.terms["marketObjectCodeOfRateReset"].value,
                "rateSpread": c.terms["rateSpread"].value,
                "cycleAnchorDateOfInterestPayment": c.terms["cycleAnchorDateOfInterestPayment"].value,
            } if "marketObjectCodeOfRateReset" in c.terms else {}
        ),
        counterpartyID=c.terms["counterpartyID"].value,
        creatorID=c.terms["creatorID"].value,
    )
    for c in asset_portfolio.contracts
])
flat_df_lln = pd.DataFrame({
    "date": [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS + 1)],
    "value": [policy.technical_rate] * (HORIZON_YEARS + 1),
})
from awesome_actus_lib.models.RiskFactor import ReferenceIndex as _RF
rf_lln = _RF(marketObjectCode=MARKET_CODE, source=flat_df_lln)
service_lln = PublicActusService()
det_assets_lln = service_lln.generateEvents(scaled_assets, riskFactors=[rf_lln])
fr_dist_lln = np.array([
    ALMAnalysis(
        assets_cf=det_assets_lln, liabilities_cf=lp, flat_rate=policy.technical_rate,
    ).funding_ratio(as_of=BASE_DATE)
    for lp in liab_paths_lln
])
# Reference variance at original scale, using same number of paths for fairness.
liab_paths_ref = simulate_liability_paths(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS, n_paths=n_lln,
    seed=SEED_LIAB, mortality_filter="all",
)
fr_dist_ref = np.array([
    ALMAnalysis(
        assets_cf=det_assets, liabilities_cf=lp, flat_rate=policy.technical_rate,
    ).funding_ratio(as_of=BASE_DATE)
    for lp in liab_paths_ref
])
var_ref = float(np.var(fr_dist_ref, ddof=1))
var_lln = float(np.var(fr_dist_lln, ddof=1))
factor = var_ref / var_lln if var_lln > 0 else float("inf")
print(f"  (c) LLN factor (variance scale ×1 / scale ×10) = {factor:.2f} "
      f"(expected ~10).")
if not (8.0 <= factor <= 12.0):
    print(f"      WARNING: LLN factor {factor:.2f} outside [8, 12] tolerance.")

# (d) Reproducibility: identical seed → identical liabilities CFS list.
liab_paths_d1 = simulate_liability_paths(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS, n_paths=min(N_PATHS, 30),
    seed=SEED_LIAB, mortality_filter="all",
)
liab_paths_d2 = simulate_liability_paths(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS, n_paths=min(N_PATHS, 30),
    seed=SEED_LIAB, mortality_filter="all",
)
for i, (la, lb) in enumerate(zip(liab_paths_d1, liab_paths_d2)):
    ea = la.events_df.sort_values(["contractId", "time", "type"]).reset_index(drop=True)
    eb = lb.events_df.sort_values(["contractId", "time", "type"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ea, eb)
print(f"  (d) seed reproducibility OK ({len(liab_paths_d1)} paths identical).")

# (e) Risk additivity: |Var(full) - [Var(assets) + Var(mort_all)]| / Var(full) < 5%
res = attr["additivity_residual"]
assert res < 0.05, \
    f"(e) additivity residual {res:.2%} > 5% — streams may not be independent"
print(f"  (e) additivity residual = {res:.4%} < 5% OK.")


# =============================================================================
# 5. PLOTS
# =============================================================================
print("\n[5] Plots")
print("-" * 40)

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "stage5b")
os.makedirs(_FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# --- Plot 1: histograms of all five FR distributions ----------------------
fig1, ax1 = plt.subplots(figsize=(11, 5))
colors = {
    "baseline": "#264653",
    "mortality_retirees_only": "#2a9d8f",
    "mortality_all_cohorts": "#e9c46a",
    "assets_only": "#f4a261",
    "full": "#e76f51",
}
for name, dist in attr["dists"].items():
    if np.std(dist, ddof=1) < 1e-12:
        ax1.axvline(np.mean(dist) * 100, color=colors[name], linestyle=":",
                    linewidth=1.5, label=f"{name} (det.)")
    else:
        ax1.hist(dist * 100, bins=30, alpha=0.45, color=colors[name],
                 edgecolor="black", label=name)
ax1.set_title(f"Stage 5b — Funding-ratio distributions across 5 configs ({N_PATHS} paths)")
ax1.set_xlabel("Funding ratio (%)")
ax1.set_ylabel("Scenarios")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "v1_fr_distributions_by_config.png")

# --- Plot 2: stacked variance contribution -------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))
var_retiree_mort = attr["var_mortality_retirees"]
var_active_mort = max(0.0, attr["var_active_mortality_empirical"])
var_assets_ = attr["var_assets_only"]
parts = [var_assets_, var_retiree_mort, var_active_mort]
labels = ["assets", "retiree mortality", "active mortality"]
bar_colors = ["#f4a261", "#2a9d8f", "#e9c46a"]
bottom = 0.0
for val, lbl, col in zip(parts, labels, bar_colors):
    ax2.bar(["Variance components"], [val], bottom=bottom, label=lbl, color=col)
    bottom += val
ax2.axhline(attr["var_full"], color="red", linestyle="--",
            linewidth=1.0, label="Var(full) observed")
ax2.set_title("Stage 5b — Variance attribution (additive decomposition)")
ax2.set_ylabel("Var(FR)")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig2, "v2_variance_attribution.png")

plt.show()
