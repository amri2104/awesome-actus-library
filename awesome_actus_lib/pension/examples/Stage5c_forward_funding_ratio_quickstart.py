"""Stage5c_forward_funding_ratio_quickstart.py

Stage 5c — forward stochastic solvency funding ratio (Art. 44 BVV2) on top of Stage 5b:
  - Stage-4 deterministic liability backbone + stochastic mortality paths
  - Stochastic ASSET side: StochasticALMAnalysis (Hull-White + GBM equity)
  - Forward solvency funding ratio: PC_t,i and V_t,i coupled per path via LiabilityPath
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from awesome_actus_lib import PAM, Portfolio
from awesome_actus_lib.stochastic_rates import CurveCalibrator
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    Stage4Dynamics,
    EntryPolicy,
    FundingRatioAnalysis,
    StochasticALMAnalysis,
    StochasticFundingRatioAnalysis,
    ek2001_2005,
    simulate_liability_paths,
)


HORIZON_YEARS = 30
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("STAGE5C_N_PATHS", "60"))
SEED_ASSETS = 42
SEED_LIAB = 4242


print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 5c — forward stochastic solvency funding ratio")
print("=" * 78)


# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side)
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1965",
    birth_year=1965,
    headcount=120,
    gross_salary=115_000.0,
    accrued_savings=520_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1972",
    birth_year=1972,
    headcount=140,
    gross_salary=108_000.0,
    accrued_savings=320_000.0,
    gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1980",
    birth_year=1980,
    headcount=160,
    gross_salary=100_000.0,
    accrued_savings=180_000.0,
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
    headcount=15,
    gross_salary=80_000.0,
    gender="unisex",
)

dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)
# Generational EK 2001-2005 (1.25%/yr improvement from 2003): lifts ä_x to a
# realistic level. This is the analysis where it matters most — the forward
# PC_t,i values retired survivors via ä_x at each horizon date.
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)
print(f"  3 active cohorts + 1 retired cohort, EntryPolicy 15/y, EK 2001-2005 (generational)")


# =============================================================================
# 1B. PORTFOLIO DEFINITION (asset side) — small mix: one fixed bond + one cash account
# =============================================================================
print("\n[1B] Portfolio Definition — assets (mixed fixed + cash, small for speed)")
print("-" * 40)

_INTEREST_ANCHOR = "2026-01-01T00:00:00"

asset_portfolio = Portfolio([
    PAM(
        contractID="FIX_2040",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=180_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2040-12-31T00:00:00",
        nominalInterestRate=0.020,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="CASH_ACCOUNT",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=60_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00",
        nominalInterestRate=0.0,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE,
        cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset=MARKET_CODE,
        rateSpread=0.0,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
])

BOND_BOOK = 180_000_000.0
CASH_BOOK = 60_000_000.0
EQUITY_SLEEVE_0 = 60_000_000.0
print(f"  Bonds (book) : CHF {BOND_BOOK:>15,.0f}")
print(f"  Cash  (book) : CHF {CASH_BOOK:>15,.0f}")
print(f"  Equity (MTM) : CHF {EQUITY_SLEEVE_0:>15,.0f}")


# =============================================================================
# 2. CURVE CALIBRATION
# =============================================================================
print("\n[2] Curve calibration")
print("-" * 40)
curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)


# =============================================================================
# 3. STOCHASTIC LIABILITY PATHS (with headcount trace)
# =============================================================================
print(f"\n[3] Liability paths (n={N_PATHS}, stochastic mortality, headcount trace ON)")
print("-" * 40)
liab_paths = simulate_liability_paths(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=dynamics,
    mortality=mortality,
    horizon_years=HORIZON_YEARS,
    n_paths=N_PATHS,
    seed=SEED_LIAB,
    mortality_filter="all",
    return_headcount_log=True,
)
print(f"  Built {len(liab_paths)} LiabilityPath objects, "
      f"trace[0] has {len(liab_paths[0].headcount_trace)} (year,cohort) rows.")


# =============================================================================
# 4A. STOCHASTIC ASSETS (HW + GBM)
# =============================================================================
print(f"\n[4A] StochasticALM run (Hull-White + GBM, n={N_PATHS})")
print("-" * 40)
salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths[0].cashflows,  # not used by Stage 5c, but required.
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": 0.05, "sigma": 0.15},
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=BASE_DATE,
    market_code=MARKET_CODE,
    seed=SEED_ASSETS,
    verbose=False,
)
salm.run()
print(f"  Done. asset_cfs[0] events={len(salm._asset_cfs[0].events_df)}")

sfra = StochasticFundingRatioAnalysis(
    salm=salm,
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
    liab_paths=liab_paths,
)


# =============================================================================
# 4B. DETERMINISTIC ASSETS (sigma=0) — for the "gap closed" assert
# =============================================================================
print(f"\n[4B] Deterministic-assets twin (sigma_HW=0, sigma_eq=0, n={N_PATHS})")
print("-" * 40)
salm_det = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths[0].cashflows,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.0},
    equity_params={"S0": 100.0, "mu": 0.05, "sigma": 0.0},
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=BASE_DATE,
    market_code=MARKET_CODE,
    seed=SEED_ASSETS,
    verbose=False,
)
salm_det.run()
sfra_det_assets = StochasticFundingRatioAnalysis(
    salm=salm_det,
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
    liab_paths=liab_paths,
)
print("  Done.")


# =============================================================================
# 5. VERIFY — three inline asserts
# =============================================================================
print("\n[VERIFY] Inline asserts")
print("-" * 40)

# ---- (a) t0 collapse --------------------------------------------------------
dga_t0 = FundingRatioAnalysis(
    fund=liability_fund,
    mortality=mortality,
    pension_assets=BOND_BOOK + CASH_BOOK + EQUITY_SLEEVE_0,
).funding_ratio_t0()
fwd_t0 = sfra.funding_ratio_distribution(BASE_DATE)
mean_fwd_t0 = float(np.mean(fwd_t0))
std_fwd_t0 = float(np.std(fwd_t0, ddof=1))
print(f"  (a) Funding ratio_t0 (deterministic): {dga_t0:.6%}")
print(f"      Forward DG @ t0 (mean)          : {mean_fwd_t0:.6%}  "
      f"(std={std_fwd_t0:.2e})")
assert abs(mean_fwd_t0 - dga_t0) < 1e-9, (
    f"(a) t0 collapse failed: forward {mean_fwd_t0} vs deterministic {dga_t0}"
)
assert std_fwd_t0 < 1e-9, (
    f"(a) t0 collapse: forward DG @ t0 should be path-invariant, got std={std_fwd_t0:.2e}"
)
print("      => Assert (a) OK — forward surface collapses to t0 snapshot.")

# ---- (b) gap closed: variance > 0 with deterministic assets ----------------
as_of_b = f"{START_YEAR + 10}-01-01"
fwd_det = sfra_det_assets.funding_ratio_distribution(as_of_b)
var_det = float(np.var(fwd_det, ddof=1))
print(f"\n  (b) Deterministic assets + stochastic mortality, as_of={as_of_b}")
print(f"      Var(DG)                         : {var_det:.6e}")
print(f"      mean                            : {float(np.mean(fwd_det)):.4%}")
print(f"      std                             : {float(np.std(fwd_det, ddof=1)):.4%}")
assert var_det > 1e-10, (
    f"(b) Var(DG) should be strictly positive, got {var_det:.2e} — "
    "stage-5c coupling is not wiring stochastic mortality into the solvency lens."
)
print("      => Assert (b) OK — solvency funding ratio now responds to longevity risk.")

# ---- (c) coupling: high-mortality vs low-mortality path --------------------
as_of_c = f"{START_YEAR + HORIZON_YEARS - 1}-01-01"
# Order paths by surviving retirees at end-of-horizon (lower = more deaths).
surviving_retirees = np.array([
    sum(r["headcount"] for r in lp.headcount_trace
        if r["year"] == START_YEAR + HORIZON_YEARS - 1 and r["status"] == "retired")
    for lp in liab_paths
])
idx_high_mort = int(np.argmin(surviving_retirees))   # most deaths
idx_low_mort = int(np.argmax(surviving_retirees))    # fewest deaths

pc_high = sfra._path_PC(as_of_c, idx_high_mort)
pc_low = sfra._path_PC(as_of_c, idx_low_mort)


def _cum_pension_outflow(lp, end_year: int) -> float:
    df = lp.cashflows.events_df
    if df.empty:
        return 0.0
    end_ts = pd.to_datetime(f"{end_year}-12-31")
    mask = (df["type"] == "PENSION_PAYMENT") & (pd.to_datetime(df["time"]) <= end_ts)
    return float(df.loc[mask, "payoff"].sum())  # negative number


pen_high = _cum_pension_outflow(liab_paths[idx_high_mort], START_YEAR + HORIZON_YEARS - 1)
pen_low = _cum_pension_outflow(liab_paths[idx_low_mort], START_YEAR + HORIZON_YEARS - 1)
print(f"\n  (c) Coupling check, as_of={as_of_c}")
print(f"      surviving retirees: high-mort path = {surviving_retirees[idx_high_mort]:.1f}, "
      f"low-mort path = {surviving_retirees[idx_low_mort]:.1f}")
print(f"      PC (high-mort)   : CHF {pc_high:>15,.0f}")
print(f"      PC (low-mort )   : CHF {pc_low:>15,.0f}")
print(f"      cum pension out (high-mort, negative): CHF {pen_high:>15,.0f}")
print(f"      cum pension out (low-mort , negative): CHF {pen_low:>15,.0f}")
assert pc_high < pc_low, (
    f"(c) coupling: PC on high-mortality path ({pc_high:,.0f}) should be smaller "
    f"than on low-mortality path ({pc_low:,.0f})."
)
# Outflows are negative; smaller magnitude on high-mort means pen_high > pen_low.
assert pen_high > pen_low, (
    f"(c) coupling: cumulative pension outflow magnitude on high-mort path "
    f"should be smaller (|{pen_high:,.0f}| < |{pen_low:,.0f}|)."
)
print("      => Assert (c) OK — same survivor path drives PC and pension outflows.")


# =============================================================================
# 5. PLOTS
# =============================================================================
print("\n[5] Plots")
print("-" * 40)

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "stage5c")
os.makedirs(_FIG_DIR, exist_ok=True)

fan_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 2)]
fan = sfra.fan_chart_data(fan_dates)

fig, ax = plt.subplots(figsize=(11, 5))
xf = fan.index
ax.fill_between(xf, fan["p5"] * 100, fan["p95"] * 100, color="#2a9d8f",
                alpha=0.15, label="p5-p95")
ax.fill_between(xf, fan["p25"] * 100, fan["p75"] * 100, color="#2a9d8f",
                alpha=0.35, label="p25-p75")
ax.plot(xf, fan["p50"] * 100, color="#264653", marker="o", linewidth=2,
        label="Median (p50)")
ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100% solvency")
ax.set_title(f"Stage 5c — Forward Solvency Funding-Ratio fan chart "
             f"(coupled mortality + assets, {N_PATHS} paths)")
ax.set_xlabel("Valuation date")
ax.set_ylabel("Funding Ratio (%)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
_path = os.path.join(_FIG_DIR, "v1_forward_fr_fan_chart.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")
plt.show()
