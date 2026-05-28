"""Stage5a_Phase2_multi_asset_quickstart.py

Stage 5a Phase 2 — Multi-Asset-Allocation (Equities via MTM & GBM outside ACTUS):
  - Stage-4 deterministic liability stream
  - Diversified asset portfolio:
      - ACTUS Portfolio (70%): Fixed bonds (120M) + Floating bonds (80M) + Cash MM (80M)
      - Equity Sleeve (30%): CHF 120M outside ACTUS, marked-to-market using GBM simulated paths
  - Stochastic block: Hull-White short-rate paths (ACTUS) + GBM paths (Equity Sleeves)
  - Plausible solvency Deckungsgrad (~100% to 130%) and tail-risk metrics (VaR, Expected Shortfall)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.stochastic_rates import CurveCalibrator, GBMModel
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    DynamicFundSimulator,
    Stage4Dynamics,
    EntryPolicy,
    ALMAnalysis,
    DeckungsgradAnalysis,
    PensionierungsverlustAnalysis,
    StochasticALMAnalysis,
    StochasticDeckungsgradAnalysis,
    ek2001_2005,
)

HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
N_PATHS = int(os.environ.get("STAGE5_PHASE2_N_PATHS", "200"))
SEED = 42

print("=" * 80)
print("PENSION ALM MULTI-ASSET CASE STUDY: Diversified Portfolio vs Liabilities")
print("           Stage 5a Phase 2 — Solvency Deckungsgrad via GBM MTM")
print("=" * 80)

# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side) — same fund as Stage 4/5a
# =============================================================================
print("\n[1A] Portfolio Definition — Liabilities (Stage 4 Dynamic setup)")
print("-" * 50)

policy = PensionPolicy()
liability_fund = PensionFund(policy=policy, start_date=START_DATE)

# Comprehensive active population to get realistic Vorsorgekapital (~375M)
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1960", birth_year=1960, headcount=90,
    gross_salary=120_000.0, accrued_savings=600_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1963", birth_year=1963, headcount=100,
    gross_salary=115_000.0, accrued_savings=520_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1966", birth_year=1966, headcount=110,
    gross_salary=115_000.0, accrued_savings=450_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970", birth_year=1970, headcount=150,
    gross_salary=110_000.0, accrued_savings=350_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1972", birth_year=1972, headcount=130,
    gross_salary=108_000.0, accrued_savings=320_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1975", birth_year=1975, headcount=140,
    gross_salary=105_000.0, accrued_savings=270_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1978", birth_year=1978, headcount=140,
    gross_salary=100_000.0, accrued_savings=220_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0, status="retired",
    annual_pension=31_380.0, gender="f",
))

entry_policy = EntryPolicy(entry_age=25, headcount=20, gross_salary=80_000.0, gender="unisex")
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)
mortality = ek2001_2005()

simulator = DynamicFundSimulator(
    fund=liability_fund, entry_policy=entry_policy,
    dynamics=dynamics, mortality=mortality,
)
liability_cfs = simulator.run(horizon_years=HORIZON_YEARS)
print(f"  Liabilities ready: {len(liability_cfs.events_df)} events")

# =============================================================================
# 1B. PORTFOLIO DEFINITION (asset side) — Diversified Portfolio (50% Bonds / 30% Equities / 20% Cash)
# =============================================================================
print("\n[1B] Portfolio Definition — Diversified Assets")
print("-" * 50)

_INTEREST_ANCHOR = "2026-01-01T00:00:00"

# ACTUS Portfolio: CHF 280,000,000 total (70% of 400M)
asset_portfolio = Portfolio([
    # Fixed-rate Bond (notional CHF 120,000,000)
    PAM(
        contractID="FIX_BOND_2035",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=120_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2035-12-31T00:00:00",
        nominalInterestRate=0.020,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    # Floating-rate Bond (notional CHF 80,000,000, tied to "IR_SCENARIO")
    PAM(
        contractID="FLT_BOND_2035",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=80_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2035-12-31T00:00:00",
        nominalInterestRate=0.0,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE,
        cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset="IR_SCENARIO",
        rateSpread=0.0060,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0",
        counterpartyID="Bank-01",
        creatorID="PK_ALM",
    ),
    # Cash / Money Market (notional CHF 80,000,000, tied to "IR_SCENARIO")
    PAM(
        contractID="CASH_ACCOUNT",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=80_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00",
        nominalInterestRate=0.0,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE,
        cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset="IR_SCENARIO",
        rateSpread=0.0,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    )
])

# Equity Sleeve runs completely OUTSIDE ACTUS to implement mark-to-market revaluation
EQUITY_SLEEVE_0 = 120_000_000.0  # CHF 120,000,000 (30% of 400M)
BOND_BOOK = 200_000_000.0        # CHF 120M fixed + 80M floating
CASH_BOOK = 80_000_000.0         # CHF 80M

print(f"  ACTUS asset portfolio created with {len(asset_portfolio.contracts)} contracts:")
print(f"    - Fixed Bonds:   CHF 120,000,000")
print(f"    - Floating Bonds: CHF  80,000,000")
print(f"    - Money Market:   CHF  80,000,000")
print(f"    - Total ACTUS assets: CHF 280,000,000 (70%)")
print(f"  Equity Sleeve (Runs outside ACTUS):")
print(f"    - Equities (MTM): CHF 120,000,000 (30%)")
print(f"  Total Portfolio Value @ t0: CHF 400,000,000 (100%)")

# =============================================================================
# 2. CALIBRATION & MONTE CARLO SETUP
# =============================================================================
print("\n[2] Calibration & Monte Carlo Setup (Hull-White + GBM)")
print("-" * 50)

# Calibrate initial yield curve
curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)

# Setup joint stochastic analysis (HW for rates, GBM for equity)
salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liability_cfs,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,  # sticky technical rate
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": 0.05, "sigma": 0.15},  # GBM equities (5% return, 15% volatility)
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=START_DATE,
    market_code="IR_SCENARIO",
    seed=SEED,
)

print(f"  Simulating {N_PATHS} paths over {HORIZON_YEARS} years...")
salm.run()
print("  Monte Carlo simulation complete!")

# Initialize the new Stochastic Solvency Deckungsgrad Analysis
sdga = StochasticDeckungsgradAnalysis(
    salm=salm,
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
)

# Initialize standard ALM analysis (for comparing the two metrics)
alm_det = ALMAnalysis(
    assets_cf=salm._asset_cfs[0],
    liabilities_cf=liability_cfs,
    flat_rate=policy.technical_rate,
)

# =============================================================================
# 3. COMPARISON OF TWO METRICS (LIQUIDITY vs SOLVENCY)
# =============================================================================
print("\n[3] Comparison of the Two Metrics (Liquidity net-CF vs Solvency DG)")
print("-" * 50)

VK_t0 = sdga.vorsorgekapital_t0
print(f"  Vorsorgekapital (VK) @ t0 (deterministisch): CHF {VK_t0:,.2f}")

# Metric 1: Deterministic net-CF diagnostic (Liquidity View)
fr_t0_det = alm_det.funding_ratio(as_of=START_DATE)
print(f"  1. Net-CF funding_ratio (Liquidity View) @ t0:  {fr_t0_det:.2%} (Generic Cashflow NPV)")

# Metric 2: Stochastic Solvency Deckungsgrad (Solvency View)
dg_t0_det = sdga.deckungsgrad_distribution(as_of=START_DATE)[0]
print(f"  2. Solvency Deckungsgrad (Solvency View) @ t0: {dg_t0_det:.2%} (Art. 44 BVV2)")

for valuation_date in ("2025-01-01", "2035-01-01", "2045-01-01"):
    summary = sdga.summary(valuation_date)
    print(f"\n  Solvency Deckungsgrad (Art. 44) Distribution @ {valuation_date}:")
    print(f"    Mean:                 {summary['mean']:.2%}")
    print(f"    Std Dev:              {summary['std']:.2%}")
    print(f"    Median (p50):         {summary['p50']:.2%}")
    print(f"    95% Solvency Floor:   {summary['p5']:.2%}")
    print(f"    Expected Shortfall:   {summary['dg_es5']:.2%}")
    print(f"    Deficit Probability:  {summary['p_underfunded']:.1%}")

# =============================================================================
# 4. MANDATORY VERIFICATION TESTS
# =============================================================================
print("\n[4] Mandatory Verification Tests")
print("-" * 50)

# A. Falsification Assert: sigma = 0.0 must yield degenerate distribution (std < 1e-6)
print("  Running Falsification Test (sigma=0.0)...")
salm_degenerate = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liability_cfs,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": 0.05, "sigma": 0.0},  # ZERO VOLATILITY
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=START_DATE,
    market_code="IR_SCENARIO",
    seed=SEED,
    verbose=False,
)
salm_degenerate.run()

sdga_degenerate = StochasticDeckungsgradAnalysis(
    salm=salm_degenerate,
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
)
dist_degen = sdga_degenerate.deckungsgrad_distribution("2035-01-01")
std_degen = np.std(dist_degen, ddof=1)
print(f"    - Degenerate (sigma=0.0) std: {std_degen:.4e}")
assert std_degen < 1e-6, f"Falsification: degenerate std is too large ({std_degen:.2e})"

dist_active = sdga.deckungsgrad_distribution("2035-01-01")
std_active = np.std(dist_active, ddof=1)
print(f"    - Active (sigma=0.15) std:     {std_active:.4f}")
assert std_active > 0.001, f"Falsification: active std should be positive ({std_active:.4f})"
print("    => Assert [a] (Degenerate vs Active dispersion) OK.")

# B. Equity Sensitivity: DG mean must increase monotonically with equity_sleeve_0
print("  Running Equity Sensitivity Test...")
sdga_more_equity = StochasticDeckungsgradAnalysis(
    salm=salm,
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0 + 50_000_000.0,  # Increase equity sleeve
)
mean_original = np.mean(sdga.deckungsgrad_distribution("2035-01-01"))
mean_more_equity = np.mean(sdga_more_equity.deckungsgrad_distribution("2035-01-01"))
print(f"    - Mean with CHF 120M equity: {mean_original:.4f}")
print(f"    - Mean with CHF 170M equity: {mean_more_equity:.4f}")
assert mean_more_equity > mean_original, "Sensitivity: larger equity sleeve did not increase mean DG"
print("    => Assert [b] (Equity sensitivity) OK.")

# C. GBM Convergence: E[S_T] ≈ S0 * exp(mu * T)
print("  Running GBM Convergence Test...")
sim_eq = salm.equity_simulation
S0 = sim_eq.rates[0, 0]
mu = 0.05
T = HORIZON_YEARS
expected_S_T = S0 * np.exp(mu * T)
actual_S_T = np.mean(sim_eq.terminal_distribution())
rel_diff = abs(actual_S_T - expected_S_T) / expected_S_T
print(f"    - Expected E[S_T]: {expected_S_T:.2f}")
print(f"    - Actual E[S_T]:   {actual_S_T:.2f} (diff = {rel_diff:.2%})")
max_margin = 0.50 if N_PATHS < 100 else 0.15
assert rel_diff < max_margin, f"GBM Convergence: actual E[S_T] diverges too much ({rel_diff:.2%})"
print("    => Assert [c] (GBM convergence) OK.")

# D. Reproducibility: same seed -> identical equity paths
print("  Running Reproducibility Test...")
gbm_a = GBMModel(S0=100.0, mu=0.05, sigma=0.15, seed=SEED)
gbm_b = GBMModel(S0=100.0, mu=0.05, sigma=0.15, seed=SEED)
sim_a = gbm_a.simulate(T=10.0, M=120, I=64)
sim_b = gbm_b.simulate(T=10.0, M=120, I=64)
assert np.allclose(sim_a.rates, sim_b.rates), "Reproducibility: paths under same seed differ"
print("    => Assert [d] (Seed reproducibility) OK.")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
print("\n[5] Plot Generation")
print("-" * 50)

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "stage5a_phase2")
os.makedirs(_FIG_DIR, exist_ok=True)

def _save(fig, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {path}")

# --- Plot 1: Equity Price Paths (GBM) ---
fig1, ax1 = plt.subplots(figsize=(11, 5))
times = salm.equity_simulation.times
prices = salm.equity_simulation.rates
ax1.plot(times, prices[:, :20], color="#2a9d8f", alpha=0.3)
ax1.plot(times, salm.equity_simulation.mean_path(), color="#264653", linewidth=2.5, label="Mean Path")
ax1.plot(times, salm.equity_simulation.percentile_path(5), color="#e76f51", linestyle="--", label="5th Percentile")
ax1.plot(times, salm.equity_simulation.percentile_path(95), color="#e9c46a", linestyle="--", label="95th Percentile")
ax1.set_title(f"Stage 5a Phase 2 — Simulated Equity Index Paths (GBM, mu=5%, sigma=15%, {N_PATHS} paths)")
ax1.set_xlabel("Years")
ax1.set_ylabel("Index Level")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "v1_equity_paths_gbm.png")

# --- Plot 2: Stochastic Solvency Deckungsgrad Distribution @ t10 ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
dist10 = sdga.deckungsgrad_distribution("2035-01-01")
mean10 = np.mean(dist10)
mean_degen10 = np.mean(dist_degen)
ax2.hist(dist10 * 100, bins=30, color="#e9c46a", edgecolor="black", alpha=0.8, label="Active Solvency (sigma=0.15)")
ax2.axvline(100, color="red", linestyle="--", linewidth=1.5, label="100% Solvency")
ax2.axvline(mean10 * 100, color="#264653", linestyle="-", linewidth=2, label=f"Active Mean ({mean10:.2%})")
ax2.axvline(mean_degen10 * 100, color="#e76f51", linestyle=":", linewidth=2.5, label=f"Degen Mean ({mean_degen10:.2%})")
ax2.set_title(f"Stage 5a Phase 2 — Stochastic Solvency Deckungsgrad Distribution @ 2035-01-01")
ax2.set_xlabel("Solvency Deckungsgrad (%)")
ax2.set_ylabel("Scenarios")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig2, "v2_stoch_fr_hist_t10.png")

# --- Plot 3: Quantile Fan Chart over Horizon ---
fan_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 5)]
fan = sdga.fan_chart_data(fan_dates)

fig3, ax3 = plt.subplots(figsize=(11, 5))
xf = fan.index
ax3.fill_between(xf, fan["p5"] * 100, fan["p95"] * 100, color="#2a9d8f", alpha=0.15, label="p5-p95")
ax3.fill_between(xf, fan["p25"] * 100, fan["p75"] * 100, color="#2a9d8f", alpha=0.35, label="p25-p75")
ax3.plot(xf, fan["p50"] * 100, color="#264653", marker="o", linewidth=2, label="Median (p50)")
ax3.axhline(100, color="red", linestyle="--", linewidth=1, label="100% Solvency")
ax3.set_title("Stage 5a Phase 2 — Solvency Deckungsgrad Fan Chart (Bonds, Equities MTM, Cash)")
ax3.set_ylabel("Solvency Deckungsgrad (%)")
ax3.set_xlabel("Valuation Date")
ax3.legend()
ax3.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig3, "v3_stoch_fr_fan_multi_asset.png")

print("\nAll plots generated and saved successfully!")
print("=" * 80)
