"""Stage 6 going-concern allocation-shift experiment
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
    GoingConcernFundingRatioAnalysis,
    RebalancingPolicy,
    SanierungsPolicy,
    StochasticALMAnalysis,
    StochasticFundingRatioAnalysis,
    ek2001_2005,
    simulate_liability_paths,
)
from awesome_actus_lib.pension.analysis.going_concern import BONDS, EQUITY


HORIZON_YEARS = 30
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("STAGE6_N_PATHS", "60"))
SEED_ASSETS = 42
SEED_LIAB = 4242
SEED_EQ = 777

EQ_RETURN = 0.03
EQ_SIGMA = 0.10
# Book yield on the incremental BONDS holdings above the initial book value;
# the original book value keeps earning only the ACTUS coupons.
BOND_YIELD = 0.02

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO_ROOT, "output", "stage_6")

print("=" * 78)
print("PENSION ALM CASE STUDY - Stage 6 - going-concern allocation experiment")
print("=" * 78)


print("\n[1A] Portfolio definition - liabilities (Stage-4 book)")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
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
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)
print("  8 active cohorts + RETIRED_1955, EntryPolicy 20/y, "
      "EK 2001-2005 (generational)")


print("\n[1B] Calibration - DG_t0 = 107.6%")
print("-" * 40)

VK_T0 = FundingRatioAnalysis(liability_fund, mortality, 0.0).pension_capital_t0()
INITIAL_ASSETS = 1.076 * VK_T0

# Start allocation = scenario B target mix: 40% fixed income / 60% equity.
BOND_BOOK = 0.30 * INITIAL_ASSETS
CASH_BOOK = 0.10 * INITIAL_ASSETS
EQUITY_SLEEVE_0 = 0.60 * INITIAL_ASSETS

print(f"  VK_t0          : CHF {VK_T0:>15,.0f}")
print(f"  initial_assets : CHF {INITIAL_ASSETS:>15,.0f}  (= 1.076 x VK_t0)")
print(f"  Bonds (book)   : CHF {BOND_BOOK:>15,.0f}  (30%)")
print(f"  Cash  (book)   : CHF {CASH_BOOK:>15,.0f}  (10%)")
print(f"  Equity (MTM)   : CHF {EQUITY_SLEEVE_0:>15,.0f}  (60%)")

_INTEREST_ANCHOR = "2026-01-01T00:00:00"

asset_portfolio = Portfolio([
    PAM(
        contractID="FIX_2040",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=BOND_BOOK,
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
        notionalPrincipal=CASH_BOOK,
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


print("\n[2] Curve calibration")
print("-" * 40)
curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)


print(f"\n[3] Liability paths (n={N_PATHS}, stochastic mortality, trace ON)")
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
print(f"  Built {len(liab_paths)} LiabilityPath objects.")


print(f"\n[4] StochasticALM run (Hull-White + GBM, n={N_PATHS})")
print("-" * 40)
salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths[0].cashflows,  # not used here, but required.
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": EQ_RETURN, "sigma": EQ_SIGMA},
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


print("\n[5] Scenarios")
print("-" * 40)

_COMMON = dict(
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
    liab_paths=liab_paths,
)

# Legacy 5c run kept only as a regression reference (assert 1), not a scenario.
ref_5c = StochasticFundingRatioAnalysis(salm=salm, **_COMMON)

# A' buy-and-hold: frequency_years > horizon, so no rebalancing ever fires.
scen_Ap = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON,
    rebalancing=RebalancingPolicy(
        target_weights={BONDS: 0.40, EQUITY: 0.60},
        frequency_years=HORIZON_YEARS + 1,
    ),
    equity_return=EQ_RETURN,
    equity_sigma=EQ_SIGMA,
    equity_seed=SEED_EQ,
    bond_yield=BOND_YIELD,
)

scen_B = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON,
    rebalancing=RebalancingPolicy(target_weights={BONDS: 0.40, EQUITY: 0.60}),
    equity_return=EQ_RETURN,
    equity_sigma=EQ_SIGMA,
    equity_seed=SEED_EQ,
    bond_yield=BOND_YIELD,
)

scen_C = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON,
    rebalancing=RebalancingPolicy(
        target_weights={BONDS: 0.40, EQUITY: 0.60},
        shift_at_year=5,
        shift_weights={BONDS: 0.60, EQUITY: 0.40},
    ),
    equity_return=EQ_RETURN,
    equity_sigma=EQ_SIGMA,
    equity_seed=SEED_EQ,
    bond_yield=BOND_YIELD,
)

# D - like B plus a DG-conditional recovery contribution; D - B isolates it.
scen_D = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON,
    rebalancing=RebalancingPolicy(target_weights={BONDS: 0.40, EQUITY: 0.60}),
    equity_return=EQ_RETURN,
    equity_sigma=EQ_SIGMA,
    equity_seed=SEED_EQ,
    bond_yield=BOND_YIELD,
    sanierung=SanierungsPolicy(trigger_dg=1.00, exit_dg=1.00, sb_factor=0.5),
)

scenarios = {
    "A' (buy-and-hold, GC engine)": scen_Ap,
    "B (fixed-mix 40/60)": scen_B,
    "C (shift 60/40 from year 5)": scen_C,
    "D (B + recovery)": scen_D,
}
for name in scenarios:
    print(f"  {name}")
print("  (Legacy 5c: regression reference only, not in the comparison)")


print("\n[VERIFY] Spec asserts")
print("-" * 40)

# Assert 1: rebalancing=None, sanierung=None reproduces the legacy 5c run.
gc_off = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON, rebalancing=None, sanierung=None,
)
for d in (BASE_DATE, f"{START_YEAR + 10}-01-01", f"{START_YEAR + 25}-01-01"):
    np.testing.assert_array_equal(
        gc_off.funding_ratio_distribution(d),
        ref_5c.funding_ratio_distribution(d),
        err_msg=f"Regression vs 5c violated at {d}",
    )
print("  => Assert 1 OK - rebalancing=None, sanierung=None identical to 5c.")

dg_t0 = ref_5c.funding_ratio_distribution(BASE_DATE)
assert abs(float(np.mean(dg_t0)) - 1.076) < 1e-9, (
    f"DG_t0 calibration missed: {float(np.mean(dg_t0)):.6%} instead of 107.6%"
)
print(f"  => Calibration OK - DG_t0 = {float(np.mean(dg_t0)):.4%} (target 107.6%).")

_LAST_DATE = f"{START_YEAR + HORIZON_YEARS}-01-01"
wp_ap = scen_Ap.weight_path(0, _LAST_DATE)
assert not bool(wp_ap["rebalanced"].any()), (
    "A' (frequency_years > horizon) must never rebalance"
)
print("  => A' configuration OK - no rebalancing date within the horizon.")

# Assert 3: after each rebalance |w - target| < 1e-12.
for p_i in (0, N_PATHS // 2, N_PATHS - 1):
    wp = scen_B.weight_path(p_i, _LAST_DATE)
    reb = wp[wp["rebalanced"]]
    dev = np.max(np.abs(np.r_[reb["w_BONDS"] - reb["target_BONDS"],
                              reb["w_EQUITY"] - reb["target_EQUITY"]]))
    assert dev < 1e-12, f"Path {p_i}: |w - target| = {dev:.2e} >= 1e-12"
print("  => Assert 3 OK - post-rebalance |w - target| < 1e-12 (3 paths checked).")

# Assert 5: C documents the shift from year 5.
wp_c = scen_C.weight_path(0, _LAST_DATE)
reb_c = wp_c[wp_c["rebalanced"]].set_index("year_index")
assert all(abs(reb_c.loc[k, "w_BONDS"] - 0.40) < 1e-12 for k in range(1, 5)), (
    "Scenario C: before year 5 must be 40/60"
)
assert all(
    abs(reb_c.loc[k, "w_BONDS"] - 0.60) < 1e-12
    for k in range(5, HORIZON_YEARS + 1)
), "Scenario C: from year 5 must be 60/40"
print("  => Assert 5 OK - weight path C: 40/60 -> 60/40 from year 5.")

# Assert 2: SB only when DG_{t-1} < trigger; >0 in D, == 0 otherwise.
sb_events = [scen_D.gc_events(i, _LAST_DATE) for i in range(N_PATHS)]
all_sb = pd.concat(sb_events, ignore_index=True)
assert len(all_sb) > 0 and float(all_sb["payoff"].sum()) > 0.0, (
    "Scenario D: total SB must be > 0 - trigger fires on no path "
    "(check DG_t0 calibration)"
)
assert (all_sb["dg_prev"] < 1.00).all(), (
    "SB bookings only allowed in path-years with DG_{t-1} < trigger"
)
for _name, _scen in (("A'", scen_Ap), ("B", scen_B), ("C", scen_C)):
    _tot = sum(
        float(_scen.gc_events(i, _LAST_DATE)["payoff"].sum())
        for i in range(N_PATHS)
    )
    assert _tot == 0.0, f"Scenario {_name}: total SB must be == 0, is {_tot}"
print(f"  => Assert 2 OK - SB only when DG_t-1 < 100% "
      f"({len(all_sb)} SB path-years in D, total CHF {float(all_sb['payoff'].sum())/1e6:,.1f}m); "
      f"A'/B/C: 0.")

# Assert 4: D is B + a pure asset inflow under the same shocks, so DG_T >= B.
_dist_D_T = scen_D.funding_ratio_distribution(_LAST_DATE)
_dist_B_T = scen_B.funding_ratio_distribution(_LAST_DATE)
assert float(np.mean(_dist_D_T)) >= float(np.mean(_dist_B_T)), (
    "mean(DG_T) in D must be >= B - SB acts in the wrong direction"
)
assert np.all(_dist_D_T >= _dist_B_T - 1e-15), (
    "D is B + pure asset inflow under the same shocks - DG_T must be "
    "path-wise >= B"
)
print(f"  => Assert 4 OK - mean(DG_T): D = {float(np.mean(_dist_D_T)):.2%} >= "
      f"B = {float(np.mean(_dist_B_T)):.2%} (path-wise >=).")


print("\n[6] Plots")
print("-" * 40)
os.makedirs(OUT_DIR, exist_ok=True)

fan_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS + 1, 2)]
dists = {
    name: {d: scen.funding_ratio_distribution(d) for d in fan_dates}
    for name, scen in scenarios.items()
}

_QUANTILES = (5, 25, 50, 75, 95)
_COLORS = {
    "A' (buy-and-hold, GC engine)": "#264653",
    "B (fixed-mix 40/60)": "#2a9d8f",
    "C (shift 60/40 from year 5)": "#e76f51",
    "D (B + recovery)": "#7b2cbf",
}
_x = [np.datetime64(d) for d in fan_dates]

# v1: DG quantile fan per scenario
for tag, (name, scen_dists) in zip(("Aprime", "B", "C", "D"), dists.items()):
    q = {p: np.array([np.percentile(scen_dists[d], p) for d in fan_dates])
         for p in _QUANTILES}
    fig, ax = plt.subplots(figsize=(11, 5))
    col = _COLORS[name]
    ax.fill_between(_x, q[5] * 100, q[95] * 100, color=col, alpha=0.15, label="p5-p95")
    ax.fill_between(_x, q[25] * 100, q[75] * 100, color=col, alpha=0.35, label="p25-p75")
    ax.plot(_x, q[50] * 100, color=col, marker="o", linewidth=2, label="Median (p50)")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100% funding")
    ax.set_title(f"Stage 6 - DG quantile fan scenario {name} ({N_PATHS} paths)")
    ax.set_xlabel("Valuation date")
    ax.set_ylabel("Funding ratio (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    _path = os.path.join(OUT_DIR, f"v1_dg_fan_{tag}.png")
    fig.savefig(_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {_path}")

# v2: P(DG_t < 100%) for all scenarios
fig, ax = plt.subplots(figsize=(11, 5))
for name, scen_dists in dists.items():
    p_under = [float(np.mean(scen_dists[d] < 1.0)) for d in fan_dates]
    ax.plot(_x, np.array(p_under) * 100, marker="o", linewidth=2,
            color=_COLORS[name], label=name)
ax.set_title(f"Stage 6 - underfunding probability P(DG_t < 100%) "
             f"({N_PATHS} paths)")
ax.set_xlabel("Valuation date")
ax.set_ylabel("P(DG < 100%) (%)")
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
_path = os.path.join(OUT_DIR, "v2_p_underfunded.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")

# v3: weight paths B vs C (post-rebalance, path 0)
wp_b = scen_B.weight_path(0, _LAST_DATE)
reb_b = wp_b[wp_b["rebalanced"]]
reb_c = wp_c[wp_c["rebalanced"]]
fig, ax = plt.subplots(figsize=(11, 5))
ax.step(reb_b["date"], reb_b["w_EQUITY"] * 100, where="post", linewidth=2,
        color=_COLORS["B (fixed-mix 40/60)"], label="B: w_EQUITY")
ax.step(reb_c["date"], reb_c["w_EQUITY"] * 100, where="post", linewidth=2,
        color=_COLORS["C (shift 60/40 from year 5)"], label="C: w_EQUITY")
ax.step(reb_b["date"], reb_b["w_BONDS"] * 100, where="post", linewidth=1.5,
        linestyle="--", color=_COLORS["B (fixed-mix 40/60)"], label="B: w_BONDS")
ax.step(reb_c["date"], reb_c["w_BONDS"] * 100, where="post", linewidth=1.5,
        linestyle="--", color=_COLORS["C (shift 60/40 from year 5)"], label="C: w_BONDS")
ax.axvline(np.datetime64(f"{START_YEAR + 5}-01-01"), color="grey",
           linestyle=":", linewidth=1.5, label="Shift from year 5")
ax.set_title("Stage 6 - weight paths (post-rebalance) B vs C")
ax.set_xlabel("Rebalancing date")
ax.set_ylabel("Weight (%)")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(ncol=2)
plt.tight_layout()
_path = os.path.join(OUT_DIR, "v3_weight_paths_B_vs_C.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")

# v4: SB statistics (scenario D)
cum_sb = np.array([float(ev["payoff"].sum()) for ev in sb_events])
years_active = np.array([len(ev) for ev in sb_events])
share_active = float(years_active.sum()) / (N_PATHS * HORIZON_YEARS)

_sb_years = list(range(START_YEAR + 1, START_YEAR + HORIZON_YEARS + 1))
_year_frac = [
    sum(1 for ev in sb_events if not ev.empty
        and (pd.to_datetime(ev["time"]).dt.year == y).any()) / N_PATHS
    for y in _sb_years
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.bar(_sb_years, np.array(_year_frac) * 100, color="#7b2cbf", alpha=0.8)
ax1.set_title(f"Share of paths with SB per year "
              f"(active path-years total: {share_active:.1%})")
ax1.set_xlabel("Payment year")
ax1.set_ylabel("Share of paths (%)")
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle="--", alpha=0.5)
ax2.hist(cum_sb / 1e6, bins=15, color="#7b2cbf", alpha=0.8)
ax2.axvline(float(np.mean(cum_sb)) / 1e6, color="black", linestyle="--",
            linewidth=1.5, label=f"Mean CHF {float(np.mean(cum_sb))/1e6:,.0f}m")
ax2.set_title("Cumulative SB per path (distribution)")
ax2.set_xlabel("Cumulative SB (CHF m)")
ax2.set_ylabel("Number of paths")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
fig.suptitle(f"Stage 6 - recovery contributions scenario D "
             f"(trigger=100%, sb_factor=0.5, {N_PATHS} paths)")
plt.tight_layout()
_path = os.path.join(OUT_DIR, "v4_sb_statistics.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")


print("\n[7] Summary @ horizon end")
print("-" * 40)
d_T = fan_dates[-1]
print(f"  Valuation date {d_T}:")
for name, scen_dists in dists.items():
    dist = scen_dists[d_T]
    print(f"    {name:<32} mean={float(np.mean(dist)):8.2%}  "
          f"p5={float(np.percentile(dist, 5)):8.2%}  "
          f"P(DG<1)={float(np.mean(dist < 1.0)):6.1%}")

plt.show()
