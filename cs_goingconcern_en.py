"""Stage 6 going-concern figures and JSON summary for Case Study I.

Scenarios on the same liability paths and equity shocks, so differences are
pure strategy effects: buy-and-hold, fixed-mix 40/60, shift to 60/40 from
year 5, and fixed-mix plus a DG-conditional recovery contribution.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awesome_actus_lib import PAM, Portfolio
from awesome_actus_lib.stochastic_rates import CurveCalibrator
from awesome_actus_lib.pension import (
    PensionPolicy, Cohort, PensionFund, Stage4Dynamics, EntryPolicy,
    FundingRatioAnalysis, GoingConcernFundingRatioAnalysis,
    RebalancingPolicy, SanierungsPolicy, StochasticALMAnalysis,
    ek2001_2005, simulate_liability_paths,
)
from awesome_actus_lib.pension.analysis.going_concern import BONDS, EQUITY

HORIZON_YEARS = 30
START_DATE = "2025-01-01T00:00:00"
START_YEAR = 2025
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("GC_N_PATHS", "60"))
SEED_ASSETS, SEED_LIAB, SEED_EQ = 42, 4242, 777
EQ_RETURN, EQ_SIGMA, BOND_YIELD = 0.03, 0.10, 0.02

OUT = "/Users/amrighodra/Bachelorarbeit/AAL/output/thesis_figures"
os.makedirs(OUT, exist_ok=True)

policy = PensionPolicy()
fund = PensionFund(policy=policy, start_date=START_DATE)
_cohorts = [
    ("ACTIVE_1960", 1960, 90, 120_000.0, 600_000.0),
    ("ACTIVE_1963", 1963, 100, 115_000.0, 520_000.0),
    ("ACTIVE_1966", 1966, 110, 115_000.0, 450_000.0),
    ("ACTIVE_1970", 1970, 150, 110_000.0, 350_000.0),
    ("ACTIVE_1972", 1972, 130, 108_000.0, 320_000.0),
    ("ACTIVE_1975", 1975, 140, 105_000.0, 270_000.0),
    ("ACTIVE_1978", 1978, 140, 100_000.0, 220_000.0),
    ("ACTIVE_1990", 1990, 200, 90_000.0, 80_000.0),
]
for cid, by, hc, sal, agh in _cohorts:
    fund.add_cohort(Cohort(cohort_id=cid, birth_year=by, headcount=hc,
                           gross_salary=sal, accrued_savings=agh, gender="unisex"))
fund.add_cohort(Cohort(cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
                       gross_salary=0.0, accrued_savings=0.0, status="retired",
                       annual_pension=31_380.0, gender="f"))

entry = EntryPolicy(entry_age=25, headcount=20, gross_salary=80_000.0, gender="unisex")
dynamics = Stage4Dynamics(salary_growth=0.01,
                          conversion_rate_path={2025: 0.0523, 2029: 0.0510},
                          threshold_index_period=5)
mort = ek2001_2005(improvement_rate=0.0125, base_year=2003)

VK_T0 = FundingRatioAnalysis(fund, mort, 0.0).pension_capital_t0()
INITIAL = 1.076 * VK_T0
BOND_BOOK, CASH_BOOK, EQUITY0 = 0.30 * INITIAL, 0.10 * INITIAL, 0.60 * INITIAL
print(f"VK_t0={VK_T0:,.0f}  initial_assets={INITIAL:,.0f} (DG_t0=107.6%)")

asset_portfolio = Portfolio([
    PAM(contractID="FIX_2040", statusDate=START_DATE, contractDealDate=START_DATE,
        currency="CHF", notionalPrincipal=BOND_BOOK, initialExchangeDate=START_DATE,
        maturityDate="2040-12-31T00:00:00", nominalInterestRate=0.020,
        dayCountConvention="30E360", contractRole="RPA",
        cycleOfInterestPayment="P1YL1", counterpartyID="SNB", creatorID="PK_ALM"),
    PAM(contractID="CASH_ACCOUNT", statusDate=START_DATE, contractDealDate=START_DATE,
        currency="CHF", notionalPrincipal=CASH_BOOK, initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00", nominalInterestRate=0.0,
        dayCountConvention="30E360", contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE, cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset=MARKET_CODE, rateSpread=0.0,
        cycleAnchorDateOfInterestPayment="2026-01-01T00:00:00",
        cycleOfInterestPayment="P1YL0", counterpartyID="SNB", creatorID="PK_ALM"),
])

curve_t = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_r = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_t, spot_rates=curve_r)

print(f"Simulating {N_PATHS} liability paths + assets ...")
liab_paths = simulate_liability_paths(
    fund=fund, entry_policy=entry, dynamics=dynamics, mortality=mort,
    horizon_years=HORIZON_YEARS, n_paths=N_PATHS, seed=SEED_LIAB,
    mortality_filter="all", return_headcount_log=True)

salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio, liabilities_cf=liab_paths[0].cashflows,
    calibrator=calibrator, technical_rate=policy.technical_rate, model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": EQ_RETURN, "sigma": EQ_SIGMA},
    n_paths=N_PATHS, horizon_years=HORIZON_YEARS, steps_per_year=12,
    base_date=BASE_DATE, market_code=MARKET_CODE, seed=SEED_ASSETS, verbose=False)
salm.run()

_COMMON = dict(fund=fund, mortality=mort, bond_book=BOND_BOOK, cash_book=CASH_BOOK,
               equity_sleeve_0=EQUITY0, liab_paths=liab_paths)
common_roll = dict(equity_return=EQ_RETURN, equity_sigma=EQ_SIGMA,
                   equity_seed=SEED_EQ, bond_yield=BOND_YIELD)

scen = {
    "Buy-and-Hold": GoingConcernFundingRatioAnalysis(
        salm, **_COMMON, rebalancing=RebalancingPolicy(
            target_weights={BONDS: 0.40, EQUITY: 0.60},
            frequency_years=HORIZON_YEARS + 1), **common_roll),
    "Fixed-Mix 40/60": GoingConcernFundingRatioAnalysis(
        salm, **_COMMON, rebalancing=RebalancingPolicy(
            target_weights={BONDS: 0.40, EQUITY: 0.60}), **common_roll),
    "Shift to 60/40 (yr 5)": GoingConcernFundingRatioAnalysis(
        salm, **_COMMON, rebalancing=RebalancingPolicy(
            target_weights={BONDS: 0.40, EQUITY: 0.60}, shift_at_year=5,
            shift_weights={BONDS: 0.60, EQUITY: 0.40}), **common_roll),
    "Fixed-Mix + recovery": GoingConcernFundingRatioAnalysis(
        salm, **_COMMON, rebalancing=RebalancingPolicy(
            target_weights={BONDS: 0.40, EQUITY: 0.60}),
        sanierung=SanierungsPolicy(trigger_dg=1.00, exit_dg=1.00, sb_factor=0.5),
        **common_roll),
}
COL = {"Buy-and-Hold": "#264653", "Fixed-Mix 40/60": "#2a9d8f",
       "Shift to 60/40 (yr 5)": "#e76f51", "Fixed-Mix + recovery": "#7b2cbf"}

fan_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS + 1, 2)]
dists = {nm: {d: s.funding_ratio_distribution(d) for d in fan_dates}
         for nm, s in scen.items()}
_x = [np.datetime64(d) for d in fan_dates]
_LAST = f"{START_YEAR + HORIZON_YEARS}-01-01"

# Fig A: median DG + p5-p95 band, all scenarios
fig, ax = plt.subplots(figsize=(11, 5.4))
for nm, sd in dists.items():
    med = np.array([np.percentile(sd[d], 50) for d in fan_dates]) * 100
    p5 = np.array([np.percentile(sd[d], 5) for d in fan_dates]) * 100
    p95 = np.array([np.percentile(sd[d], 95) for d in fan_dates]) * 100
    ax.plot(_x, med, color=COL[nm], marker="o", markersize=3, linewidth=2, label=nm)
    ax.fill_between(_x, p5, p95, color=COL[nm], alpha=0.08)
ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100% coverage")
ax.set_title(f"Case Study I — Forward funding ratio by allocation strategy "
             f"(median, p5–p95 band, {N_PATHS} paths)")
ax.set_xlabel("Valuation date"); ax.set_ylabel("Funding ratio (%)")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(ncol=2)
plt.tight_layout()
fA = os.path.join(OUT, "cs1_gc_median_bands.png"); fig.savefig(fA, dpi=150, bbox_inches="tight")
print("Saved", fA)

# Fig B: P(DG<100%)
fig, ax = plt.subplots(figsize=(11, 5))
for nm, sd in dists.items():
    pu = [float(np.mean(sd[d] < 1.0)) * 100 for d in fan_dates]
    ax.plot(_x, pu, color=COL[nm], marker="o", linewidth=2, label=nm)
ax.set_title(f"Case Study I — Probability of underfunding P(DG$_t$ < 100%) ({N_PATHS} paths)")
ax.set_xlabel("Valuation date"); ax.set_ylabel("P(DG < 100%) (%)")
ax.set_ylim(-2, 102); ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
plt.tight_layout()
fB = os.path.join(OUT, "cs1_gc_p_underfunded.png"); fig.savefig(fB, dpi=150, bbox_inches="tight")
print("Saved", fB)

# Fig C: weight paths BH vs Fixed-Mix vs Shift
wp_fm = scen["Fixed-Mix 40/60"].weight_path(0, _LAST)
wp_sh = scen["Shift to 60/40 (yr 5)"].weight_path(0, _LAST)
reb_fm = wp_fm[wp_fm["rebalanced"]]; reb_sh = wp_sh[wp_sh["rebalanced"]]
fig, ax = plt.subplots(figsize=(11, 5))
ax.step(reb_fm["date"], reb_fm["w_EQUITY"]*100, where="post", linewidth=2,
        color=COL["Fixed-Mix 40/60"], label="Fixed-Mix: equity")
ax.step(reb_sh["date"], reb_sh["w_EQUITY"]*100, where="post", linewidth=2,
        color=COL["Shift to 60/40 (yr 5)"], label="Shift: equity")
ax.axvline(np.datetime64(f"{START_YEAR+5}-01-01"), color="grey", linestyle=":",
           linewidth=1.5, label="shift at year 5")
ax.set_title("Case Study I — Post-rebalance equity weight: Fixed-Mix vs Shift (path 0)")
ax.set_xlabel("Rebalancing date"); ax.set_ylabel("Equity weight (%)")
ax.set_ylim(0, 100); ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
plt.tight_layout()
fC = os.path.join(OUT, "cs1_gc_weight_paths.png"); fig.savefig(fC, dpi=150, bbox_inches="tight")
print("Saved", fC)

# summary
summary = {}
for nm, sd in dists.items():
    dist_T = sd[_LAST]
    summary[nm] = {
        "mean_DG_T": float(np.mean(dist_T)),
        "p5_DG_T": float(np.percentile(dist_T, 5)),
        "p50_DG_T": float(np.percentile(dist_T, 50)),
        "P_under_T": float(np.mean(dist_T < 1.0)),
    }
sb_events = [scen["Fixed-Mix + recovery"].gc_events(i, _LAST) for i in range(N_PATHS)]
all_sb = pd.concat(sb_events, ignore_index=True)
cum_sb = np.array([float(ev["payoff"].sum()) for ev in sb_events])
summary["_recovery"] = {
    "n_paths": N_PATHS,
    "paths_with_SB": int(sum(len(ev) > 0 for ev in sb_events)),
    "total_SB": float(all_sb["payoff"].sum()) if len(all_sb) else 0.0,
    "mean_SB_per_path": float(np.mean(cum_sb)),
    "DG_T_FM": summary["Fixed-Mix 40/60"]["mean_DG_T"],
    "DG_T_recovery": summary["Fixed-Mix + recovery"]["mean_DG_T"],
    "VK_t0": VK_T0, "initial_assets": INITIAL,
}
with open(os.path.join(OUT, "cs1_gc_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print("\n=== Summary @ horizon end ({}): ===".format(_LAST))
for nm in scen:
    s = summary[nm]
    print(f"  {nm:<24} mean={s['mean_DG_T']:7.2%}  p5={s['p5_DG_T']:7.2%}  "
          f"P(DG<1)={s['P_under_T']:6.1%}")
print(f"  Recovery: {summary['_recovery']['paths_with_SB']}/{N_PATHS} paths pay SB, "
      f"total CHF {summary['_recovery']['total_SB']/1e6:,.1f}m")
print("Saved", os.path.join(OUT, "cs1_gc_summary.json"))
print("DONE.")
