"""Stage 5a phase 1 - stochastic asset side (floaters): deterministic baseline plus Hull-White MC ALM."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from awesome_actus_lib import PAM, Portfolio, PublicActusService, ReferenceIndex
from awesome_actus_lib.stochastic_rates import CurveCalibrator, create_model
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    DynamicFundSimulator,
    Stage4Dynamics,
    EntryPolicy,
    ALMAnalysis,
    FundingRatioAnalysis,
    RetirementLossAnalysis,
    StochasticALMAnalysis,
    ek2001_2005,
)


HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("STAGE5A_N_PATHS", "200"))
SEED = 42


print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 5a — stochastic asset side (floaters)")
print("=" * 78)

print("\n[1A] Portfolio Definition — liabilities (same fund as Stage 4)")
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

# Generational EK 2001-2005: improvement applies to both projected headcounts and the ä_x valuation.
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)

print(f"  Active cohorts: 8 birth-year buckets, retirement age 65")
print(f"  Retired cohort: RETIRED_1955, 80 members, pension CHF 31,380/year")
print(f"  EntryPolicy: 20 entrants/year, age 25, salary CHF 80,000")
print(f"  Stage4Dynamics: salary_growth={dynamics.salary_growth:.2%}, "
      f"CR path={dynamics.conversion_rate_path}, "
      f"threshold_index_period={dynamics.threshold_index_period}y")
print(f"  Mortality:    EK 2001-2005 period table")
print(f"  Horizon:      {HORIZON_YEARS} years")


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

service = PublicActusService()


print("\n[2] Event Generation")
print("-" * 40)

flat_rate_df = pd.DataFrame({
    "date": [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS + 1)],
    "value": [policy.technical_rate] * (HORIZON_YEARS + 1),
})
flat_ref = ReferenceIndex(marketObjectCode=MARKET_CODE, source=flat_rate_df)

asset_cfs = service.generateEvents(asset_portfolio, riskFactors=[flat_ref])
print(f"  Asset CashFlowStream ready (deterministic baseline @ "
      f"{policy.technical_rate:.2%}): {len(asset_cfs.events_df)} events")

simulator = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=dynamics,
    mortality=mortality,
)
liability_cfs = simulator.run(horizon_years=HORIZON_YEARS)
print(f"  Liability CashFlowStream ready: {len(liability_cfs.events_df)} events")

cohort_hc_df = pd.DataFrame(simulator.cohort_headcount_log)


print("\n[3] Deterministic ALM (sticky flat discount = technical_rate)")
print("-" * 40)

alm = ALMAnalysis(
    assets_cf=asset_cfs,
    liabilities_cf=liability_cfs,
    flat_rate=policy.technical_rate,
)

net_table = alm.net_liquidity(freq="YE")
print("\n  Variant 1 — Net Liquidity (year-end, first 5 rows):")
print(net_table.head().to_string())

fr_t0_det = alm.funding_ratio(as_of="2025-01-01")
print(f"\n  Variant 2 — Net-CF diagnostic @ 2025-01-01: {fr_t0_det:.2%}")

fr_path = alm.funding_ratio_path(["2025-01-01", "2030-01-01", "2035-01-01"])
print("  Net-CF diagnostic path:")
print(fr_path.to_string())


print("\n[3A-DG] Funding ratio @ t0 (Art. 44 BVV2)")
print("-" * 40)

vv = sum(c.terms["notionalPrincipal"].value for c in asset_portfolio.contracts)
fra = FundingRatioAnalysis(
    fund=liability_fund,
    mortality=mortality,
    pension_assets=vv,
)
PC_check = fra.pension_capital_t0()
dg_t0 = fra.funding_ratio_t0()
print(f"  Pension assets   (Σ bond notionals): CHF {vv:>15,.2f}")
print(f"  Pension capital_t0:                  CHF {PC_check:>15,.2f}")
print(f"  Funding ratio_t0:                    {dg_t0:>15.2%}")


print("\n[3B] Retirement-loss analysis — baseline (5.23% -> 5.10%)")
print("-" * 40)

rla_base = RetirementLossAnalysis(
    liability_cf=liability_cfs,
    policy=policy,
    mortality=mortality,
)
loss_df = rla_base.summary()
base_total = loss_df["loss_total"].sum()
print(f"  Σ baseline loss: CHF {base_total:,.2f}")


print("\n[3C] Stochastic ALM run (Hull-White)")
print("-" * 40)

curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)
print(f"  Base curve: {curve_rates[0]:.2%} (1Y) -> {curve_rates[-1]:.2%} (10Y), "
      f"flat-forward beyond 10Y")

salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liability_cfs,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=BASE_DATE,
    market_code=MARKET_CODE,
    seed=SEED,
)
salm.run()

for as_of in ("2025-01-01", "2035-01-01"):
    s = salm.summary(as_of)
    print(f"\n  Funding ratio @ {as_of}:")
    print(f"    mean={s['mean']:.2%}  std={s['std']:.2%}  "
          f"p5={s['p5']:.2%}  p50={s['p50']:.2%}  p95={s['p95']:.2%}")
    print(f"    ES(worst 5%)={s['fr_es5']:.2%}  P(DG<100%)={s['p_underfunded']:.1%}")


print("\n[VERIFY] Inline asserts")
print("-" * 40)

dist0 = salm.funding_ratio_distribution("2025-01-01")
assert dist0.shape == (N_PATHS,), f"(a) shape: {dist0.shape}"
print(f"  (a) distribution length = {dist0.size} OK.")

assert np.std(dist0, ddof=1) > 0.0, "(b) zero dispersion with sigma>0"
print(f"  (b) dispersion > 0 (std={np.std(dist0, ddof=1):.4%}) OK.")

# (c) sigma -> 0 collapses the rate paths
hw_det = create_model("hull_white", r0=0.0176, a=0.15, sigma=1e-12,
                      calibrator=calibrator, seed=SEED)
sim_det = hw_det.simulate(T=float(HORIZON_YEARS), M=HORIZON_YEARS * 12, I=32)
assert float(sim_det.std_path()[-1]) < 1e-8, \
    f"(c) sigma->0 not deterministic: {sim_det.std_path()[-1]:.2e}"
print(f"  (c) sigma->0 collapses paths (terminal std={sim_det.std_path()[-1]:.2e}) OK.")

# (d) same seed -> identical paths
hw_a = create_model("hull_white", r0=0.0176, a=0.15, sigma=0.01,
                    calibrator=calibrator, seed=SEED)
hw_b = create_model("hull_white", r0=0.0176, a=0.15, sigma=0.01,
                    calibrator=calibrator, seed=SEED)
ra = hw_a.simulate(T=10.0, M=120, I=64).rates
rb = hw_b.simulate(T=10.0, M=120, I=64).rates
assert np.allclose(ra, rb), "(d) seed reproducibility violated"
print(f"  (d) seed reproducibility OK (max|Δ|={np.max(np.abs(ra - rb)):.2e}).")

# (e) stochastic mean near deterministic baseline @ t0 (sticky discount, small floater fraction)
mean_dg = float(np.mean(dist0))
rel_e = abs(mean_dg - fr_t0_det) / abs(fr_t0_det)
assert rel_e < 0.20, \
    f"(e) stoch mean vs det baseline diverges: {rel_e:.2%}"
print(f"  (e) stoch mean ({mean_dg:.2%}) vs det baseline ({fr_t0_det:.2%}) "
      f"rel.diff = {rel_e:.2%} OK.")


print("\n[4] Plots")
print("-" * 40)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_FIG_DIR = os.path.join(_REPO_ROOT, "output", "stage_5a")
os.makedirs(_FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


fig1, ax1 = plt.subplots(figsize=(12, 5))
years = [str(idx.year) for idx in net_table.index]
x = range(len(years))
width = 0.27
ax1.bar([i - width for i in x], net_table["netLiquidity_assets"],
        width=width, label="Assets", color="#2a9d8f")
ax1.bar(x, net_table["netLiquidity_liabilities"],
        width=width, label="Liabilities", color="#e76f51")
ax1.bar([i + width for i in x], net_table["net"],
        width=width, label="Net", color="#264653")
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_title("Stage 5a — Net Liquidity per Year (deterministic baseline)")
ax1.set_ylabel("CHF")
ax1.set_xticks(list(x))
ax1.set_xticklabels(years, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "v1_net_liquidity.png")

fig2, ax2 = plt.subplots(figsize=(12, 5))
colors = {"ACTIVE_1990": "#2a9d8f", "ACTIVE_1970": "#e9c46a", "RETIRED_1955": "#e76f51"}
for cid in ["ACTIVE_1990", "ACTIVE_1970", "RETIRED_1955"]:
    sub = cohort_hc_df[cohort_hc_df["cohort_id"] == cid]
    if not sub.empty:
        ax2.plot(sub["year"], sub["headcount"], marker="o", markersize=3,
                 label=cid, color=colors.get(cid))
ax2.set_title("Stage 5a — Cohort Headcount Decline")
ax2.set_ylabel("Members (fractional)")
ax2.set_xlabel("Year")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
plt.tight_layout()
_save(fig2, "v2_cohort_headcount_decline.png")

fig3, ax3 = plt.subplots(figsize=(11, 5))
plot_df = loss_df.sort_values("year")
ax3.plot(plot_df["year"], plot_df["applied_uws"] * 100,
         marker="o", color="#e76f51", label="Applied CR (Stage4Dynamics)")
ax3.plot(plot_df["year"], plot_df["technical_uws"] * 100,
         marker="s", color="#2a9d8f", label="Technical CR = 1/ä_x")
ax3.axhline(policy.conversion_rate * 100, color="grey", linestyle=":",
            linewidth=0.8, label=f"policy.conversion_rate ({policy.conversion_rate:.2%})")
ax3.set_title("Stage 5a — Applied vs Technical CR per Retirement Event")
ax3.set_ylabel("Conversion rate in %")
ax3.set_xlabel("Retirement year")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend()
plt.tight_layout()
_save(fig3, "v3_uws_path.png")

fig4, ax4 = plt.subplots(figsize=(11, 5))
agg = (loss_df.groupby("year")["loss_total"].sum()
       .sort_index())
ax4.bar(agg.index.astype(str), agg.values, color="#e76f51")
ax4.axhline(0, color="black", linewidth=0.6)
ax4.set_title("Stage 5a — Retirement loss per year (Σ across all cohorts)")
ax4.set_ylabel("Loss in CHF")
ax4.set_xlabel("Retirement year")
ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig4, "v4_retirement_loss_per_year.png")

fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.hist(dist0 * 100, bins=30, color="#2a9d8f", edgecolor="black", alpha=0.8)
ax5.axvline(100, color="red", linestyle="--", linewidth=1.2, label="100% coverage")
ax5.axvline(mean_dg * 100, color="#264653", linestyle="-", linewidth=1.5, label="stoch mean")
ax5.axvline(fr_t0_det * 100, color="#e76f51", linestyle="-", linewidth=1.5, label="det baseline")
ax5.set_title(f"Stage 5a — Stochastic FR distribution @ t0 ({N_PATHS} paths, Hull-White)")
ax5.set_xlabel("Funding ratio (%)")
ax5.set_ylabel("Scenarios")
ax5.legend()
ax5.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig5, "v5_stoch_fr_hist_t0.png")

fan_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 5)]
fan = salm.fan_chart_data(fan_dates)

fig6, ax6 = plt.subplots(figsize=(11, 5))
xf = fan.index
ax6.fill_between(xf, fan["p5"] * 100, fan["p95"] * 100, color="#2a9d8f", alpha=0.20, label="p5-p95")
ax6.fill_between(xf, fan["p25"] * 100, fan["p75"] * 100, color="#2a9d8f", alpha=0.40, label="p25-p75")
ax6.plot(xf, fan["p50"] * 100, color="#264653", marker="o", label="median")
ax6.axhline(100, color="red", linestyle="--", linewidth=0.9, label="100% coverage")
ax6.set_title("Stage 5a — Stochastic FR fan chart (sticky discount, stochastic floater resets)")
ax6.set_ylabel("Funding ratio (%)")
ax6.set_xlabel("Valuation date")
ax6.legend()
ax6.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig6, "v6_stoch_fr_fan.png")

plt.show()
