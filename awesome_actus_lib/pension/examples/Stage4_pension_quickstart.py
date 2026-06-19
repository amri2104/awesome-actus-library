"""Stage 4 deterministic time-varying parameters: salary growth, falling conversion-rate path, threshold indexation, RetirementLossAnalysis (applied CR vs technical CR = 1/ä_x)."""

import os

import pandas as pd

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.analysis.liquidity import LiquidityAnalysis
from awesome_actus_lib.analysis.value import ValueAnalysis
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    ClosedFundSimulator,
    OpenFundSimulator,
    DynamicFundSimulator,
    Stage4Dynamics,
    EntryPolicy,
    ALMAnalysis,
    FundingRatioAnalysis,
    RetirementLossAnalysis,
    annuity_due,
    technical_uws,
    MortalityTable,
    ek2001_2005,
)


HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])


print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 4 — dynamic salary + CR path + KPI")
print("=" * 78)

print("\n[1A] Portfolio Definition — liabilities (dynamic parameters)")
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
    conversion_rate_path={
        2025: 0.0523,
        2029: 0.0510,
    },
    threshold_index_period=5,
)

# Generational projection lifts ä_x to a realistic level (fair CR ~4.9% @65), so the baseline path produces a loss instead of the period table's artefactual gain.
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)

print(f"  Active cohorts: 8 birth-year buckets, retirement age 65")
print(f"  Retired cohort: RETIRED_1955, 80 members, pension CHF 31,380/year")
print(f"  EntryPolicy: 20 entrants/year, age 25, salary CHF 80,000")
print(f"  Stage4Dynamics: salary_growth={dynamics.salary_growth:.2%}, "
      f"CR path={dynamics.conversion_rate_path}, "
      f"threshold_index_period={dynamics.threshold_index_period}y")
print(f"  Mortality:    EK 2001-2005 generational (improvement 1.25%/yr from 2003)")
print(f"  Horizon:      {HORIZON_YEARS} years")


print("\n[1B] Portfolio Definition — assets")
print("-" * 40)

asset_portfolio = Portfolio([
    PAM(
        contractID="BOND_2030",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=160_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00",
        nominalInterestRate=0.020,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM"
    ),
    PAM(
        contractID="BOND_2040",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=140_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2040-12-31T00:00:00",
        nominalInterestRate=0.025,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM"
    ),
    PAM(
        contractID="BOND_2050",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=100_000_000,
        initialExchangeDate=START_DATE,
        maturityDate="2050-12-31T00:00:00",
        nominalInterestRate=0.030,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM"
    )
])

print(f"  Asset 1: CHF 160,000,000 bond (PAM), 2.0% coupon, matures 2030")
print(f"  Asset 2: CHF 140,000,000 bond (PAM), 2.5% coupon, matures 2040")
print(f"  Asset 3: CHF 100,000,000 bond (PAM), 3.0% coupon, matures 2050")
print(f"  Total notional: CHF 400,000,000")

service = PublicActusService()


print("\n[2] Event Generation")
print("-" * 40)

asset_cfs = service.generateEvents(asset_portfolio)
print(f"  Asset CashFlowStream ready: {len(asset_cfs.events_df)} events")

simulator = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=dynamics,
    mortality=mortality,
)
liability_cfs = simulator.run(horizon_years=HORIZON_YEARS)
print(f"  Liability CashFlowStream ready: {len(liability_cfs.events_df)} events")

cohort_hc_df = pd.DataFrame(simulator.cohort_headcount_log)

print("\n[3] Merging assets and liabilities")
print("-" * 40)

from awesome_actus_lib.models.cashFlowStream import CashFlowStream
from awesome_actus_lib.pension.fund import _DummyPortfolio

alm = ALMAnalysis(
    assets_cf=asset_cfs,
    liabilities_cf=liability_cfs,
    flat_rate=policy.technical_rate,
)

net_table = alm.net_liquidity(freq="YE")
print("\n  Variant 1 — Net Liquidity (year-end, first 5 rows):")
print(net_table.head().to_string())

fr_t0 = alm.funding_ratio(as_of="2025-01-01")
print(f"\n  Variant 2 — Net-CF diagnostic @ 2025-01-01: {fr_t0:.2%}")

fr_path = alm.funding_ratio_path(["2025-01-01", "2030-01-01", "2035-01-01"])
print("  Net-CF diagnostic path:")
print(fr_path.to_string())


def _df_to_raw_response(events_df):
    raw = []
    for cid, grp in events_df.groupby("contractId"):
        events = grp.drop(columns=["contractId"]).to_dict(orient="records")
        raw.append({"contractID": cid, "events": events, "children": []})
    return raw


def merge_cashflow_streams(asset_cfs, liability_cfs):
    combined_raw = (
        _df_to_raw_response(asset_cfs.events_df)
        + _df_to_raw_response(liability_cfs.events_df)
    )
    combined_contracts = (
        list(asset_cfs.portfolio.contracts)
        + list(liability_cfs.portfolio.contracts)
    )
    return CashFlowStream(
        portfolio=_DummyPortfolio(combined_contracts),
        riskFactors=asset_cfs.riskFactors,
        raw_response=combined_raw,
    )


combined_cfs = merge_cashflow_streams(asset_cfs, liability_cfs)
print(f"\n  Variant 3 — Combined CashFlowStream: {len(combined_cfs.events_df)} events "
      f"across {len(combined_cfs.portfolio)} contracts/cohorts")

# Art. 44 BVV2 funding ratio at t0, computed from the fund rather than from events.
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

print(f"\n  Breakdown per cohort:")
breakdown = fra.breakdown()
print(breakdown.to_string(index=False,
                          float_format=lambda x: f"{x:,.4f}"))

print("\n[3B] Retirement-loss analysis — baseline (5.23% -> 5.10%)")
print("-" * 40)

rla_base = RetirementLossAnalysis(
    liability_cf=liability_cfs,
    policy=policy,
    mortality=mortality,
)
loss_df = rla_base.summary()
print(loss_df.to_string(index=False,
                           float_format=lambda x: f"{x:,.4f}"))
base_total = loss_df["loss_total"].sum()
print(f"\n  Σ baseline loss: CHF {base_total:,.2f}")

print("\n[3C] Retirement-loss analysis — stress (6.8% flat, no offset)")
print("-" * 40)

stress_dynamics = Stage4Dynamics(
    salary_growth=dynamics.salary_growth,
    conversion_rate_path={START_YEAR: 0.068},
    threshold_index_period=dynamics.threshold_index_period,
)
stress_sim = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=stress_dynamics,
    mortality=mortality,
)
stress_cf = stress_sim.run(horizon_years=HORIZON_YEARS)
rla_stress = RetirementLossAnalysis(
    liability_cf=stress_cf,
    policy=policy,
    mortality=mortality,
)
stress_df = rla_stress.summary()
stress_total = stress_df["loss_total"].sum()
print(f"  Σ stress loss:   CHF {stress_total:,.2f}")
print(f"  Σ baseline loss: CHF {base_total:,.2f}")
print(f"  Difference:      CHF {stress_total - base_total:,.2f}")


print("\n[VERIFY] Inline asserts")
print("-" * 40)

# (a) annuity_due reference values
synthetic = MortalityTable(
    q_male={60: 0.0, 61: 0.0, 62: 0.0},
    q_female={60: 0.0, 61: 0.0, 62: 0.0},
)
a_ref = annuity_due(60, "m", 0.0, synthetic, 62)
assert abs(a_ref - 3.0) < 1e-12, f"(a) annuity_due reference failed: {a_ref}"

a_deg = annuity_due(100, "m", 0.0176, ek2001_2005(), 100)
assert abs(a_deg - 1.0) < 1e-12, f"(a2) degenerate annuity_due failed: {a_deg}"

tbl = MortalityTable(
    q_male={0: 0.5, 1: 0.0},
    q_female={0: 0.5, 1: 0.0},
)
a3 = annuity_due(0, "m", 0.10, tbl, 1)
assert abs(a3 - 1.4545454545454546) < 1e-12, f"(a3) annuity_due failed: {a3}"
print(f"  (a) annuity_due references OK.")

# (b) DynamicFundSimulator with neutral dynamics must equal OpenFundSimulator
stage4_neutral_dynamics = Stage4Dynamics(
    salary_growth=0.0,
    conversion_rate_path=None,
    threshold_index_period=1,
)
sim_stage4_neutral = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=stage4_neutral_dynamics,
    mortality=mortality,
)
cf_stage4_neutral = sim_stage4_neutral.run(horizon_years=HORIZON_YEARS)
sim_stage3 = OpenFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    mortality=mortality,
)
cf_stage3 = sim_stage3.run(horizon_years=HORIZON_YEARS)

sum_s4 = float(cf_stage4_neutral.events_df["payoff"].sum())
sum_s3 = float(cf_stage3.events_df["payoff"].sum())
rel = abs(sum_s4 - sum_s3) / (abs(sum_s3) + 1e-30)
assert rel < 1e-9, f"(b) Stage-3 identity violated (rel. diff {rel:.2e})"
print(f"  (b) Stage-3 identity OK (rel. diff {rel:.2e}).")

violations = stress_df[
    (stress_df["applied_uws"] > stress_df["technical_uws"])
    & (stress_df["loss_per_capita"] <= 0.0)
]
assert violations.empty, f"(c) sign violation in stress"
assert stress_total > 0, f"(c2) stress total not positive: {stress_total}"
assert base_total <= stress_total, f"(c3) baseline must be <= stress"
print(f"  (c) Stress sign + ordering OK.")

solo_fund = PensionFund(policy=policy, start_date=START_DATE)
solo_fund.add_cohort(Cohort(
    cohort_id="SOLO",
    birth_year=1990,
    headcount=10,
    gross_salary=80_000.0,
    accrued_savings=100_000.0,
    gender="unisex",
))
pc_solo = FundingRatioAnalysis(
    fund=solo_fund,
    mortality=None,
    pension_assets=0.0,
).pension_capital_t0()
assert pc_solo == 1_000_000.0, f"(d) single-cohort PC failed: {pc_solo}"
print(f"  (d) Single-cohort PC = {pc_solo:,.2f} OK.")

assert dg_t0 > 0 and dg_t0 < 10, f"(e) funding ratio_t0 unrealistic: {dg_t0:.4f}"
print(f"  (e) Funding ratio_t0 = {dg_t0:.4%}. OK.")

# (f) retiree pc_contribution must equal pension * headcount * annuity-due
ret_row = breakdown[breakdown["cohort_id"] == "RETIRED_1955"].iloc[0]
ret_cohort = next(c for c in liability_fund.cohorts if c.cohort_id == "RETIRED_1955")
assert ret_row["annuity_due"] > 1.0
expected_vk = ret_cohort.annual_pension * ret_cohort.headcount * ret_row["annuity_due"]
rel_f = abs(ret_row["pc_contribution"] - expected_vk) / abs(expected_vk)
assert rel_f < 1e-9
print(f"  (f) Retiree pc_contribution matches pension*hc*ä OK.")


print("\n[4] Plots")
print("-" * 40)

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_FIG_DIR = os.path.join(_REPO_ROOT, "output", "stage_4")
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
ax1.set_title("Stage 4 — Net Liquidity per Year")
ax1.set_ylabel("CHF")
ax1.set_xticks(list(x))
ax1.set_xticklabels(years, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "v1_net_liquidity.png")

fr_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 5)]
fr_series = alm.funding_ratio_path(fr_dates)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(fr_series.index, fr_series.values, marker="o", color="#264653")
ax2.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="100% coverage")
ax2.set_title("Stage 4 — Funding Ratio Path")
ax2.set_ylabel("Funding Ratio")
ax2.set_xlabel("Valuation Date")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
plt.tight_layout()
_save(fig2, "v2_funding_ratio_path.png")

fig3 = combined_cfs.plot(title="Stage 4 — Combined Asset + Liability Cashflows",
                         return_fig=True)
if fig3 is not None:
    _save(fig3, "v3_combined_cashflows.png")

fig4, ax4 = plt.subplots(figsize=(12, 5))
colors = {"ACTIVE_1990": "#2a9d8f", "ACTIVE_1970": "#e9c46a", "RETIRED_1955": "#e76f51"}
for cid in ["ACTIVE_1990", "ACTIVE_1970", "RETIRED_1955"]:
    sub = cohort_hc_df[cohort_hc_df["cohort_id"] == cid]
    if not sub.empty:
        ax4.plot(sub["year"], sub["headcount"], marker="o", markersize=3,
                 label=cid, color=colors.get(cid))
ax4.set_title("Stage 4 — Cohort Headcount Decline")
ax4.set_ylabel("Members (fractional)")
ax4.set_xlabel("Year")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()
plt.tight_layout()
_save(fig4, "v4_cohort_headcount_decline.png")

fig5, ax5 = plt.subplots(figsize=(11, 5))
plot_df = loss_df.sort_values("year")
ax5.plot(plot_df["year"], plot_df["applied_uws"] * 100,
         marker="o", color="#e76f51", label="Applied CR (Stage4Dynamics)")
ax5.plot(plot_df["year"], plot_df["technical_uws"] * 100,
         marker="s", color="#2a9d8f", label="Technical CR = 1/ä_x")
ax5.axhline(policy.conversion_rate * 100, color="grey", linestyle=":",
            linewidth=0.8, label=f"policy.conversion_rate ({policy.conversion_rate:.2%})")
ax5.set_title("Stage 4 — Applied vs Technical CR per Retirement Event")
ax5.set_ylabel("Conversion rate in %")
ax5.set_xlabel("Retirement year")
ax5.grid(True, linestyle="--", alpha=0.5)
ax5.legend()
plt.tight_layout()
_save(fig5, "v5_uws_path.png")

fig6, ax6 = plt.subplots(figsize=(11, 5))
agg = (loss_df.groupby("year")["loss_total"].sum()
       .sort_index())
ax6.bar(agg.index.astype(str), agg.values, color="#e76f51")
ax6.axhline(0, color="black", linewidth=0.6)
ax6.set_title("Stage 4 — Retirement loss per year (Σ across all cohorts)")
ax6.set_ylabel("Loss in CHF")
ax6.set_xlabel("Retirement year")
ax6.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig6, "v6_retirement_loss_per_year.png")

plt.show()
