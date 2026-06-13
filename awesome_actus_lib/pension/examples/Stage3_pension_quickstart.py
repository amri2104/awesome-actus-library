"""Stage3_pension_quickstart.py

Stage 3 — deterministic retiree mortality (EK 2001-2005):
  - MortalityTable + ek2001_2005() factory (Swiss period table)
  - Cohort.gender + EntryPolicy.gender (opt-in, default "unisex")
  - Optional mortality= parameter on ClosedFundSimulator/OpenFundSimulator
  - cohort_headcount_log (per-cohort headcount decay over time)
"""

import os

import pandas as pd

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    ClosedFundSimulator,
    ALMAnalysis,
    ek2001_2005,
)


HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])


print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 3 — retiree mortality (EK 2001-2005)")
print("=" * 78)

# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side) — Stage 3: closed fund + mortality
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities (closed fund + mortality)")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990",
    birth_year=1990,
    headcount=200,
    gross_salary=90_000.0,
    accrued_savings=80_000.0,
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
    cohort_id="RETIRED_1955",
    birth_year=1955,
    headcount=80,
    gross_salary=0.0,
    accrued_savings=0.0,
    status="retired",
    annual_pension=31_380.0,
    gender="f",
))

mortality = ek2001_2005()

print(f"  Cohort 1: 200 members, born 1990, active, unisex, AGH/head CHF 80,000")
print(f"  Cohort 2: 150 members, born 1970, active, unisex, AGH/head CHF 350,000")
print(f"  Cohort 3:  80 members, born 1955, retired, female, pension CHF 31,380/year")
print(f"  Total members: 430  (closed fund — no new entrants)")
print(f"  Mortality:    EK 2001-2005 period table (retirees only, deterministic)")


# =============================================================================
# 1B. PORTFOLIO DEFINITION (asset side)
# =============================================================================
print("\n[1B] Portfolio Definition — assets")
print("-" * 40)

asset_portfolio = Portfolio([
    PAM(
        contractID="BOND_2030",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=40_000_000,
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
        notionalPrincipal=35_000_000,
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
        notionalPrincipal=25_000_000,
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

print(f"  Asset 1: CHF 40,000,000 bond (PAM), 2.0% coupon, matures 2030")
print(f"  Asset 2: CHF 35,000,000 bond (PAM), 2.5% coupon, matures 2040")
print(f"  Asset 3: CHF 25,000,000 bond (PAM), 3.0% coupon, matures 2050")
print(f"  Total notional: CHF 100,000,000")

service = PublicActusService()


# =============================================================================
# 2. EVENT GENERATION
# =============================================================================
print("\n[2] Event Generation")
print("-" * 40)

asset_cfs = service.generateEvents(asset_portfolio)
print(f"  Asset CashFlowStream ready: {len(asset_cfs.events_df)} events")

simulator = ClosedFundSimulator(liability_fund, mortality=mortality)
liability_cfs = simulator.run(horizon_years=HORIZON_YEARS)
print(f"  Liability CashFlowStream ready: {len(liability_cfs.events_df)} events")

# Stage 3 specific: per-cohort headcount log
cohort_hc_df = pd.DataFrame(simulator.cohort_headcount_log)
retired_first = cohort_hc_df[cohort_hc_df["cohort_id"] == "RETIRED_1955"].head(1)
retired_last = cohort_hc_df[cohort_hc_df["cohort_id"] == "RETIRED_1955"].tail(1)
print(f"  RETIRED_1955 headcount @ year 1 : {retired_first['headcount'].iloc[0]:.4f}")
print(f"  RETIRED_1955 headcount @ year 40: {retired_last['headcount'].iloc[0]:.4f}")

# =============================================================================
# 3. MERGING ASSETS AND LIABILITIES
# =============================================================================
print("\n[3] Merging assets and liabilities")
print("-" * 40)

from awesome_actus_lib.models.cashFlowStream import CashFlowStream
from awesome_actus_lib.pension.fund import _DummyPortfolio

alm = ALMAnalysis(
    assets_cf=asset_cfs,
    liabilities_cf=liability_cfs,
    flat_rate=policy.technical_rate,
)

# --- Variant 1: net liquidity per year ------------------------------------
net_table = alm.net_liquidity(freq="YE")
print("\n  Variant 1 — Net Liquidity (year-end, first 5 rows):")
print(net_table.head().to_string())

# --- Variant 2: funding ratio (PV comparison) -----------------------------
fr_t0 = alm.funding_ratio(as_of="2025-01-01")
print(f"\n  Variant 2 — Funding Ratio @ 2025-01-01: {fr_t0:.2%}")

fr_path = alm.funding_ratio_path(["2025-01-01", "2030-01-01", "2035-01-01"])
print("  Funding Ratio path:")
print(fr_path.to_string())

# --- Variant 3: combined CashFlowStream ----------------------------------
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


# =============================================================================
# 4. PLOTS
# =============================================================================
print("\n[4] Plots")
print("-" * 40)

import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_FIG_DIR = os.path.join(_REPO_ROOT, "output", "stage_3")
os.makedirs(_FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# --- Plot 1: Net Liquidity (Variant 1) -----------------------------------
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
ax1.set_title("Stage 3 — Net Liquidity per Year")
ax1.set_ylabel("CHF")
ax1.set_xticks(list(x))
ax1.set_xticklabels(years, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "v1_net_liquidity.png")

# --- Plot 2: Funding Ratio Path (Variant 2) ------------------------------
fr_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 5)]
fr_series = alm.funding_ratio_path(fr_dates)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(fr_series.index, fr_series.values, marker="o", color="#264653")
ax2.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="100% coverage")
ax2.set_title("Stage 3 — Funding Ratio Path")
ax2.set_ylabel("Funding Ratio")
ax2.set_xlabel("Valuation Date")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
plt.tight_layout()
_save(fig2, "v2_funding_ratio_path.png")

# --- Plot 3: Combined CashFlowStream (Variant 3) -------------------------
fig3 = combined_cfs.plot(title="Stage 3 — Combined Asset + Liability Cashflows",
                         return_fig=True)
if fig3 is not None:
    _save(fig3, "v3_combined_cashflows.png")

# --- Plot 4 (Stage 3 specific): Cohort headcount decline ------------------
fig4, ax4 = plt.subplots(figsize=(12, 5))
colors = {"ACTIVE_1990": "#2a9d8f", "ACTIVE_1970": "#e9c46a", "RETIRED_1955": "#e76f51"}
for cid in cohort_hc_df["cohort_id"].unique():
    sub = cohort_hc_df[cohort_hc_df["cohort_id"] == cid]
    ax4.plot(sub["year"], sub["headcount"], marker="o", markersize=3,
             label=cid, color=colors.get(cid))
ax4.set_title("Stage 3 — Cohort Headcount Decline")
ax4.set_ylabel("Members (fractional)")
ax4.set_xlabel("Year")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()
plt.tight_layout()
_save(fig4, "v4_cohort_headcount_decline.png")

plt.show()
