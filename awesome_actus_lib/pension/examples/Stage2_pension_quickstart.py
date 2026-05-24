"""Stage2_pension_quickstart.py

Working file — we build this up step by step.

Run from the repo root:
    .venv/bin/python awesome_actus_lib/pension/examples/Stage2_pension_quickstart.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: lets you run this file directly from the examples/ folder
# without installing the package.
# ---------------------------------------------------------------------------
try:
    # Asset side (plain AAL) + the analysis tools
    from awesome_actus_lib import PAM, Portfolio, PublicActusService
    from awesome_actus_lib.analysis.liquidity import LiquidityAnalysis
    from awesome_actus_lib.analysis.value import ValueAnalysis

    # Liability side (the pension extension) — the 7 user-facing classes
    from awesome_actus_lib.pension import (
        PensionPolicy,
        Cohort,
        PensionFund,
        ClosedFundSimulator,
        OpenFundSimulator,
        EntryPolicy,
        ALMAnalysis,
    )
except ImportError:
    _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from awesome_actus_lib import PAM, Portfolio, PublicActusService
    from awesome_actus_lib.analysis.liquidity import LiquidityAnalysis
    from awesome_actus_lib.analysis.value import ValueAnalysis
    from awesome_actus_lib.pension import (
        PensionPolicy,
        Cohort,
        PensionFund,
        ClosedFundSimulator,
        OpenFundSimulator,
        EntryPolicy,
        ALMAnalysis,
    )


print("=" * 70)
print("PENSION ALM CASE STUDY: BVG liabilities vs bond portfolio")
print("=" * 70)

# =============================================================================
# 1A. PORTFOLIO DEFINITION (asset side)
# =============================================================================
print("\n[1A] Portfolio Definition — assets")
print("-" * 40)

asset_portfolio = Portfolio([
    PAM(
        contractID="BOND_2030",
        statusDate="2025-01-01T00:00:00",
        contractDealDate="2025-01-01T00:00:00",
        currency="CHF",
        notionalPrincipal=40_000_000,
        initialExchangeDate="2025-01-01T00:00:00",
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
        statusDate="2025-01-01T00:00:00",
        contractDealDate="2025-01-01T00:00:00",
        currency="CHF",
        notionalPrincipal=35_000_000,
        initialExchangeDate="2025-01-01T00:00:00",
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
        statusDate="2025-01-01T00:00:00",
        contractDealDate="2025-01-01T00:00:00",
        currency="CHF",
        notionalPrincipal=25_000_000,
        initialExchangeDate="2025-01-01T00:00:00",
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
# 1B. PORTFOLIO DEFINITION (liability side)  —  Stage 2: open fund
# =============================================================================
print("\n[1B] Portfolio Definition — liabilities (Stage 2: open fund)")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date="2025-01-01T00:00:00",
)
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990",
    birth_year=1990,
    headcount=200,
    gross_salary=90_000.0,
    accrued_savings=80_000.0,
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970",
    birth_year=1970,
    headcount=150,
    gross_salary=110_000.0,
    accrued_savings=350_000.0,
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1955",
    birth_year=1955,
    headcount=80,
    gross_salary=0.0,
    accrued_savings=600_000.0,
))

# Stage 2 addition: deterministic annual new entrants
entry_policy = EntryPolicy(
    entry_age=25,
    headcount=20,
    gross_salary=80_000.0,
)

print(f"  Cohort 1: 200 members, born 1990, AGH/head CHF 80,000")
print(f"  Cohort 2: 150 members, born 1970, AGH/head CHF 350,000")
print(f"  Cohort 3:  80 members, born 1955, AGH/head CHF 600,000")
print(f"  Total starting members: 430")
print(f"  Entry policy: 20 new members/year, age 25, gross salary CHF 80,000")


# =============================================================================
# 2. EVENT GENERATION
# =============================================================================
print("\n[2] Event Generation")
print("-" * 40)

asset_cfs = service.generateEvents(asset_portfolio)
print(f"  Asset CashFlowStream ready: {len(asset_cfs.events_df)} events")

simulator = OpenFundSimulator(liability_fund, entry_policy)
liability_cfs = simulator.run(horizon_years=40)
print(f"  Liability CashFlowStream ready: {len(liability_cfs.events_df)} events")

# Stage 2 extra: headcount evolution log
headcount_df = pd.DataFrame(simulator.headcount_log)
print(f"  Headcount @ year 1 :  active={headcount_df.iloc[0]['active']:>4}, "
      f"retired={headcount_df.iloc[0]['retired']:>4}")
print(f"  Headcount @ year 40:  active={headcount_df.iloc[-1]['active']:>4}, "
      f"retired={headcount_df.iloc[-1]['retired']:>4}")

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

fr_path = alm.deckungsgrad_path(["2025-01-01", "2030-01-01", "2035-01-01"])
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

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
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
ax1.set_title("Variant 1 — Net Liquidity per Year (Stage 2: open fund)")
ax1.set_ylabel("CHF")
ax1.set_xticks(list(x))
ax1.set_xticklabels(years, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "stage2_v1_net_liquidity.png")

# --- Plot 2: Funding Ratio Path (Variant 2) ------------------------------
fr_dates = [f"{y}-01-01" for y in range(2025, 2061, 5)]
fr_series = alm.deckungsgrad_path(fr_dates)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(fr_series.index, fr_series.values, marker="o", color="#264653")
ax2.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="100% coverage")
ax2.set_title("Variant 2 — Funding Ratio Path (Stage 2: open fund)")
ax2.set_ylabel("Funding Ratio")
ax2.set_xlabel("Valuation Date")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
plt.tight_layout()
_save(fig2, "stage2_v2_funding_ratio_path.png")

# --- Plot 3: Combined CashFlowStream (Variant 3) -------------------------
fig3 = combined_cfs.plot(title="Variant 3 — Combined Asset + Liability Cashflows (Stage 2)",
                         return_fig=True)
if fig3 is not None:
    _save(fig3, "stage2_v3_combined_cashflows.png")

# --- Plot 4 (Stage 2 specific): Headcount evolution ----------------------
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(headcount_df["year"], headcount_df["active"], marker="o",
         label="Active", color="#2a9d8f")
ax4.plot(headcount_df["year"], headcount_df["retired"], marker="s",
         label="Retired", color="#e76f51")
ax4.set_title("Stage 2 — Headcount Evolution (active vs retired)")
ax4.set_ylabel("Members")
ax4.set_xlabel("Year")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()
plt.tight_layout()
_save(fig4, "stage2_v4_headcount_evolution.png")

plt.show()
