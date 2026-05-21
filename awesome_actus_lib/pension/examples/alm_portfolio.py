"""ALM portfolio example: PAM bond ladder + BVG liability engine.

Defines a 5-bond CHF maturity ladder as the asset side, runs it through the
ACTUS server, and combines it with Stage 1 (closed) and Stage 2 (open) liability
streams via ALMAnalysis.  Prints funding ratio, Deckungsgrad path, and net
liquidity; optionally saves plots.

Run from repo root:
    .venv/bin/python awesome_actus_lib/pension/examples/alm_portfolio.py
    .venv/bin/python awesome_actus_lib/pension/examples/alm_portfolio.py --stage 1
    .venv/bin/python awesome_actus_lib/pension/examples/alm_portfolio.py --stage 2
    .venv/bin/python awesome_actus_lib/pension/examples/alm_portfolio.py --no-plots

Requires network access to the ZHAW public ACTUS server.
"""

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.pension import (
    ALMAnalysis,
    ClosedFundSimulator,
    Cohort,
    EntryPolicy,
    OpenFundSimulator,
    PensionFund,
    PensionPolicy,
)
from awesome_actus_lib.pension.analysis.plots import plot_alm_all

OUTPUT_DIR = Path(__file__).parent / "output"

# BVG technical rate — used for Deckungsgrad (funding ratio) discounting
FLAT_RATE = 0.0176

# Deckungsgrad snapshot dates
DG_DATES = [
    "2025-01-01", "2030-01-01", "2035-01-01",
    "2040-01-01", "2050-01-01", "2060-01-01",
]


def _build_asset_portfolio() -> Portfolio:
    """CHF PAM bond ladder calibrated to ~CHF 115M total notional.

    Five fixed-coupon bullet bonds spanning maturities 2027–2050.
    contractRole='RPA' (Receive Positive Amounts) — fund holds the bonds.
    cycleOfInterestPayment='P1YL1' generates annual coupon IP events.
    """
    status = "2025-01-01T00:00:00"
    common = dict(
        contractDealDate=status,
        statusDate=status,
        initialExchangeDate=status,
        contractRole="RPA",
        counterpartyID="SNB",
        creatorID="PK_ALM",
        currency="CHF",
        dayCountConvention="30E360",
        cycleOfInterestPayment="P1YL1",
    )
    bonds = [
        PAM(contractID="BOND_2027", maturityDate="2027-12-31T00:00:00",
            notionalPrincipal=15_000_000.0, nominalInterestRate=0.020, **common),
        PAM(contractID="BOND_2030", maturityDate="2030-12-31T00:00:00",
            notionalPrincipal=25_000_000.0, nominalInterestRate=0.022, **common),
        PAM(contractID="BOND_2035", maturityDate="2035-12-31T00:00:00",
            notionalPrincipal=30_000_000.0, nominalInterestRate=0.025, **common),
        PAM(contractID="BOND_2040", maturityDate="2040-12-31T00:00:00",
            notionalPrincipal=25_000_000.0, nominalInterestRate=0.028, **common),
        PAM(contractID="BOND_2050", maturityDate="2050-12-31T00:00:00",
            notionalPrincipal=20_000_000.0, nominalInterestRate=0.031, **common),
    ]
    return Portfolio(bonds)


def _build_liability_stage1(policy: PensionPolicy):
    """Stage 1 (closed fund): 3 cohorts, 40-year horizon."""
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    fund.add_cohort(Cohort("COHORT_1990", 1990, 200, 90_000.0, 80_000.0))
    fund.add_cohort(Cohort("COHORT_1970", 1970, 150, 110_000.0, 350_000.0))
    fund.add_cohort(Cohort("COHORT_1955", 1955, 80, 0.0, 600_000.0))
    return ClosedFundSimulator(fund).run(horizon_years=40)


def _build_liability_stage2(policy: PensionPolicy):
    """Stage 2 (open fund): same 3 cohorts + 50 new entrants/yr at age 30."""
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    fund.add_cohort(Cohort("COHORT_1990", 1990, 200, 90_000.0, 80_000.0))
    fund.add_cohort(Cohort("COHORT_1970", 1970, 150, 110_000.0, 350_000.0))
    fund.add_cohort(Cohort("COHORT_1955", 1955, 80, 0.0, 600_000.0))
    entry = EntryPolicy(entry_age=30, headcount=50, gross_salary=80_000.0)
    sim = OpenFundSimulator(fund, entry)
    cf = sim.run(horizon_years=40)
    return cf


def _print_alm_kpis(assets_cf, liabilities_cf, label: str) -> dict:
    """Run ALMAnalysis and print key KPIs. Returns result dict for plotting."""
    alm = ALMAnalysis(assets_cf, liabilities_cf, flat_rate=FLAT_RATE)

    fr = alm.funding_ratio("2025-01-01")
    net_liq = alm.net_liquidity(freq="YE")
    dg_path = alm.deckungsgrad_path(DG_DATES)

    print(f"\n  --- {label} ---")
    print(f"  Funding ratio (2025-01-01): {fr:.1%}")
    print(f"  Deckungsgrad path:")
    for d, v in dg_path.items():
        flag = " ⚠ UNDERCOVER" if v < 1.0 else ""
        print(f"    {pd.to_datetime(d).year}: {v:.1%}{flag}")

    # Net liquidity summary: first 10 years
    print(f"  Net liquidity (first 10 years):")
    nl10 = net_liq[["netLiquidity_assets", "netLiquidity_liabilities", "net"]].head(10)
    nl10.index = pd.to_datetime(nl10.index).year
    nl10.index.name = "year"
    print(nl10.to_string())

    return {"alm": alm, "net_liq": net_liq, "dg_path": dg_path, "fr": fr}


def main() -> None:
    parser = argparse.ArgumentParser(description="ALM portfolio: PAM bonds + BVG liabilities")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="Which liability stage to analyse (default: both)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    total_notional = 15_000_000 + 25_000_000 + 30_000_000 + 25_000_000 + 20_000_000

    print("ALM portfolio analysis")
    print("-" * 60)
    print(f"Asset portfolio : 5 PAM bonds, total notional CHF {total_notional:,.0f}")
    print(f"Discount rate   : {FLAT_RATE:.2%} (BVG technical rate)")
    print("Generating asset cashflows via PublicActusService...")

    service = PublicActusService()
    asset_portfolio = _build_asset_portfolio()
    assets_cf = service.generateEvents(asset_portfolio)

    n_asset_events = len(assets_cf.events_df)
    print(f"  Asset events generated: {n_asset_events} rows")

    policy = PensionPolicy()

    if args.stage in ("1", "both"):
        print("\nBuilding Stage 1 (closed fund) liabilities...")
        liab_cf = _build_liability_stage1(policy)
        r1 = _print_alm_kpis(assets_cf, liab_cf, "Stage 1 — Closed Fund")
        if not args.no_plots:
            plot_alm_all(r1["net_liq"], r1["dg_path"], OUTPUT_DIR, prefix="alm_stage1")

    if args.stage in ("2", "both"):
        print("\nBuilding Stage 2 (open fund) liabilities...")
        liab_cf = _build_liability_stage2(policy)
        r2 = _print_alm_kpis(assets_cf, liab_cf, "Stage 2 — Open Fund (+50 entrants/yr)")
        if not args.no_plots:
            plot_alm_all(r2["net_liq"], r2["dg_path"], OUTPUT_DIR, prefix="alm_stage2")

    print("\nDone.")


if __name__ == "__main__":
    main()
