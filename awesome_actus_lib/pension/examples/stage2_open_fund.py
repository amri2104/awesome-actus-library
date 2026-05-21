"""Stage 2 verification script — open-fund BVG liability engine.

Run from repo root:
    .venv/bin/python awesome_actus_lib/pension/examples/stage2_open_fund.py
    .venv/bin/python awesome_actus_lib/pension/examples/stage2_open_fund.py --no-plots

Or as a module:
    python -m awesome_actus_lib.pension.examples.stage2_open_fund
"""

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from awesome_actus_lib.pension import (
    ClosedFundSimulator,
    Cohort,
    EntryPolicy,
    OpenFundSimulator,
    PensionFund,
    PensionPolicy,
)
from awesome_actus_lib.pension.analysis.plots import plot_all_stage2

OUTPUT_DIR = Path(__file__).parent / "output"


def test_new_entrant_first_year() -> None:
    """New entrant with accrued_savings=0 generates correct events in year 1."""
    policy = PensionPolicy()
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    entry = EntryPolicy(entry_age=30, headcount=50, gross_salary=80_000.0)
    sim = OpenFundSimulator(fund, entry)
    cf = sim.run(horizon_years=1)
    df = cf.events_df

    assert set(df["contractId"].unique()) == {"ENTRY_2025_AGE30"}, \
        f"Unexpected contractIds: {set(df['contractId'].unique())}"

    insured = policy.insured_salary(80_000.0)
    hc = 50
    expected_sav = policy.savings_rate(30) * insured * hc
    expected_risk = policy.risk_contribution_rate * insured * hc
    expected_admin = -policy.admin_cost_per_member * hc

    actual_sav = df[df["type"] == "SAV_CONTRIB"]["payoff"].sum()
    actual_risk = df[df["type"] == "RISK_CONTRIB"]["payoff"].sum()
    actual_admin = df[df["type"] == "ADMIN_COST"]["payoff"].sum()
    actual_interest = df[df["type"] == "INTEREST_CREDIT"]["payoff"].sum()

    assert abs(actual_sav - expected_sav) < 1e-6, \
        f"SAV_CONTRIB mismatch: {actual_sav:.4f} vs {expected_sav:.4f}"
    assert abs(actual_risk - expected_risk) < 1e-6, \
        f"RISK_CONTRIB mismatch: {actual_risk:.4f} vs {expected_risk:.4f}"
    assert abs(actual_admin - expected_admin) < 1e-6, \
        f"ADMIN_COST mismatch: {actual_admin:.4f} vs {expected_admin:.4f}"
    assert abs(actual_interest) < 1e-9, \
        f"INTEREST_CREDIT payoff must be 0 (accrued_savings=0 at entry), got {actual_interest}"

    print(f"  [t1] SAV_CONTRIB={actual_sav:,.2f}  RISK={actual_risk:,.2f}"
          f"  ADMIN={actual_admin:,.2f}  INTEREST_payoff={actual_interest:.2f} — all correct")


def test_sav_contrib_grows_linearly() -> None:
    """Each year adds one cohort; total SAV_CONTRIB grows N*base each year."""
    policy = PensionPolicy()
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    entry = EntryPolicy(entry_age=30, headcount=10, gross_salary=80_000.0)
    sim = OpenFundSimulator(fund, entry)
    cf = sim.run(horizon_years=5)
    df = cf.events_df
    df["year"] = pd.to_datetime(df["time"]).dt.year

    insured = policy.insured_salary(80_000.0)
    # All cohorts are ages 30..34 within the 5-year window — same savings bracket (10%)
    base_sav = policy.savings_rate(30) * insured * 10

    for yr_offset in range(5):
        sim_year = 2025 + yr_offset
        n_cohorts = yr_offset + 1
        expected = n_cohorts * base_sav
        actual = df[(df["year"] == sim_year) & (df["type"] == "SAV_CONTRIB")]["payoff"].sum()
        rel_err = abs(actual - expected) / expected
        assert rel_err < 1e-6, \
            f"Year {sim_year}: expected SAV_CONTRIB={expected:.2f}, actual={actual:.2f}"

    print(f"  [t2] SAV_CONTRIB grows {base_sav:,.2f} → {5 * base_sav:,.2f} over 5 years — correct")


def test_headcount_log() -> None:
    """headcount_log records correct active/retired totals per year."""
    policy = PensionPolicy()
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    entry = EntryPolicy(entry_age=30, headcount=10, gross_salary=80_000.0)
    sim = OpenFundSimulator(fund, entry)
    sim.run(horizon_years=5)

    assert len(sim.headcount_log) == 5, \
        f"Expected 5 log entries, got {len(sim.headcount_log)}"

    for i, record in enumerate(sim.headcount_log):
        sim_year = 2025 + i
        expected_active = (i + 1) * 10
        assert record["year"] == sim_year, f"Wrong year at index {i}: {record}"
        assert record["active"] == expected_active, \
            f"Year {sim_year}: active={record['active']}, expected={expected_active}"
        assert record["retired"] == 0, \
            f"Year {sim_year}: retired={record['retired']}, expected=0"

    hc_df = pd.DataFrame(sim.headcount_log)
    assert set(hc_df.columns) >= {"year", "active", "retired"}
    print(f"  [t3] headcount_log correct: active grows 10→50, retired=0 throughout")
    print(f"       {hc_df.to_string(index=False)}")


def test_stage1_backward_compat() -> None:
    """ClosedFundSimulator AGH recurrence still exact; OpenFundSimulator does not mutate fund."""
    policy = PensionPolicy()

    # Stage 1 AGH recurrence
    fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    fund.add_cohort(Cohort(
        cohort_id="COHORT_1995",
        birth_year=1995,
        headcount=100,
        gross_salary=80_000.0,
        accrued_savings=50_000.0,
    ))
    cf = ClosedFundSimulator(fund).run(horizon_years=36)
    df = cf.events_df
    retirement_rows = df[df["type"] == "RETIREMENT_CONV"]
    assert len(retirement_rows) == 1

    insured = policy.insured_salary(80_000.0)
    agh = 50_000.0
    for age in range(30, 65):
        agh = agh * (1 + policy.applied_interest_rate) + policy.savings_rate(age) * insured
    actual_agh = retirement_rows.iloc[0]["agh_per_capita_at_conversion"]
    rel_err = abs(actual_agh - agh) / agh
    assert rel_err < 1e-6, f"Stage 1 AGH mismatch: actual={actual_agh:.4f}, expected={agh:.4f}"

    # Fund immutability: OpenFundSimulator must not mutate fund.cohorts
    fund2 = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    fund2.add_cohort(Cohort("COHORT_1990", 1990, 200, 90_000.0, 80_000.0))
    ep = EntryPolicy(entry_age=30, headcount=50, gross_salary=80_000.0)
    OpenFundSimulator(fund2, ep).run(horizon_years=10)
    assert len(fund2.cohorts) == 1, \
        f"OpenFundSimulator mutated fund.cohorts: {len(fund2.cohorts)} cohorts after run"

    print(f"  [t4] Stage 1 AGH@65={actual_agh:,.2f} (match); fund2.cohorts unchanged at 1")


def _build_plot_scenario():
    """Mixed scenario for illustrative plots: closed baseline vs open fund."""
    policy = PensionPolicy()

    closed_fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    closed_fund.add_cohort(Cohort("COHORT_1990", 1990, 200, 90_000.0, 80_000.0))
    closed_fund.add_cohort(Cohort("COHORT_1970", 1970, 150, 110_000.0, 350_000.0))
    closed_fund.add_cohort(Cohort("COHORT_1955", 1955, 80, 0.0, 600_000.0))
    closed_cf = ClosedFundSimulator(closed_fund).run(horizon_years=40)

    open_fund = PensionFund(policy=policy, start_date=date(2025, 1, 1))
    open_fund.add_cohort(Cohort("COHORT_1990", 1990, 200, 90_000.0, 80_000.0))
    open_fund.add_cohort(Cohort("COHORT_1970", 1970, 150, 110_000.0, 350_000.0))
    open_fund.add_cohort(Cohort("COHORT_1955", 1955, 80, 0.0, 600_000.0))
    entry = EntryPolicy(entry_age=30, headcount=50, gross_salary=80_000.0)
    open_sim = OpenFundSimulator(open_fund, entry)
    open_cf = open_sim.run(horizon_years=40)

    return closed_cf, open_cf, pd.DataFrame(open_sim.headcount_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 open-fund verification")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print("Stage 2 open-fund verification")
    print("-" * 60)
    print("Test 1: new entrant first-year events (accrued_savings=0 at entry)")
    test_new_entrant_first_year()
    print()
    print("Test 2: SAV_CONTRIB grows linearly as cohorts accumulate")
    test_sav_contrib_grows_linearly()
    print()
    print("Test 3: headcount_log tracks active/retired correctly")
    test_headcount_log()
    print()
    print("Test 4: Stage 1 backward compatibility and fund immutability")
    test_stage1_backward_compat()
    print()
    print("All Stage 2 checks passed.")

    if not args.no_plots:
        print()
        print("Generating Stage 2 plots...")
        closed_cf, open_cf, hc_df = _build_plot_scenario()
        plot_all_stage2(
            events_df=open_cf.events_df,
            headcount_df=hc_df,
            closed_events_df=closed_cf.events_df,
            output_dir=OUTPUT_DIR,
            prefix="stage2",
        )


if __name__ == "__main__":
    main()
