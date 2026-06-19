"""Stage 6 sensitivity variations over two going-concern levers:
equity_sigma (scenario B) and sanierung.trigger_dg (scenario D).
Same setup as the Stage 6 quickstart, with 60 paths.
"""

import os

import numpy as np

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
    ek2001_2005,
    simulate_liability_paths,
)
from awesome_actus_lib.pension.analysis.going_concern import BONDS, EQUITY

HORIZON_YEARS = 30
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])
BASE_DATE = "2025-01-01"
MARKET_CODE = "IR_SCENARIO"
N_PATHS = int(os.environ.get("STAGE6_VARIATIONS_N_PATHS", "60"))
SEED_ASSETS = 42
SEED_LIAB = 4242
SEED_EQ = 777
EQ_RETURN = 0.03
BOND_YIELD = 0.02

EQ_SIGMA_GRID = (0.05, 0.10, 0.15)
TRIGGER_GRID = (0.95, 1.00)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "output", "stage_6")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 78)
print(f"STAGE 6 - sensitivities (N_PATHS={N_PATHS})")
print("=" * 78)


policy = PensionPolicy()
liability_fund = PensionFund(policy=policy, start_date=START_DATE)
for c in (
    Cohort(cohort_id="ACTIVE_1960", birth_year=1960, headcount=90,
           gross_salary=120_000.0, accrued_savings=600_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1963", birth_year=1963, headcount=100,
           gross_salary=115_000.0, accrued_savings=520_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1966", birth_year=1966, headcount=110,
           gross_salary=115_000.0, accrued_savings=450_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1970", birth_year=1970, headcount=150,
           gross_salary=110_000.0, accrued_savings=350_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1972", birth_year=1972, headcount=130,
           gross_salary=108_000.0, accrued_savings=320_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1975", birth_year=1975, headcount=140,
           gross_salary=105_000.0, accrued_savings=270_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1978", birth_year=1978, headcount=140,
           gross_salary=100_000.0, accrued_savings=220_000.0, gender="unisex"),
    Cohort(cohort_id="ACTIVE_1990", birth_year=1990, headcount=200,
           gross_salary=90_000.0, accrued_savings=80_000.0, gender="unisex"),
    Cohort(cohort_id="RETIRED_1955", birth_year=1955, headcount=80,
           gross_salary=0.0, accrued_savings=0.0, status="retired",
           annual_pension=31_380.0, gender="f"),
):
    liability_fund.add_cohort(c)

entry_policy = EntryPolicy(entry_age=25, headcount=20,
                           gross_salary=80_000.0, gender="unisex")
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={2025: 0.0523, 2029: 0.0510},
    threshold_index_period=5,
)
mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)

VK_T0 = FundingRatioAnalysis(liability_fund, mortality, 0.0).pension_capital_t0()
INITIAL_ASSETS = 1.076 * VK_T0
BOND_BOOK = 0.30 * INITIAL_ASSETS
CASH_BOOK = 0.10 * INITIAL_ASSETS
EQUITY_SLEEVE_0 = 0.60 * INITIAL_ASSETS
print(f"[Setup] VK_t0 = CHF {VK_T0:,.0f}, initial_assets = 1.076 x VK_t0")

_INTEREST_ANCHOR = "2026-01-01T00:00:00"
asset_portfolio = Portfolio([
    PAM(
        contractID="FIX_2040", statusDate=START_DATE,
        contractDealDate=START_DATE, currency="CHF",
        notionalPrincipal=BOND_BOOK, initialExchangeDate=START_DATE,
        maturityDate="2040-12-31T00:00:00", nominalInterestRate=0.020,
        dayCountConvention="30E360", contractRole="RPA",
        cycleOfInterestPayment="P1YL1", counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="CASH_ACCOUNT", statusDate=START_DATE,
        contractDealDate=START_DATE, currency="CHF",
        notionalPrincipal=CASH_BOOK, initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00", nominalInterestRate=0.0,
        dayCountConvention="30E360", contractRole="RPA",
        cycleAnchorDateOfRateReset=START_DATE, cycleOfRateReset="P1YL0",
        marketObjectCodeOfRateReset=MARKET_CODE, rateSpread=0.0,
        cycleAnchorDateOfInterestPayment=_INTEREST_ANCHOR,
        cycleOfInterestPayment="P1YL0", counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
])

calibrator = CurveCalibrator.from_market(
    times=np.array([1, 2, 3, 4, 5, 7, 10], dtype=float),
    spot_rates=np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021],
                        dtype=float),
)

print(f"[Setup] Liability paths (n={N_PATHS}) ...")
liab_paths = simulate_liability_paths(
    fund=liability_fund, entry_policy=entry_policy, dynamics=dynamics,
    mortality=mortality, horizon_years=HORIZON_YEARS, n_paths=N_PATHS,
    seed=SEED_LIAB, mortality_filter="all", return_headcount_log=True,
)
print(f"[Setup] StochasticALM (Hull-White + GBM, n={N_PATHS}) ...")
salm = StochasticALMAnalysis(
    asset_portfolio=asset_portfolio,
    liabilities_cf=liab_paths[0].cashflows,
    calibrator=calibrator,
    technical_rate=policy.technical_rate,
    model="hull_white",
    model_params={"r0": 0.0176, "a": 0.15, "sigma": 0.01},
    equity_params={"S0": 100.0, "mu": EQ_RETURN, "sigma": 0.10},
    n_paths=N_PATHS,
    horizon_years=HORIZON_YEARS,
    steps_per_year=12,
    base_date=BASE_DATE,
    market_code=MARKET_CODE,
    seed=SEED_ASSETS,
    verbose=False,
)
salm.run()

_COMMON = dict(
    fund=liability_fund, mortality=mortality,
    bond_book=BOND_BOOK, cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0, liab_paths=liab_paths,
)
_FIX_MIX = {BONDS: 0.40, EQUITY: 0.60}
D_T = f"{START_YEAR + HORIZON_YEARS}-01-01"


def _kpis(scen):
    dist = scen.funding_ratio_distribution(D_T)
    return (float(np.mean(dist)), float(np.percentile(dist, 5)),
            float(np.mean(dist < 1.0)))


# Variation 1: equity_sigma, scenario B (fixed-mix 40/60)
print("\n[Var 1] equity_sigma - scenario B")
rows_sigma = []
for sig in EQ_SIGMA_GRID:
    scen = GoingConcernFundingRatioAnalysis(
        salm, **_COMMON,
        rebalancing=RebalancingPolicy(target_weights=dict(_FIX_MIX)),
        equity_return=EQ_RETURN, equity_sigma=sig, equity_seed=SEED_EQ,
        bond_yield=BOND_YIELD,
    )
    m, p5, pu = _kpis(scen)
    rows_sigma.append((sig, m, p5, pu))
    print(f"  equity_sigma={sig:.2f}: mean={m:7.2%}  p5={p5:7.2%}  "
          f"P(DG<1)={pu:6.1%}")

# Variation 2: sanierung.trigger_dg, scenario D (B + recovery contribution)
print("\n[Var 2] trigger_dg - scenario D")
rows_trigger = []
for trig in TRIGGER_GRID:
    scen = GoingConcernFundingRatioAnalysis(
        salm, **_COMMON,
        rebalancing=RebalancingPolicy(target_weights=dict(_FIX_MIX)),
        equity_return=EQ_RETURN, equity_sigma=0.10, equity_seed=SEED_EQ,
        bond_yield=BOND_YIELD,
        sanierung=SanierungsPolicy(trigger_dg=trig, exit_dg=trig,
                                   sb_factor=0.5),
    )
    m, p5, pu = _kpis(scen)
    rows_trigger.append((trig, m, p5, pu))
    print(f"  trigger_dg={trig:.2f}: mean={m:7.2%}  p5={p5:7.2%}  "
          f"P(DG<1)={pu:6.1%}")

# Write the sensitivities markdown table.
lines = []
lines.append("# Stage 6 - sensitivities\n")
lines.append(f"- Script: `examples/Stage6_variations.py`, "
             f"**N_PATHS = {N_PATHS}**, horizon {HORIZON_YEARS} years, "
             f"KPIs at {D_T}")
lines.append(f"- Setup identical to the quickstart (seeds {SEED_ASSETS}/"
             f"{SEED_LIAB}/{SEED_EQ}, DG_t0 = 107.6%), existing "
             f"public API only")
lines.append("- Note: smaller run than the canonical 200 - numbers "
             "are not 1:1 comparable with `summary_kpis.md`\n")

lines.append("## equity_sigma - scenario B (fixed-mix 40/60)\n")
lines.append("| equity_sigma | mean DG | p5 | P(DG<100%) |")
lines.append("|---|---:|---:|---:|")
for sig, m, p5, pu in rows_sigma:
    base = "  *(base case B)*" if abs(sig - 0.10) < 1e-12 else ""
    lines.append(f"| {sig:.2f}{base} | {m:.1%} | {p5:.1%} | {pu:.1%} |")

lines.append("\n## sanierung.trigger_dg - scenario D (B + recovery, "
             "sb_factor=0.5)\n")
lines.append("| trigger_dg | mean DG | p5 | P(DG<100%) |")
lines.append("|---|---:|---:|---:|")
for trig, m, p5, pu in rows_trigger:
    base = "  *(base case D)*" if abs(trig - 1.00) < 1e-12 else ""
    lines.append(f"| {trig:.2f}{base} | {m:.1%} | {p5:.1%} | {pu:.1%} |")

_path = os.path.join(OUT_DIR, "sensitivities.md")
with open(_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nSaved: {_path}")
