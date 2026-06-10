"""Stage5d_going_concern_quickstart.py

Stage 5d (Baustein 2) — Allokations-Shift-Experiment auf dem going-concern
Asset-Roll (Spec: docs/stage5d_going_concern_spec.md):

  A  Legacy 5c: kein Reinvestment, Buy-and-hold (Referenz / Regression)
  B  Reinvest + Fixed-Mix 40% BONDS / 60% EQUITY, jaehrlich, equity_sigma=0.10
  C  wie B, Shift auf 60/40 ab Jahr 5 ("mortgage->bonds"-Beispiel)

Kalibrierung per Spec: Stage-4-Bestand (8 aktive Kohorten + RETIRED_1955,
EntryPolicy 20/Jahr), initial_assets = 1.076 x VK_t0 (Complementa YE 2023),
Start-Allokation 40/60 = Zielmix von B.

Szenario D (Sanierung) folgt mit Baustein 3.
"""

import os

import numpy as np
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
N_PATHS = int(os.environ.get("STAGE5D_N_PATHS", "60"))
SEED_ASSETS = 42
SEED_LIAB = 4242
SEED_EQ = 777

# Equity-Annahmen konsistent ueber alle Szenarien: A nutzt die GBM-MTM-Linse
# von 5c, B/C den Buchwert-Roll — gleiche Drift/Vol, damit der Vergleich
# Reinvestment + Rebalancing isoliert und nicht Return-Annahmen.
EQ_RETURN = 0.03
EQ_SIGMA = 0.10

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO_ROOT, "output", "stage_5d")

print("=" * 78)
print("PENSION ALM CASE STUDY — Stage 5d — going-concern Allokations-Experiment")
print("=" * 78)


# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side) — Stage-4-Bestand (Spec)
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities (Stage-4-Bestand)")
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
print("  8 aktive Kohorten + RETIRED_1955, EntryPolicy 20/y, "
      "EK 2001-2005 (generational)")


# =============================================================================
# 1B. KALIBRIERUNG — initial_assets = 1.076 x VK_t0 (Complementa YE 2023)
# =============================================================================
print("\n[1B] Kalibrierung — DG_t0 = 107.6%")
print("-" * 40)

VK_T0 = FundingRatioAnalysis(liability_fund, mortality, 0.0).pension_capital_t0()
INITIAL_ASSETS = 1.076 * VK_T0

# Start-Allokation = Zielmix von Szenario B: 40% fixed income (Bonds+Cash,
# im Roll der BONDS-Bucket) / 60% Equity. So startet der GC-Roll on-target.
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


# =============================================================================
# 2. CURVE CALIBRATION
# =============================================================================
print("\n[2] Curve calibration")
print("-" * 40)
curve_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
curve_rates = np.array([0.014, 0.015, 0.016, 0.017, 0.0176, 0.019, 0.021], dtype=float)
calibrator = CurveCalibrator.from_market(times=curve_tenors, spot_rates=curve_rates)


# =============================================================================
# 3. STOCHASTIC LIABILITY PATHS (with headcount trace)
# =============================================================================
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


# =============================================================================
# 4. STOCHASTIC ASSETS (Hull-White + GBM) — gemeinsame Basis aller Szenarien
# =============================================================================
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


# =============================================================================
# 5. SZENARIEN A / B / C
# =============================================================================
print("\n[5] Szenarien")
print("-" * 40)

_COMMON = dict(
    fund=liability_fund,
    mortality=mortality,
    bond_book=BOND_BOOK,
    cash_book=CASH_BOOK,
    equity_sleeve_0=EQUITY_SLEEVE_0,
    liab_paths=liab_paths,
)

scen_A = StochasticFundingRatioAnalysis(salm=salm, **_COMMON)

scen_B = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON,
    rebalancing=RebalancingPolicy(target_weights={BONDS: 0.40, EQUITY: 0.60}),
    equity_return=EQ_RETURN,
    equity_sigma=EQ_SIGMA,
    equity_seed=SEED_EQ,
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
)

scenarios = {
    "A (Legacy 5c, buy-and-hold)": scen_A,
    "B (Fix-Mix 40/60)": scen_B,
    "C (Shift 60/40 ab Jahr 5)": scen_C,
}
for name in scenarios:
    print(f"  {name}")


# =============================================================================
# VERIFY — Spec-Asserts (Stil wie Stage 1)
# =============================================================================
print("\n[VERIFY] Spec-Asserts")
print("-" * 40)

# ---- Spec-Assert 1: rebalancing=None, sanierung=None == 5c ------------------
gc_off = GoingConcernFundingRatioAnalysis(
    salm, **_COMMON, rebalancing=None, sanierung=None,
)
for d in (BASE_DATE, f"{START_YEAR + 10}-01-01", f"{START_YEAR + 25}-01-01"):
    np.testing.assert_array_equal(
        gc_off.funding_ratio_distribution(d),
        scen_A.funding_ratio_distribution(d),
        err_msg=f"Regression vs 5c verletzt bei {d}",
    )
print("  => Assert 1 OK — rebalancing=None, sanierung=None identisch zu 5c.")

# ---- DG_t0-Kalibrierung ------------------------------------------------------
dg_t0 = scen_A.funding_ratio_distribution(BASE_DATE)
assert abs(float(np.mean(dg_t0)) - 1.076) < 1e-9, (
    f"DG_t0-Kalibrierung verfehlt: {float(np.mean(dg_t0)):.6%} statt 107.6%"
)
print(f"  => Kalibrierung OK — DG_t0 = {float(np.mean(dg_t0)):.4%} (Soll 107.6%).")

# ---- Spec-Assert 3: nach jedem Rebalance |w - target| < 1e-12 ----------------
_LAST_DATE = f"{START_YEAR + HORIZON_YEARS}-01-01"
for p_i in (0, N_PATHS // 2, N_PATHS - 1):
    wp = scen_B.weight_path(p_i, _LAST_DATE)
    reb = wp[wp["rebalanced"]]
    dev = np.max(np.abs(np.r_[reb["w_BONDS"] - reb["target_BONDS"],
                              reb["w_EQUITY"] - reb["target_EQUITY"]]))
    assert dev < 1e-12, f"Pfad {p_i}: |w - target| = {dev:.2e} >= 1e-12"
print("  => Assert 3 OK — post-rebalance |w - target| < 1e-12 (3 Pfade geprueft).")

# ---- Spec-Assert 5: C dokumentiert den Shift ab Jahr 5 -----------------------
wp_c = scen_C.weight_path(0, _LAST_DATE)
reb_c = wp_c[wp_c["rebalanced"]].set_index("year_index")
assert all(abs(reb_c.loc[k, "w_BONDS"] - 0.40) < 1e-12 for k in range(1, 5)), (
    "Szenario C: vor Jahr 5 muss 40/60 gelten"
)
assert all(
    abs(reb_c.loc[k, "w_BONDS"] - 0.60) < 1e-12
    for k in range(5, HORIZON_YEARS + 1)
), "Szenario C: ab Jahr 5 muss 60/40 gelten"
print("  => Assert 5 OK — Gewichtspfad C: 40/60 -> 60/40 ab Jahr 5.")


# =============================================================================
# 6. DISTRIBUTIONS + PLOTS  ->  output/stage_5d/
# =============================================================================
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
    "A (Legacy 5c, buy-and-hold)": "#264653",
    "B (Fix-Mix 40/60)": "#2a9d8f",
    "C (Shift 60/40 ab Jahr 5)": "#e76f51",
}
_x = [np.datetime64(d) for d in fan_dates]

# ---- v1: DG-Quantilfaecher je Szenario ---------------------------------------
for tag, (name, scen_dists) in zip("ABC", dists.items()):
    q = {p: np.array([np.percentile(scen_dists[d], p) for d in fan_dates])
         for p in _QUANTILES}
    fig, ax = plt.subplots(figsize=(11, 5))
    col = _COLORS[name]
    ax.fill_between(_x, q[5] * 100, q[95] * 100, color=col, alpha=0.15, label="p5-p95")
    ax.fill_between(_x, q[25] * 100, q[75] * 100, color=col, alpha=0.35, label="p25-p75")
    ax.plot(_x, q[50] * 100, color=col, marker="o", linewidth=2, label="Median (p50)")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100% Deckung")
    ax.set_title(f"Stage 5d — DG-Quantilfaecher Szenario {name} ({N_PATHS} Pfade)")
    ax.set_xlabel("Stichtag")
    ax.set_ylabel("Deckungsgrad (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    _path = os.path.join(OUT_DIR, f"v1_dg_fan_{tag}.png")
    fig.savefig(_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {_path}")

# ---- v2: P(DG_t < 100%) — alle Szenarien in einem Bild -----------------------
fig, ax = plt.subplots(figsize=(11, 5))
for name, scen_dists in dists.items():
    p_under = [float(np.mean(scen_dists[d] < 1.0)) for d in fan_dates]
    ax.plot(_x, np.array(p_under) * 100, marker="o", linewidth=2,
            color=_COLORS[name], label=name)
ax.set_title(f"Stage 5d — Unterdeckungswahrscheinlichkeit P(DG_t < 100%) "
             f"({N_PATHS} Pfade)")
ax.set_xlabel("Stichtag")
ax.set_ylabel("P(DG < 100%) (%)")
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
_path = os.path.join(OUT_DIR, "v2_p_underfunded.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")

# ---- v3: Gewichtspfade B vs C (post-rebalance, Pfad 0) ------------------------
wp_b = scen_B.weight_path(0, _LAST_DATE)
reb_b = wp_b[wp_b["rebalanced"]]
reb_c = wp_c[wp_c["rebalanced"]]
fig, ax = plt.subplots(figsize=(11, 5))
ax.step(reb_b["date"], reb_b["w_EQUITY"] * 100, where="post", linewidth=2,
        color=_COLORS["B (Fix-Mix 40/60)"], label="B: w_EQUITY")
ax.step(reb_c["date"], reb_c["w_EQUITY"] * 100, where="post", linewidth=2,
        color=_COLORS["C (Shift 60/40 ab Jahr 5)"], label="C: w_EQUITY")
ax.step(reb_b["date"], reb_b["w_BONDS"] * 100, where="post", linewidth=1.5,
        linestyle="--", color=_COLORS["B (Fix-Mix 40/60)"], label="B: w_BONDS")
ax.step(reb_c["date"], reb_c["w_BONDS"] * 100, where="post", linewidth=1.5,
        linestyle="--", color=_COLORS["C (Shift 60/40 ab Jahr 5)"], label="C: w_BONDS")
ax.axvline(np.datetime64(f"{START_YEAR + 5}-01-01"), color="grey",
           linestyle=":", linewidth=1.5, label="Shift ab Jahr 5")
ax.set_title("Stage 5d — Gewichtspfade (post-rebalance) B vs C")
ax.set_xlabel("Rebalancing-Termin")
ax.set_ylabel("Gewicht (%)")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(ncol=2)
plt.tight_layout()
_path = os.path.join(OUT_DIR, "v3_weight_paths_B_vs_C.png")
fig.savefig(_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {_path}")


# =============================================================================
# 7. SUMMARY
# =============================================================================
print("\n[7] Summary @ Horizontende")
print("-" * 40)
d_T = fan_dates[-1]
print(f"  Stichtag {d_T}:")
for name, scen_dists in dists.items():
    dist = scen_dists[d_T]
    print(f"    {name:<32} mean={float(np.mean(dist)):8.2%}  "
          f"p5={float(np.percentile(dist, 5)):8.2%}  "
          f"P(DG<1)={float(np.mean(dist < 1.0)):6.1%}")

plt.show()
