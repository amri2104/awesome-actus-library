"""Stage5a_pension_quickstart.py

Stage 5a — Monte-Carlo ALM under stochastic short rates on top of Stage 1-4:
  - Reuse Stage 4 setup (DynamicFundSimulator + DeckungsgradAnalysis +
    PensionierungsverlustAnalysis).
  - Add Hull-White stochastic short-rate model + per-path discounting.
  - Monte-Carlo distributions for t0 NPVs and DG-Pfad at valuation dates.
"""

import os
import warnings

import numpy as np
import pandas as pd

from awesome_actus_lib import PAM, Portfolio, PublicActusService
from awesome_actus_lib.pension import (
    PensionPolicy,
    Cohort,
    PensionFund,
    OpenFundSimulator,
    DynamicFundSimulator,
    Stage4Dynamics,
    EntryPolicy,
    ALMAnalysis,
    DeckungsgradAnalysis,
    PensionierungsverlustAnalysis,
    FundProjection,
    StochasticALMAnalysis,
    annuity_due,
    MortalityTable,
    ek2001_2005,
)
from awesome_actus_lib.stochastic_rates import (
    CurveCalibrator,
    create_model,
)


# ---------------------------------------------------------------------------
# Configurable horizon, start date, MC config
# ---------------------------------------------------------------------------
HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])

N_PATHS = 500
N_STEPS_PER_YEAR = 12
SEED = 42


print("=" * 70)
print("PENSION ALM CASE STUDY: BVG liabilities vs bond portfolio")
print("                       Stage 5a — stochastic short rates (Hull-White)")
print("=" * 70)

# =============================================================================
# 1A. LIABILITIES (Stage 4 reuse: dense cohort grid + dynamics + mortality)
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities (Stage 4 setup)")
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
mortality = ek2001_2005()

print(f"  {len(liability_fund.cohorts)} cohorts, EntryPolicy, Stage4Dynamics, EK 0105 mortality")
print(f"  Horizon: {HORIZON_YEARS} years")

# =============================================================================
# 1B. ASSETS (Stage 4 reuse: VK-scaled 3-bond ladder at DG_t0 ≈ 107 %)
# =============================================================================
print("\n[1B] Portfolio Definition — assets (VK-scaled)")
print("-" * 40)

VK_t0 = DeckungsgradAnalysis(
    fund=liability_fund,
    mortality=mortality,
    vorsorgevermoegen=0.0,
).vorsorgekapital_t0()
TARGET_DG = 1.07
total_notional = round(TARGET_DG * VK_t0)
notional_2030 = round(0.40 * total_notional)
notional_2040 = round(0.35 * total_notional)
notional_2050 = total_notional - notional_2030 - notional_2040

asset_portfolio = Portfolio([
    PAM(
        contractID="BOND_2030",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=notional_2030,
        initialExchangeDate=START_DATE,
        maturityDate="2030-12-31T00:00:00",
        nominalInterestRate=0.020,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="BOND_2040",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=notional_2040,
        initialExchangeDate=START_DATE,
        maturityDate="2040-12-31T00:00:00",
        nominalInterestRate=0.025,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
    PAM(
        contractID="BOND_2050",
        statusDate=START_DATE,
        contractDealDate=START_DATE,
        currency="CHF",
        notionalPrincipal=notional_2050,
        initialExchangeDate=START_DATE,
        maturityDate="2050-12-31T00:00:00",
        nominalInterestRate=0.030,
        dayCountConvention="30E360",
        contractRole="RPA",
        cycleOfInterestPayment="P1YL1",
        counterpartyID="SNB",
        creatorID="PK_ALM",
    ),
])
print(f"  VK_t0 = CHF {VK_t0:>15,.2f}   Σ Bond-Notional = CHF {total_notional:>13,}")
print(f"  Bond ladder: {notional_2030:,} (2030) / {notional_2040:,} (2040) / {notional_2050:,} (2050)")

service = PublicActusService()

# =============================================================================
# 1C. STOCHASTIC SETUP — Hull-White on a hardcoded CHF plateau
# =============================================================================
print("\n[1C] Stochastic Setup — Hull-White short-rate model")
print("-" * 40)

# Hardcoded CHF plateau (illustrative — replace with real SNB curve in practice)
curve_tenors = [1.0, 2.0, 5.0, 10.0]
curve_rates = [0.012, 0.014, 0.018, 0.022]
r0 = curve_rates[0]

cal = CurveCalibrator.from_market(
    times=curve_tenors,
    spot_rates=curve_rates,
    extrapolation_mode="flat",
)

HW_A = 0.10
HW_SIGMA = 0.008

# Suppress the SimulationResult warning about negative rates (Gaussian model).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    model = create_model(
        "hull_white",
        r0=r0,
        a=HW_A,
        sigma=HW_SIGMA,
        calibrator=cal,
        seed=SEED,
    )
    sim = model.simulate(
        T=float(HORIZON_YEARS),
        M=HORIZON_YEARS * N_STEPS_PER_YEAR,
        I=N_PATHS,
    )

stats = sim.statistics()
print(f"  Market curve (tenors → spot rate):")
for t, r in zip(curve_tenors, curve_rates):
    print(f"     {int(t):>2}Y → {r:.2%}")
print(f"  Model: Hull-White (a={HW_A}, σ={HW_SIGMA}, seed={SEED})")
print(f"  Simulation: I={N_PATHS} paths, T={HORIZON_YEARS}y, M={HORIZON_YEARS * N_STEPS_PER_YEAR} steps")
print(f"  Terminal rate: mean={stats['terminal_mean']:.4%}, std={stats['terminal_std']:.4%}")
print(f"  Range: min={stats['min_rate']:.4%}, max={stats['max_rate']:.4%}")


# =============================================================================
# 2. EVENT GENERATION (Stage 4 reuse)
# =============================================================================
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


# =============================================================================
# 3. DETERMINISTIC ALM (Stage 4 reuse — diagnostic baseline)
# =============================================================================
print("\n[3] Deterministic ALM (Stage 4 baseline)")
print("-" * 40)

alm = ALMAnalysis(
    assets_cf=asset_cfs,
    liabilities_cf=liability_cfs,
    flat_rate=policy.technical_rate,
)
fr_t0_det = alm.funding_ratio(as_of="2025-01-01")
print(f"  Netto-CF-Diagnostik (flat technical_rate): {fr_t0_det:.2%}")


# =============================================================================
# 3A-DG. DETERMINISTIC DECKUNGSGRAD @ t0 (Stage 4 reuse)
# =============================================================================
print("\n[3A-DG] Deckungsgrad @ t0 — deterministic baseline")
print("-" * 40)

vv_det = sum(c.terms["notionalPrincipal"].value for c in asset_portfolio.contracts)
dga = DeckungsgradAnalysis(
    fund=liability_fund,
    mortality=mortality,
    vorsorgevermoegen=vv_det,
)
dg_t0_det = dga.deckungsgrad_t0()
print(f"  Vorsorgevermögen   = CHF {vv_det:>15,.2f}")
print(f"  Vorsorgekapital_t0 = CHF {dga.vorsorgekapital_t0():>15,.2f}")
print(f"  Deckungsgrad_t0    = {dg_t0_det:>15.2%}")


# =============================================================================
# 3B. PENSIONIERUNGSVERLUST — BASELINE (Stage 4 reuse, condensed)
# =============================================================================
print("\n[3B] Pensionierungsverlust — BASELINE (Stage 4)")
print("-" * 40)
pva_base = PensionierungsverlustAnalysis(
    liability_cf=liability_cfs,
    policy=policy,
    mortality=mortality,
)
base_total = pva_base.summary()["verlust_total"].sum()
print(f"  Σ BASELINE-Verlust: CHF {base_total:,.2f}")


# =============================================================================
# 3C. PENSIONIERUNGSVERLUST — STRESS (Stage 4 reuse, condensed)
# =============================================================================
print("\n[3C] Pensionierungsverlust — STRESS (Stage 4)")
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
stress_df = PensionierungsverlustAnalysis(
    liability_cf=stress_cf,
    policy=policy,
    mortality=mortality,
).summary()
stress_total = stress_df["verlust_total"].sum()
print(f"  Σ STRESS-Verlust:   CHF {stress_total:,.2f}")


# =============================================================================
# 3D. STOCHASTIC ALM — Monte-Carlo (Stage 5a CORE)
# =============================================================================
print("\n[3D] Stochastic ALM — Monte-Carlo distributions (Stage 5a)")
print("-" * 40)

projection = FundProjection.from_simulator(
    simulator=simulator,
    liability_cf=liability_cfs,
    entry_policy=entry_policy,
)

salm = StochasticALMAnalysis(
    asset_cf=asset_cfs,
    liability_cf=liability_cfs,
    projection=projection,
    policy=policy,
    mortality=mortality,
    model=model,
    sim=sim,
    base_date=pd.to_datetime(START_DATE),
)

# --- t0 NPV distributions -----------------------------------------------
dist_assets = salm.asset_npv_t0_distribution()
dist_liab = salm.liability_npv_t0_distribution()
dist_fr = salm.funding_ratio_t0_distribution()

def _fmt(arr):
    q = StochasticALMAnalysis.quantiles(arr, qs=(0.05, 0.5, 0.95))
    return (f"mean={np.mean(arr):>15,.2f}  std={np.std(arr):>13,.2f}  "
            f"P5={q[0.05]:>13,.2f}  P50={q[0.5]:>13,.2f}  P95={q[0.95]:>13,.2f}")

print(f"  Asset NPV @ t0 ({N_PATHS} paths):")
print(f"     {_fmt(dist_assets)}")
print(f"  Liability NPV @ t0 ({N_PATHS} paths):")
print(f"     {_fmt(dist_liab)}")
print(f"  Funding Ratio (Netto-CF) @ t0 ({N_PATHS} paths):")
fr_q = StochasticALMAnalysis.quantiles(dist_fr, qs=(0.05, 0.5, 0.95))
print(f"     mean={np.mean(dist_fr):>10.2%}  P5={fr_q[0.05]:>10.2%}  "
      f"P50={fr_q[0.5]:>10.2%}  P95={fr_q[0.95]:>10.2%}")

# VaR/ES on asset NPV loss (loss = mean - path)
loss_assets = np.mean(dist_assets) - dist_assets
var95 = StochasticALMAnalysis.var(loss_assets, alpha=0.95)
es95 = StochasticALMAnalysis.es(loss_assets, alpha=0.95)
print(f"  Asset NPV loss vs MC mean: VaR95 = CHF {var95:,.2f}  ES95 = CHF {es95:,.2f}")

# --- DG path at valuation dates -----------------------------------------
val_years = [2025, 2030, 2035, 2040, 2045]
dg_path = salm.deckungsgrad_path(val_years)
agg = (
    dg_path.groupby("year")["dg"]
    .agg(["mean", "std",
          lambda s: np.percentile(s, 5),
          lambda s: np.percentile(s, 50),
          lambda s: np.percentile(s, 95),
          lambda s: float((s < 1.0).mean())])
    .rename(columns={"<lambda_0>": "P5", "<lambda_1>": "P50",
                     "<lambda_2>": "P95", "<lambda_3>": "Pr(DG<1.0)"})
)
print(f"\n  Deckungsgrad-Pfad — per Stichtag (over {N_PATHS} paths):")
print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# VERIFIKATION — Inline-Asserts (Stage 4 (a..f) + Stage 5a (g..j))
# =============================================================================
print("\n[VERIFY] Inline-Asserts")
print("-" * 40)

# (a)/(a2)/(a3): annuity_due references
synthetic = MortalityTable(
    q_male={60: 0.0, 61: 0.0, 62: 0.0},
    q_female={60: 0.0, 61: 0.0, 62: 0.0},
)
a_ref = annuity_due(60, "m", 0.0, synthetic, 62)
assert abs(a_ref - 3.0) < 1e-12
print(f"  (a) annuity_due(60,'m',i=0,q=0,term=62) = {a_ref} (=3.0). OK.")

a_deg = annuity_due(100, "m", 0.0176, ek2001_2005(), 100)
assert abs(a_deg - 1.0) < 1e-12
print(f"  (a2) annuity_due(100,'m',EK0105,term=100) = {a_deg} (=1.0). OK.")

tbl_a3 = MortalityTable(
    q_male={0: 0.5, 1: 0.0},
    q_female={0: 0.5, 1: 0.0},
)
a3 = annuity_due(0, "m", 0.10, tbl_a3, 1)
assert abs(a3 - 1.4545454545454546) < 1e-12
print(f"  (a3) annuity_due(0,'m',i=10%,q0=0.5,term=1) = {a3} (=1.4545...). OK.")

# (b) Stage-3-Identität (Stage 4 = Stage 2/3 with neutral dynamics)
neutral_dyn = Stage4Dynamics(
    salary_growth=0.0,
    conversion_rate_path=None,
    threshold_index_period=1,
)
neutral_sim = DynamicFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    dynamics=neutral_dyn,
    mortality=mortality,
).run(horizon_years=HORIZON_YEARS)
ref_sim = OpenFundSimulator(
    fund=liability_fund,
    entry_policy=entry_policy,
    mortality=mortality,
).run(horizon_years=HORIZON_YEARS)
sum_n = float(neutral_sim.events_df["payoff"].sum())
sum_r = float(ref_sim.events_df["payoff"].sum())
rel_b = abs(sum_n - sum_r) / (abs(sum_r) + 1e-30)
assert rel_b < 1e-9
print(f"  (b) Σ payoff Stage4-neutral == Stage3 (rel diff {rel_b:.2e}). OK.")

# (c)/(c2)/(c3): stress sign + ordering
violations = stress_df[
    (stress_df["applied_uws"] > stress_df["technical_uws"])
    & (stress_df["verlust_per_capita"] <= 0.0)
]
assert violations.empty
print(f"  (c) Vorzeichen STRESS: 0 Verletzungen. OK.")
assert stress_total > 0
print(f"  (c2) Σ STRESS-Verlust > 0: CHF {stress_total:,.2f}. OK.")
assert base_total <= stress_total
print(f"  (c3) BASELINE {base_total:,.2f} ≤ STRESS {stress_total:,.2f}. OK.")

# (d) DG sanity
solo_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
solo_fund.add_cohort(Cohort(
    cohort_id="SOLO",
    birth_year=1990,
    headcount=10,
    gross_salary=80_000.0,
    accrued_savings=100_000.0,
    gender="unisex",
))
vk_solo = DeckungsgradAnalysis(
    fund=solo_fund,
    mortality=None,
    vorsorgevermoegen=0.0,
).vorsorgekapital_t0()
assert vk_solo == 1_000_000.0
print(f"  (d) VK(only-active, AGH=100k, hc=10) = {vk_solo:,.2f}. OK.")

# (e) DG_t0 ∈ [1.00, 1.15]
assert 1.00 <= dg_t0_det <= 1.15
print(f"  (e) DG_t0 = {dg_t0_det:.2%} ∈ [1.00, 1.15]. OK.")

# (f) retired vk_beitrag
breakdown = dga.breakdown()
ret_row = breakdown[breakdown["cohort_id"] == "RETIRED_1955"].iloc[0]
ret_cohort = next(c for c in liability_fund.cohorts if c.cohort_id == "RETIRED_1955")
assert ret_row["annuity_due"] > 1.0
exp_vk = ret_cohort.annual_pension * ret_cohort.headcount * ret_row["annuity_due"]
assert abs(ret_row["vk_beitrag"] - exp_vk) / abs(exp_vk) < 1e-9
print(f"  (f) RETIRED_1955 ä={ret_row['annuity_due']:.4f}, vk_beitrag exakt. OK.")

# -------------------------------------------------------------------------
# Stage 5a asserts
# -------------------------------------------------------------------------
# (g) Zero-vola sanity: Hull-White with σ=1e-12 -> MC NPV ≈ deterministic NPV
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    model_zero = create_model(
        "hull_white",
        r0=r0,
        a=HW_A,
        sigma=1e-12,
        calibrator=cal,
        seed=SEED,
    )
    sim_zero = model_zero.simulate(
        T=float(HORIZON_YEARS),
        M=HORIZON_YEARS * N_STEPS_PER_YEAR,
        I=50,
    )
salm_zero = StochasticALMAnalysis(
    asset_cf=asset_cfs,
    liability_cf=liability_cfs,
    projection=projection,
    policy=policy,
    mortality=mortality,
    model=model_zero,
    sim=sim_zero,
    base_date=pd.to_datetime(START_DATE),
)
dist_a_zero = salm_zero.asset_npv_t0_distribution()
spread = float(np.std(dist_a_zero) / (abs(np.mean(dist_a_zero)) + 1e-30))
assert spread < 1e-3, f"(g) Zero-vola NPV must be constant across paths, std/mean={spread:.2e}"
print(f"  (g) Zero-vola NPV across paths: std/mean = {spread:.2e} < 1e-3. OK.")

# (h) MC convergence (I=500 vs I=1000)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    model_big = create_model(
        "hull_white",
        r0=r0,
        a=HW_A,
        sigma=HW_SIGMA,
        calibrator=cal,
        seed=SEED,
    )
    sim_big = model_big.simulate(
        T=float(HORIZON_YEARS),
        M=HORIZON_YEARS * N_STEPS_PER_YEAR,
        I=1000,
    )
salm_big = StochasticALMAnalysis(
    asset_cf=asset_cfs,
    liability_cf=liability_cfs,
    projection=projection,
    policy=policy,
    mortality=mortality,
    model=model_big,
    sim=sim_big,
    base_date=pd.to_datetime(START_DATE),
)
mean_500 = float(np.mean(dist_assets))
mean_1000 = float(np.mean(salm_big.asset_npv_t0_distribution()))
rel_h = abs(mean_1000 - mean_500) / (abs(mean_500) + 1e-30)
assert rel_h < 0.05, f"(h) MC mean diverges between I=500 and I=1000: rel={rel_h:.2%}"
print(f"  (h) MC convergence: mean I=500 vs I=1000 rel diff = {rel_h:.2%} < 5%. OK.")

# (i) Quantile ordering per stichtag
for vyear in val_years:
    sub = dg_path[dg_path["year"] == vyear]["dg"].values
    p5, p50, p95 = np.percentile(sub, [5, 50, 95])
    assert p5 <= p50 <= p95, f"(i) ordering violated at {vyear}: {p5}, {p50}, {p95}"
print(f"  (i) P5 ≤ P50 ≤ P95 per Stichtag (alle {len(val_years)} Stichtage). OK.")

# (j) DG @ t=START_YEAR (path-aggregate) ≈ deterministic DG_t0 (same order)
dg_t0_mc = float(np.mean(dg_path[dg_path["year"] == START_YEAR]["dg"]))
rel_j = abs(dg_t0_mc - dg_t0_det) / (abs(dg_t0_det) + 1e-30)
assert rel_j < 0.20, (
    f"(j) DG_t0 MC={dg_t0_mc} vs det={dg_t0_det}, rel diff {rel_j:.2%}"
)
print(f"  (j) DG @ t0 MC mean = {dg_t0_mc:.4f} vs deterministic = {dg_t0_det:.4f} "
      f"(rel {rel_j:.2%}). OK.")

# (k) Maturing tranche must move bond->cash, not bond->void.
# Between consecutive Stichtage, mean Vorsorgevermögen must not collapse
# by more than 30%. A bond maturing within the interval converts to par cash
# (CSH account), so total VV evolves smoothly rather than dropping by the
# notional. Without the CSH cash account, VV drops by the full notional of
# any maturing tranche — that artefact is what assert (k) guards against.
vv_means = []
years_sorted = sorted(val_years)
for y in years_sorted:
    vv_means.append(float(np.mean(dg_path[dg_path["year"] == y]["vv"])))
for i in range(1, len(years_sorted)):
    y_prev, y_curr = years_sorted[i - 1], years_sorted[i]
    vv_prev, vv_curr = vv_means[i - 1], vv_means[i]
    rel_drop = (vv_prev - vv_curr) / (abs(vv_prev) + 1e-30)
    assert rel_drop < 0.30, (
        f"(k) Vorsorgevermögen collapse between {y_prev}->{y_curr}: "
        f"{vv_prev:,.0f} -> {vv_curr:,.0f} (drop {rel_drop:.2%}). "
        f"Maturing tranche likely lost to void instead of cash."
    )
print(f"  (k) Smooth VV evolution across {years_sorted}: "
      f"max consecutive drop < 30%. OK.")


# =============================================================================
# 4. PLOTS — Stage 5a-specific (Stage 4 plots intentionally not duplicated)
# =============================================================================
print("\n[4] Plots")
print("-" * 40)

import matplotlib.pyplot as plt

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "stage5a")
os.makedirs(_FIG_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# --- Plot 7: Short-rate fan chart (5/50/95 percentile bands) -------------
fig7, ax7 = plt.subplots(figsize=(11, 5))
years_axis = [START_YEAR + t for t in sim.times]
mean_path = sim.mean_path()
p05 = sim.percentile_path(5)
p95 = sim.percentile_path(95)
ax7.fill_between(years_axis, p05 * 100, p95 * 100, color="#264653", alpha=0.25,
                 label="5–95 % band")
ax7.plot(years_axis, mean_path * 100, color="#264653", linewidth=1.6, label="Mean path")
ax7.axhline(0, color="black", linewidth=0.4)
ax7.set_title("Stage 5a — Short-Rate Fan Chart (Hull-White, I=500)")
ax7.set_ylabel("Short rate (%)")
ax7.set_xlabel("Year")
ax7.grid(True, linestyle="--", alpha=0.4)
ax7.legend()
plt.tight_layout()
_save(fig7, "v7_short_rate_fan.png")

# --- Plot 8: Asset NPV distribution at t0 --------------------------------
fig8, ax8 = plt.subplots(figsize=(11, 5))
ax8.hist(dist_assets / 1e6, bins=40, color="#2a9d8f", edgecolor="black", alpha=0.75)
ax8.axvline(np.mean(dist_assets) / 1e6, color="#264653", linewidth=1.8, label="Mean")
ax8.axvline(np.percentile(dist_assets, 5) / 1e6, color="#e76f51", linestyle="--",
            label="P5")
ax8.axvline(np.percentile(dist_assets, 95) / 1e6, color="#e9c46a", linestyle="--",
            label="P95")
ax8.set_title("Stage 5a — Asset NPV Distribution @ t0 (I=500)")
ax8.set_ylabel("Frequency")
ax8.set_xlabel("Asset NPV (Mio CHF)")
ax8.grid(True, axis="y", linestyle="--", alpha=0.4)
ax8.legend()
plt.tight_layout()
_save(fig8, "v8_asset_npv_distribution.png")

# --- Plot 9: DG fan chart at Stichtage -----------------------------------
fig9, ax9 = plt.subplots(figsize=(11, 5))
years_plot = sorted(val_years)
p5_arr = [np.percentile(dg_path[dg_path["year"] == y]["dg"], 5) for y in years_plot]
p50_arr = [np.percentile(dg_path[dg_path["year"] == y]["dg"], 50) for y in years_plot]
p95_arr = [np.percentile(dg_path[dg_path["year"] == y]["dg"], 95) for y in years_plot]
ax9.fill_between(years_plot, p5_arr, p95_arr, color="#2a9d8f", alpha=0.25,
                 label="5–95 % band")
ax9.plot(years_plot, p50_arr, color="#264653", marker="o", linewidth=1.8, label="P50")
ax9.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="100 % coverage")
ax9.set_title("Stage 5a — Deckungsgrad Fan Chart (Hull-White)")
ax9.set_ylabel("Deckungsgrad")
ax9.set_xlabel("Stichtag")
ax9.grid(True, linestyle="--", alpha=0.4)
ax9.legend()
plt.tight_layout()
_save(fig9, "v9_dg_fan.png")

# --- Plot 10: Pr(DG<1.0) per Stichtag ------------------------------------
fig10, ax10 = plt.subplots(figsize=(11, 5))
pr_under = [float((dg_path[dg_path["year"] == y]["dg"] < 1.0).mean())
            for y in years_plot]
ax10.bar([str(y) for y in years_plot], np.array(pr_under) * 100,
         color="#e76f51", edgecolor="black")
ax10.set_title("Stage 5a — Pr(Deckungsgrad < 100 %) per Stichtag")
ax10.set_ylabel("Probability (%)")
ax10.set_xlabel("Stichtag")
ax10.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
_save(fig10, "v10_pr_underfunded.png")

plt.show()
