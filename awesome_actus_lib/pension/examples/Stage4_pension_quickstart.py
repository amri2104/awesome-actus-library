"""Stage4_pension_quickstart.py

Stage 4 — deterministic time-varying parameters on top of Stage 1/2/3:
  - salary growth (annual, geometric)
  - falling applied UWS path (per retirement year)
  - BVG-threshold indexation in k-year steps
  - PensionierungsverlustAnalysis (applied UWS vs technical UWS = 1/ä_x)

Run from the repo root:
    .venv/bin/python awesome_actus_lib/pension/examples/Stage4_pension_quickstart.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: lets you run this file directly from the examples/ folder
# without installing the package.
# ---------------------------------------------------------------------------
try:
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
        DeckungsgradAnalysis,
        PensionierungsverlustAnalysis,
        annuity_due,
        technical_uws,
        MortalityTable,
        ek2001_2005,
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
        DynamicFundSimulator,
        Stage4Dynamics,
        EntryPolicy,
        ALMAnalysis,
        DeckungsgradAnalysis,
        PensionierungsverlustAnalysis,
        annuity_due,
        technical_uws,
        MortalityTable,
        ek2001_2005,
    )


# ---------------------------------------------------------------------------
# Configurable horizon and start date (used everywhere below).
# ---------------------------------------------------------------------------
HORIZON_YEARS = 40
START_DATE = "2025-01-01T00:00:00"
START_YEAR = int(START_DATE[:4])


print("=" * 70)
print("PENSION ALM CASE STUDY: BVG liabilities vs bond portfolio")
print("                       Stage 4 — dynamic salary + UWS path + KPI")
print("=" * 70)

# =============================================================================
# 1A. PORTFOLIO DEFINITION (liability side) — Stage 4: dynamic parameters
# Liabilities are defined FIRST so that the asset notional in [1B] can be
# calibrated against the Vorsorgekapital_t0 (Art-44-BVV2-near coverage).
# =============================================================================
print("\n[1A] Portfolio Definition — liabilities (Stage 4: dynamic parameters)")
print("-" * 40)

policy = PensionPolicy()

liability_fund = PensionFund(
    policy=policy,
    start_date=START_DATE,
)
# Active cohorts spread across birth years 1960..1990 so that retirements
# (age 65) fall into 2025, 2028, 2031, 2035, 2037, 2040, 2043, 2055 — a
# dense Verlust-per-year curve crossing the held-forward UWS step (2029).
# AGH per capita scales roughly with age (more accrual = older cohort).
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1960",  # retires 2025 (Variant B: auto-conv at year 1)
    birth_year=1960, headcount=90,
    gross_salary=120_000.0, accrued_savings=600_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1963",  # retires 2028
    birth_year=1963, headcount=100,
    gross_salary=115_000.0, accrued_savings=520_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1966",  # retires 2031
    birth_year=1966, headcount=110,
    gross_salary=115_000.0, accrued_savings=450_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1970",  # retires 2035
    birth_year=1970, headcount=150,
    gross_salary=110_000.0, accrued_savings=350_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1972",  # retires 2037
    birth_year=1972, headcount=130,
    gross_salary=108_000.0, accrued_savings=320_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1975",  # retires 2040
    birth_year=1975, headcount=140,
    gross_salary=105_000.0, accrued_savings=270_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1978",  # retires 2043
    birth_year=1978, headcount=140,
    gross_salary=100_000.0, accrued_savings=220_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="ACTIVE_1990",  # retires 2055
    birth_year=1990, headcount=200,
    gross_salary=90_000.0, accrued_savings=80_000.0, gender="unisex",
))
liability_fund.add_cohort(Cohort(
    cohort_id="RETIRED_1955",
    birth_year=1955, headcount=80,
    gross_salary=0.0, accrued_savings=0.0,
    status="retired", annual_pension=31_380.0, gender="f",
))

entry_policy = EntryPolicy(
    entry_age=25,
    headcount=20,
    gross_salary=80_000.0,
    gender="unisex",
)

# BASELINE scenario (calibrated, study-backed):
#   5.23% = average umhüllender UWS @65 across Swiss PK (Complementa Risiko
#   Check-up 2024), projected to 5.10% by 2029. The applied UWS in the model
#   is applied to the TOTAL AGH (Anrechnungsprinzip: PK lower the umhüllender
#   UWS below the legal 6.8% obligatorisch UWS because the obligatorisch
#   guarantee is absorbed in aggregate).
# Held-forward semantics: 5.23% from 2025, 5.10% from 2029 onwards.
dynamics = Stage4Dynamics(
    salary_growth=0.01,
    conversion_rate_path={
        2025: 0.0523,
        2029: 0.0510,
    },
    threshold_index_period=5,
)

mortality = ek2001_2005()

print(f"  Active cohorts (8 birth-year buckets, retire ages all 65):")
for c in liability_fund.cohorts:
    if c.status == "active":
        retire_year = c.birth_year + policy.retirement_age
        print(f"    {c.cohort_id}: {c.headcount} members, salary CHF "
              f"{c.gross_salary:,.0f}, AGH/head CHF {c.accrued_savings:,.0f}, "
              f"retires {retire_year}")
print(f"  Retired cohort:")
for c in liability_fund.cohorts:
    if c.status == "retired":
        print(f"    {c.cohort_id}: {c.headcount} members, pension CHF "
              f"{c.annual_pension:,.0f}/year, gender {c.gender}")
print(f"  EntryPolicy: 20 entrants/year, age 25, salary CHF 80,000, unisex")
print(f"  Stage4Dynamics (BASELINE — umhüllender UWS, studiengestützt):")
print(f"     salary_growth          = {dynamics.salary_growth:.2%} (geometric, annual)")
print(f"     threshold_index_period = {dynamics.threshold_index_period} years")
print(f"     conversion_rate_path   = {dynamics.conversion_rate_path}")
print(f"                              (held-forward: 5.23% from 2025, 5.10% from 2029)")
print(f"  Mortality:    EK 2001-2005 period table (retirees only, deterministic)")
print(f"  Horizon:      {HORIZON_YEARS} years")


# =============================================================================
# 1B. PORTFOLIO DEFINITION (asset side) — bond ladder calibrated to VK_t0.
# Notionals are sized at 1.07 × Vorsorgekapital_t0 (Complementa-region target
# Deckungsgrad ≈ 107 %), split 40 % / 35 % / 25 % across the three maturities.
# =============================================================================
print("\n[1B] Portfolio Definition — assets (VK-skaliert)")
print("-" * 40)

VK_t0 = DeckungsgradAnalysis(
    fund=liability_fund,
    mortality=mortality,
    vorsorgevermoegen=0.0,           # placeholder for the sizing step
).vorsorgekapital_t0()
TARGET_DG = 1.07
total_notional = round(TARGET_DG * VK_t0)
notional_2030 = round(0.40 * total_notional)
notional_2040 = round(0.35 * total_notional)
notional_2050 = total_notional - notional_2030 - notional_2040   # round-safe split

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
        creatorID="PK_ALM"
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
        creatorID="PK_ALM"
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
        creatorID="PK_ALM"
    )
])

print(f"  Vorsorgekapital_t0:  CHF {VK_t0:>15,.2f}")
print(f"  Target Deckungsgrad: {TARGET_DG:.2%} (Complementa-region)")
print(f"  Σ Bond-Notionals:    CHF {total_notional:>15,.0f}")
print(f"  Asset 1: CHF {notional_2030:>13,.0f} bond (PAM), 2.0% coupon, matures 2030 (40%)")
print(f"  Asset 2: CHF {notional_2040:>13,.0f} bond (PAM), 2.5% coupon, matures 2040 (35%)")
print(f"  Asset 3: CHF {notional_2050:>13,.0f} bond (PAM), 3.0% coupon, matures 2050 (25%)")

service = PublicActusService()


# =============================================================================
# 2. EVENT GENERATION
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

cohort_hc_df = pd.DataFrame(simulator.cohort_headcount_log)
retired_first = cohort_hc_df[cohort_hc_df["cohort_id"] == "RETIRED_1955"].head(1)
retired_last = cohort_hc_df[cohort_hc_df["cohort_id"] == "RETIRED_1955"].tail(1)
print(f"  RETIRED_1955 headcount @ year 1 : {retired_first['headcount'].iloc[0]:.4f}")
print(f"  RETIRED_1955 headcount @ year {HORIZON_YEARS}: {retired_last['headcount'].iloc[0]:.4f}")

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

net_table = alm.net_liquidity(freq="YE")
print("\n  Variant 1 — Net Liquidity (year-end, first 5 rows):")
print(net_table.head().to_string())

fr_t0 = alm.funding_ratio(as_of="2025-01-01")
print(f"\n  Variant 2 — Netto-CF-Diagnostik @ 2025-01-01: {fr_t0:.2%}")
print(f"  (KEIN Deckungsgrad — diskontiert Netto-Liability-CF inkl. Beiträge.")
print(f"   Echter Art-44-BVV2-naher DG: siehe Abschnitt [3A-DG].)")

fr_path = alm.deckungsgrad_path(["2025-01-01", "2030-01-01", "2035-01-01"])
print("  Netto-CF-Diagnostik path (NICHT der DG):")
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

# =============================================================================
# 3A-DG. ART-44-BVV2-NAHER DECKUNGSGRAD @ t0 (aus dem Fund, nicht aus Events)
# Echter DG: Vorsorgevermögen / Vorsorgekapital_t0
#   Aktive  -> AGH × headcount  (Buchwert)
#   Rentner -> pension × headcount × ä_alter  (PV laufender Renten)
# Kalibrierbar gegen Complementa-Median ~107% (Risiko Check-up 2024).
# =============================================================================
print("\n[3A-DG] Deckungsgrad @ t0 (Art. 44 BVV2, t0-only)")
print("-" * 40)

vv = sum(c.terms["notionalPrincipal"].value for c in asset_portfolio.contracts)
dga = DeckungsgradAnalysis(
    fund=liability_fund,
    mortality=mortality,
    vorsorgevermoegen=vv,
)
VK_check = dga.vorsorgekapital_t0()
dg_t0 = dga.deckungsgrad_t0()
print(f"  Vorsorgevermögen  (Σ Bond-Notionals): CHF {vv:>15,.2f}")
print(f"  Vorsorgekapital_t0:                   CHF {VK_check:>15,.2f}")
print(f"  Deckungsgrad_t0:                      {dg_t0:>15.2%}")
print(f"\n  Kontrast zur Netto-CF-Diagnostik:")
print(f"     Deckungsgrad_t0 (Art-44-nah):      {dg_t0:>10.2%}")
print(f"     funding_ratio   (Netto-CF-PV):     {fr_t0:>10.2%}")
print(f"  -> Zahlen weichen ab, weil funding_ratio Beiträge im Zähler hat und")
print(f"     mix-sensitiv ist. Deckungsgrad_t0 ist die aufsichtsnahe Grösse.")

print(f"\n  Breakdown je Cohort:")
breakdown = dga.breakdown()
print(breakdown.to_string(index=False,
                          float_format=lambda x: f"{x:,.4f}"))

# =============================================================================
# 3B. PENSIONIERUNGSVERLUST-ANALYSE — BASELINE (umhüllender UWS, kalibriert)
# =============================================================================
print("\n[3B] Pensionierungsverlust-Analyse — BASELINE (5.23% → 5.10%)")
print("-" * 40)

pva_base = PensionierungsverlustAnalysis(
    liability_cf=liability_cfs,
    policy=policy,
    mortality=mortality,
)
verlust_df = pva_base.summary()
print(verlust_df.to_string(index=False,
                           float_format=lambda x: f"{x:,.4f}"))
base_total = verlust_df["verlust_total"].sum()
print(f"\n  Σ BASELINE-Verlust: CHF {base_total:,.2f}")
print(f"  (negativ = Gewinn aus Fonds-Sicht: applied UWS < technical UWS)")

# =============================================================================
# 3C. PENSIONIERUNGSVERLUST-ANALYSE — STRESS (6.8% flach aufs Gesamtkapital)
# =============================================================================
# Stress-Illustration der Studienaussage "ein zu hoch angesetzter UWS führt zu
# Pensionierungsverlusten". 6.8% ist der gesetzliche BVG-MINDEST-UWS auf den
# OBLIGATORISCHEN Teil. Hier auf das GESAMTKAPITAL angewendet — bewusst OHNE
# Anrechnungsprinzip. Das ist KEINE reale umhüllende Kasse, kein Baseline-Wert,
# sondern ein illustratives Szenario, um den Mechanismus sichtbar zu machen.
print("\n[3C] Pensionierungsverlust-Analyse — STRESS (6.8% flach, ohne Anrechnung)")
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
pva_stress = PensionierungsverlustAnalysis(
    liability_cf=stress_cf,
    policy=policy,
    mortality=mortality,
)
stress_df = pva_stress.summary()
print(stress_df.to_string(index=False,
                          float_format=lambda x: f"{x:,.4f}"))
stress_total = stress_df["verlust_total"].sum()
print(f"\n  Σ STRESS-Verlust:   CHF {stress_total:,.2f}")

# --- Kontrast: Baseline vs Stress ----------------------------------------
print("\n  --- Kontrast Baseline vs Stress ---")
print(f"  Σ BASELINE-Verlust: CHF {base_total:>20,.2f}  (5.23% → 5.10%)")
print(f"  Σ STRESS-Verlust:   CHF {stress_total:>20,.2f}  (6.8% flach)")
print(f"  Differenz:          CHF {stress_total - base_total:>20,.2f}  "
      f"(Mehrverlust durch UWS-Stress)")


# =============================================================================
# VERIFIKATION — Inline-Asserts
# =============================================================================
print("\n[VERIFY] Inline-Asserts")
print("-" * 40)

# --- (a) annuity_due reference: synthetic q=0 table, i=0, age=60, term=62 -----
synthetic = MortalityTable(
    q_male={60: 0.0, 61: 0.0, 62: 0.0},
    q_female={60: 0.0, 61: 0.0, 62: 0.0},
)
a_ref = annuity_due(60, "m", 0.0, synthetic, 62)
assert abs(a_ref - 3.0) < 1e-12, f"(a) annuity_due reference failed: {a_ref}"
print(f"  (a) annuity_due(60, 'm', i=0, q=0, term=62) = {a_ref:.12f} (=3.0, exact). OK.")

# --- (a2) Degenerate: age == terminal_age ⇒ K=0 ⇒ ä = 1.0 -------------------
a_deg = annuity_due(100, "m", 0.0176, ek2001_2005(), 100)
assert abs(a_deg - 1.0) < 1e-12, f"(a2) degenerate annuity_due failed: {a_deg}"
print(f"  (a2) annuity_due(100, 'm', EK0105, term=100) = {a_deg:.12f} (=1.0). OK.")

# --- (a3) Diskontierung UND Mortalitätsgewichtung handgerechnet --------------
# tbl: q_0 = 0.5, q_1 = 0.0; i = 10%; age=0, term=1 ⇒ K=1
# ä = v^0*1 + v^1*(1-0.5) = 1 + (1/1.1)*0.5 = 1.4545454545454...
tbl = MortalityTable(q_male={0: 0.5, 1: 0.0}, q_female={0: 0.5, 1: 0.0})
a3 = annuity_due(0, "m", 0.10, tbl, 1)
assert abs(a3 - 1.4545454545454546) < 1e-12, f"(a3) annuity_due failed: {a3}"
print(f"  (a3) annuity_due(0,'m',i=10%,q0=0.5,term=1) = {a3:.12f} (=1.454545..., exakt). OK.")

# --- (b) Stage-3-Identität: Stage-4 mit Null-Dynamiken == Stage-2 Sim --------
# Vergleichslauf: gleiche fund/entry_policy/mortality, aber OpenFundSimulator
# (Stage-2-Code, der mit mortality erweitert ist und Stage-3 reproduziert).
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
assert rel < 1e-9, (
    f"(b) Stage-3-Identität verletzt: payoff-Summe Stage4-neutral={sum_s4} "
    f"vs Stage3={sum_s3} (rel. Diff. {rel:.2e})"
)
print(f"  (b) Σ payoff Stage4-neutral={sum_s4:,.2f} == Stage3={sum_s3:,.2f} "
      f"(rel. diff {rel:.2e}). OK.")

alm_s4n = ALMAnalysis(asset_cfs, cf_stage4_neutral, flat_rate=policy.technical_rate)
alm_s3 = ALMAnalysis(asset_cfs, cf_stage3, flat_rate=policy.technical_rate)
fr_s4n = alm_s4n.funding_ratio("2025-01-01")
fr_s3 = alm_s3.funding_ratio("2025-01-01")
rel_fr = abs(fr_s4n - fr_s3) / (abs(fr_s3) + 1e-30)
assert rel_fr < 1e-9, (
    f"(b) funding_ratio Identität verletzt: Stage4-neutral={fr_s4n} "
    f"vs Stage3={fr_s3} (rel. Diff. {rel_fr:.2e})"
)
print(f"  (b) FR Stage4-neutral={fr_s4n:.6f} == Stage3={fr_s3:.6f} "
      f"(rel. diff {rel_fr:.2e}). OK.")

# --- (c) Vorzeichen auf dem STRESS-Df (echter Verlust, applied>technical) ----
violations = stress_df[
    (stress_df["applied_uws"] > stress_df["technical_uws"])
    & (stress_df["verlust_per_capita"] <= 0.0)
]
assert violations.empty, (
    f"(c) Sign violation in STRESS: {len(violations)} rows with "
    f"applied>technical but verlust_per_capita<=0:\n{violations}"
)
n_loss_rows = ((stress_df["applied_uws"] > stress_df["technical_uws"])
               & (stress_df["verlust_per_capita"] > 0.0)).sum()
print(f"  (c) Vorzeichen-Konsistenz STRESS: {n_loss_rows}/{len(stress_df)} "
      f"Zeilen mit applied>technical und verlust_pc>0. OK.")

# --- (c2) STRESS produziert echten Verlust + ist verlustreicher als BASELINE -
assert stress_total > 0, f"(c2) STRESS-Verlust nicht positiv: {stress_total}"
print(f"  (c2) Σ STRESS-Verlust = CHF {stress_total:,.2f} > 0. OK.")

assert base_total <= stress_total, (
    f"(c3) Base-Verlust {base_total} muss <= Stress-Verlust {stress_total} sein."
)
print(f"  (c3) Σ BASELINE {base_total:,.2f} <= Σ STRESS {stress_total:,.2f}. OK.")

# --- (d) DeckungsgradAnalysis single-cohort sanity ---------------------------
# Only-active fund, AGH=100k, hc=10 -> vorsorgekapital_t0() = 1_000_000 exakt.
solo_fund = PensionFund(policy=policy, start_date=START_DATE)
solo_fund.add_cohort(Cohort(
    cohort_id="SOLO", birth_year=1990, headcount=10,
    gross_salary=80_000.0, accrued_savings=100_000.0, gender="unisex",
))
vk_solo = DeckungsgradAnalysis(
    fund=solo_fund, mortality=None, vorsorgevermoegen=0.0,
).vorsorgekapital_t0()
assert vk_solo == 1_000_000.0, f"(d) single-cohort VK failed: {vk_solo}"
print(f"  (d) VK(only-active, AGH=100k, hc=10) = {vk_solo:,.2f} (=1,000,000). OK.")

# --- (e) Skalierter DG @ t0 liegt in [1.00, 1.15] (Complementa-Region) -------
assert 1.00 <= dg_t0 <= 1.15, (
    f"(e) Deckungsgrad_t0 ausserhalb [1.00, 1.15]: {dg_t0:.4f}"
)
print(f"  (e) Deckungsgrad_t0 = {dg_t0:.4%} in [1.00, 1.15]. OK.")

# --- (f) Rentner-Beitrag: RETIRED_1955 -> ä > 1 und vk_beitrag == pension*hc*ä
ret_row = breakdown[breakdown["cohort_id"] == "RETIRED_1955"].iloc[0]
ret_cohort = next(c for c in liability_fund.cohorts if c.cohort_id == "RETIRED_1955")
assert ret_row["annuity_due"] > 1.0, (
    f"(f) RETIRED_1955 annuity_due muss > 1, got {ret_row['annuity_due']}"
)
expected_vk = ret_cohort.annual_pension * ret_cohort.headcount * ret_row["annuity_due"]
rel_f = abs(ret_row["vk_beitrag"] - expected_vk) / abs(expected_vk)
assert rel_f < 1e-9, (
    f"(f) vk_beitrag != pension*hc*ä: got {ret_row['vk_beitrag']}, "
    f"expected {expected_vk}, rel. diff {rel_f:.2e}"
)
print(f"  (f) RETIRED_1955 ä={ret_row['annuity_due']:.4f} > 1, "
      f"vk_beitrag={ret_row['vk_beitrag']:,.2f} == pension*hc*ä "
      f"(rel. diff {rel_f:.2e}). OK.")


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
ax1.set_title("Variant 1 — Net Liquidity per Year (Stage 4: dynamic salary + UWS path)")
ax1.set_ylabel("CHF")
ax1.set_xticks(list(x))
ax1.set_xticklabels(years, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig1, "stage4_v1_net_liquidity.png")

# --- Plot 2: Funding Ratio Path (Variant 2) ------------------------------
fr_dates = [f"{y}-01-01" for y in range(START_YEAR, START_YEAR + HORIZON_YEARS, 5)]
fr_series = alm.deckungsgrad_path(fr_dates)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(fr_series.index, fr_series.values, marker="o", color="#264653")
ax2.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="100% coverage")
ax2.set_title("Variant 2 — Funding Ratio Path (Stage 4)")
ax2.set_ylabel("Funding Ratio")
ax2.set_xlabel("Valuation Date")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
plt.tight_layout()
_save(fig2, "stage4_v2_funding_ratio_path.png")

# --- Plot 3: Combined CashFlowStream (Variant 3) -------------------------
fig3 = combined_cfs.plot(title="Variant 3 — Combined Asset + Liability Cashflows (Stage 4)",
                         return_fig=True)
if fig3 is not None:
    _save(fig3, "stage4_v3_combined_cashflows.png")

# --- Plot 4: Cohort headcount decline (Stage 3 inheritance) --------------
fig4, ax4 = plt.subplots(figsize=(12, 5))
colors = {"ACTIVE_1990": "#2a9d8f", "ACTIVE_1970": "#e9c46a", "RETIRED_1955": "#e76f51"}
for cid in ["ACTIVE_1990", "ACTIVE_1970", "RETIRED_1955"]:
    sub = cohort_hc_df[cohort_hc_df["cohort_id"] == cid]
    if not sub.empty:
        ax4.plot(sub["year"], sub["headcount"], marker="o", markersize=3,
                 label=cid, color=colors.get(cid))
ax4.set_title("Stage 4 — Cohort Headcount Decline (retirees decay, actives flat until retirement)")
ax4.set_ylabel("Members (fractional)")
ax4.set_xlabel("Year")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()
plt.tight_layout()
_save(fig4, "stage4_v4_cohort_headcount_decline.png")

# --- Plot 5 (Stage 4 specific): UWS path — applied vs technical ----------
fig5, ax5 = plt.subplots(figsize=(11, 5))
plot_df = verlust_df.sort_values("year")
ax5.plot(plot_df["year"], plot_df["applied_uws"] * 100,
         marker="o", color="#e76f51", label="Applied UWS (Stage4Dynamics)")
ax5.plot(plot_df["year"], plot_df["technical_uws"] * 100,
         marker="s", color="#2a9d8f", label="Technical UWS = 1/ä_x")
ax5.axhline(policy.conversion_rate * 100, color="grey", linestyle=":",
            linewidth=0.8, label=f"policy.conversion_rate ({policy.conversion_rate:.2%})")
ax5.set_title("Stage 4 — Applied vs Technical UWS per Retirement Event")
ax5.set_ylabel("UWS in %")
ax5.set_xlabel("Pensionierungsjahr")
ax5.grid(True, linestyle="--", alpha=0.5)
ax5.legend()
plt.tight_layout()
_save(fig5, "stage4_v5_uws_path.png")

# --- Plot 6 (Stage 4 specific): Pensionierungsverlust per Jahr ----------
fig6, ax6 = plt.subplots(figsize=(11, 5))
agg = (verlust_df.groupby("year")["verlust_total"].sum()
       .sort_index())
ax6.bar(agg.index.astype(str), agg.values, color="#e76f51")
ax6.axhline(0, color="black", linewidth=0.6)
ax6.set_title("Stage 4 — Pensionierungsverlust pro Jahr (Σ über alle Cohorts)")
ax6.set_ylabel("Verlust in CHF")
ax6.set_xlabel("Pensionierungsjahr")
ax6.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
_save(fig6, "stage4_v6_pensionierungsverlust_per_year.png")

plt.show()
