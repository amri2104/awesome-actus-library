"""Case Study II: a pure-ACTUS counterfactual of the pension liabilities.

ACTUS can emit the payment mechanics as a term-certain ANN annuity but cannot
apply biometric survival, so it overstates the obligation relative to the
mortality-weighted actuarial annuity the BVG engine computes. The gap is
quantified per age and at fund level.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awesome_actus_lib import ANN, PAM, Portfolio, PublicActusService
from awesome_actus_lib.pension import (
    PensionPolicy,
    ek2001_2005,
    annuity_due,
)

OUT = "/Users/amrighodra/Bachelorarbeit/AAL/output/thesis_figures"
os.makedirs(OUT, exist_ok=True)

START_DATE = "2025-01-01T00:00:00"
START_YEAR = 2025
policy = PensionPolicy()
i_tech = policy.technical_rate
OMEGA = policy.terminal_age
mort = ek2001_2005(improvement_rate=0.0125, base_year=2003)

KPI = {"meta": {"technical_rate": i_tech, "terminal_age": OMEGA,
                "start_year": START_YEAR}}

print("=" * 78)
print("CASE STUDY II  --  ACTUS-only counterfactual")
print("=" * 78)


def v(i):
    return 1.0 / (1.0 + i)


def annuity_due_certain(n_years, i):
    """Term-certain annuity-due over (n_years+1) payments at rate i, no mortality."""
    vv = v(i)
    return sum(vv ** k for k in range(n_years + 1))


print("\n[1] Annuity factor by age: ACTUS term-certain vs BVG mortality-weighted")
print("-" * 60)

ages = [65, 70, 75, 80, 85]
rows = []
for x in ages:
    a_mort = annuity_due(x, "f", i_tech, mort, OMEGA, cal_year=START_YEAR)
    a_cert = annuity_due_certain(OMEGA - x, i_tech)
    rows.append({
        "age": x,
        "years_to_omega": OMEGA - x,
        "a_mortality_BVG": a_mort,
        "a_termcertain_ACTUS": a_cert,
        "overstatement_pct": 100.0 * (a_cert / a_mort - 1.0),
    })
age_df = pd.DataFrame(rows)
print(age_df.to_string(index=False, float_format=lambda z: f"{z:,.4f}"))
KPI["annuity_by_age"] = age_df.to_dict(orient="records")


print("\n[2] ACTUS ANN proof-of-mechanics (live engine)")
print("-" * 60)

P = 40_000.0
HC = 100
x0 = 70
n = OMEGA - x0
MATURITY = f"{START_YEAR + n}-12-31T00:00:00"

# Choose N = P * a_immediate(n,i) so the level ANN installment is ~ P.
a_imm = sum(v(i_tech) ** k for k in range(1, n + 1))
N_ann = P * a_imm
ann = ANN(
    contractID="PENSION_70",
    statusDate=START_DATE,
    contractDealDate=START_DATE,
    currency="CHF",
    notionalPrincipal=round(N_ann, 2),
    initialExchangeDate=START_DATE,
    maturityDate=MATURITY,
    nominalInterestRate=i_tech,
    dayCountConvention="30E360",
    contractRole="RPL",
    cycleAnchorDateOfInterestPayment=f"{START_YEAR + 1}-01-01T00:00:00",
    cycleOfInterestPayment="P1YL0",
    cycleAnchorDateOfPrincipalRedemption=f"{START_YEAR + 1}-01-01T00:00:00",
    cycleOfPrincipalRedemption="P1YL0",
    counterpartyID="PENSIONER",
    creatorID="PK_ALM",
)

service = PublicActusService()
ann_cfs = service.generateEvents(Portfolio([ann]))
edf = ann_cfs.events_df.copy()
edf["time"] = pd.to_datetime(edf["time"])
# annual payment = principal-redemption + interest-payment legs
pay_mask = edf["type"].isin(["PR", "IP"])
annual_pay = (edf[pay_mask]
              .assign(year=lambda d: d["time"].dt.year)
              .groupby("year")["payoff"].sum())
inst = annual_pay.abs()
inst = inst[inst > 1e-6]
print(f"  ANN notional set to CHF {N_ann:,.0f} so the level installment ~ CHF {P:,.0f}")
print(f"  ACTUS emitted {len(edf)} events; {len(inst)} annual installments.")
print(f"  Mean ACTUS installment: CHF {inst.mean():,.2f}  (target CHF {P:,.0f})")

# Discount the ACTUS-emitted installments at the technical rate -> ACTUS PV.
pv_actus = 0.0
for yr, amt in inst.items():
    k = yr - START_YEAR
    pv_actus += amt * v(i_tech) ** k
pv_actus_per_capita_scaled = pv_actus / inst.mean() * P
print(f"  ACTUS-only PV of the emitted stream (rate {i_tech:.2%}): "
      f"CHF {pv_actus:,.0f}")

a_cert_70 = annuity_due_certain(n, i_tech)
a_mort_70 = annuity_due(x0, "f", i_tech, mort, OMEGA, cal_year=START_YEAR)
pv_termcertain = P * a_cert_70
pv_actuarial = P * a_mort_70
print(f"  Analytic term-certain PV  (no mortality, ACTUS view): CHF {pv_termcertain:,.0f}")
print(f"  Analytic actuarial PV     (mortality, BVG view)     : CHF {pv_actuarial:,.0f}")
print(f"  Overstatement of pure ACTUS: "
      f"{100*(pv_termcertain/pv_actuarial-1):.1f}%")

KPI["actus_proof"] = {
    "pension_per_capita": P, "age": x0, "headcount": HC,
    "ann_notional": N_ann, "n_payments": int(len(inst)),
    "mean_actus_installment": float(inst.mean()),
    "pv_actus_emitted": float(pv_actus),
    "a_termcertain": a_cert_70, "a_mortality": a_mort_70,
    "pv_termcertain_per_capita": pv_termcertain,
    "pv_actuarial_per_capita": pv_actuarial,
    "overstatement_pct": 100 * (pv_termcertain / pv_actuarial - 1),
}


print("\n[3] Fund-level pensioner book: BVG vs ACTUS-only liability and DG")
print("-" * 60)

# Closed pensioner book: 5 cohorts, all female, pension 40k, 100 heads each.
book = [{"age": x, "pension": 40_000.0, "hc": 100} for x in ages]
L_bvg = 0.0
L_actus = 0.0
for c in book:
    a_m = annuity_due(c["age"], "f", i_tech, mort, OMEGA, cal_year=START_YEAR)
    a_c = annuity_due_certain(OMEGA - c["age"], i_tech)
    L_bvg += c["pension"] * c["hc"] * a_m
    L_actus += c["pension"] * c["hc"] * a_c

# Assets calibrated so the (correct) BVG funding ratio is a realistic 105%.
ASSETS = 1.05 * L_bvg
DG_bvg = ASSETS / L_bvg
DG_actus = ASSETS / L_actus
print(f"  Liability (BVG, mortality)      : CHF {L_bvg:,.0f}")
print(f"  Liability (ACTUS, term-certain) : CHF {L_actus:,.0f}  "
      f"(+{100*(L_actus/L_bvg-1):.1f}%)")
print(f"  Assets (calibrated to DG_BVG=105%): CHF {ASSETS:,.0f}")
print(f"  Funding ratio  -- BVG engine    : {DG_bvg:.1%}")
print(f"  Funding ratio  -- ACTUS-only    : {DG_actus:.1%}  "
      f"(spurious underfunding of {100*(DG_bvg-DG_actus):.1f} pp)")

KPI["fund_level"] = {
    "L_bvg": L_bvg, "L_actus": L_actus,
    "L_overstatement_pct": 100 * (L_actus / L_bvg - 1),
    "assets": ASSETS, "DG_bvg": DG_bvg, "DG_actus": DG_actus,
    "DG_gap_pp": 100 * (DG_bvg - DG_actus),
}


print("\n[4] Figures")
print("-" * 60)

# Fig 1: annuity factor by age (term-certain vs mortality)
fig, ax = plt.subplots(figsize=(9.5, 5))
xpos = np.arange(len(ages))
w = 0.38
ax.bar(xpos - w/2, age_df["a_termcertain_ACTUS"], width=w,
       label="ACTUS term-certain  $\\ddot a^{\\,certain}$", color="#e76f51")
ax.bar(xpos + w/2, age_df["a_mortality_BVG"], width=w,
       label="BVG mortality-weighted  $\\ddot a_x$", color="#2a9d8f")
for i, r in age_df.iterrows():
    ax.text(i, r["a_termcertain_ACTUS"] + 0.2,
            f"+{r['overstatement_pct']:.0f}%", ha="center", va="bottom",
            fontsize=9, color="#9d2c1f")
ax.set_xticks(xpos)
ax.set_xticklabels([f"age {a}" for a in ages])
ax.set_ylabel("Annuity factor $\\ddot a$ (years)")
ax.set_title("Case Study II — Annuity factor: ACTUS term-certain vs BVG actuarial\n"
             f"(female, technical rate {i_tech:.2%}, EK 2001–2005 generational, to age {OMEGA})")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
f1 = os.path.join(OUT, "cs2_annuity_by_age.png")
fig.savefig(f1, dpi=150, bbox_inches="tight")
print(f"  Saved: {f1}")

# Fig 2: pension cashflow profile, age-70 cohort
years = list(range(START_YEAR + 1, START_YEAR + n + 1))
certain_stream = [P * HC for _ in years]
kpx = 1.0
exp_stream = []
for k, yr in enumerate(years):
    p = mort.survival_probability(x0 + k, "f", START_YEAR + k)
    kpx *= p
    exp_stream.append(P * HC * kpx)

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(years, exp_stream, certain_stream, color="#e76f51", alpha=0.20,
                label="Phantom payments to the deceased\n(ACTUS-only overstatement)")
ax.plot(years, certain_stream, color="#e76f51", linewidth=2,
        label="ACTUS term-certain (ANN, no mortality)")
ax.plot(years, exp_stream, color="#2a9d8f", linewidth=2, marker="o", markersize=3,
        label="BVG expected (survival-weighted)")
ax.set_xlabel("Payment year")
ax.set_ylabel("Annual pension outflow (CHF)")
ax.set_title(f"Case Study II — Pension outflow of a 100-head age-{x0} cohort\n"
             "ACTUS pays to age 100 regardless of survival; BVG decrements by mortality")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
f2 = os.path.join(OUT, "cs2_cashflow_profile.png")
fig.savefig(f2, dpi=150, bbox_inches="tight")
print(f"  Saved: {f2}")

# Fig 3: fund-level liability + funding ratio
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
axL.bar(["BVG\n(mortality)", "ACTUS-only\n(term-certain)"],
        [L_bvg/1e6, L_actus/1e6], color=["#2a9d8f", "#e76f51"])
axL.set_ylabel("Liability (CHF m)")
axL.set_title("Pensioner-book liability")
axL.grid(True, axis="y", linestyle="--", alpha=0.5)
axR.bar(["BVG\nengine", "ACTUS-only"],
        [DG_bvg*100, DG_actus*100], color=["#2a9d8f", "#e76f51"])
axR.axhline(100, color="red", linestyle="--", linewidth=1, label="100% coverage")
axR.set_ylabel("Funding ratio (%)")
axR.set_title("Reported funding ratio (same assets)")
axR.legend()
axR.grid(True, axis="y", linestyle="--", alpha=0.5)
fig.suptitle("Case Study II — Same obligations, two models: ACTUS-only overstates "
             "the liability and misreports solvency")
plt.tight_layout()
f3 = os.path.join(OUT, "cs2_fund_level.png")
fig.savefig(f3, dpi=150, bbox_inches="tight")
print(f"  Saved: {f3}")

with open(os.path.join(OUT, "cs2_kpis.json"), "w") as fh:
    json.dump(KPI, fh, indent=2)
print(f"  Saved KPI JSON: {os.path.join(OUT, 'cs2_kpis.json')}")
print("\nDONE.")
