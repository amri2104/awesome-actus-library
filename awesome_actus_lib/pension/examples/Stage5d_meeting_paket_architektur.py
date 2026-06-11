"""Stage5d_meeting_paket_architektur.py — Meeting-Paket Block 3.

Erzeugt das Architekturdiagramm der Pension-Extension als PNG
(output/meeting_paket/architektur.png, 16:10, 200 dpi) — reines
matplotlib, keine neuen Dependencies, keine Modelllogik.

Ebenen (oben -> unten):
  1. ACTUS-Engine (Assets) und BVG-Engine (Liabilities)
  2. gemeinsames CashFlowStream-Schema
  3. AAL-Analyse (unveraendert) und Stage-5c-Kopplung (VK_t, Asset-Roll, DG_t)
  4. Stage-5d-Container (Rebalancing, Sanierungs-Trigger) mit DG_t- und
     SB-Inflow-Pfeilen
  + rot gestrichelt: Minderverzinsung -> aendert Liability-Events -> Outlook
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "output", "meeting_paket")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

FIG_W, FIG_H = 16.0, 10.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

C_ASSET = "#dbe9f6"     # hellblau
C_LIAB = "#fdebd0"      # hellorange
C_SCHEMA = "#eaeaea"    # hellgrau
C_AAL = "#e8f6e8"       # hellgruen
C_5C = "#f3e8f6"        # helllila
C_5D = "#fff6db"        # hellgelb
EDGE = "#444444"


def box(x, y, w, h, title, lines, fc, title_fs=15, line_fs=12.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor=EDGE, linewidth=1.4, zorder=2,
    ))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - i * 0.40, ln, ha="center",
                va="center", fontsize=line_fs, zorder=3)


def arrow(p0, p1, color=EDGE, lw=1.8, style="-|>", ls="-", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=18, linewidth=lw,
        color=color, linestyle=ls, zorder=z,
        connectionstyle=f"arc3,rad={rad}",
    ))


# --- Ebene 1: die beiden Engines --------------------------------------------
box(0.6, 8.0, 6.6, 1.7, "ACTUS-Engine — Asset-Seite",
    ["PAM-Kontrakte (Bonds, Cash) in Portfolio",
     "PublicActusService.generateEvents(...)"], C_ASSET)
box(8.8, 8.0, 6.6, 1.7, "BVG-Engine — Liability-Seite",
    ["PensionPolicy · Cohort · PensionFund · Stage4Dynamics",
     "Closed/Open/DynamicFundSimulator · MortalityTable"], C_LIAB)

# --- Ebene 2: gemeinsames Event-Schema ---------------------------------------
box(2.6, 6.1, 10.8, 1.3, "Gemeinsames CashFlowStream-Schema",
    ["events_df: time · type · payoff · contractId   (+ = Zufluss, − = Abfluss)"],
    C_SCHEMA)
arrow((3.9, 8.0), (6.0, 7.46))
arrow((12.1, 8.0), (10.0, 7.46))

# --- Ebene 3: Analyse-Schicht -------------------------------------------------
box(0.6, 3.8, 6.6, 1.8, "AAL-Analyse (unverändert)",
    ["LiquidityAnalysis · ValueAnalysis · IncomeAnalysis",
     "ALMAnalysis: Netto-Liquidität, NPV-Funding-Ratio"], C_AAL)
box(8.8, 3.8, 6.6, 1.8, "Stage-5c-Kopplung (generate-then-pair)",
    ["simulate_liability_paths → LiabilityPath (Survivor-Trace)",
     "VK_t pro Pfad · Asset-Roll (Buchwert) · DG_t = V_t / VK_t"], C_5C)
arrow((6.0, 6.1), (3.9, 5.66))
arrow((10.0, 6.1), (12.1, 5.66))

# --- Ebene 4: Stage-5d-Container ----------------------------------------------
box(3.4, 0.5, 9.2, 2.3, "Stage 5d — GoingConcernFundingRatioAnalysis",
    [], C_5D)
box(3.8, 0.75, 4.1, 1.35, "RebalancingPolicy",
    ["Reinvest + Fixed-Mix,", "Shift ab Jahr t"], "white", title_fs=13.5, line_fs=12)
box(8.1, 0.75, 4.1, 1.35, "SanierungsPolicy",
    ["DG$_{t-1}$ < Trigger ⇒", "SB$_t$ = f · SAV_CONTRIB$_t$"], "white",
    title_fs=13.5, line_fs=12)

# DG_t-Feedback in den 5d-Container (in-loop verfuegbar)
arrow((11.2, 3.8), (10.2, 2.86))
ax.text(11.45, 3.35, "DG$_t$ pro Pfad (in-loop)", fontsize=12.5, color=EDGE)

# SB-Inflow zurueck in den Asset-Roll (reiner Asset-Inflow, Log SANIERUNG_SB)
arrow((12.6, 1.6), (13.6, 4.0), rad=-0.35, color="#1a7a1a", lw=2.0)
ax.text(13.75, 2.3, "SB-Inflow\n(nur Assets,\nLog: SANIERUNG_SB)",
        fontsize=12.5, color="#1a7a1a")

# Rebalancing wirkt auf den Asset-Roll (Buchtransfer BONDS <-> EQUITY)
arrow((5.0, 2.86), (9.4, 3.8), rad=0.18)
ax.text(4.3, 3.25, "Buchtransfer\nBONDS ↔ EQUITY", fontsize=12.5, color=EDGE)

# --- Outlook: Minderverzinsung (NICHT gebaut) ---------------------------------
# Der rote Pfeil schneidet bewusst die 5c-Kopplung: Minderverzinsung wirkt
# aufs AGH, also auf die Liability-Event-Generierung — genau diese Schicht
# wuerde brechen (Pfade muessten neu generiert werden).
arrow((3.0, 1.95), (10.2, 8.0), rad=-0.18, color="#c1121f", lw=2.2, ls="--")
ax.text(0.35, 2.55,
        "Outlook — nicht gebaut:\nMinderverzinsung\n→ ändert Liability-Events\n"
        "→ bricht generate-then-pair",
        fontsize=11.5, color="#c1121f", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#c1121f", linestyle="--"))

ax.set_title(
    "Architektur: ACTUS-Assets + BVG-Engine über ein gemeinsames Event-Schema "
    "— Stage 5c koppelt, Stage 5d steuert",
    fontsize=17, fontweight="bold", pad=14,
)

_path = os.path.join(OUT_DIR, "architektur.png")
fig.savefig(_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {_path}")
