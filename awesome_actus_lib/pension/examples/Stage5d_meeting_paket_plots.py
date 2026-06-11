"""Stage5d_meeting_paket_plots.py — Meeting-Paket Block 2 (Folien-Plots).

Liest die Rohdaten des kanonischen 200-Pfad-Laufs
(output/meeting_paket/meeting_run200.pkl, erzeugt von
Stage5d_meeting_paket_run.py) und regeneriert die vier Kernplots in
Praesentationsqualitaet — reine Darstellung, keine Logik, kein Re-Run.

Vorgaben: 200 dpi, weisser Hintergrund, Schriftgroesse >= 14, konsistente
Szenario-Farben (A' grau, B blau, C gruen, D orange), Titel als Aussage.

Output -> output/meeting_paket/:
  dg_fan_C.png, p_underfunded_alle.png, weight_paths_B_vs_C.png,
  sb_kostenverteilung.png
"""

import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(_REPO, "output", "meeting_paket")

with open(os.path.join(OUT_DIR, "meeting_run200.pkl"), "rb") as fh:
    run = pickle.load(fh)

N_PATHS = run["n_paths"]
START_YEAR = run["start_year"]
fan_dates = run["fan_dates"]
dists = run["dists"]
d_T = fan_dates[-1]

KEY_A = "A' (Buy-and-Hold, GC-Engine)"
KEY_B = "B (Fix-Mix 40/60)"
KEY_C = "C (Shift 60/40 ab Jahr 5)"
KEY_D = "D (B + Sanierung)"

# Konsistente Szenario-Farben fuer das ganze Deck.
COLORS = {
    KEY_A: "#7f7f7f",   # grau
    KEY_B: "#1f77b4",   # blau
    KEY_C: "#2ca02c",   # gruen
    KEY_D: "#ff7f0e",   # orange
}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

_x = [np.datetime64(d) for d in fan_dates]


def _save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Plot 1 — DG-Quantilfaecher Szenario C
# =============================================================================
q = {p: np.array([np.percentile(dists[KEY_C][d], p) for d in fan_dates])
     for p in (5, 25, 50, 75, 95)}
col = COLORS[KEY_C]

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.fill_between(_x, q[5] * 100, q[95] * 100, color=col, alpha=0.15,
                label="90%-Band (p5–p95)")
ax.fill_between(_x, q[25] * 100, q[75] * 100, color=col, alpha=0.35,
                label="50%-Band (p25–p75)")
ax.plot(_x, q[50] * 100, color=col, marker="o", linewidth=2.5,
        label="Median (p50)")
ax.axhline(100, color="red", linestyle="--", linewidth=1.5,
           label="100% Deckung")
fig.suptitle("Umschichtung auf 60% Bonds ab Jahr 5 verengt die "
             "Deckungsgrad-Bandbreite", fontsize=18, fontweight="bold")
ax.set_title(f"Szenario C — DG-Quantilfächer, {N_PATHS} Pfade "
             f"(Median am Horizont: {q[50][-1]:.0%})", color="#555555")
ax.set_xlabel("Stichtag")
ax.set_ylabel("Deckungsgrad (%)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
fig.tight_layout()
_save(fig, "dg_fan_C.png")


# =============================================================================
# Plot 2 — P(DG_t < 100%), alle Szenarien
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6.5))
for key in (KEY_A, KEY_B, KEY_C, KEY_D):
    p_under = np.array([float(np.mean(dists[key][d] < 1.0)) for d in fan_dates])
    ax.plot(_x, p_under * 100, marker="o", linewidth=2.5,
            color=COLORS[key], label=f"{key}  (Ende: {p_under[-1]:.0%})")
p_d_end = float(np.mean(dists[KEY_D][d_T] < 1.0))
p_b_end = float(np.mean(dists[KEY_B][d_T] < 1.0))
fig.suptitle("Sanierungsbeiträge senken das Unterdeckungsrisiko stärker "
             "als jede Allokationswahl", fontsize=18, fontweight="bold")
ax.set_title(f"P(DG < 100%) je Stichtag, {N_PATHS} Pfade — am Horizont: "
             f"D {p_d_end:.0%} vs. B {p_b_end:.0%}", color="#555555")
ax.set_xlabel("Stichtag")
ax.set_ylabel("P(DG < 100%) (%)")
ax.set_ylim(-2, 102)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
fig.tight_layout()
_save(fig, "p_underfunded_alle.png")


# =============================================================================
# Plot 3 — Gewichtspfade B vs C (post-rebalance, Pfad 0)
# =============================================================================
reb_b = run["wp_b"][run["wp_b"]["rebalanced"]]
reb_c = run["wp_c"][run["wp_c"]["rebalanced"]]

fig, ax = plt.subplots(figsize=(12, 6.5))
# B als breites halbtransparentes Band unten, C als schmale Linie oben —
# so bleiben beide sichtbar, wo die Gewichte exakt uebereinanderliegen.
ax.step(reb_b["date"], reb_b["w_EQUITY"] * 100, where="post", linewidth=7,
        alpha=0.45, color=COLORS[KEY_B], label="B: Aktienquote")
ax.step(reb_b["date"], reb_b["w_BONDS"] * 100, where="post", linewidth=5,
        alpha=0.45, linestyle="--", color=COLORS[KEY_B], label="B: Bondquote")
ax.step(reb_c["date"], reb_c["w_EQUITY"] * 100, where="post", linewidth=2.5,
        color=COLORS[KEY_C], label="C: Aktienquote")
ax.step(reb_c["date"], reb_c["w_BONDS"] * 100, where="post", linewidth=2,
        linestyle="--", color=COLORS[KEY_C], label="C: Bondquote")
ax.axvline(np.datetime64(f"{START_YEAR + 5}-01-01"), color="grey",
           linestyle=":", linewidth=2, label="Shift ab Jahr 5")
fig.suptitle("Strategie C schichtet ab Jahr 5 von 60% auf 40% Aktien um — "
             "B hält den Fixed-Mix", fontsize=18, fontweight="bold")
ax.set_title("Gewichte nach jedem Rebalancing-Termin (Pfad 0, "
             "repräsentativ für alle Pfade)", color="#555555")
ax.set_xlabel("Rebalancing-Termin")
ax.set_ylabel("Gewicht (%)")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(ncol=2, loc="upper right")
fig.tight_layout()
_save(fig, "weight_paths_B_vs_C.png")


# =============================================================================
# Plot 4 — SB-Kostenverteilung (Szenario D)
# =============================================================================
cum_sb = np.asarray(run["cum_sb"], dtype=float)
sb_years = run["sb_years"]
year_frac = np.asarray(run["year_frac"], dtype=float)
col_d = COLORS[KEY_D]
sb_med = float(np.median(cum_sb)) / 1e6
sb_p95 = float(np.percentile(cum_sb, 95)) / 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.bar(sb_years, year_frac * 100, color=col_d, alpha=0.85)
ax1.set_title(f"Anteil Pfade mit SB je Jahr\n"
              f"(aktive Pfadjahre gesamt: {run['share_active']:.0%})")
ax1.set_xlabel("Zahlungsjahr")
ax1.set_ylabel("Anteil Pfade (%)")
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.hist(cum_sb / 1e6, bins=15, color=col_d, alpha=0.85)
ax2.axvline(sb_med, color="black", linestyle="--", linewidth=2,
            label=f"Median CHF {sb_med:,.0f} Mio.")
ax2.axvline(sb_p95, color="#c1121f", linestyle=":", linewidth=2,
            label=f"p95 CHF {sb_p95:,.0f} Mio.")
ax2.set_title("Kumulierte SB pro Pfad (Verteilung)")
ax2.set_xlabel("Kumulierte SB (CHF Mio.)")
ax2.set_ylabel("Anzahl Pfade")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

fig.suptitle(f"Was die Sanierung kostet: meist moderat "
             f"(Median CHF {sb_med:,.0f} Mio.), im Tail teuer "
             f"(p95 CHF {sb_p95:,.0f} Mio.)",
             fontsize=18, fontweight="bold")
fig.tight_layout()
_save(fig, "sb_kostenverteilung.png")

print("Block 2 fertig.")
