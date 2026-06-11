"""Stage5d_meeting_paket_run.py — Meeting-Paket Block 1 (kanonischer Lauf).

Fuehrt den UNVERAENDERTEN Stage5d_going_concern_quickstart mit N_PATHS=200
aus (ueber die bestehende Umgebungsvariable STAGE5D_N_PATHS — gleiche
Seeds-Struktur SEED_ASSETS=42 / SEED_LIAB=4242 / SEED_EQ=777, gleiche
Szenarien A'/B/C/D, alle Spec-Asserts laufen mit). Dieses Skript enthaelt
KEINE Modelllogik: es importiert den Quickstart als Modul, misst die
Laufzeit und liest danach nur dessen Modul-Variablen aus.

Output -> output/meeting_paket/:
  - summary_kpis.md      KPI-Tabelle pro Szenario am Horizontende
  - meeting_run200.pkl   Rohdaten fuer die Plot-Skripte (Block 2)
"""

import importlib.util
import os
import pickle
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
# N_PATHS=200 fuer den kanonischen Lauf; der Quickstart liest STAGE5D_N_PATHS.
os.environ["STAGE5D_N_PATHS"] = os.environ.get("MEETING_N_PATHS", "200")

import numpy as np

_EXAMPLES = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_EXAMPLES, "..", "..", ".."))
sys.path.insert(0, _REPO)

OUT_DIR = os.path.join(_REPO, "output", "meeting_paket")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 1. Quickstart unveraendert ausfuehren (inkl. aller Spec-Asserts)
# =============================================================================
print("=" * 78)
print("MEETING-PAKET BLOCK 1 — kanonischer Stage-5d-Lauf, N_PATHS="
      + os.environ["STAGE5D_N_PATHS"])
print("=" * 78)

_t0 = time.time()
_spec = importlib.util.spec_from_file_location(
    "stage5d_quickstart_run",
    os.path.join(_EXAMPLES, "Stage5d_going_concern_quickstart.py"),
)
qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qs)  # laeuft komplett durch — Assert-Fehler brechen hier ab
RUNTIME_S = time.time() - _t0

print(f"\n[Meeting-Paket] Quickstart komplett durchgelaufen, alle Asserts gruen.")
print(f"[Meeting-Paket] Laufzeit: {RUNTIME_S:,.1f} s")


# =============================================================================
# 2. KPIs am Horizontende aus den Quickstart-Variablen ableiten
# =============================================================================
d_T = qs.fan_dates[-1]

rows = []
for name, scen_dists in qs.dists.items():
    dist = scen_dists[d_T]
    rows.append({
        "szenario": name,
        "mean": float(np.mean(dist)),
        "p5": float(np.percentile(dist, 5)),
        "p50": float(np.percentile(dist, 50)),
        "p95": float(np.percentile(dist, 95)),
        "p_under": float(np.mean(dist < 1.0)),
    })

cum_sb = qs.cum_sb                      # kumulierte SB pro Pfad (CHF), Szenario D
share_active = qs.share_active          # Anteil aktiver SB-Pfadjahre
sb_median = float(np.median(cum_sb))
sb_mean = float(np.mean(cum_sb))
sb_p95 = float(np.percentile(cum_sb, 95))

lines = []
lines.append("# Meeting-Paket — Summary-KPIs (kanonischer Lauf)\n")
lines.append(f"- Skript: `examples/Stage5d_going_concern_quickstart.py` "
             f"(unveraendert, via `STAGE5D_N_PATHS`)")
lines.append(f"- Pfade: **N_PATHS = {qs.N_PATHS}**, Horizont: "
             f"{qs.HORIZON_YEARS} Jahre (t0 = {qs.BASE_DATE}, "
             f"Horizontende = {d_T})")
lines.append(f"- Seeds: SEED_ASSETS={qs.SEED_ASSETS}, SEED_LIAB={qs.SEED_LIAB}, "
             f"SEED_EQ={qs.SEED_EQ} (identisch zum Quickstart)")
lines.append(f"- Annahmen: equity_return={qs.EQ_RETURN:.0%}, "
             f"equity_sigma={qs.EQ_SIGMA:.0%}, bond_yield={qs.BOND_YIELD:.0%}, "
             f"DG_t0 = 107.6% (kalibriert)")
lines.append(f"- **Alle Spec-Asserts gruen** (1, 2, 3, 4, 5, DG_t0-Kalibrierung, "
             f"A'-Konfiguration); Laufzeit des Gesamtlaufs: "
             f"**{RUNTIME_S:,.1f} s**\n")

lines.append(f"## Deckungsgrad am Horizontende ({d_T})\n")
lines.append("| Szenario | mean | p5 | p50 | p95 | P(DG<100%) |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['szenario']} | {r['mean']:.1%} | {r['p5']:.1%} | "
        f"{r['p50']:.1%} | {r['p95']:.1%} | {r['p_under']:.1%} |"
    )

lines.append("\n## Sanierungsbeitraege Szenario D "
             "(trigger_dg=1.00, sb_factor=0.5)\n")
lines.append("| KPI | Wert |")
lines.append("|---|---:|")
lines.append(f"| Anteil aktiver SB-Pfadjahre (von {qs.N_PATHS} Pfaden x "
             f"{qs.HORIZON_YEARS} Jahren) | {share_active:.1%} |")
lines.append(f"| Median kumulierte SB pro Pfad | CHF {sb_median/1e6:,.1f} Mio. |")
lines.append(f"| Mittel kumulierte SB pro Pfad | CHF {sb_mean/1e6:,.1f} Mio. |")
lines.append(f"| p95 kumulierte SB pro Pfad | CHF {sb_p95/1e6:,.1f} Mio. |")
lines.append("\nHinweis: SB werden als reiner Asset-Inflow gebucht und separat "
             "als `SANIERUNG_SB` geloggt (`gc_events`), nie im Liability-Stream "
             "— D minus B ist die reine Massnahmenwirkung (gleiche Seeds, "
             "gleiche Liability-Pfade).")

_kpi_path = os.path.join(OUT_DIR, "summary_kpis.md")
with open(_kpi_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"[Meeting-Paket] Saved: {_kpi_path}")


# =============================================================================
# 3. Rohdaten fuer Block 2 (Plots) sichern — reine Darstellungs-Daten
# =============================================================================
payload = {
    "n_paths": qs.N_PATHS,
    "horizon_years": qs.HORIZON_YEARS,
    "start_year": qs.START_YEAR,
    "fan_dates": qs.fan_dates,
    "dists": qs.dists,                  # {szenario: {datum: np.ndarray}}
    "wp_b": qs.wp_b,                    # Gewichtspfad B (Pfad 0)
    "wp_c": qs.wp_c,                    # Gewichtspfad C (Pfad 0)
    "cum_sb": qs.cum_sb,                # kumulierte SB pro Pfad (D)
    "sb_years": qs._sb_years,           # Kalenderjahre der SB-Statistik
    "year_frac": qs._year_frac,         # Anteil Pfade mit SB je Jahr
    "share_active": qs.share_active,
    "runtime_s": RUNTIME_S,
}
_pkl_path = os.path.join(OUT_DIR, "meeting_run200.pkl")
with open(_pkl_path, "wb") as fh:
    pickle.dump(payload, fh)
print(f"[Meeting-Paket] Saved: {_pkl_path}")

print("\n[Meeting-Paket] Block 1 fertig.")
