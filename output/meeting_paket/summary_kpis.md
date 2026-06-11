# Meeting-Paket — Summary-KPIs (kanonischer Lauf)

- Skript: `examples/Stage5d_going_concern_quickstart.py` (unveraendert, via `STAGE5D_N_PATHS`)
- Pfade: **N_PATHS = 200**, Horizont: 30 Jahre (t0 = 2025-01-01, Horizontende = 2055-01-01)
- Seeds: SEED_ASSETS=42, SEED_LIAB=4242, SEED_EQ=777 (identisch zum Quickstart)
- Annahmen: equity_return=3%, equity_sigma=10%, bond_yield=2%, DG_t0 = 107.6% (kalibriert)
- **Alle Spec-Asserts gruen** (1, 2, 3, 4, 5, DG_t0-Kalibrierung, A'-Konfiguration); Laufzeit des Gesamtlaufs: **57.5 s**

## Deckungsgrad am Horizontende (2055-01-01)

| Szenario | mean | p5 | p50 | p95 | P(DG<100%) |
|---|---:|---:|---:|---:|---:|
| A' (Buy-and-Hold, GC-Engine) | 150.7% | 55.6% | 130.5% | 283.4% | 30.0% |
| B (Fix-Mix 40/60) | 149.3% | 56.6% | 137.4% | 266.3% | 26.0% |
| C (Shift 60/40 ab Jahr 5) | 139.4% | 65.2% | 129.8% | 225.0% | 25.0% |
| D (B + Sanierung) | 159.1% | 80.1% | 143.3% | 266.3% | 15.5% |

## Sanierungsbeitraege Szenario D (trigger_dg=1.00, sb_factor=0.5)

| KPI | Wert |
|---|---:|
| Anteil aktiver SB-Pfadjahre (von 200 Pfaden x 30 Jahren) | 20.4% |
| Median kumulierte SB pro Pfad | CHF 13.0 Mio. |
| Mittel kumulierte SB pro Pfad | CHF 21.7 Mio. |
| p95 kumulierte SB pro Pfad | CHF 80.2 Mio. |

Hinweis: SB werden als reiner Asset-Inflow gebucht und separat als `SANIERUNG_SB` geloggt (`gc_events`), nie im Liability-Stream — D minus B ist die reine Massnahmenwirkung (gleiche Seeds, gleiche Liability-Pfade).
