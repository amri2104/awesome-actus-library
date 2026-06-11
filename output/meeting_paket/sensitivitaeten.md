# Meeting-Paket — Sensitivitaeten (Stage 5d)

- Skript: `examples/Stage5d_variations.py`, **N_PATHS = 60**, Horizont 30 Jahre, KPIs am 2055-01-01
- Setup identisch zum Quickstart (Seeds 42/4242/777, DG_t0 = 107.6%), nur bestehende oeffentliche API
- Achtung: kleinerer Lauf als der kanonische 200er — Zahlen sind nicht 1:1 mit `summary_kpis.md` vergleichbar

## equity_sigma — Szenario B (Fix-Mix 40/60)

| equity_sigma | mean DG | p5 | P(DG<100%) |
|---|---:|---:|---:|
| 0.05 | 145.7% | 95.1% | 10.0% |
| 0.10  *(Basisfall B)* | 138.2% | 50.9% | 35.0% |
| 0.15 | 131.3% | 19.5% | 55.0% |

## sanierung.trigger_dg — Szenario D (B + Sanierung, sb_factor=0.5)

| trigger_dg | mean DG | p5 | P(DG<100%) |
|---|---:|---:|---:|
| 0.95 | 146.6% | 71.1% | 25.0% |
| 1.00  *(Basisfall D)* | 149.7% | 76.3% | 25.0% |
