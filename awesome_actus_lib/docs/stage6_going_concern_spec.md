# Stage 6 — Going Concern (Goal 2): Implementation Note

## Ziel & Scope

Ausschreibung Goal 2 verlangt, dass Allokationswechsel zwischen Anlageklassen
"adequately simulatable in the risk analysis" sind. Umsetzung in vier
Bausteinen — **alle opt-in**, bei `rebalancing=None` und `sanierung=None`
bleibt Stage 5c **byte-identisch** (Regressionsprinzip wie λ=0 /
cal_year=None).

1. Reinvestment + Fixed-Mix-Rebalancing auf Zielgewichte (Stufe 1)
2. Allokations-Shift-Experiment (Szenarien A'/B/C/D)
3. DG-konditionaler Sanierungsbeitrag als reiner Asset-Inflow (minimales
   Stufe-2-Element)
4. Plots/Dashboard für Review und Thesis-Discussion

**NICHT bauen:** Minder-/Nullverzinsung (verändert AGH → Liability-Events →
bricht generate-then-pair). Nur als Outlook dokumentieren, mit der
Architektur-Regel unten.

## Architektur-Regel (Kernaussage — auch für Thesis-Discussion)

> DG-Feedback ist genau dann mit der generate-then-pair-Architektur
> vereinbar, wenn die Massnahme keine Liability-Events erzeugt oder
> verändert. Sanierungsbeiträge (Art. 65d BVG) werden nicht dem
> Altersguthaben gutgeschrieben ⇒ reiner Asset-Inflow ⇒ kompatibel. Der
> sequenzielle per-Pfad-Asset-Roll wird in Stage 6 in der Subclass
> `GoingConcernFundingRatioAnalysis` eingeführt; dort ist DG_t in-loop
> verfügbar. Minder-/Nullverzinsung wirkt aufs AGH ⇒ Liability-Pfade
> müssten neu generiert werden ⇒ Outlook (Stufe 2 voll).

## Implementierung

**Ort:** neues Modul `awesome_actus_lib/pension/analysis/going_concern.py` +
Beispiel `examples/Stage6_going_concern_quickstart.py`.

**Disziplin:** `stochastic_funding_ratio.py` NICHT verändern (höchstens einen
Hook extrahieren, falls unvermeidbar); bevorzugt Subclass.

**WICHTIG:** Vor dem Codieren den realen Hookpunkt verifizieren — wie
`StochasticFundingRatioAnalysis._path_V` den Asset-Roll heute genau bucht
(skalarer Buchwert? Return-Quelle aus den ACTUS-PAM-Events?). Die Spec geht
von einem sequenziellen Jahres-Roll mit verfügbarem VK_t pro Pfad aus
(5c-Stand).

```python
@dataclass(frozen=True)
class RebalancingPolicy:
    target_weights: dict[str, float]      # z.B. {"BONDS": 0.40, "EQUITY": 0.60}; assert sum == 1
    frequency_years: int = 1
    cost_bps: float = 0.0                 # optional, default kostenlos
    shift_at_year: int | None = None      # optional: ab Jahr t neue Gewichte
    shift_weights: dict[str, float] | None = None

@dataclass(frozen=True)
class SanierungsPolicy:
    trigger_dg: float = 1.00              # aktiv wenn DG_{t-1} < trigger
    exit_dg: float = 1.00                 # Hysterese: exit_dg >= trigger_dg
    sb_factor: float = 0.5                # SB_t = sb_factor * SAV_CONTRIB_t (dokumentierte Vereinfachung)
    max_years: int | None = None

class GoingConcernFundingRatioAnalysis(StochasticFundingRatioAnalysis):
    def __init__(self, ..., rebalancing: RebalancingPolicy | None = None,
                 sanierung: SanierungsPolicy | None = None,
                 equity_return: float = 0.03,          # Buchwert-Drift, Mark-to-Model-Vereinfachung
                 equity_sigma: float = 0.0,            # optionaler erwartungstreuer lognormaler Equity-Schock
                 equity_seed: int | None = None,       # reproduzierbare Equity-Pfadkopplung
                 bond_yield: float | None = None,      # Yield auf inkrementellem BONDS-Bestand
                 initial_weights: dict[str, float] = ...):
        ...
```

**Per-Pfad-Roll (Override von `_path_V`), Reihenfolge pro Jahr:**

1. **Accrue** je Bucket: BONDS aus den ACTUS-PAM-`IP`-Events der
   Originalkontrakte plus optionalem `bond_yield` auf dem inkrementellen
   BONDS-Bestand oberhalb des Anfangsbuchwerts; EQUITY mit `equity_return`
   und optionalem erwartungstreuem lognormalem Jahresschock (`equity_sigma`,
   `equity_seed`).
2. **Net-Liability-CF** aus dem gepairten `LiabilityPath`
   zuführen/entnehmen, proportional zu aktuellen Gewichten.
3. **Sanierung:** wenn aktiv (DG_{t-1} < trigger, Hysterese via exit_dg,
   max_years beachten): `SB_t = sb_factor * SAV_CONTRIB_t` als Asset-Inflow
   buchen. Log als `SANIERUNG_SB` in einen **separaten** gc_event-Log pro
   Pfad — NICHT in den Liability-Stream (Provenienz sauber halten).
4. **Rebalance** auf target_weights alle `frequency_years` (Buchtransfer;
   `cost_bps` optional abziehen). Ab `shift_at_year` gelten `shift_weights`.
5. **DG_t = ΣA_t / VK_t** speichern.
6. `RISK_CONTRIB` weiterhin aus dem Roll filtern (Fix vom Mai nicht
   regressieren!).

**Payroll-Frage:** SB ∝ Lohnsumme wäre exakt, Lohnsumme ist aber nicht
direkt im Event-Stream. Pragmatische, dokumentierte Wahl: SB ∝ SAV_CONTRIB
(Sparbeiträge), `sb_factor` als freier Parameter. In Limitations erwähnen.

## Kalibrierung fürs Beispiel

- Bestand aus dem Stage-4-Beispiel übernehmen (8 aktive Kohorten +
  RETIRED_1955, EntryPolicy 20/Jahr).
- `initial_assets` so setzen, dass **DG_t0 ≈ 107.6%** (Complementa YE 2023)
  — nicht 170% wie im 5c-Beispiel, sonst feuert der Sanierungs-Trigger auf
  keinem Pfad und Szenario D ist leer.

## Experiment (Quickstart-Skript)

| Szenario | Setup |
|---|---|
| ref_5c | Legacy Stage 5c: reine Regressionsreferenz, kein Strategievergleich |
| A' | echtes Buy-and-Hold in der Going-Concern-Engine; keine Rebalancing-Termine im Horizont |
| B | Reinvest + Fixed-Mix 40% BONDS / 60% EQUITY, jährlich |
| C | wie B, Shift auf 60/40 ab Jahr 5 ("mortgage→bonds"-Beispiel der Ausschreibung) |
| D | wie B + SanierungsPolicy(trigger=1.00, sb_factor=0.5) |

`Stage6_going_concern_quickstart.py` vergleicht A'/B/C/D. Legacy Stage 5c
wird nur noch für den Assert verwendet:
`rebalancing=None, sanierung=None` muss identisch zur Stage-5c-Verteilung
bleiben.

**Outputs (PNG + Dashboard-Tab):**

- DG-Quantilfächer (5/25/50/75/95%) je Szenario über t
- P(DG_t < 100%)-Kurve, alle Szenarien in einem Plot
- SB-Statistik: Anteil aktiver Pfad-Jahre, kumulierte SB in CHF (Verteilung)
- Gewichtspfade B vs. C (zeigt den Shift sichtbar)

## Akzeptanz-Asserts (ins Skript, Stil wie Stage 1)

1. `rebalancing=None, sanierung=None` ⇒ Ergebnisse identisch zu 5c
   (`np.allclose` auf DG-Matrix; ideal byte-gleiche CSV).
2. SB-Buchungen nur in Jahren mit DG_{t-1} < trigger; Summe SB > 0 in D,
   == 0 in B/C.
3. Nach jedem Rebalance: |w − target| < 1e-12 (vor Kosten).
4. Sanity: mean(DG_T) in D ≥ B (SB wirkt in die richtige Richtung).
5. C: Gewichtspfad dokumentiert den Shift ab Jahr 5.

## Dokumentierte Limitations

- Equity als Buchwert-Drift mit optionalem lognormalem Jahresschock.
- SB ∝ Sparbeiträge (`SAV_CONTRIB`) statt exakter Lohnsumme.
- Keine Minder-/Nullverzinsung, weil diese AGH und Liability-Events ändern
  würde (→ Outlook mit Architektur-Regel).
- Keine Transaktionskosten-Kalibrierung.

## Implementierter Stand

- `equity_sigma` erzeugt erwartungstreue lognormale Jahresschocks um
  `equity_return`; `equity_seed` koppelt die Equity-Pfade reproduzierbar über
  A'/B/C/D.
- `bond_yield` ist eine Buch-Yield-Approximation auf den positiven
  inkrementellen BONDS-Bestand. Die Original-Bonds bleiben durch ihre
  ACTUS-`IP`-Events verankert.
- Sanierungsbeiträge werden als `SANIERUNG_SB` in einem separaten
  `gc_events(...)`-Log geführt.
- Der Liability-Stream bleibt unverändert; `SANIERUNG_SB` wird nie als
  Liability-Event geschrieben.

## Bewusst Nicht Umgesetzt

Kein obli/überobli-Split, keine Steuern/Verwaltungskosten, keine
Minderverzinsung, keine Kostenkalibrierung.
