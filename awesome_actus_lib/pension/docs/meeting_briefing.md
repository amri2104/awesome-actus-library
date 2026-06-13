# Meeting-Briefing — Pension-ALM-Extension (Stages 1–5d)

Stand: kanonischer Lauf mit `N_PATHS=200`
(`examples/Stage6_meeting_paket_run.py`, alle Spec-Asserts grün,
Laufzeit 57.5 s). Zahlen in diesem Dokument stammen aus
`output/meeting_paket/summary_kpis.md`.

---

## 1. Ziel 1 — Was gebaut wurde (Stages 1–5c) und warum so

### Was

| Stage | Inhalt | Klassen / Dateien |
|---|---|---|
| 1 | Geschlossener Bestand, deterministische Verzinsung, jährliche Events | `PensionPolicy` (`policy.py`), `Cohort` (`cohort.py`), `PensionFund` (`fund.py`), `ClosedFundSimulator` (`simulators/Stage1_closed.py`), Event-Typen in `events.py` |
| 2 | Offener Bestand: deterministische Neueintritte | `OpenFundSimulator` (`simulators/Stage2_open.py`), `EntryPolicy` (`entry.py`) |
| 3 | Deterministische Sterblichkeit (EK 2001–2005, pro Geschlecht) | `MortalityTable` + `ek2001_2005()` (`mortality.py`) |
| 4 | Dynamik: Lohnwachstum, UWS-Pfad (held-forward), Schwellen-Indexierung; t0-Deckungsgrad und Pensionierungsverlust | `DynamicFundSimulator` + `Stage4Dynamics` (`simulators/Stage4_dynamic.py`), `FundingRatioAnalysis` (`analysis/funding_ratio.py`), `RetirementLossAnalysis` / `annuity_due` / `technical_uws` (`analysis/conversion.py`), `ALMAnalysis` (`analysis/alm.py`) |
| 5a | Stochastische Asset-Seite: Hull-White/Vasicek/CIR/Ho-Lee-Zinspfade treiben ACTUS-Kontrakte; GBM-Equity-Sleeve | `StochasticALMAnalysis` (`analysis/stochastic_alm.py`), `CurveCalibrator` (`stochastic_rates/`) |
| 5b | Stochastische Liabilities: binomiale Kohorten-Mortalität pro Pfad; Varianz-Zerlegung | `simulate_liability_paths` + `LiabilityPath` (`simulators/liability_paths.py`), `RiskAttribution` (`analysis/risk_attribution.py`) |
| 5c | Forward-Solvenz-DG als (Zeit × Pfade)-Fläche: `DG_t,i = V_t,i / VK_t,i`, beide Seiten pfadabhängig | `StochasticFundingRatioAnalysis` (`analysis/stochastic_funding_ratio.py`) |

### Warum jede Architekturentscheidung so fiel

**ANN-Kategorienfehler.** Eine BVG-Altersrente als ACTUS-`ANN`-Kontrakt
abzubilden wäre ein Kategorienfehler: `ANN` ist ein amortisierender
*Kredit*vertrag mit vertraglich fixiertem Zahlungsplan. Die BVG-Rente ist
dagegen überlebensabhängig (biometrisch kontingent) und reglementarisch
parametrisiert (UWS-Pfad, Mindestzins, Koordinationsabzug) — ihre Events
folgen keinem Kontraktschema, sondern Regeln über einem Bestand. Deshalb
gibt es keine "ACTUS-Liability-Kontrakte", sondern eine BVG-Engine.

**Behavior-Risikoklasse.** ACTUS trennt Kontrakt-Terme von Risikofaktoren;
nicht-marktliche Kontingenzen (Verhalten, Biometrie) gehören in die
Risikofaktor-Schicht. Genau so ist es gebaut: Sterblichkeit lebt als
`MortalityTable`-Risikofaktor (deterministisch in Stage 3, binomial pro
Pfad in 5b), der die Event-Generierung des Simulators treibt — sie wird
nicht in Kontrakt-Terme gepresst.

**Gemeinsames Event-Schema.** Beide Seiten konvergieren auf das gleiche
AAL-Objekt `CashFlowStream` (`events_df`: `time · type · payoff ·
contractId`; `cohort_id` fungiert als `contractID`). Konsequenz: die
bestehende AAL-Analyse-Schicht (`LiquidityAnalysis`, `ValueAnalysis`,
`IncomeAnalysis`, `ALMAnalysis`) konsumiert Assets und Liabilities
unverändert — kein einziger AAL-Core-File wurde angefasst.

**generate-then-pair.** Stochastik ist als *vorab generierte* Pfadmengen
organisiert: `simulate_liability_paths` (Seed 4242) erzeugt n
Liability-Pfade, `StochasticALMAnalysis` (Seed 42) n Asset-Pfade; gepaart
wird nur über den Index i. Kopplungsinvariante (5c): derselbe
Survivor-Pfad treibt gleichzeitig die Renten-Outflows im Zähler und die
Survivor-Counts im Nenner `VK_t,i` — zwei unabhängige Ziehungen für
Zähler und Nenner wären ein Bug. Die Seeds bleiben getrennt, weil Zins-
und idiosynkratisches Sterblichkeitsrisiko hier echte unabhängige
Risikoquellen sind.

---

## 2. Ziel 2 — Was Stage 6 gebaut hat und warum

### Was (alles in `analysis/going_concern.py`, opt-in)

- **Jahres-Roll:** `GoingConcernFundingRatioAnalysis` führt den
  per-Pfad-Asset-Roll sequenziell Jahr für Jahr (Accrue → Netto-Liability-CF
  → Sanierung → Rebalance → `DG_t` speichern); damit ist `DG_t` *in-loop*
  verfügbar.
- **`RebalancingPolicy`:** Reinvest + Fixed-Mix auf `target_weights`
  (Frequenz, optionale `cost_bps`), optional `shift_at_year` /
  `shift_weights` für den Allokations-Shift.
- **`equity_sigma` / `equity_seed`:** erwartungstreue lognormale
  Jahresschocks um `equity_return`; ein Generator pro Pfad
  (`SeedSequence.spawn`), damit A'/B/C/D pfadweise gekoppelt sind.
- **`bond_yield`:** Buch-Yield auf dem *inkrementellen* BONDS-Bestand über
  dem Anfangsbuchwert (Anker: Complementa-Ertragsrendite ≈ 2.1%).
- **`SanierungsPolicy`:** `SB_t = sb_factor · SAV_CONTRIB_t`, aktiv wenn
  `DG_{t-1} < trigger_dg` (Hysterese via `exit_dg`, Kappung `max_years`);
  Buchung als reiner Asset-Inflow, Log separat als `SANIERUNG_SB` über
  `gc_events(...)` — nie im Liability-Stream.
- **Szenarien:** A' (Buy-and-Hold in der GC-Engine, `frequency_years` >
  Horizont), B (Fix-Mix 40/60), C (Shift auf 60/40 ab Jahr 5),
  D (B + Sanierung). Legacy 5c bleibt reine Regressions-Referenz.

### Warum

- **5c war one-shot → Feedback unmöglich.** Der 5c-Roll bucht die Flows in
  einem Schritt pro Stichtag; ein DG-konditionaler Eingriff (Sanierung,
  Rebalancing) braucht aber `DG_{t-1}` *während* des Rolls. Der
  sequenzielle Jahres-Roll der Subclass liefert genau das — ohne
  `stochastic_funding_ratio.py` zu verändern (Disziplin aus der Spec).
- **Architektur-Regel (Kernsatz):** DG-Feedback ist genau dann mit
  generate-then-pair vereinbar, wenn die Massnahme **keine
  Liability-Events erzeugt oder verändert**. Sanierungsbeiträge (Art. 65d
  BVG) werden nicht dem AGH gutgeschrieben ⇒ reiner Asset-Inflow ⇒
  kompatibel. Minder-/Nullverzinsung wirkt aufs AGH ⇒ Liability-Pfade
  müssten konditional neu generiert werden ⇒ bewusst NICHT gebaut, nur
  als Outlook dokumentiert (rote Linie im Architekturdiagramm).
- **`bond_yield` wegen geschlossenem ACTUS-Buchbestand:** Reinvestierte
  Mittel erzeugen keine neuen ACTUS-Kontrakte (das wäre ein
  ACTUS-Round-Trip pro Pfadjahr); die Originalkontrakte verdienen weiter
  ihre echten ACTUS-`IP`-Coupons, nur der inkrementelle Bestand bekommt
  die dokumentierte Buch-Yield-Approximation.
- **Regressionsdisziplin:** jeder neue Parameter ist opt-in; mit
  `rebalancing=None, sanierung=None` ist das Ergebnis byte-identisch zu
  Stage 5c (Spec-Assert 1, zusätzlich pytest-Smoke
  `tests/test_stage6_going_concern.py`).

---

## 3. Verteidigungssätze

### Fünf Aussagen, die ich sicher machen darf

1. "Die Asset-Seite ist ACTUS-nativ; die BVG-Liability-Seite ist eine
   dokumentierte Domain-Engine, die in dasselbe `CashFlowStream`-Schema
   emittiert — die AAL-Analyse-Schicht blieb unverändert."
2. "Der regulatorische Deckungsgrad ist ab Stage 5c eine
   (Zeit × Pfade)-Fläche, in der derselbe Survivor-Pfad Zähler und Nenner
   treibt — das ist die zentrale Kopplungsinvariante."
3. "Jede 5d-Erweiterung ist opt-in: mit `rebalancing=None` und
   `sanierung=None` ist das Ergebnis byte-identisch zu Stage 5c, per
   Assert in jedem Lauf geprüft."
4. "Sanierungsbeiträge sind architektur-kompatibel, weil sie als reiner
   Asset-Inflow keine Liability-Events erzeugen oder verändern — sie
   werden separat als `SANIERUNG_SB` geloggt, nie im Liability-Stream."
5. "D minus B ist die reine Massnahmenwirkung, weil beide Szenarien
   dieselben Seeds und dieselben Liability-Pfade teilen — am Horizont
   pfadweise DG_D ≥ DG_B (Assert 4)."

### Fünf Aussagen, die ich vermeiden muss

1. Nie **"die Liabilities sind ACTUS-nativ modelliert"** — sie sind es
   bewusst nicht; das ist die dokumentierte Thesis-Boundary.
2. Nie **"bond_yield ist kalibriert"** — es ist eine dokumentierte
   Buch-Yield-Approximation mit Plausibilitätsanker (Complementa ≈ 2.1%),
   kein kalibriertes Reinvestment-Modell.
3. Nie **"Complementa validiert das Modell"** — `DG_t0 = 107.6%` ist
   Kalibrierung per Konstruktion (initial_assets = 1.076 × VK_t0), kein
   unabhängiger Benchmark.
4. Nie **"das ersetzt eine ALM-Studie / misst ökonomisches Kapital"** —
   es fehlen Hinterbliebenen-/IV-Leistungen, aktuelle Tafeln und echtes
   Reinvestment (siehe Abschnitt 4).
5. Nie **"Equity/Diskontierung sind marktkonsistent"** — Equity ist
   Buchwert-Drift mit erwartungstreuem lognormalem Jahresschock, die
   ä_x-Diskontierung bleibt sticky beim technischen Zins; eine
   P/Q-masskonsistente Bewertung war explizit out of scope.

---

## 4. Nutzung durch eine echte Pensionskasse

### Welche Inputs sie liefern muss → welche Klassen das aufnehmen

| Input der Kasse | Nimmt auf |
|---|---|
| Bestandsdaten pro Kohorte: Geburtsjahr, Headcount, Lohn, AGH, Status, laufende Rente, Geschlecht | `Cohort(...)` → `PensionFund.add_cohort` |
| Reglement-Parameter: UWS (+ beschlossener Pfad), technischer Zins, Verzinsung, Sparstaffel, Koordinationsabzug, Min/Max-Lohn, Verwaltungskosten | `PensionPolicy(...)`, UWS-Pfad/Lohnwachstum/Indexierung in `Stage4Dynamics(...)` |
| Erwartete Neueintritte (Alter, Anzahl/Jahr, Lohn) | `EntryPolicy(...)` |
| Bond-/Cash-Buch als ACTUS-Terme: Nominal, Kupon, Laufzeit, Zahlungszyklen pro Position | `PAM(...)`-Kontrakte in `Portfolio([...])` |
| Annahmen: Sterbetafel + Improvement, Zinskurve, Zinsmodell-Parameter, Equity-Erwartung/-Vol, Startallokation, Marktwert der Anlagen | `ek2001_2005(improvement_rate, base_year)`, `CurveCalibrator.from_market`, `StochasticALMAnalysis(...)`, `GoingConcernFundingRatioAnalysis(...)` (`bond_book`, `cash_book`, `equity_sleeve_0`, `equity_*`, `bond_yield`) |
| Strategie-Fragen: Zielmix, Shift-Zeitpunkt, Sanierungsregeln | `RebalancingPolicy(...)`, `SanierungsPolicy(...)` |

### Welche Outputs sie bekommt

DG-Verteilungen über die Zeit (`funding_ratio_distribution(as_of)`),
Fan-Charts und `P(DG<100%)`-Kurven pro Strategie, Gewichtspfade
(`weight_path`), SB-Kostenstatistik (`gc_events`), Pensionierungsverlust
fair vs. angewandt (`RetirementLossAnalysis.summary()`),
Netto-Liquiditäts-CF (`ALMAnalysis.net_liquidity`), VK-Zerlegung
(`FundingRatioAnalysis.breakdown()`), Varianz-Zerlegung
(`RiskAttribution.summary()`).

### Ehrlicher Rahmen

Das Tool ist ein **Prototyp für Strukturfragen und eine Zweitmeinung**
("Was passiert mit dem DG-Fächer, wenn wir ab Jahr 5 umschichten oder ab
DG<100% sanieren?") — **kein Ersatz für kommerzielle ALM-Studien.** Es
fehlen: Hinterbliebenen- und IV-Leistungen (Risikobeiträge sind Inflow
ohne modellierte Gegenleistung), aktuelle Tafeln (EK 2001–2005 +
pauschales Improvement statt BVG 2020 generational), echtes Reinvestment
(Buch-Yield-Approximation statt neuer Kontrakte), Kostenkalibrierung,
obli/überobli-Split, Wertschwankungsreserven-Mechanik.

---

## 5. KPI-Liste (je ein Satz Bedeutung)

- **DG_t** — der Art-44-BVV2-nahe Deckungsgrad pro Pfad und Stichtag:
  Vorsorgevermögen geteilt durch Vorsorgekapital, die regulatorische
  Kopfkennzahl des gesamten Modells.
- **P(DG_t < 100%)** — Anteil der Pfade in Unterdeckung je Stichtag: die
  Wahrscheinlichkeitssicht auf das Sanierungsrisiko (am Horizont: D 15.5%
  vs. B 26.0%).
- **Quantile (p5/p25/p50/p75/p95)** — der DG-Fächer: Median als zentrale
  Erwartung, p5 als Downside, p95 als Upside; Basis aller Fan-Charts.
- **Pensionierungsverlust fair vs. angewandt** — pro Pensionierung
  `AGH · (UWS_angewandt · ä_x − 1)`: was die Kasse verliert (oder
  gewinnt), weil der angewandte UWS nicht dem fairen `1/ä_x` entspricht.
- **Netto-Liquiditäts-CF** — Zuflüsse minus Abflüsse pro Periode über
  beide Seiten: deckt das Cash-Management-Risiko ab, das der DG nicht
  zeigt.
- **VK-Zerlegung** — Vorsorgekapital aufgeteilt in Aktiven-AGH und
  Rentner-Barwert (`breakdown()`): zeigt, welcher Block die Verpflichtung
  treibt.
- **SB-Kostenstatistik** — Anteil aktiver SB-Pfadjahre (20.4%) plus
  Verteilung der kumulierten SB pro Pfad (Median CHF 13 Mio., p95 CHF 80
  Mio.): der Preis der Sanierungswirkung.
- **RiskAttribution** — Varianz-Zerlegung der Funding-Ratio nach
  Risikoquelle (nur Assets, nur Mortalität, voll): beantwortet, welcher
  Kanal die Streuung dominiert.

---

## 6. Stress-/Variations-Katalog (heute schon, ohne Codeänderung)

| Hebel | Parameter (öffentliche API) |
|---|---|
| Zinsmodell | `StochasticALMAnalysis(model=...)`: `hull_white`, `vasicek`, `cir`, `ho_lee` + `model_params` |
| Zinskurve | `CurveCalibrator.from_market(times, spot_rates)` |
| Tafel-Improvement | `ek2001_2005(improvement_rate=..., base_year=...)` |
| Equity-Annahmen | `equity_return`, `equity_sigma`, `equity_seed` |
| Buch-Yield | `bond_yield` |
| UWS-Pfad / Lohn / Indexierung | `Stage4Dynamics(conversion_rate_path={...}, salary_growth=..., threshold_index_period=...)` |
| Allokations-Szenarien | `RebalancingPolicy(target_weights, frequency_years, cost_bps, shift_at_year, shift_weights)` |
| Sanierungs-Trigger/-Höhe | `SanierungsPolicy(trigger_dg, exit_dg, sb_factor, max_years)` |
| Umfang | `N_PATHS` (Env `STAGE6_N_PATHS`), Seeds, `horizon_years` |

**Degenerations-Prinzip:** jede Variation kollabiert per Assert auf ihren
Basisfall — `rebalancing=None, sanierung=None` ⇒ byte-identisch 5c;
`frequency_years > Horizont` ⇒ echtes Buy-and-Hold (A', nie ein
Rebalancing-Termin); `equity_sigma=0` ⇒ deterministische Drift;
`bond_yield=None` ⇒ nur die echten ACTUS-Coupons;
`improvement_rate=0` ⇒ rohe Periodentafel;
`Stage4Dynamics(0.0, None, 1)` ⇒ Stage 3 byte-identisch. Damit ist jede
Stress-Variante gegen den Basisfall abgesichert, nicht nur plausibel.
