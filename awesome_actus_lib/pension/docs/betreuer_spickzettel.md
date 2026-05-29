---
editor_options: 
  markdown: 
    wrap: 72
---

# Spickzettel — Pension Extension (Betreuer-Gespräch)

> Vorbereitung für die Besprechung. Aufbau: zuerst die zentrale Idee in
> einem Satz, dann Architektur, Stages, Funktionsweise, die Beispiele
> (mit dem grössten Erklärungsbedarf), bewusst nicht umgesetzte
> Funktionen, und am Ende eine Liste wahrscheinlicher Rückfragen mit
> Antworten.

------------------------------------------------------------------------

## 0. Die zentrale Idee in einem Satz

> **Die Pension Extension ergänzt AAL um die *Passivseite*
> (Verpflichtungen) einer BVG-Pensionskasse — und zwar so, dass am Ende
> derselbe `CashFlowStream`-Typ herauskommt wie auf der Aktivseite,
> sodass die bestehende AAL-Analyse-Schicht unverändert auf beiden
> Seiten funktioniert.**

Warum überhaupt eine Extension und keine ACTUS-Contracts?

| Seite | Wie modelliert | Wer erzeugt Events | Output |
|------------------|------------------|------------------|------------------|
| **Aktiv (schon in AAL)** | ACTUS-Verträge (z. B. `PAM` für Anleihen) | ACTUS-Engine remote (`PublicActusService`) | `CashFlowStream` |
| **Passiv (diese Extension)** | **Regeln + Population** (`PensionPolicy` + `Cohort` + `PensionFund`) | **lokaler Python-Simulator** | `CashFlowStream` |

**Kernargument:** BVG-Rentenregeln (Umwandlungssatz, Koordinationsabzug,
Sparstaffel, Mindestzins …) sind **nicht** im ACTUS-Standard. Die
ACTUS-Engine kann sie also nicht erzeugen. Lösung: Man beschreibt nicht
*Verträge*, sondern *Regeln + Versichertenbestand*, und ein eigener
Simulator emittiert die Events Jahr für Jahr. Das Ergebnis ist bewusst
derselbe Objekttyp — die Symmetrie ist der ganze Trick.

------------------------------------------------------------------------

## 1. Architektur & Dateistruktur

Modul: `awesome_actus_lib/pension/`

### 1.1 Drei Bausteine (die Eingabe)

| Baustein | Datei | Rolle | Mutabilität |
|------------------|------------------|------------------|------------------|
| `PensionPolicy` | `policy.py` | **Regelwerk** der Kasse (BVG-Parameter: Mindestzins 1.25 %, angewandter Zins 2.0 %, technischer Zins 1.76 %, UWS 5.23 %, Rücktrittsalter 65, Sparstaffel 7/10/15/18 %, Koordinationsabzug, BVG-Lohngrenzen) | **frozen** (immutable) |
| `Cohort` | `cohort.py` | **eine Gruppe** gleichartiger Versicherter (`headcount`, `birth_year`, `gross_salary`, `accrued_savings` = AGH, `status`, `annual_pension`, `gender`) | veränderbar |
| `PensionFund` | `fund.py` | **Container**: bündelt Policy + `start_date` + Liste von Cohorts | veränderbar |

### 1.2 Simulatoren (die Engine) — Vererbungskette

```         
ClosedFundSimulator              (Stage 1)  simulators/Stage1_closed.py
        └── OpenFundSimulator    (Stage 2)  simulators/Stage2_open.py
                └── DynamicFundSimulator (Stage 4)  simulators/Stage4_dynamic.py
```

-   Jede Stufe **erbt** und überschreibt nur das Nötigste (Open fügt
    Eintritte hinzu, Dynamic überschreibt nur `_emit_active` +
    `_emit_retirement`).
-   **Stochastische Sterblichkeit (Stage 5b)** lebt direkt in der
    Basisklasse `ClosedFundSimulator` und wird per Flags
    (`stochastic_mortality=True`, `mortality_filter`, `rng`) aktiviert →
    alle Subklassen erben sie.
-   `simulate_liability_paths(...)` (`simulators/liability_paths.py`)
    ist der Monte-Carlo-Wrapper: erzeugt `n_paths` unabhängige
    Passiv-Pfade.

### 1.3 Analyse-Schicht — `pension/analysis/`

| Klasse / Funktion | Datei | Frage, die sie beantwortet |
|------------------------|------------------------|------------------------|
| `ALMAnalysis` | `alm.py` | Netto-Liquidität pro Jahr; **Liquiditäts-Deckungsgrad** (Netto-CF-PV-Verhältnis, **Diagnose**) |
| `FundingRatioAnalysis` | `funding_ratio.py` | **Solvenz-Deckungsgrad \@ t0** (Art.-44-BVV2-nah): Aktiven / Vorsorgekapital |
| `annuity_due`, `technical_uws`, `RetirementLossAnalysis` | `conversion.py` | **Verrentungsverlust-KPI**: angewandter UWS vs. technisch korrekter UWS = 1/ä_x |
| `StochasticALMAnalysis` | `stochastic_alm.py` | Verteilung des Deckungsgrads über stochastische Zins-/Equity-Pfade |
| `StochasticFundingRatioAnalysis` | `stochastic_funding_ratio.py` | stochastischer Solvenz-DG (t0-eingefroren **oder** forward-projiziert, 5c) |
| `RiskAttribution` | `risk_attribution.py` | **Varianzzerlegung** des Deckungsgrads auf Kanäle (Zins vs. Sterblichkeit) |

### 1.4 Wiederverwendete Schicht — `stochastic_rates/`

Aus der **Vorgänger-PA** vendored, **unverändert** genutzt:
`CurveCalibrator.from_market`, `create_model` (Vasicek / CIR / Ho-Lee /
**Hull-White**), `GBMModel` (Equity), `simulation_to_reference_index`
(adaptiert einen simulierten Pfad als AAL-`ReferenceIndex` für
ACTUS-Floater).

------------------------------------------------------------------------

## 2. Die Stages (das Rückgrat der Arbeit)

**Leitprinzip:** Jede Stage fügt **genau eine** Fähigkeit additiv hinzu.
Defaults reproduzieren die Vorstufe **numerisch identisch** —
abgesichert durch Inline-Asserts in den Beispielen (z. B. Stage-4-Assert
„(b)": Stage 4 mit Null-Dynamik == Stage 3, relative Differenz \< 1e-9).

| Stage | Neue Fähigkeit | Schlüssel-Klasse(n) | Doku |
|------------------|------------------|------------------|------------------|
| **1** | Geschlossener Fonds, jährliche Schritte, keine Eintritte, keine Sterblichkeit | `ClosedFundSimulator` | `stage1.md` |
| **2** | Offener Fonds: deterministische jährliche Neueintritte | `OpenFundSimulator` + `EntryPolicy` | `stage2.md` |
| **3** | Deterministische **Rentner-Sterblichkeit** (EK 2001–2005, erwartetes Überleben, nur Rentner) | `MortalityTable`, `ek2001_2005()` | `stage3.md` |
| **4** | **Deterministisch zeitvariable** Parameter: Lohnwachstum, fallender UWS-Pfad, Schwellen-Indexierung + **Verrentungsverlust-KPI** + **Solvenz-DG \@ t0** | `DynamicFundSimulator`, `Stage4Dynamics`, `RetirementLossAnalysis`, `FundingRatioAnalysis` | `stage4.md` |
| **5a-1** | **Stochastische Aktivseite**: Hull-White-Kurzzins treibt Floater-Resets; „sticky" Flat-Diskontierung | `StochasticALMAnalysis` | `stage5.md` |
| **5a-2** | **Equity-Sleeve** (GBM) + pfadabhängiger Solvenz-DG | `StochasticFundingRatioAnalysis` | `stage5.md` |
| **5b** | **Stochastische Passivseite**: binomiale Kohorten-Übergänge + **Varianzattribution** über Kanäle | `simulate_liability_paths`, `RiskAttribution` | `stage5.md` |
| **5c** | **Forward** stochastischer Solvenz-DG: VK_t **pfadabhängig** über Zeit (schliesst die Lücke, dass der Solvenz-DG bis dahin invariant gegenüber Sterblichkeit war) | `LiabilityPath`, `StochasticFundingRatioAnalysis(liab_paths=…)` | `stage5c.md` |

> **Hinweis zu 5c:** `stage5c.md` ist als *Design-Spec* („Bauplan
> zuerst") formuliert, **aber der Code ist umgesetzt**:
> `simulate_liability_paths(..., return_headcount_log=True)` liefert
> `List[LiabilityPath]`, und `StochasticFundingRatioAnalysis` hat den
> `liab_paths`-Parameter. Es gibt ein lauffähiges Beispiel +
> Fan-Chart-Figures.

------------------------------------------------------------------------

## 3. Funktionsweise des Simulators (Schritt für Schritt)

`run(horizon_years)` der Basisklasse — das Herzstück:

```         
für jedes Simulationsjahr y (start.year … start.year + horizon − 1):
    Event-Datum = y-12-31
    für jede Cohort:
        age = y − birth_year
        wenn age >= terminal_age (100):  überspringen
        _step_cohort(...)        →  emittiert die Jahres-Events
        _apply_mortality(...)    →  Headcount-Abbau NACH den Events
        cohort_headcount_log anhängen
→ raw_response (gleiche Struktur wie ACTUS-Output) → CashFlowStream
```

**`_step_cohort` verzweigt nach Status/Alter:**

-   **aktiv & age \< 65** → `_emit_active`: 4 Events (`SAV_CONTRIB +`,
    `RISK_CONTRIB +`, `INTEREST_CREDIT 0`, `ADMIN_COST −`); intern:
    `AGH += Zins + neue Sparbeiträge`.
-   **aktiv & age \>= 65** → `_emit_retirement`: einmalig
    `RETIREMENT_CONV` (Rente = AGH × UWS, Status kippt auf „retired") +
    danach `PENSION_PAYMENT`.
-   **retired** → `_emit_pension`: `PENSION_PAYMENT −` (= −Rente ×
    Headcount).

### 3.1 Event-Typen (`events.py`)

Vorzeichen: **+ = Zufluss** zur Kasse, **− = Abfluss**.

| Event | Wann | Vorzeichen |
|------------------------|------------------------|------------------------|
| `SAV_CONTRIB` | aktiv, age \< 65 | \+ |
| `RISK_CONTRIB` | aktiv, age \< 65 | \+ |
| `INTEREST_CREDIT` | aktiv, age \< 65 | 0 (Buchung; trägt `agh_per_capita`) |
| `ADMIN_COST` | aktiv, age \< 65 | − |
| `RETIREMENT_CONV` | einmalig bei Pensionierung | 0 (Buchung; trägt UWS-Felder) |
| `PENSION_PAYMENT` | retired | − |

### 3.2 Zwei wichtige Konventionen (typische Prüf-Stellen)

1.  **„Emit-then-decrement"** (Stage 3+): Die Rente für Jahr `y` wird
    mit dem Headcount **zu Jahresbeginn** gezahlt; **danach** wird der
    Headcount um die erwarteten/gezogenen Toten reduziert. Aktuarielle
    Konvention: erst die Lebenden zahlen, dann am Jahresende die
    Überlebenden zählen.
2.  **Sterblichkeit erzeugt keinen neuen Event-Typ** — sie ist eine
    Zustandsänderung der Population; ihr Effekt erscheint implizit als
    schrumpfende `PENSION_PAYMENT`-Beträge. Der explizite Verlauf steht
    in `cohort_headcount_log`.

### 3.3 Aktiv- und Passivseite zusammenführen (`ALMAnalysis`, 3 Varianten)

| Variante | Output | Frage |
|------------------------|------------------------|------------------------|
| 1 — Netto-Liquidität | `DataFrame`/Jahr | Reicht der Cash, um Renten zu zahlen? |
| 2 — Funding Ratio (PV) | Skalar/Series | Decken die Aktiven die Passiven (PV) am Stichtag? |
| 3 — kombinierter Stream | `CashFlowStream` | Wie sieht das Gesamtbild als ein Objekt aus? |

------------------------------------------------------------------------

## 4. Die Beispiele (grösster Erklärungsbedarf)

**Alle Quickstarts haben dasselbe Skelett** — das ist der rote Faden,
den du zuerst zeigen solltest:

```         
[1A] Passivseite definieren  (Policy + Cohorts + Fund)
[1B] Aktivseite definieren   (PAM-Anleihen-Portfolio)
[2]  Event-Generierung       (Aktiv: ACTUS-Service / Passiv: Simulator)
[3]  Merging & ALM-Analyse   (Netto-Liquidität, Deckungsgrad, kombiniert)
[VERIFY] Inline-Asserts      (statt pytest — Projektkonvention!)
[4/5] Plots                  (in examples/figures/<stage>/ gespeichert)
```

> **Methodischer Punkt fürs Gespräch:** Verifikation läuft über
> **Inline-Asserts in den Beispielskripten**, nicht über ein separates
> pytest-Setup. Jedes Beispiel ist gleichzeitig Demo *und*
> Regressionstest.

### 4.1 Stage 1 — `Stage1_pension_quickstart.py` (Fundament)

-   **Passiv:** 3 aktive Cohorts (200 / 150 / 80 Köpfe, Jg. 1990 / 1970
    / 1955). Die 1955er sind mit 70 schon über 65 → demonstrieren die
    **automatische Verrentung** (Variante B: Status „active", aber
    Simulator löst `RETIREMENT_CONV` selbst aus).
-   **Aktiv:** 3 PAM-Anleihen 40M / 35M / 25M = **100M CHF**, Kupons
    2.0/2.5/3.0 %.
-   **Zeigt:** die drei Merging-Varianten + die zwei Plots
    (Netto-Liquidität, Funding-Ratio-Pfad).
-   **Botschaft:** Passiv-`CashFlowStream` geht 1:1 in dieselbe
    AAL-Toolchain wie die Aktivseite.

### 4.2 Stage 4 — `Stage4_pension_quickstart.py` (die KPI-Story)

-   **Passiv:** 8 aktive Jahrgangs-Cohorts + 1 Rentner-Cohort
    (RETIRED_1955, 80 Köpfe, 31'380 CHF/Jahr), 20 Eintritte/Jahr.
-   **`Stage4Dynamics`:** Lohnwachstum 1 %/Jahr, **UWS-Pfad {2025: 5.23
    %, 2029: 5.10 %}** (held-forward, kalibriert an Complementa Risiko
    Check-up 2024), Schwellen-Indexierung alle 5 Jahre.
-   **Sterblichkeit:**
    `ek2001_2005(improvement_rate=0.0125, base_year=2003)` — also
    **Generationentafel** (siehe §6, wichtiger Punkt!).
-   **Zwei zentrale Auswertungen:**
    -   `FundingRatioAnalysis` → **Solvenz-DG \@ t0 ≈ 105 %** (400M
        Anleihen-Nominale gegen VK_t0 ≈ 380M; Aktive: AGH×Köpfe,
        Rentner: Rente×Köpfe×ä_x).
    -   `RetirementLossAnalysis` → **Verrentungsverlust**: pro
        Pensionierung `Verlust = AGH × (angewandter_UWS × ä_x − 1)`.
        Zwei Szenarien: **Baseline** (5.23→5.10 %) vs. **Stress** (6.8 %
        flach = gesetzlicher Mindest-UWS *ohne* Anrechnungsprinzip →
        erzwingt positiven Verlust).
-   **Plots:** UWS angewandt vs. technisch (1/ä_x), Verlust pro Jahr,
    Kohorten-Headcount-Rückgang.

### 4.3 Stage 5b — `Stage5b_mortality_quickstart.py` (Stochastik-Showcase)

-   **Gleicher Fonds wie Stage 4**, gemischtes Anleihen-Portfolio (2
    fix + 2 Floater, 400M, **65 % fix / 35 % variabel** = 260M fix /
    140M variabel).

-   **`RiskAttribution`** fährt einen **Fünf-Konfigurations-Sweep**:

    | Konfiguration             | Aktivseite                | Passivseite           |
    |---------------------------|---------------------------|-----------------------|
    | `baseline`                | deterministisch           | deterministisch       |
    | `mortality_retirees_only` | deterministisch           | binomial, nur Rentner |
    | `mortality_all_cohorts`   | deterministisch           | binomial, alle        |
    | `assets_only`             | stochastisch (Hull-White) | deterministisch       |
    | `full`                    | stochastisch              | binomial, alle        |

-   **Varianzzerlegung:** Var(Zins) + Var(Sterblichkeit) ≈ Var(full).
    Der **`additivity_residual`** (\< 5 %) ist der Sanity-Check, dass
    die Zins- und Sterblichkeits-RNG-Ketten wirklich unabhängig sind
    (getrennte Seeds: `seed_assets=42`, `seed_liabilities=4242`).

-   **Schöner Diagnose-Check „(c)" (Warnung bei Faktor ∉ [8, 12], kein
    harter Assert):** Skaliert man alle Cohorts ×10 (Renten **und** AGH
    ÷10 → VK invariant), muss die Sterblichkeitsvarianz um \~Faktor 10
    sinken — das ist das **Gesetz der grossen Zahl** (idiosynkratische
    Sterblichkeit diversifiziert weg), als Zahl gezeigt.

> Wenn du nur **ein** Beispiel im Detail zeigen willst: **Stage 4**
> (KPI-Story, regulatorisch greifbar). Wenn du den methodischen Anspruch
> zeigen willst: **Stage 5b** (Varianzattribution +
> Falsifikations-Asserts).

------------------------------------------------------------------------

## 5. Bewusst NICHT umgesetzt (zeigt Prioritätensetzung)

| Nicht umgesetzt | Warum bewusst ausgelassen |
|------------------------------------|------------------------------------|
| **Witwen-/Witwer-/Waisenrente** (Hinterlassenenrenten) | Würde Joint-Life-Annuitäten + Begünstigten-Modellierung erfordern; verändert ä_x. Bewusst zurückgestellt, um den Fokus auf die Kern-ALM-Mechanik zu halten. **Bekannte Konsequenz:** ä_x ist „single-life" → `technical_uws` ist nach oben verzerrt → der Baseline-UWS wirkt im KPI **konservativer** als er ist. Im Bericht dokumentiert. |
| **Aktiven-Sterblichkeit *mit* Leistungen** | Stage 5b zieht zwar binomial auch Aktive ab, zahlt aber kein Todesfallkapital → leichte Überschätzung des Deckungsgrads bei hoher Aktiven-Sterblichkeit. Offen: „Was passiert mit dem verfallenden AGH eines verstorbenen Aktiven?" |
| **Invaliditäts-Dekremente** | Risikobeiträge werden als Zufluss gebucht, die korrespondierenden Abflüsse nicht modelliert. |
| **Asset-Rebalancing / Mark-to-Market der Anleihen** | Statisches Buy-and-Hold-Anleihen-Portfolio; nur der Equity-Sleeve ist MTM (GBM). |
| **Stochastische Diskontierung** | Beide Seiten werden mit flachem, „sticky" technischem Zins diskontiert (regulatorische Konvention). Pfad-spezifische Diskontfaktoren bewusst ausgeklammert. |
| **Sanierungsmassnahmen** (Zins-Ablation, Sanierungsbeiträge) | „Goal 2"-Gebiet; bräuchte Betreuer-Scoping und bricht die generate-then-pair-Architektur (Passivpfad würde vom realisierten Aktiv-DG abhängen). |
| **Systemisches Langlebigkeitsrisiko** | Nur idiosynkratische Sterblichkeit; keine kalenderjahr-stochastischen Trend-Faktoren (Aro/Pennanen). |

------------------------------------------------------------------------

## 6. Wahrscheinliche Rückfragen + kurze Antworten

**F: Warum modellierst du die Passivseite nicht mit ACTUS-Contracts?** →
BVG-Rentenregeln sind nicht im ACTUS-Standard. Daher: Regeln +
Population → eigener Simulator → *gleicher* `CashFlowStream`-Output,
damit die AAL-Analyse unverändert nutzbar ist.

**F: Du hast zwei „Deckungsgrade" — was ist der Unterschied?** → (1)
**Liquiditäts-DG** in `ALMAnalysis.funding_ratio`: PV(Aktiv-CF) /
\|PV(Netto-Passiv-CF)\|. Ist **mix-sensitiv** (Beiträge im Zähler des
Netto-CF blähen ihn auf) → bleibt als **Diagnose** drin, ist *nicht* der
regulatorische DG. (2) **Solvenz-DG** in `FundingRatioAnalysis`:
Vorsorgevermögen / Vorsorgekapital_t0 (Art.-44-BVV2-nah). Die Beispiele
drucken beide nebeneinander, damit die Divergenz sichtbar ist.

**F: Warum nutzt du für ä_x eine Generationentafel, für den
Headcount-Abbau aber die rohe Periodentafel?** *(subtilster Punkt — sei
darauf vorbereitet)* → `MortalityTable.survival_probability` projiziert
generational **nur**, wenn `cal_year` übergeben wird. `annuity_due`
reicht `cal_year` durch (→ realistische ä_x, fairer UWS \~4.9 % @65
statt artefaktische \~5.5 % auf der veralteten Periodentafel — sonst
zeigt die Baseline einen *Gewinn* statt eines Verlustes). Der
Headcount-Abbau im Simulator übergibt **kein** `cal_year` → rohe
Periodentafel. Bewusste Trennung: Bewertung (ä_x, VK) soll
vorausschauende Langlebigkeit abbilden; der Überlebenspfad bleibt auf
der Periodentafel. Im Code-Kommentar als „harmless for liquidity-lens"
dokumentiert.

**F: Was bedeutet „sticky" Diskontierung in der Stochastik?** →
Floater-Resets sind stochastisch (jeder Pfad liest den Kupon vom
simulierten Kurzzins ab), aber **diskontiert** wird auf beiden Seiten
mit dem flachen technischen Zins — unabhängig vom Pfad. Das ist die
regulatorische Stickiness-Konvention (qualitativ in FRP 4 motiviert);
konsistente Marktbewertung wäre eine Folgestufe.

**F: Warum ist der Solvenz-DG in 5a/5b invariant gegenüber
stochastischer Sterblichkeit, und was macht 5c?** → In 5a-P2/5b ist
`VK_t0` einmalig zu t0 berechnet (nur t0-Kohortengrössen +
deterministisches ä_x) → Sterblichkeit bewegt den Nenner nicht. **5c**
macht `VK_t,i` über `LiabilityPath.headcount_trace` **pfad- und
zeitabhängig**, mit der **Kopplungs-Invariante**: derselbe
Überlebenspfad `i` treibt Nenner (VK) und Renten-Abflüsse im Zähler.
Dann reagiert der regulatorische Headline-DG endlich auf
Langlebigkeitsrisiko.

**F: Wie stellst du Reproduzierbarkeit/Unabhängigkeit der Pfade
sicher?** → `numpy.random.SeedSequence(seed).spawn(n_paths)` pro Pfad,
und pro Cohort nochmals gespawnt — so ändert das Umschalten von
`mortality_filter` („retirees"↔„all") nicht die Draws der bereits
beteiligten Cohorts. Aktiv- und Passiv-Seite nutzen getrennte
Master-Seeds (Unabhängigkeits-Annahme der Varianzzerlegung).

**F: Wie hast du getestet, dass jede Stage die Vorstufe nicht kaputt
macht?** → Backward-Compat-Asserts in den Beispielen: Stage N mit
neutralen Parametern == Stage N−1, numerisch bis \~1e-9 (z. B.
Stage-4-Assert „(b)" gegen `OpenFundSimulator`).

------------------------------------------------------------------------

## 7. 30-Sekunden-Zusammenfassung (falls du schnell einsteigen musst)

„Ich habe AAL um die Passivseite einer BVG-Kasse erweitert. Statt
ACTUS-Verträgen beschreibe ich Regeln plus Versichertenbestand; ein
eigener Simulator erzeugt daraus Jahr-für-Jahr-Events im *gleichen*
`CashFlowStream`-Format wie die Aktivseite, sodass die bestehende
AAL-Analyse-Schicht auf beiden Seiten läuft. Darauf habe ich in
**additiven Stages** aufgebaut: geschlossener Fonds → offener Fonds →
Sterblichkeit → deterministische Dynamik mit Verrentungsverlust-KPI →
und schliesslich die stochastische Schicht (Hull-White-Zinsen +
Equity-GBM auf der Aktivseite, binomiale Sterblichkeit auf der
Passivseite) mit einer Varianzattribution, die zeigt, wie viel des
Deckungsgrad-Risikos vom Zins- bzw. vom Sterblichkeitskanal kommt.
Hinterlassenenrenten habe ich bewusst ausgeklammert, um den Fokus auf
die Kern-ALM-Mechanik zu halten."
