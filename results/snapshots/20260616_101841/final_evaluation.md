# Final Model Evaluation – Business Case

_Erstellt am 2026-06-16 10:17_

---

## 1. Business Case (Aufgabenstellung)

> *„Verspätungen frühzeitig erkennen. Klassifikation, Airline Delay Dataset,*
> *bessere Ressourcenplanung.*

### Operative Übersetzung

Wenn das Modell einen Flug als wahrscheinlich verspätet klassifiziert, kann der
Carrier / Airport Operator:

- **Personal aufstocken** (Gate-Agents, Bodenpersonal, Catering) – Kosten ~50-200 € pro Stunde pro Person
- **Passagiere proaktiv informieren** – reduziert Beschwerden und Re-Check-Workload
- **Crew-Rotation vorausschauend anpassen** – vermeidet Domino-Effekte auf nachfolgende Flüge
- **Ground-Equipment reservieren** (Busse, Pushback, De-Icing) – Engpässe vermeiden

**Wirtschaftliche Annahmen (grobe Schätzung):**

| Komponente | Kosten pro Vorhersage |
|---|---:|
| Vorbereitungs-Aktion (Personal/Equipment) | 500 € |
| Vermiedene Verspätungs-Kosten (PAX-Re-Booking, Hotel, EU261) | 2000 € |
| Falschalarm-Kosten (verschwendete Ressourcen) | 200 € |

Damit ist das Modell **lohnend**, sobald die **Precision ≥ ~10 %** ist – was bei
Val-Base-Rate von 22.0 % und unserer erreichten Precision von
> 40 % **deutlich überschritten** ist.

## 2. Modell-Vergleich

| Modell | PR-AUC | ROC-AUC | F1@opt | P@opt | R@opt | Brier |
|---|---:|---:|---:|---:|---:|---:|
| logreg | 0.4549 | 0.7521 | 0.5079 | 0.4273 | 0.6260 | 0.2389 |
| random_forest | 0.5268 | 0.7783 | 0.5278 | 0.4613 | 0.6169 | 0.2291 |
| xgboost | 0.4853 | 0.7706 | 0.5169 | 0.4492 | 0.6086 | 0.1425 |

**Empfehlung: `random_forest`** – beste Diskrimination (PR-AUC = 0.5268).

## 3. Threshold-Analyse

Drei operativ relevante Threshold-Szenarien:

| Modell | Threshold-Strategie | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| logreg | t=0.5 (default) | 0.327 | 0.829 | 0.469 |
| random_forest | t=0.5 (default) | 0.350 | 0.803 | 0.488 |
| xgboost | t=0.5 (default) | 0.608 | 0.219 | 0.322 |
| | | | | |
| logreg | F1-optimal | 0.427 | 0.626 | 0.508 |
| random_forest | F1-optimal | 0.461 | 0.617 | 0.528 |
| xgboost | F1-optimal | 0.449 | 0.609 | 0.517 |
| | | | | |
| logreg | Precision ≥ 0.7 | 0.700 | 0.006 | 0.012 |
| random_forest | Precision ≥ 0.7 | 0.701 | 0.212 | 0.326 |
| xgboost | Precision ≥ 0.7 | 0.702 | 0.104 | 0.181 |

### Operativer Use-Case

**Szenario 1 – Konservativ (Precision maximieren):**
Threshold so, dass Precision ≥ 0.7. Recall wird niedrig, aber jeder Alarm
ist sehr verlässlich → nützlich für kostenintensive Aktionen (z. B. Crew-Re-Routing).

**Szenario 2 – Ausgewogen (F1-maximal):**
Bester Kompromiss aus Precision und Recall. Gut für Standard-Ressourcen-Planung.

**Szenario 3 – Aggressiv (Recall maximieren):**
Niedriger Threshold, fast alle Verspätungen werden gefangen → viele False Positives.
Nur sinnvoll, wenn Falschalarm-Kosten << verhinderte Verspätungskosten.

## 4. Lift über Baseline

Base-Rate im Val: 21.95%.

- **logreg**: PR-AUC 0.455 = 2.07× Base-Rate
- **random_forest**: PR-AUC 0.527 = 2.40× Base-Rate
- **xgboost**: PR-AUC 0.485 = 2.21× Base-Rate

## 5. Wetter-Hebel

Vergleich zur Baseline ohne Wetter (Snapshot `20260616_090104`):

| Metrik | Ohne Wetter | Mit Wetter | Δ |
|---|---:|---:|---:|
| PR-AUC | 0.3147 | 0.5268 | +0.2121 |
| Precision@opt | 0.2991 | 0.4613 | +0.1622 |
| Recall@opt | 0.4409 | 0.6169 | +0.1760 |

Caveat: Die Val-Perioden sind unterschiedlich (Baseline: 2026-Q1/Q2, neu: 2025-Q3).
Ein Teil des Sprungs könnte saisonal bedingt sein.

## 6. Daten-Herkunft & Limitationen

**Datenquellen:**
- BTS TranStats Detailed Statistics (manuell exportiert, 4 Tabellen, 2 Jahre)
- NCEI ISD-Lite Hourly (Wetter, 14497 Stunden, 2024-01-01 bis 2025-08-27)

**Limitationen:**

1. **Single-Carrier-Scope**: nur United Airlines am IAD. Carrier-übergreifende
   Effekte (z. B. NAS-Delay, ATC-Last) sind indirekt nur über Arrival-Aggregate
   sichtbar.
2. **Wetter-Coverage-Limit**: NCEI publiziert mit ~6 Wochen Verzug. Modell kann
   erst ab 2025-Q3 sauber auf Wetter trainieren; 2026 ist (Stand 2026-06-16)
   wetterlos.
3. **Tail-Number fehlt** (99 % NaN) → keine Aircraft-Rotation-Features (Vorflug-Delay).
4. **Kein Flugzeugtyp**, **kein Gate-Readiness** in BTS → keine Strecken-Equipment-Features.
5. **Kein NAS-Load / ATC-Delay-Indikator** → systemische Verlangsamung nur über
   Arrival-Aggregate erfasst.
6. **Temporal-Split-Bias**: Val nur auf Sommer (2025-07/08), Winter (Sturm, Eis)
   und Thanksgiving/Christmas-Spitzen sind im Val nicht abgedeckt.

## 7. Reproduzierbarkeit

```powershell
# Setup
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'

# Wetter (falls nicht vorhanden)
python -m src.weather_download --start-year 2024 --end-year 2025

# Feature-Pipeline
python src/02_data_preparation.py

# Diese Evaluation
python src/final_evaluation.py
```

## 8. Empfehlung an den Business

**random_forest** ist das beste Modell für diesen Use-Case:

- **PR-AUC = 0.527** (vs. 0.140 Base-Rate, **+2.4× Lift**)
- **Precision = 0.461** am F1-Threshold
- **Recall = 0.617** – fast zwei Drittel der echten Verspätungen werden gefangen
- **Brier = 0.229** – moderat kalibriert, Wahrscheinlichkeiten interpretierbar

**Empfohlene nächste Schritte:**

1. **Threshold pro Use-Case** wählen (Precision vs. Recall Trade-off)
2. **Real-Time-Inference-API** auf Basis von `src/inference.py`
3. **Operative Pilotphase** mit 1-2 Monaten Beobachtung; KPIs:
   - True-Positive-Rate der vorhergesagten Verspätungen
   - Kosten pro verhinderter Verspätung (EUR)
   - False-Positive-Rate
4. **Re-Training** wenn NCEI-Wetter weiter publiziert ist (alle 4 Wochen)
5. **Drift-Monitoring** der Feature-Distribution (besonders Wetter & Arrival-Aggregate)

---

_Automatisch generiert von `src/final_evaluation.py` am 2026-06-16 10:17._