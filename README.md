# ISE-02-Mini-Case – Flugverspätungs-Vorhersage (IAD)

Mini-Case zur Vorhersage von Flugverspätungen am Washington Dulles International (IAD).
Methode: **CRISP-DM** (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment).

## Business Problem

> "Verspätungen frühzeitig erkennen." — Klassifikation, Airline Delay Dataset, bessere Ressourcenplanung.

## Datenquelle

- **BTS TranStats – On-Time Departures** (manuell heruntergeladen)
- Web: <https://www.transtats.bts.gov/ONTIME/Departures.aspx>
- Airport: IAD (Washington Dulles International)
- Carrier (in den vorhandenen Daten): **United Airlines (UA)**
- Zeitraum: **2025-01-01 bis 2026-04-30** (16 Monate, 47.217 Flüge)

## Projektstruktur

```
ISE-02-Mini-Case/
├── data/
│   ├── raw/                    # Manuelle Downloads (BTS)
│   │   ├── Detailed_Statistics_Departures.csv    (4.9 MB, 47.217 Flüge)
│   │   ├── Detailed_Statistics_Arrivals.csv      (4.9 MB, 47.212 Flüge)
│   │   ├── Detailed_Statistics_Cancellation.csv  (12.5 KB, 356 Events)
│   │   └── Detailed_Statistics_Diversion.csv     (5.3 KB, 121 Events)
│   └── processed/              # Aufbereitete Daten (Phase 3)
├── docs/
│   ├── Aufgabenstellung.md
│   ├── data_understanding.md   # CRISP-DM Phase 1+2
│   ├── data_preparation.md     # CRISP-DM Phase 3
│   └── model_evaluation.md     # CRISP-DM Phase 5
├── logs/                       # Skript-Logs
├── models/                     # Trainierte Modelle (joblib)
├── results/                    # Metriken + Plots
├── notebooks/                  # Jupyter Notebooks (Phase 4+5)
│   ├── 01_baseline_logreg.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_xgboost.ipynb
│   └── 04_compare_models.ipynb
├── src/                        # Daten-Pipeline
│   ├── 01_business_data_understanding.py
│   ├── 02_data_preparation.py
│   ├── _sanity_check.py
│   └── inference.py            # CLI/API-Inference (Phase 6)
├── examples_flights.csv        # Demo-Batch für Inference
├── examples_predictions.csv    # Demo-Output (generiert)
├── .venv/                      # Virtuelle Umgebung (nicht in Git)
├── requirements.txt
└── README.md
```

## Setup

```powershell
# Einmalig (venv wurde bereits angelegt)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CRISP-DM-Status

| Phase | Status | Output |
|---|---|---|
| 1. Business Understanding | erledigt | `docs/data_understanding.md` Abschnitt 1 |
| 2. Data Understanding | erledigt | `docs/data_understanding.md` Abschnitt 2 |
| 3. Data Preparation | erledigt | `docs/data_preparation.md` + `data/processed/*.parquet` |
| 4. Modeling | erledigt | `notebooks/01-03*.ipynb` + `models/*.joblib` + `results/*.png` |
| 5. Evaluation | erledigt | `notebooks/04_compare_models.ipynb` + `docs/model_evaluation.md` |
| 6. Deployment | offen | – |

## Wichtigste Erkenntnisse aus Phase 1+2

- **47.217 Flüge** von 01.01.2025 bis 30.04.2026
- **Klassen-Verteilung**: 15.7 % verspätet (≥ 15 Min) vs. 84.3 % pünktlich → **Imbalance**
- **63 verschiedene Zielflughäfen** (Top: DEN, SFO, LAX, MCO, ORD)
- **Tail Number** zu 99 % fehlend → **nicht als Feature nutzbar**
- **5 Delay-Kategorien** (Carrier/Weather/NAS/Security/Late Aircraft) sind von BTS **nachträglich attribuiert** → Leakage-Risiko, deshalb ausschließen
- **Vorgeschlagene Target-Variable**: `departure_delay_min >= 15`
- **Vorgeschlagene Features**: Datum, sched_dep_time, dest_airport, sched_elapsed_min
- **Carrier konstant** (nur UA) → vereinfacht die Modellierung

## Modell-Ergebnisse (Phase 4+5)

| Modell | PR-AUC | ROC-AUC | F1 (optimal) | Brier | Precision (opt) | Recall (opt) |
|---|---:|---:|---:|---:|---:|---:|
| LogReg Baseline | 0.276 | 0.681 | 0.336 | 0.215 | 0.293 | 0.396 |
| Random Forest    | 0.301 | 0.691 | 0.353 | **0.182** | 0.294 | 0.441 |
| **XGBoost**      | **0.315** | **0.694** | **0.356** | 0.191 | **0.299** | **0.441** |

**Empfehlung: XGBoost** (PR-AUC = 0.315, 2.2x Lift über Baseline 0.14).

Kein Modell erreicht Precision ≥ 0.7 mit sinnvollem Recall – das ist eine **strukturelle Eigenschaft** der Aufgabe (seltene, schwer vorhersagbare Events). Wichtigste Features:
1. `origin_daily_arrival_delay_mean` (verspätete Vorgänger-Ankunft)
2. `cancellations_on_day` (Carrier-Probleme am Tag)
3. `tod_sin` / `sched_dep_min_of_day` (Tageszeit)

## Skripte

### `src/01_business_data_understanding.py`

Lädt die 3 BTS-CSVs (mit korrekter Behandlung der 9-zeiligen Metadaten-Köpfe),
berechnet Schlüssel-Kennzahlen und erzeugt den Markdown-Bericht.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python src\01_business_data_understanding.py
```

Output: `docs/data_understanding.md`

### `src/02_data_preparation.py`

Lädt alle 4 BTS-CSVs, bereinigt sie, leitet 23 Features ab, erstellt die
Target-Variable und führt den temporalen Train/Val-Split durch.

```powershell
python src\02_data_preparation.py
```

Output:
- `data/processed/iad_flights_train.parquet` (35.158 Zeilen)
- `data/processed/iad_flights_val.parquet` (12.059 Zeilen)
- `data/processed/feature_metadata.json`
- `docs/data_preparation.md`

## Selbst getroffene Entscheidungen (zu validieren)

1. **Carrier = UA only (FESTLEGUNG)** — alle vier BTS-Tabellen enthalten ausschließlich United-Airlines-Flüge am IAD. Limitation wird im Bericht dokumentiert.
2. **Verfrühte Flüge (< 0 Min) = Klasse 0** — BTS-Konvention; operativ pünktlich.
3. **Cancelled/Diverted = Drop** für die Hauptaufgabe; separat analysierbar.
4. **2025 = Training, 2026 = Validation** (temporal split, falls genug Daten).
5. **Keine Airborne_Time-Tabelle** verfügbar — wird in der Modellierung nicht verwendet.

## Nächste Schritte (Phase 6: Deployment)

1. **CLI-Wrapper** für manuelle Predictions (Python-Skript) → `src/inference.py` ✓
2. **REST-API** (FastAPI) für Realtime-Predictions
3. **Model Monitoring**: Drift-Detection für Feature-Distribution
4. **Re-Training-Pipeline**: monatliches Retraining mit neuen BTS-Daten

### Inference-Skript (`src/inference.py`)

Nimmt einen oder mehrere Flüge und liefert Verspätungs-Wahrscheinlichkeit + Label.
Aktuell nutzt das Skript das trainierte **XGBoost**-Modell, kann aber auch
LogReg/RF laden. Wetterdaten sind als Input-Feld vorbereitet, werden aber noch
nicht als Feature genutzt (Backwards-Compat bis das nächste Training Wetter
integriert).

```powershell
# Demo-Prediction
python -m src.inference --demo

# Einzelner Flug (JSON-String)
python -m src.inference --flight-json '{"carrier_code":"UA","date":"2026-06-15","flight_number":"944","dest_airport":"LHR","sched_dep_time":"18:30","sched_elapsed_min":420}'

# Einzelner Flug (JSON-Datei)
python -m src.inference --flight-json tmp_flight.json

# Batch (CSV → CSV)
python -m src.inference --flight-csv examples_flights.csv --output-csv examples_predictions.csv

# Andere Modelle / Threshold
python -m src.inference --model models/random_forest.joblib --threshold 0.3 --flight-json tmp_flight.json
```

Programmatisch:

```python
from src.inference import FlightPredictor
p = FlightPredictor()  # default: XGBoost, threshold 0.5
print(p.predict_one({
    "carrier_code": "UA",
    "date": "2026-06-15",
    "flight_number": "944",
    "dest_airport": "LHR",
    "sched_dep_time": "18:30",
    "sched_elapsed_min": 420,
}))
```

## Verfügbare Daten (4 Tabellen)

| Tabelle | Zeilen | Beschreibung |
|---|---:|---|
| `Detailed_Statistics_Departures.csv` | 47.217 | Abflüge ab IAD (Haupttarget: `departure_delay_min`) |
| `Detailed_Statistics_Arrivals.csv` | 47.212 | Ankünfte nach IAD (Sekundär-Analyse, evtl. Feature-Quelle) |
| `Detailed_Statistics_Cancellation.csv` | 356 | Stornierungen |
| `Detailed_Statistics_Diversion.csv` | 121 | Umleitungen |

Airborne_Time existiert nicht und wird nicht verwendet.
