# ISE-02-Mini-Case – Flugverspätungs-Vorhersage (IAD)

Mini-Case zur Vorhersage von Flugverspätungen am Washington Dulles International (IAD).
Methode: **CRISP-DM** (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment).

**Status: Alle Phasen abgeschlossen. Finaler Bericht: `docs/final_evaluation.md`.**

## Business Problem

> "Verspätungen frühzeitig erkennen." — Klassifikation, Airline Delay Dataset, bessere Ressourcenplanung.

## Datenquelle

- **BTS TranStats – On-Time Statistics** (manuell heruntergeladen, 4 Tabellen)
- Web: <https://www.transtats.bts.gov/ONTIME/Departures.aspx>
- Airport: IAD (Washington Dulles International)
- Carrier: **United Airlines (UA)** (alle Tabellen)
- Zeitraum: **2024-01-01 bis 2025-12-31** (~95 000 Flüge nach Cleaning)
- **Wetter**: NCEI ISD-Lite (Bulk, KIAD Station USAF 724050), 2024-01-01 bis 2025-08-27

## Projektstruktur

```
ISE-02-Mini-Case/
├── data/
│   ├── raw/                    # Manuelle Downloads (BTS)
│   │   ├── Detailed_Statistics_Departures.csv    (4.9 MB, 47.217 Flüge)
│   │   ├── Detailed_Statistics_Arrivals.csv      (4.9 MB, 47.212 Flüge)
│   │   ├── Detailed_Statistics_Cancellation.csv  (12.5 KB, 356 Events)
│   │   └── Detailed_Statistics_Diversion.csv     (5.3 KB, 121 Events)
│   ├── external/               # Exogene Daten (z. B. Wetter, ab Phase 6)
│   │   └── weather/            # NCEI ISD Hourly (siehe src/weather_download.py)
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
│   ├── 03_weather_download.py  # NCEI ISD Hourly (Phase 6, optional)
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
| 4. Modeling | erledigt | `src/final_evaluation.py` + `models/*.joblib` + `results/final_*.png` |
| 5. Evaluation | erledigt | `results/final_metrics.json` + `docs/final_evaluation.md` |
| 6. Deployment | Skeleton | `src/inference.py` (CLI) – operative Pilotphase ausstehend |

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

### Final – 24 Features, mit Wetter, 2025-Q3 Val (Snapshot 20260616_101841)

| Modell | PR-AUC | ROC-AUC | F1 (optimal) | Brier | Precision (opt) | Recall (opt) |
|---|---:|---:|---:|---:|---:|---:|
| LogReg | 0.4549 | 0.7521 | 0.5079 | 0.2389 | 0.4273 | 0.6260 |
| **Random Forest** | **0.5268** | **0.7783** | **0.5278** | 0.2291 | **0.4613** | 0.6169 |
| XGBoost | 0.4853 | 0.7706 | 0.5169 | **0.1425** | 0.4492 | 0.6086 |

**Empfehlung: Random Forest** – beste Diskrimination (PR-AUC 0.527, 2.4× Lift).
XGBoost hat das beste Brier-Score (besser kalibriert) – Wahl je nach Use-Case.

**Business Case**: Bei Precision ≥ 0.7 (kostenintensive Aktionen) erreichen alle
drei Modelle rund 70 % Precision; Random Forest dabei mit höchstem Recall (21 %).

Wichtigste Features:
1. `origin_daily_arrival_delay_mean` (verspätete Vorgänger-Ankunft)
2. `cancellations_on_day` (Carrier-Probleme am Tag)
3. `wind_kts` (Wetter)
4. `temp_c` (Wetter)
5. `tod_sin` (Tageszeit)

### Baseline (vor Wetter, Snapshot 20260616_090104) – 27 Features, 2026-Q1/Q2 Val

| Modell | PR-AUC | F1@opt | Precision@opt | Recall@opt |
|---|---:|---:|---:|---:|
| LogReg | 0.276 | 0.336 | 0.293 | 0.396 |
| Random Forest | 0.301 | 0.353 | 0.294 | 0.441 |
| XGBoost | 0.315 | 0.356 | 0.299 | 0.441 |

**Verbesserung Final vs. Baseline: PR-AUC +0.21, Precision@opt +0.16.**
Wetter + Reduktion + größerer Trainingsumfang haben das Modell **deutlich** verbessert.

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
5. **Wetter integrieren** (Phase 6) → `src/weather_download.py` ✓ (Download bereit, Re-Training offen)

### Snapshot Baseline-Stand

Vor der Wetter-Integration wurde der Modell-Stand eingefroren unter
`results/snapshots/20260616_090104/` (alle Metriken + Plots). So lässt sich
später exakt vergleichen, was die Wetter-Features an zusätzlichem Lift bringen.

### Wetter-Download (`src/weather_download.py`)

Bezieht stündliche Wetterdaten für IAD (USAF `724050`, WMO `13743`) aus
**NCEI ISD-Lite** (Public Domain, ~0.5–1 MB/Jahr als .gz).

```powershell
# Standard: 2024..aktuelles Jahr
python -m src.weather_download

# Zeitraum explizit
python -m src.weather_download --start-year 2024 --end-year 2026
```

Output:
- `data/external/weather/raw/isd_lite_KIAD_<YYYY>.csv` (Roh, resumefähig)
- `data/external/weather/iad_isd_hourly.csv` (vereinheitlicht; Spalten: `station,
  ts_utc, temp_c, dewpoint_c, wind_kts, wind_dir, pressure_hpa, precip_mm,
  cloud_cover_flag`)

Hinweise:
- ISD-Lite ist Public Domain, kein API-Key nötig.
- NCEI publiziert laufendes Jahr mit ~2–6 Wochen Verzögerung; das Skript
  überspringt nicht verfügbare Jahre automatisch.
- Für **Live-Vorhersage** (Inferenz) nicht dieses Skript nutzen, sondern
  Open-Meteo / NWS (siehe `docs/inference.md`).

Nächste Schritte nach dem Download:
1. **Join-Skript** (`src/04_weather_join.py`) baut pro Flug das passende Stundenwetter
   zur scheduled departure time.
2. Re-Training der Notebooks mit den zusätzlichen Wetter-Features.
3. `feature_metadata.json` + Lookup-Tabellen im `inference.py` erweitern.

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
