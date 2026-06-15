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
│   └── data_understanding.md   # Output von src/01_business_data_understanding.py
├── logs/                       # Skript-Logs
├── src/
│   └── 01_business_data_understanding.py
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
| 4. Modeling | offen | – |
| 5. Evaluation | offen | – |
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

## Nächste Schritte (Phase 3)

1. Saubere Pipeline: laden → bereinigen → Features ableiten → speichern als `data/processed/iad_flights.parquet`
2. Feature-Engineering: Datums-Features, Strecken-Features, Holiday-Flags (US-Feiertage)
3. Train/Validation-Split
4. Baseline-Modell: Logistic Regression
5. Iteration: Random Forest → XGBoost/LightGBM
6. Modell-Evaluation mit PR-AUC, F1, Brier Score

## Verfügbare Daten (4 Tabellen)

| Tabelle | Zeilen | Beschreibung |
|---|---:|---|
| `Detailed_Statistics_Departures.csv` | 47.217 | Abflüge ab IAD (Haupttarget: `departure_delay_min`) |
| `Detailed_Statistics_Arrivals.csv` | 47.212 | Ankünfte nach IAD (Sekundär-Analyse, evtl. Feature-Quelle) |
| `Detailed_Statistics_Cancellation.csv` | 356 | Stornierungen |
| `Detailed_Statistics_Diversion.csv` | 121 | Umleitungen |

Airborne_Time existiert nicht und wird nicht verwendet.
