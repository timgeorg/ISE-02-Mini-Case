# CRISP-DM Phase 3: Data Preparation

_Erstellt am 2026-06-15 17:04_

---

## Input-Output

| Phase | Input | Output |
|---|---|---|
| Phase 2 (Data Understanding) | `data/raw/Detailed_Statistics_*.csv` | `docs/data_understanding.md` |
| **Phase 3 (Data Preparation)** | `data/raw/*.csv` | `data/processed/iad_flights_{train,val}.parquet`<br>`docs/data_preparation.md`<br>`logs/data_preparation.log` |

## Pipeline-Schritte

```
Roh-CSV  ->  load_bts_csv  ->  parse_types  ->  dropna  ->
add_date_features  ->  add_time_features  ->  add_arrival_aggregate  ->
add_cancellation  ->  make_target  ->  temporal_split  ->
add_frequency_features (Train-only Ref)  ->  save parquet
```

## Getroffene Entscheidungen (Übernahme aus Phase 1+2)

| # | Entscheidung | Auswirkung in Pipeline |
|---|---|---|
| 1 | Carrier = United Airlines (UA) only | Filter `carrier_code == 'UA'` (alle Daten ohnehin UA) |
| 2 | Threshold = 15 Min | `delay_label = (departure_delay_min >= 15)` |
| 3 | Verfrühte = pünktlich | Verfrühte Flüge bekommen `delay_label = 0` |
| 4 | Cancelled/Diverted = Drop | `dropna(subset=['departure_delay_min'])` |
| 5 | 2025 = Train, 2026 = Val | `split_date = 2026-01-01` |

## Feature-Liste

Insgesamt **23 Features**, eingeteilt in Gruppen:

| Gruppe | Features | Beschreibung |
|---|---|---|
| **Datum (zyklisch)** | `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Sinus/Cosinus-Encoding für Wochentag & Monat (Modell erkennt Zyklizität) |
| **Datum (linear)** | `dayofweek`, `day`, `month`, `weekofyear`, `quarter`, `is_weekend`, `is_holiday` | Roh-Encoding |
| **Tageszeit (zyklisch)** | `tod_sin`, `tod_cos` | Sinus/Cosinus für Minuten seit Mitternacht |
| **Tageszeit (linear)** | `sched_dep_hour`, `sched_dep_minute`, `sched_dep_min_of_day`, `time_of_day` | Roh-Encoding + 4-Bucket (Nacht/Morgen/Mittag/Abend) |
| **Strecke** | `sched_elapsed_min` | Proxy für Distanz (Kurz-/Langstrecke) |
| **Frequency-Encoding** | `flight_combo_freq`, `dest_freq` | Häufigkeit der (carrier, dest, flight_number)-Kombination bzw. Destination |
| **Arrival-Aggregate** | `origin_daily_arrival_delay_mean`, `origin_daily_arrival_n` | Mittlere Ankunftsverspätung an diesem Tag von dieser Origin |
| **Cancellation-Aggregate** | `cancellations_on_day` | Anzahl Stornierungen am gleichen Tag (IAD-weit) |

## Explizit ausgeschlossene Features (Leakage-Schutz)

| Spalte | Grund |
|---|---|
| `actual_dep_time`, `actual_elapsed_min`, `wheels_off_time`, `taxi_out_min` | post-hoc Info |
| Alle `delay_*_min` (Carrier/Weather/NAS/Security/Late Aircraft) | BTS attribuiert diese nachträglich |
| `tail_number` | 99 % Missing |
| `arrival_delay_min` | Target Leakage wenn Flugzeug ankommt |

## Temporal-Split

- **Split-Datum:** 2026-01-01
- **Train:** 35,158 Zeilen, 2025-01-01 -> 2025-12-31
- **Val:**   12,059 Zeilen, 2026-01-01 -> 2026-04-30
- **Train Klassen-Balance:** 16.27 % verspätet
- **Val Klassen-Balance:**   13.96 % verspätet

> **Anmerkung zur Saisonalität:** Val enthält nur Jan-Apr, also Winter + früher Frühling. Sommerverspätungen (Gewitter) und Thanksgiving/Christmas-Spitzen sind nicht in Val enthalten. Wird in Phase 5 (Evaluation) thematisiert.

## Train vs. Val – Verteilungs-Check

| Kennzahl | Train | Val |
|---|---:|---:|
| Zeilen | 35,158 | 12,059 |
| Unique Destinations | 59 | 54 |
| Mittlere Verspätung (Min) | 10.66 | 8.40 |
| Median Verspätung | -3.00 | -3.00 |
| `delay_label=1` (Anteil) | 16.27 % | 13.96 % |
| Mittlere Abflug-Stunde | 13.44 | 13.73 |
| Wochenend-Anteil | 27.67 % | 27.57 % |
| Feiertags-Anteil | 2.95 % | 2.40 % |
| Mittlere Flugdauer (Min) | 209.3 | 208.0 |
| Mittlere origin_daily_arrival_delay | 3.71 | -0.28 |



### Unseen Destinations (Data-Drift-Risiko)

- Destinations in Val, die nicht in Train vorkommen: **4 von 54**
- Anteil der Val-Flüge zu unseen Destinations: **0.47 %** (57 Flüge)

Diese Flüge können vom Modell nicht sinnvoll eingeschätzt werden, weil das `dest_freq` und `flight_combo_freq`-Encoding 0/fehlend ist.

## Output-Dateien

| Datei | Größe (typisch) | Inhalt |
|---|---|---|
| `data/processed/iad_flights_train.parquet` | ~5-10 MB | Trainingsdaten mit Features + Target |
| `data/processed/iad_flights_val.parquet`   | ~2-4 MB | Validierungsdaten |
| `data/processed/feature_metadata.json` | <10 KB | Feature-Liste + Split-Statistiken |

## Nächste Schritte (CRISP-DM Phase 4: Modeling)

1. **Baseline-Modell:** Logistic Regression (mit `class_weight='balanced'`)
2. **Baum-Modelle:** Random Forest, dann XGBoost/LightGBM
3. **Hyperparameter-Tuning:** Optuna oder GridSearch
4. **Evaluation:** PR-AUC, F1, Brier Score, Confusion Matrix
5. **Feature-Importance:** SHAP-Werte für die Top-Features
6. **Modell-Persistierung:** Joblib/Pickle
