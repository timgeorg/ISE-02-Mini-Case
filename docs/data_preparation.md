# CRISP-DM Phase 3: Data Preparation

_Erstellt am 2026-06-15 17:14_

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

Insgesamt **27 Features**, eingeteilt in Gruppen:

| Gruppe | Features | Beschreibung |
|---|---|---|
| **Datum (zyklisch)** | `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Sinus/Cosinus-Encoding für Wochentag & Monat (Modell erkennt Zyklizität) |
| **Datum (linear)** | `dayofweek`, `day`, `month`, `weekofyear`, `quarter`, `is_weekend`, `is_holiday`, `is_school_break`, `is_free_day` | Roh-Encoding. `is_holiday` = federal, `is_school_break` = regionale Schulferien (DC, Fairfax, Montgomery), `is_free_day` = Wochenende OR holiday OR school_break |
| **Tageszeit (zyklisch)** | `tod_sin`, `tod_cos` | Sinus/Cosinus für Minuten seit Mitternacht |
| **Tageszeit (linear)** | `sched_dep_hour`, `sched_dep_minute`, `sched_dep_min_of_day`, `time_of_day` | Roh-Encoding + 4-Bucket (Nacht/Morgen/Mittag/Abend) |
| **Strecke** | `sched_elapsed_min` | Proxy für Distanz (Kurz-/Langstrecke) |
| **Frequency-Encoding** | `flight_combo_freq`, `dest_freq` | Häufigkeit der (carrier, dest, flight_number)-Kombination bzw. Destination |
| **Arrival-Aggregate (1 Tag)** | `origin_daily_arrival_delay_mean`, `origin_daily_arrival_n` | Mittlere Ankunftsverspätung an diesem Tag von dieser Origin |
| **Arrival-Aggregate (7-Tage-Rolling)** | `origin_7d_arrival_delay_mean`, `origin_7d_arrival_n` | Mittelwert/Summe der letzten 7 Tage (ohne aktuellen Tag). Glättet Ausreißer, bildet Trends ab. |
| **Cancellation-Aggregate** | `cancellations_on_day` | Anzahl Stornierungen am gleichen Tag (IAD-weit) |

### Regionale Feiertage (`is_school_break`)

Quelle: veröffentlichte Schulkalender für 3 Bezirke im IAD-Einzugsgebiet:

- **DC Public Schools** (Washington DC)
- **Fairfax County Public Schools** (Northern Virginia)
- **Montgomery County Public Schools** (Maryland Suburbs)

Kodiert als Union der Schulferien-Intervalle aller 3 Bezirke (Frühjahrs-, Sommer-, Herbst-, Winter-, Thanksgiving-Pause).

### 7-Tage-Rolling-Aggregate

Pro `(origin_airport, date)` wird der rolling-7-Tage-Mittel der `arrival_delay_min` berechnet, mit `shift(1)` damit der aktuelle Tag ausgeschlossen ist. Damit hat jeder Flug nur Zugriff auf **vorhergehende** Verspätungs-Information – kein Leakage.


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
| `delay_label=1` (Anteil) | 16.27 % | 13.96 % |
| `is_weekend` | 27.67 % | 27.57 % |
| `is_holiday` (federal) | 2.95 % | 2.40 % |
| `is_school_break` (regional) | 10.35 % | 8.67 % |
| `is_free_day` (kombi) | 38.59 % | 36.25 % |
| `origin_daily_arrival_n > 0` | 99.75 % | 99.68 % |
| `origin_7d_arrival_n > 0` | 99.45 % | 99.64 % |
| `origin_7d_arrival_delay_mean` Mittel | 3.52 | 0.06 |
| `origin_7d_arrival_delay_mean` Max | 207.57 | 170.64 |
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


### Beobachtungen zu den neuen Features (Schulferien + 7d-Rolling)

- **`is_school_break`** tritt in **10.3% der Train-Flüge** und **8.7% der Val-Flüge** auf. Damit ist der Val-Wert deutlich niedriger (Val = Jan-Apr enthaelt nur Spring Break + Winter Break), was die Modell-Performance am Anfang beeinflussen kann.
- **`is_free_day` (kombinierte Variable)** deckt Train ~39 % / Val ~36 % ab. Konsistent mit Wochenend-Quote (~28 %).
- **`origin_7d_arrival_n > 0`** fuer **99.5%** der Train-Fluege und **99.6%** der Val-Fluege. Rest: kein 7-Tage-Historie (cold start, z. B. neuer Origin oder Datenbeginn).
- **7-Tage-Mittel der Origin-Verspaetung** zeigt im Mittel ~3.5 Min Verspaetung (Train) - positiv, weil Carrier an IAD im Schnitt leicht verspaetet ist.

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

---

## Addendum 2026-06-16 (Diskussion + Plan für Phase 3.1)

### Geplante Feature-Änderungen (vor Re-Run)

**Reduktion (8 Features raus):**

| Feature | Grund |
|---|---|
| `day` | Quasi zufällig bei monatszyklischer Kodierung; leichtes Overfit-Risiko |
| `quarter` | Vollständig redundant zu `month` (4 Werte, lineare Beziehung) |
| `weekofyear` | Doppelung zur zyklischen Monatskodierung |
| `is_free_day` | Linear abhängig aus `is_weekend OR is_holiday OR is_school_break` |
| `time_of_day` (kategorisch) | Wir behalten `tod_sin/cos`; kategorische Variante war Quelle des XGB-Dtype-Problems |
| `sched_dep_minute` | Mikroverschiebung hat keinen Delay-Effekt |
| `origin_daily_arrival_n`, `origin_7d_arrival_n` | Konstanten pro Tag, nicht informativ |
| `flight_combo_freq` | Bei einer Carrier-Klasse schwankend, abhängig von `dest_freq` |

**Ergebnis: 27 → ~19 Features.**

**Hinzufügen (~6 Features):**

| Feature | Erwarteter Hebel | Begründung |
|---|---|---|
| `temp_c` zur scheduled dep hour | mittel-hoch | Wetter-Hebel: extreme Temperaturen verlangsamen den Betrieb |
| `wind_kts` zur scheduled dep hour | mittel-hoch | High Wind = Ground Stop / Runway Change |
| `precip_1h_mm` zur scheduled dep hour | mittel | Regen verlangsamt Roll-Vorgänge |
| `pressure_hpa` zur scheduled dep hour | niedrig-mittel | Drucksystem-Verschiebung korreliert mit Wetter |
| `dest_3d_arrival_delay_mean` | mittel | „War LHR gestern verspätet?" – stärkstes verfügbares Proxy-Signal |
| `is_congestion_window` | niedrig | dep_hour in {15–19} AND weekday = Bank-Stunden (Slot-Belegung) |

**Ergebnis: 19 → ~25 Features.**

### Wetter-Datenquelle (geplant)

- **Quelle:** NCEI ISD-Lite Bulk
- **URL-Schema:** `https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/<YYYY>/<USAF>-<WMO>-<YYYY>.gz`
- **IAD USAF:** 724050, **WMO:** 13743
- **Lizenz:** Public Domain (US-Regierung)
- **Downloader:** `src/weather_download.py` (Stand 2026-06-16: 14 497 Stunden, 2024-01-01 bis 2025-08-27)
- **Wartung:** automatischer Soft-Skip bei nicht-verfügbaren Jahren (NCEI-Verzug)

### Explizit nicht (Over-Engineering)

1. Aircraft Rotation (Tail-Number) – Tail-Number ist zu 99 % leer in BTS.
2. Taxiway-Belegung / ATC-Load – NAS-Daten erforderlich (FAA SWIM).
3. Flugzeugtyp / Gate-Readiness / Scheduled TAT – In BTS Detailed Statistics nicht enthalten.
4. Carrier-übergreifende Origin-Aggregate – BTS hat nur UA am IAD.
5. Komplexe Target-Encoding (z. B. nested CV) – „Simple halten" laut User.

### Reihenfolge der Implementierung

1. `02_data_preparation.py` + `inference.py` synchron anpassen (Feature-Liste).
2. `04_weather_join.py` (neu) baut Wetter-Features.
3. Notebooks `01-03` re-runnen.
4. Vergleichs-Notebook `04` mit zwei Spalten (Baseline / Mit Wetter + Reduktion).
5. Snapshot der Baseline (`results/snapshots/20260616_090104/`) als Referenz.

Ausführliche Diskussion: siehe `docs/session_2026-06-16.md`.
