# Inference (Phase 6) – `src/inference.py`

CLI + Programmatic API, um für einen oder mehrere Flüge eine
Verspätungs-Wahrscheinlichkeit vorherzusagen.

## Wetterhinweis

Wetter ist als Input-Feld vorbereitet (`weather: {source, hour_utc, temp_c, wind_kts, vis_m, precip_mm, pressure_hpa}`).
Aktuell ist es noch **kein** Modell-Feature (Modelle wurden ohne Wetter
trainiert), wird aber bei der Inferenz protokolliert, damit die Integration
später nur ein Modell-Re-Training erfordert.

Geplante Quelle:
- **Training**: NOAA NCEI ISD (Bulk-CSV, Stundenwerte, IAD Station 724030-13743)
- **Inferenz**: NWS/NDFD oder Open-Meteo (kostenlos, kein API-Key)

## Aufruf

```powershell
# Default: XGBoost, threshold 0.5
python -m src.inference --demo

# JSON-Datei
python -m src.inference --flight-json tmp_flight.json

# CSV-Batch
python -m src.inference --flight-csv examples_flights.csv --output-csv predictions.csv
```

## Input-Felder (pro Flug)

| Feld | Typ | Pflicht | Bemerkung |
|---|---|---|---|
| `carrier_code` | str | nein (default `UA`) | aktuell nur UA trainiert |
| `date` | YYYY-MM-DD | ja | Flugdatum |
| `flight_number` | str/int | nein (default `0`) | für Frequency-Encoding |
| `dest_airport` | str/IATA | nein (default `UNK`) | für Frequency-Encoding |
| `sched_dep_time` | HH:MM oder HHMM | nein (default `12:00`) | geplante Abflugszeit |
| `sched_elapsed_min` | int | nein (default `120`) | geplante Block-Time |
| `weather` | dict | nein | nur Protokoll, kein Feature |

## Post-hoc-Felder werden ignoriert

Folgende Felder werden aus dem Input **entfernt** (Leakage-Schutz):
`departure_delay_min`, `delay_label`, `is_cancelled`, `cancellation_code`,
`actual_elapsed_time`, `arrival_delay_min`, `wheels_off`, `wheels_on`,
`taxi_out`, `taxi_in`.

## Output (JSON)

```json
{
  "input": {...},
  "prob_delay": 0.6687,
  "threshold": 0.5,
  "label": 1,
  "model": "XGBClassifier",
  "model_path": "models/xgboost.joblib",
  "feature_version": {"n_features": 27, "delay_threshold_min": 15},
  "weather_used": null
}
```

`label = 1` bedeutet „Vorhersage: Verspätung ≥ 15 Min“.

## Beispiel-Output (Batch)

| flight | prob | label |
|---|---:|---:|
| 944_LHR_18:30 | 0.67 | 1 |
| 123_LAX_08:00 | 0.33 | 0 |
| 500_SFO_12:15 | 0.40 | 0 |

## Threshold-Wahl

Standard ist 0.5. Aus den `metrics.json` der Notebooks steht der
`optimal_threshold` (F1-optimal) bereit, z. B.:

```powershell
python -m src.inference --threshold 0.5375 --flight-json tmp_flight.json
```

Für Precision-orientierte Use-Cases ist 0.5+ oft der bessere Default.
