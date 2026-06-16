# Snapshot – Mit Wetter + Reduktion + Wetter-bewusster Split

**Datum:** 2026-06-16 10:04
**Inhalt:** XGBoost auf neuen Features (24) mit Wetter-Coverage
**Referenz:** `results/snapshots/20260616_090104/` (alter Stand: 27 Features, kein Wetter)

## Änderungen ggü. Baseline

| Aspekt | Baseline (20260616_090104) | Dies (20260616_100444) |
|---|---|---|
| Features | 27 (ohne Wetter) | 24 (mit Wetter, ohne Rauschen) |
| Train | 35 158 Zeilen (2025) | **49 609 Zeilen** (2024 + 2025 bis 06-30) |
| Val | 12 059 Zeilen (2026-Q1/Q2) | **5 494 Zeilen** (2025-07-01..2025-08-26) |
| Wetter-Coverage | keine | voll (NCEI ISD-Lite) |
| Split-Strategie | temporal 2025/2026 | wetter-bewusst 2024-2025 |

## Headline-Metriken (XGBoost, single config)

| Metrik | Baseline | Neu | Δ |
|---|---:|---:|---:|
| PR-AUC | 0.3147 | **0.4853** | **+0.1705** |
| ROC-AUC | 0.6938 | 0.7706 | +0.0767 |
| Brier | 0.1914 | 0.1425 | -0.0488 |
| F1@opt | 0.3564 | 0.5169 | +0.1605 |
| Precision@opt | 0.2991 | 0.4492 | +0.1501 |
| Recall@opt | 0.4409 | 0.6086 | +0.1677 |

Optimaler Threshold: 0.273 (war 0.537).

## Wichtige Caveats

1. **Val-Perioden sind nicht vergleichbar**: Baseline maß auf 2026-Q1/Q2 (Winter),
   dies auf 2025-Jul/Aug (Sommer). Sommer hat tendenziell mehr
   verspätete Flüge (höhere Baseline-Rate 22 % vs. 14 %), aber auch
   leichter vorhersagbare (Wetter schlägt stark durch).
2. **Wetter im Val explizit verfügbar** – das war der Hauptuntrieb.
3. **Größere Train-Menge** durch Hinzunahme von 2024.

## Wetter-Status

- 2024-01-01 bis 2025-08-27 04:00 UTC: vollständig (14 497 Stunden)
- 2025-08-27 04:00 UTC bis 2026: **nicht verfügbar** (NCEI-Publizierungsverzug)
- 2026 noch nicht publiziert (Stand 2026-06-16)

## Reproduktion

```powershell
# Wetter bereits heruntergeladen; falls neu:
python -m src.weather_download --start-year 2024 --end-year 2025

# Pipeline:
python src/02_data_preparation.py

# XGBoost (oder notebooks/03_xgboost.ipynb):
python src/_xgb_smoke_v2.py
```

## Nächste Schritte

- Warten auf zusätzliche BTS-2024-Roh-Dateien (vom User manuell ergänzt).
- Snapshot-Vergleich mit altem Modell.
- Doku finalisieren.
