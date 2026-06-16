# Snapshot – Baseline-Modelle (vor Wetter-Integration)

**Datum:** 2026-06-16 09:01:04
**Zweck:** Sicherung des Modell-Stands vor dem Einbau von Wetter-Features.

## Inhalt

- `*_metrics.json` – Metriken aller drei Modelle auf dem 2026-Q1/Q2-Validierungsset
- `*_pr_curve.png` – Precision-Recall-Kurven
- `*_confusion_matrix.png` – Konfusionsmatrizen
- `*_feature_importance.png` – Feature-Wichtigkeit je Modell
- `model_comparison.png` – Modellvergleich-Plot

## Reproduktion

Notebooks `notebooks/01_baseline_logreg.ipynb`, `02_random_forest.ipynb`,
`03_xgboost.ipynb` und `04_compare_models.ipynb` wurden mit den Daten aus
`data/processed/iad_flights_train.parquet` und `iad_flights_val.parquet` (Stand
2026-06-16) trainiert.

## Headline-Metriken (zur Erinnerung)

| Modell | PR-AUC | ROC-AUC | F1 (optimal) | Brier |
|---|---:|---:|---:|---:|
| LogReg | 0.276 | 0.681 | 0.336 | 0.215 |
| Random Forest | 0.301 | 0.691 | 0.353 | 0.182 |
| **XGBoost** | **0.315** | **0.694** | **0.356** | 0.191 |

Kein Precision@0.7 erreichbar – strukturelle Grenze.
