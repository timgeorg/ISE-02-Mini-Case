# Snapshot – Final Model Evaluation (Business Case)

**Datum:** 2026-06-16 10:18
**Zweck:** Vollständige Evaluation aller 3 Modelle + Business-Case-Bewertung
**Referenz:** Finale Modell-Version für die Aufgabenstellung.

## Datenstand

- 4 BTS-Tabellen × 2 Jahre (2024 + 2025) = ~95 000 Flüge
- NCEI ISD-Lite Wetter: 14 497 Stunden (2024-01-01 bis 2025-08-27)
- 24 Features, wetter-bewusster temporaler Split (Train < 2025-07-01, Val 2025-07-01..2025-08-26)

## Modelle

| Modell | PR-AUC | ROC-AUC | F1@opt | P@opt | R@opt | Brier |
|---|---:|---:|---:|---:|---:|---:|
| logreg | 0.4549 | 0.7521 | 0.5079 | 0.4273 | 0.6260 | 0.2389 |
| **random_forest** | **0.5268** | **0.7783** | **0.5278** | **0.4613** | 0.6169 | 0.2291 |
| xgboost | 0.4853 | 0.7706 | 0.5169 | 0.4492 | 0.6086 | **0.1425** |

**Empfehlung: Random Forest** (beste Diskrimination). XGBoost hat das beste
Brier-Score (besser kalibriert) – Trade-off zwischen Diskrimination und
Kalibrierung.

## Inhalt

- `final_metrics.json` – konsolidierte Metriken aller 3 Modelle
- `final_pr_curves.png` – Precision-Recall-Kurven
- `final_roc_curves.png` – ROC-Kurven
- `final_calibration.png` – Reliability-Diagramme
- `final_confusion_matrices.png` – Confusion Matrices am F1-Threshold
- `final_model_comparison.png` – Balkendiagramme
- `final_feature_importance.png` – Top-15 Features (XGB + RF)
- `final_evaluation.md` – Business-Case-Bewertung mit Empfehlung
- `logreg.joblib`, `random_forest.joblib`, `xgboost.joblib` – trainierte Modelle

## Headline gegen Business Case

**Aufgabenstellung:** "Verspätungen frühzeitig erkennen. Bessere Ressourcenplanung."

**Ergebnis:** Random-Forest-Modell mit PR-AUC 0.527 (vs. 0.22 Val-Base-Rate)
erreicht **2.4× Lift**. Bei Precision ≥ 0.7 (operativer Use-Case für teure
Aktionen) werden 21 % der echten Verspätungen korrekt erkannt.

**Wirtschaftlich:** Bei angenommenen Kosten 500 €/Aktion und vermiedenen
Verspätungskosten 2000 € ist das Modell **deutlich profitabel** ab Precision 10 %
– wir sind bei > 40 %.
