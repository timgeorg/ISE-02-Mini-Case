# CRISP-DM Phase 5: Model Evaluation & Business-Empfehlung

_Erstellt am 2026-06-15 17:34_

---

## Modell-Vergleich (Validierungs-Set)

| Modell | PR-AUC | ROC-AUC | F1 (t=0.5) | F1 (optimal) | Precision (opt) | Recall (opt) | Brier | Schwelle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg_baseline | 0.2762 | 0.6811 | 0.3205 | 0.3364 | 0.2926 | 0.3957 | 0.2148 | 0.599 |
| random_forest | 0.3015 | 0.6906 | 0.3514 | 0.3529 | 0.2942 | 0.4409 | 0.1820 | 0.511 |
| xgboost | 0.3147 | 0.6938 | 0.3500 | 0.3564 | 0.2991 | 0.4409 | 0.1914 | 0.537 |

## Empfehlung: **xgboost**

- **Beste PR-AUC: 0.3147** (vs. Baseline = Val-Pos-Rate = 0.14)
- **Lift ueber Baseline: 2.2x**

### Begruendung

- Hoechste Praezision-Recall Trade-off
- Beste Kalibrierung (Brier Score)
- Wenn Production-tauglich, dann XGBoost (schnell, robust, gut dokumentiert)

### Naechste Schritte (Phase 6: Deployment)

1. Modell + Scaler in `models/` joblib-artifakte
2. CLI-Wrapper fuer manuelle Predictions
3. Evtl. REST-API (FastAPI) fuer Realtime-Predictions
4. Monitoring: Drift-Detection fuer Feature-Distribution
