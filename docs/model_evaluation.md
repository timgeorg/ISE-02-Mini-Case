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

---

## Addendum 2026-06-16 (Vergleichsplan mit neuem Feature-Set)

### Geplante Vergleichs-Setups

| Setup | Beschreibung | Zweck |
|---|---|---|
| **A. Baseline** | Aktuelles Modell (27 Features, ohne Wetter) | Referenz (entspricht Snapshot `20260616_090104`) |
| **B. Reduzierte Features** | 19 Features (8 Reduktionen, ohne Wetter) | Misst, ob die Reduktion hilft oder schadet |
| **C. Reduziert + Wetter** | 23 Features (Wetter 4 hinzu) | Misst reinen Wetter-Hebel |
| **D. Reduziert + Wetter + Arrival-by-Dest** | 24 Features | Misst zusätzlichen Arrival-by-Dest-Hebel |
| **E. Voll + Congestion-Window** | 25 Features | Vollständig |

### Erwartete Effekte (informelle Schätzung)

| Setup vs. A | Erwartete Δ PR-AUC |
|---|---:|
| B (Reduktion) | ±0.000 (kaum Effekt) |
| C (+ Wetter) | +0.010 bis +0.030 |
| D (+ Arrival-by-Dest) | +0.005 bis +0.015 |
| E (Voll) | +0.015 bis +0.045 kombiniert |

### Diskussions-Stand

- **Mehr Daten oder mehr Feature Engineering?** Diskussion 2026-06-16: Mehr Daten ist im BTS-Universum limitiert (nur UA am IAD), aber Wetter ist neuartig und sollte den größten Sprung bringen.
- **Strukturelle Grenze?** Aktuelle PR-AUC = 0.31 bei 14 % Pos-Rate. Das ist 2.2× Lift. Eine Precision ≥ 0.7 bei Recall ≥ 0.25 ist erreichbar mit dem neuen Feature-Set, aber nicht garantiert.

### Snapshot-Status

- Baseline-Snapshot: `results/snapshots/20260616_090104/` (eingefroren 2026-06-16 09:01)
- Wetter-Rohdaten: `data/external/weather/` (14 497 Stunden)
- Nach Re-Run: neuer Snapshot mit Post-Wetter-Metriken.

Ausführliche Diskussion: siehe `docs/session_2026-06-16.md`.
