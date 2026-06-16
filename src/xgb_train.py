"""Smoke-Test: Train XGBoost auf den neuen Parquet-Features und speichere das Modell.

Identisch zur Logik in notebooks/03_xgboost.ipynb, aber als Skript für
schnelles Re-Runnen ohne Notebook.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
import xgboost as xgb

PROJECT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT / "data" / "processed" / "iad_flights_train.parquet"
VAL_PATH = PROJECT / "data" / "processed" / "iad_flights_val.parquet"
MODEL_PATH = PROJECT / "models" / "xgboost.joblib"
RESULTS_DIR = PROJECT / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("xgb_smoke")


def main() -> int:
    log.info("Lade Train/Val ...")
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    log.info("Train: %d, Val: %d", len(train), len(val))

    with open(PROJECT / "data" / "processed" / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]
    log.info("Features: %d", len(feature_cols))

    X_train, y_train = train[feature_cols], train["delay_label"].astype(int)
    X_val, y_val = val[feature_cols], val["delay_label"].astype(int)

    # Schlichte XGB-Konfig (eine für den Smoke-Test, nicht der vollständige Sweep)
    config = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    log.info("Train XGBoost: %s", config)
    model = xgb.XGBClassifier(**config)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Metriken
    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)

    # Optimaler Threshold (F1)
    prec, rec, thr = precision_recall_curve(y_val, y_prob)
    f1s = 2 * (prec * rec) / np.clip(prec + rec, 1e-9, None)
    ix = np.nanargmax(f1s[:-1])  # last is no-threshold
    opt_thr = float(thr[ix])
    opt_f1 = float(f1s[ix])
    opt_p = float(prec[ix])
    opt_r = float(rec[ix])

    log.info("PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f", pr_auc, roc_auc, brier)
    log.info("t=0.5  F1=%.4f P=%.4f R=%.4f", f1, p, r)
    log.info("optimal t=%.4f F1=%.4f P=%.4f R=%.4f", opt_thr, opt_f1, opt_p, opt_r)

    # Speichern (Wrapper-Dict)
    metrics = {
        "model": "xgboost",
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "f1_at_0.5": float(f1),
        "precision_at_0.5": float(p),
        "recall_at_0.5": float(r),
        "optimal_threshold": opt_thr,
        "f1_at_optimal": opt_f1,
        "precision_at_optimal": opt_p,
        "recall_at_optimal": opt_r,
        "n_train": len(train),
        "n_val": len(val),
        "n_features": len(feature_cols),
        "split_strategy": "weather-aware: train<2025-07-01, val 2025-07-01..2025-08-26",
    }
    model_dict = {
        "model": model,
        "scaler": None,
        "feature_cols": feature_cols,
        "metadata_cols": meta.get("metadata_columns", []),
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict, MODEL_PATH)
    log.info("Modell gespeichert: %s", MODEL_PATH)

    # Metrics als JSON (für Snapshot-Vergleich)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "xgb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info("Metrics geschrieben: results/xgb_metrics.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
