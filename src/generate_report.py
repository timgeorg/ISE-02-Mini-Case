import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(r"C:\Users\MEURERA\A\ML-itse\ISE-02-Mini-Case")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def evaluate(target_name, threshold=0.5):
    val_df = pd.read_parquet(PROCESSED_DIR / "iad_flights_val.parquet")
    exclude_cols = ["date", "carrier_code", "flight_number", "dest_airport", 
                   "sched_dep_time", "departure_delay_min", "delay_label", 
                   "delay_label_15", "delay_label_40"]
    feature_cols = [col for col in val_df.columns if col not in exclude_cols]
    X_val_raw = val_df[feature_cols]
    y_val = val_df[target_name]
    # Load imputer and transform
    imputer = joblib.load(MODELS_DIR / f"imputer_{target_name}.joblib")
    X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_cols)
    model = joblib.load(MODELS_DIR / f"ensemble_{target_name}.joblib")
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "Target": target_name, "Thresh": threshold, 
        "Recall": recall_score(y_val, preds), "Precision": precision_score(y_val, preds),
        "F1": f1_score(y_val, preds)
    }

targets = ["delay_label_15", "delay_label_40"]
results = []
for t in targets:
    results.append(evaluate(t, 0.5))
    results.append(evaluate(t, 0.2)) # High Recall scenario

df = pd.DataFrame(results)
print("\n=== FINAL PERFORMANCE REPORT (Recall-Optimized) ===")
print(df.to_string(index=False))
with open("final_performance_report.txt", "w") as f:
    f.write(df.to_string(index=False))
