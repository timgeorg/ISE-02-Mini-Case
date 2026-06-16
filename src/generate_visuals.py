"""
Visualisation Script for Flight Delay Prediction.
Generates PNGs for Precision-Recall curves and Confusion Matrices.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay

PROJECT_ROOT = Path(r"C:\Users\MEURERA\A\ML-itse\ISE-02-Mini-Case")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_results(target_name):
    print(f"Generating visuals for {target_name}...")
    
    # 1. Load Data
    val_df = pd.read_parquet(PROCESSED_DIR / "iad_flights_val.parquet")
    exclude_cols = ["date", "carrier_code", "flight_number", "dest_airport", 
                   "sched_dep_time", "departure_delay_min", "delay_label", 
                   "delay_label_15", "delay_label_40"]
    feature_cols = [col for col in val_df.columns if col not in exclude_cols]
    X_val_raw = val_df[feature_cols]
    y_val = val_df[target_name]
    
    # 2. Load Model & Imputer
    model = joblib.load(MODELS_DIR / f"ensemble_{target_name}.joblib")
    imputer = joblib.load(MODELS_DIR / f"imputer_{target_name}.joblib")
    X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_cols)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    # --- Plot 1: Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    plt.plot(recall, precision, label=f'Ensemble {target_name}', color='blue', lw=2)
    plt.axvline(x=0.9, color='red', linestyle='--', label='Recall Target 90%')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / f"pr_curve_{target_name}.png")
    plt.close()

    # --- Plot 2: Confusion Matrix at Recall >= 90% ---
    # Find threshold for 90% recall
    idx = np.where(recall >= 0.9)[0]
    best_thresh = thresholds[idx[-1]] if len(idx) > 0 else 0.5
    preds = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_val, preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Punctual', 'Delayed'], 
                yticklabels=['Punctual', 'Delayed'])
    plt.title(f'Confusion Matrix {target_name}\n(Threshold={best_thresh:.3f} for Recall >= 90%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(RESULTS_DIR / f"confusion_matrix_{target_name}.png")
    plt.close()

if __name__ == "__main__":
    targets = ["delay_label_15", "delay_label_40"]
    for t in targets:
        try:
            plot_results(t)
        except Exception as e:
            print(f"Error plotting {t}: {e}")
    print("\nVisuals saved to: " + str(RESULTS_DIR))
