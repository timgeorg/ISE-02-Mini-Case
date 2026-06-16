# ISE-02-Mini-Case Automation Script
# This script runs the full pipeline: Data Prep -> Advanced Modeling -> Performance Report

$PROJECT_ROOT = "C:\Users\MEURERA\A\ML-itse\ISE-02-Mini-Case"
cd $PROJECT_ROOT

Write-Host "--- [1/3] Starting Data Preparation (2024 -> 2025 Split) ---" -ForegroundColor Cyan
# Use the existing venv if available, otherwise use system python
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

python src\02_data_preparation.py

Write-Host "`n--- [2/3] Running Advanced Modeling (Cost-Sensitive & Ensemble) ---" -ForegroundColor Cyan
python src\advanced_modeling.py

Write-Host "`n--- [3/3] Generating Performance Report ---" -ForegroundColor Cyan
# Create a small helper script to output the table we discussed
$report_script = @"
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score

PROJECT_ROOT = Path(r"$PROJECT_ROOT")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def evaluate(target_name, threshold=0.5):
    val_df = pd.read_parquet(PROCESSED_DIR / "iad_flights_val.parquet")
    exclude_cols = ["date", "carrier_code", "flight_number", "dest_airport", 
                   "sched_dep_time", "departure_delay_min", "delay_label", 
                   "delay_label_15", "delay_label_40"]
    feature_cols = [col for col in val_df.columns if col not in exclude_cols]
    X_val, y_val = val_df[feature_cols], val_df[target_name]
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
"@

$report_script | Out-File -FilePath "src\generate_report.py" -Encoding utf8
python src\generate_report.py

Write-Host "`n--- DONE! ---" -ForegroundColor Green
Write-Host "Check 'final_performance_report.txt' for the results."
