"""
Advanced Modeling Pipeline for Flight Delay Prediction.
Implements:
1. Cost-Sensitive Learning (FN Penalty)
2. Ensemble Methods (Voting Classifier)
3. Multi-Target Evaluation (15min vs 40min)
4. Recall-Optimized Thresholding (Target >= 90%)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_cost_sensitive_xgboost(X_train, y_train, target_col="delay_label_15"):
    """
    Trains XGBoost with a balanced penalty.
    scale_pos_weight = count(negative) / count(positive)
    """
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    if target_col == "delay_label_15":
        aggressive_weight = weight * 1.2 
    elif target_col == "delay_label_40":
        aggressive_weight = weight * 1.1
    else:
        aggressive_weight = weight

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=aggressive_weight, 
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model, aggressive_weight

def train_ensemble(X_train, y_train, target_col="delay_label_15"):
    """
    Implements a Voting Classifier combining XGBoost, RF and LogReg.
    """
    # 1. Cost-sensitive XGBoost
    xgb_model, _ = train_cost_sensitive_xgboost(X_train, y_train, target_col)
    
    # 2. Random Forest (balanced)
    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. Simple LogReg (balanced)
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Voting Ensemble (Soft voting uses probabilities)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('lr', lr_model)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble

def find_recall_threshold(model, X_val, y_val, target_recall=0.90):
    """
    Finds the probability threshold required to achieve a specific recall.
    """
    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Find the largest threshold that still gives recall >= target_recall
    idx = np.where(recalls >= target_recall)[0]
    if len(idx) == 0:
        return None, 0.0
    
    best_threshold = thresholds[idx[-1]] if idx[-1] < len(thresholds) else thresholds[-1]
    return best_threshold, recalls[idx[-1]]

def run_advanced_pipeline():
    # Load data
    train_df = pd.read_parquet(PROCESSED_DIR / "iad_flights_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "iad_flights_val.parquet")
    
    # Feature columns (must match 02_data_preparation.py)
    feature_cols = [col for col in train_df.columns if col not in 
                    ["date", "carrier_code", "flight_number", "dest_airport", 
                     "sched_dep_time", "departure_delay_min", "delay_label", 
                     "delay_label_15", "delay_label_40"]]
    
    targets = ["delay_label_15", "delay_label_40"]
    
    for target in targets:
        print(f"\n--- Evaluating Target: {target} ---")
        X_train_raw = train_df[feature_cols].copy()
        y_train = train_df[target]
        X_val_raw = val_df[feature_cols].copy()
        y_val = val_df[target]
        
        # Impute NaN values (median strategy for weather features etc.)
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols)
        X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_cols)
        
        # 1. Cost-Sensitive XGBoost
        model_cs, weight = train_cost_sensitive_xgboost(X_train, y_train, target)
        
        # 2. Ensemble
        model_ens = train_ensemble(X_train, y_train, target)
        
        # 3. Threshold Optimization for Recall >= 90%
        thresh, actual_recall = find_recall_threshold(model_ens, X_val, y_val)
        
        # Final Predictions with optimized threshold
        probs = model_ens.predict_proba(X_val)[:, 1]
        preds = (probs >= thresh).astype(int)
        
        print(f"Optimized Threshold for Recall >= 90%: {thresh:.4f}")
        print(f"Actual Recall: {recall_score(y_val, preds):.4f}")
        print(f"Precision at this threshold: {precision_score(y_val, preds):.4f}")
        print(classification_report(y_val, preds))
        
        # Save models and imputer
        joblib.dump(model_cs, MODELS_DIR / f"cs_xgb_{target}.joblib")
        joblib.dump(model_ens, MODELS_DIR / f"ensemble_{target}.joblib")
        joblib.dump(imputer, MODELS_DIR / f"imputer_{target}.joblib")

if __name__ == "__main__":
    run_advanced_pipeline()
