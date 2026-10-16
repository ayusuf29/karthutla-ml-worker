"""
03_train_xgboost.py
===================
Trains a production-grade XGBoost Classifier on peatland biophysical & anthropogenic features
for Karhutla ignition risk prediction.

Constraints:
  - Input features strictly match H3FeatureVector contract in proto & risk_model.py:
      [peat_depth_meters, ground_water_level_cm, days_without_rain,
       vapor_pressure_deficit, distance_to_road_meters, distance_to_canal_meters]
  - CPU-optimized training with tree_method="hist"
  - Handles class imbalance via scale_pos_weight
  - Strict temporal split (Historical 2009/2015 -> Train, Recent 2026 -> Validation/Test)
  - Saves serialized model to ml-worker/models/karhutla_xgb_v1.json
  - Exports performance metrics to ml-worker/models/feature_metrics.json
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
xgb.compat.SKLEARN_INSTALLED = True

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainXGBoost")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "ml-worker", "models")
DATASET_PATH = os.path.join(PROCESSED_DIR, "h3_training_dataset.parquet")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "karhutla_xgb_v1.json")
METRICS_OUT_PATH = os.path.join(MODELS_DIR, "feature_metrics.json")

# Mandatory feature contract matching H3FeatureVector Protobuf definition
FEATURE_COLUMNS = [
    "peat_depth_meters",
    "ground_water_level_cm",
    "days_without_rain",
    "vapor_pressure_deficit",
    "distance_to_road_meters",
    "distance_to_canal_meters"
]
TARGET_COLUMN = "ignition_7d"

def train_model():
    t0 = time.time()
    logger.info("=================================================================")
    logger.info("Nusantara Fire Sentinel: Step 03 - Train Karhutla XGBoost Model")
    logger.info("=================================================================")

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        logger.error("Training dataset not found: %s. Run 02_merge_weather_and_firms.py first!", DATASET_PATH)
        sys.exit(1)

    logger.info("Loading training dataset from %s...", DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)
    logger.info("Loaded %d records. Total columns: %s", len(df), list(df.columns))

    # Verify required feature columns exist
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    # Temporal split: Historical (<= 2023 / 2009 & 2015) vs Recent (2026)
    train_mask = df["year"] < 2026
    val_mask = df["year"] >= 2026

    df_train = df[train_mask]
    df_val = df[val_mask]

    # In case the dataset does not have 2026 records or too few, fallback to chronological split
    if len(df_val) < 100 or df_val[TARGET_COLUMN].nunique() < 2:
        logger.warning("Val set from 2026 has insufficient samples or single class. Using 80/20 chronological split.")
        df_sorted = df.sort_values(by=["date"]).reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.8)
        df_train = df_sorted.iloc[:split_idx]
        df_val = df_sorted.iloc[split_idx:]

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train[TARGET_COLUMN].values
    X_val = df_val[FEATURE_COLUMNS].values
    y_val = df_val[TARGET_COLUMN].values

    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    pos_val = int(np.sum(y_val == 1))
    neg_val = int(np.sum(y_val == 0))

    scale_pos_weight = float(neg_train / max(1, pos_train))

    logger.info("Train set: %d rows (Pos: %d, Neg: %d, Pos Ratio: %.2f%%)",
                len(df_train), pos_train, neg_train, (pos_train / len(df_train)) * 100)
    logger.info("Val set:   %d rows (Pos: %d, Neg: %d, Pos Ratio: %.2f%%)",
                len(df_val), pos_val, neg_val, (pos_val / len(df_val)) * 100)
    logger.info("Calculated scale_pos_weight: %.3f", scale_pos_weight)

    # Initialize XGBoost Classifier with CPU histogram mode
    # Dampen scale_pos_weight slightly (sqrt or 0.7x) to avoid over-predicting false alarms
    effective_spw = np.clip(np.sqrt(scale_pos_weight) * 1.5, 1.0, 15.0)
    logger.info("Effective scale_pos_weight applied: %.3f", effective_spw)

    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=effective_spw,
        tree_method="hist",
        eval_metric=["logloss", "auc"],
        random_state=42,
        n_jobs=-1
    )

    logger.info("Starting model training on CPU...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    logger.info("XGBoost training completed.")

    # Validation evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    train_probs = model.predict_proba(X_train)[:, 1]

    val_preds_50 = (val_probs >= 0.50).astype(int)

    val_auc = float(roc_auc_score(y_val, val_probs))
    train_auc = float(roc_auc_score(y_train, train_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    train_pr_auc = float(average_precision_score(y_train, train_probs))
    val_f1 = float(f1_score(y_val, val_preds_50, zero_division=0))
    val_precision = float(precision_score(y_val, val_preds_50, zero_division=0))
    val_recall = float(recall_score(y_val, val_preds_50, zero_division=0))

    cm = confusion_matrix(y_val, val_preds_50).tolist()

    # Search for optimal F1 threshold
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_val, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info("=== EVALUATION METRICS ===")
    logger.info("Train ROC-AUC: %.4f | Val ROC-AUC: %.4f", train_auc, val_auc)
    logger.info("Train PR-AUC:  %.4f | Val PR-AUC:  %.4f", train_pr_auc, val_pr_auc)
    logger.info("Val Precision: %.4f | Val Recall:  %.4f | Val F1 (0.5): %.4f", val_precision, val_recall, val_f1)
    logger.info("Optimal Threshold: %.3f (F1: %.4f)", best_thresh, best_f1)
    logger.info("Confusion Matrix (at 0.50 threshold):\n%s", np.array(cm))

    # Feature Importance
    booster = model.get_booster()
    # Map f0, f1 to actual feature names
    score_weight = booster.get_score(importance_type="weight")
    score_gain = booster.get_score(importance_type="gain")
    score_cover = booster.get_score(importance_type="cover")

    feature_importances = {}
    for idx, name in enumerate(FEATURE_COLUMNS):
        f_key = f"f{idx}"
        feature_importances[name] = {
            "weight": float(score_weight.get(f_key, 0.0)),
            "gain": round(float(score_gain.get(f_key, 0.0)), 4),
            "cover": round(float(score_cover.get(f_key, 0.0)), 2)
        }

    # Sort by gain descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1]["gain"], reverse=True))

    logger.info("=== FEATURE IMPORTANCES (GAIN) ===")
    for feat, scores in sorted_importances.items():
        logger.info("  %-28s: Gain=%.4f, Weight=%.1f", feat, scores["gain"], scores["weight"])

    # Categorization Distribution on Val Set
    cats = []
    for p in val_probs:
        if p >= 0.85:
            cats.append("EXTREME")
        elif p >= 0.60:
            cats.append("HIGH")
        elif p >= 0.30:
            cats.append("MEDIUM")
        else:
            cats.append("LOW")
    cat_counts = pd.Series(cats).value_counts().to_dict()

    # Save Model
    logger.info("Saving trained XGBoost model to %s...", MODEL_OUT_PATH)
    model.save_model(MODEL_OUT_PATH)

    # Save Metrics JSON
    metrics_summary = {
        "model_version": "karhutla_xgb_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_engine": "CPU (XGBoost Histogram)",
        "feature_order": FEATURE_COLUMNS,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "metrics": {
            "train_roc_auc": round(train_auc, 4),
            "val_roc_auc": round(val_auc, 4),
            "train_pr_auc": round(train_pr_auc, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "val_f1_threshold_0_50": round(val_f1, 4),
            "val_precision": round(val_precision, 4),
            "val_recall": round(val_recall, 4),
            "optimal_threshold": round(best_thresh, 3),
            "optimal_f1": round(best_f1, 4),
            "confusion_matrix": cm
        },
        "risk_category_distribution_validation": cat_counts,
        "feature_importances": sorted_importances
    }

    with open(METRICS_OUT_PATH, "w") as fp:
        json.dump(metrics_summary, fp, indent=2)

    logger.info("Successfully exported metrics to %s", METRICS_OUT_PATH)
    logger.info("Step 03 completed in %.2fs total", time.time() - t0)

if __name__ == "__main__":
    train_model()
