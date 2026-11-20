
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from monitoring.evaluate_model import FEATURE_COLUMNS


# ---------------- PATHS ----------------

ROOT = Path(__file__).resolve().parent.parent

CHAMPION_PATH = ROOT / "aqi_pipeline.pkl"
CANDIDATE_PATH = ROOT / "candidate_pipeline.pkl"
HOLDOUT_PATH = ROOT / "monitoring" / "quality_gate_holdout.csv"

BACKUP_PATH = ROOT / "aqi_pipeline_backup.pkl"
DECISION_PATH = ROOT / "monitoring" / "promotion_decision.json"


# ---------------- QUALITY THRESHOLDS ----------------

MIN_MAE_IMPROVEMENT = 0.02
MAX_RMSE_INCREASE = 0.0
MAX_R2_DROP = 0.01


# ---------------- LOAD MODEL ----------------

def load_pipeline(path):

    # Make the custom transformer available to joblib
    try:
        from app.transformers import Winsorizer
        import __main__

        __main__.Winsorizer = Winsorizer

    except ImportError:
        pass

    return joblib.load(path)


# ---------------- CALCULATE METRICS ----------------

def calculate_metrics(model, X, y):

    predictions = model.predict(X)

    predictions = np.clip(predictions, 0, 500)

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ---------------- MAIN QUALITY GATE ----------------

def quality_gate():

    print("Starting AQI model quality gate...")

    # Check required files
    for path in [CHAMPION_PATH, CANDIDATE_PATH, HOLDOUT_PATH]:

        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load holdout dataset
    holdout = pd.read_csv(HOLDOUT_PATH)

    if holdout.empty:
        raise ValueError("Holdout dataset is empty.")

    if len(holdout) < 2:
        raise ValueError("At least two holdout records are required for R2.")

    X_test = holdout[FEATURE_COLUMNS]
    y_test = holdout["Actual_AQI"]

    # Load champion and candidate
    champion = load_pipeline(CHAMPION_PATH)
    candidate = load_pipeline(CANDIDATE_PATH)

    # Evaluate both on exactly the same data
    champion_metrics = calculate_metrics(
        champion, X_test, y_test
    )

    candidate_metrics = calculate_metrics(
        candidate, X_test, y_test
    )

    # Calculate MAE improvement
    champion_mae = champion_metrics["MAE"]
    candidate_mae = candidate_metrics["MAE"]

    if champion_mae > 0:
        mae_improvement = (
            champion_mae - candidate_mae
        ) / champion_mae
    else:
        mae_improvement = 0.0

    # Quality conditions
    mae_pass = (
        mae_improvement >= MIN_MAE_IMPROVEMENT
    )

    rmse_pass = (
        candidate_metrics["RMSE"]
        <= champion_metrics["RMSE"] + MAX_RMSE_INCREASE
    )

    r2_pass = (
        candidate_metrics["R2"]
        >= champion_metrics["R2"] - MAX_R2_DROP
    )

    approved = mae_pass and rmse_pass and r2_pass

    # Promotion decision
    if approved:

        # Keep a backup of the current champion
        shutil.copy2(CHAMPION_PATH, BACKUP_PATH)

        # Copy candidate to a temporary file, then replace champion
        temporary_path = ROOT / "aqi_pipeline.tmp"

        shutil.copy2(CANDIDATE_PATH, temporary_path)

        os.replace(temporary_path, CHAMPION_PATH)

        decision = "PROMOTED"

        reason = (
            "Candidate passed all quality gate conditions. "
            "It has been promoted to champion."
        )

    else:

        decision = "RETAINED_CHAMPION"

        failed_conditions = []

        if not mae_pass:
            failed_conditions.append("MAE improvement below 2%")

        if not rmse_pass:
            failed_conditions.append("RMSE increased")

        if not r2_pass:
            failed_conditions.append("R2 decreased beyond allowed limit")

        reason = (
            "Candidate failed quality gate: "
            + ", ".join(failed_conditions)
        )

    # Save decision record
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "decision": decision,

        "reason": reason,

        "holdout_records": int(len(holdout)),

        "champion_metrics": champion_metrics,

        "candidate_metrics": candidate_metrics,

        "mae_improvement_percentage": round(
            mae_improvement * 100, 2
        ),

        "quality_checks": {
            "mae_pass": bool(mae_pass),
            "rmse_pass": bool(rmse_pass),
            "r2_pass": bool(r2_pass)
        },

        "thresholds": {
            "minimum_mae_improvement": MIN_MAE_IMPROVEMENT,
            "maximum_rmse_increase": MAX_RMSE_INCREASE,
            "maximum_r2_drop": MAX_R2_DROP
        }
    }

    with open(DECISION_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print("Quality gate completed.")
    print("Decision:", decision)
    print("Reason:", reason)

    return report


if __name__ == "__main__":
    quality_gate()
