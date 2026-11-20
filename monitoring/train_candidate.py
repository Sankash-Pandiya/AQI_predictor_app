
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone

# Reuse the same data preparation and feature engineering
# functions from Task B.
from monitoring.evaluate_model import (
    load_data,
    clean_data,
    create_features,
    FEATURE_COLUMNS
)

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "aqi_pipeline.pkl"
CANDIDATE_PATH = ROOT / "candidate_pipeline.pkl"

NEW_DATA_PATH = ROOT / "data" / "new_aqi_data.csv"

HOLDOUT_PATH = ROOT / "monitoring" / "quality_gate_holdout.csv"
REPORT_PATH = ROOT / "monitoring" / "candidate_training_report.json"

TRAIN_FRACTION = 0.80
HISTORICAL_CUTOFF = pd.Timestamp("2019-12-31")


# --------------------------------------------------
# LOAD CHAMPION MODEL
# --------------------------------------------------

def load_champion():

    # Make the custom transformer available to joblib
    try:
        from app.transformers import Winsorizer
        import __main__
        __main__.Winsorizer = Winsorizer
    except ImportError:
        pass

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# CREATE CHRONOLOGICAL SPLIT
# --------------------------------------------------

def get_city_cutoffs(new_data):

    cutoffs = {}

    for city, group in new_data.groupby("City"):

        group = group.sort_values("Date")
        n = len(group)

        # Need at least two records for train and holdout
        if n < 2:
            continue

        split_index = max(
            1,
            min(n - 1, int(np.floor(n * TRAIN_FRACTION)))
        )

        cutoffs[city] = group["Date"].iloc[split_index - 1]

    return cutoffs


# --------------------------------------------------
# PREPARE TRAINING AND HOLDOUT DATA
# --------------------------------------------------

def prepare_training_data():

    # Load combined historical and incoming data
    df = load_data()

    # Apply the original cleaning and feature engineering
    df = clean_data(df)
    df = create_features(df)

    new_data = pd.read_csv(NEW_DATA_PATH)
    new_data["Date"] = pd.to_datetime(new_data["Date"])

    cutoffs = get_city_cutoffs(new_data)

    if not cutoffs:
        raise ValueError(
            "Not enough incoming records to create a train/holdout split."
        )

    # Valid incoming City-Date keys
    new_keys = set(
        zip(new_data["City"], new_data["Date"])
    )

    # Keep only genuine next-calendar-day targets
    valid_next_day = (
        df["target_date"] == df["Date"] + pd.Timedelta(days=1)
    )

    # --------------------------------------------------
    # HISTORICAL TRAINING DATA
    # --------------------------------------------------

    historical_train = df[
        (~df["_is_new"])
        & (df["Date"] <= HISTORICAL_CUTOFF)
        & (df["target_date"] <= HISTORICAL_CUTOFF)
        & valid_next_day
    ].copy()

    # --------------------------------------------------
    # INCOMING TRAINING DATA
    # --------------------------------------------------

    new_train_mask = df.apply(
        lambda row: (
            row["_is_new"]
            and row["City"] in cutoffs
            and row["Date"] <= cutoffs[row["City"]]
            and row["target_date"] <= cutoffs[row["City"]]
            and (row["City"], row["target_date"]) in new_keys
        ),
        axis=1
    )

    new_train = df[new_train_mask & valid_next_day].copy()

    # --------------------------------------------------
    # INCOMING HOLDOUT DATA
    # --------------------------------------------------

    holdout_mask = df.apply(
        lambda row: (
            row["_is_new"]
            and row["City"] in cutoffs
            and row["Date"] > cutoffs[row["City"]]
            and (row["City"], row["target_date"]) in new_keys
        ),
        axis=1
    )

    holdout = df[holdout_mask & valid_next_day].copy()

    # Combine historical and incoming training rows
    train_data = pd.concat(
        [historical_train, new_train],
        ignore_index=True
    )

    # Remove rows without a valid target
    train_data = train_data.dropna(
        subset=["target_next_day_aqi"]
    )

    holdout = holdout.dropna(
        subset=["target_next_day_aqi"]
    )

    if train_data.empty:
        raise ValueError("No valid training records were generated.")

    if holdout.empty:
        raise ValueError("No valid holdout records were generated.")

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["target_next_day_aqi"]

    X_holdout = holdout[FEATURE_COLUMNS]
    y_holdout = holdout["target_next_day_aqi"]

    return (
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        holdout,
        len(historical_train),
        len(new_train)
    )


# --------------------------------------------------
# TRAIN CANDIDATE MODEL
# --------------------------------------------------

def train_candidate():

    print("Preparing training data...")

    (
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        holdout,
        historical_count,
        new_count
    ) = prepare_training_data()

    print("Loading champion pipeline...")

    champion = load_champion()

    # Clone the complete pipeline:
    # same preprocessing and same model configuration,
    # but a fresh, unfitted candidate.
    candidate = clone(champion)

    print("Training candidate model...")

    candidate.fit(X_train, y_train)

    # --------------------------------------------------
    # SAVE CANDIDATE
    # --------------------------------------------------

    joblib.dump(
        candidate,
        CANDIDATE_PATH,
        compress=3
    )

    # --------------------------------------------------
    # SAVE HOLDOUT DATA FOR TASK D
    # --------------------------------------------------

    holdout_output = X_holdout.copy()

    holdout_output["Actual_AQI"] = y_holdout.values
    holdout_output["City"] = holdout["City"].values
    holdout_output["Date"] = holdout["target_date"].values

    holdout_output.to_csv(
        HOLDOUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # SAVE TRAINING REPORT
    # --------------------------------------------------

    report = {
        "model_type": "candidate",
        "training_method": "champion_pipeline_clone",
        "historical_training_rows": int(historical_count),
        "new_batch_training_rows": int(new_count),
        "total_training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "training_features": FEATURE_COLUMNS,
        "candidate_model_path": str(CANDIDATE_PATH)
    }

    with open(REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print("Candidate training completed.")
    print("Candidate saved:", CANDIDATE_PATH)
    print("Holdout saved:", HOLDOUT_PATH)
    print("Training report saved:", REPORT_PATH)

    # --------------------------------------------------
    # OPTIONAL MLFLOW TRACKING
    # --------------------------------------------------

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("AQI_MLOps")

        with mlflow.start_run(run_name="AQI_Candidate_Training"):

            mlflow.log_param("model_type", "candidate")
            mlflow.log_param("training_method", "pipeline_clone")
            mlflow.log_param("train_fraction", TRAIN_FRACTION)

            mlflow.log_metric(
                "historical_training_rows",
                historical_count
            )

            mlflow.log_metric(
                "new_batch_training_rows",
                new_count
            )

            mlflow.log_metric(
                "total_training_rows",
                len(X_train)
            )

            mlflow.sklearn.log_model(
                candidate,
                artifact_path="candidate_model"
            )

            mlflow.log_artifact(str(REPORT_PATH))
            mlflow.log_artifact(str(HOLDOUT_PATH))

        print("MLflow tracking completed.")

    except ImportError:
        print("MLflow is not installed. Local tracking completed.")

    except Exception as error:
        print("MLflow logging skipped:", error)


if __name__ == "__main__":
    train_candidate()
