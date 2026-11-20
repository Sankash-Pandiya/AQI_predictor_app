
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "aqi_pipeline.pkl"
OLD_DATA_PATH = ROOT / "data" / "city_day.csv"
NEW_DATA_PATH = ROOT / "data" / "new_aqi_data.csv"

REPORT_PATH = ROOT / "monitoring" / "evaluation_report.json"
PREDICTIONS_PATH = ROOT / "monitoring" / "evaluation_predictions.csv"

# --------------------------------------------------
# FEATURE CONFIGURATION
# --------------------------------------------------

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene"
]

AQI_FEATURES = [
    "AQI",
    "aqi_lag1",
    "aqi_lag2",
    "aqi_lag3",
    "aqi_delta1",
    "aqi_roll7_mean",
    "aqi_roll14_mean",
    "aqi_roll30_mean",
    "aqi_roll7_std"
]

CALENDAR_FEATURES = [
    "month_sin",
    "month_cos",
    "dayofweek",
    "is_weekend",
    "quarter"
]

FEATURE_COLUMNS = (
    POLLUTANTS
    + AQI_FEATURES
    + CALENDAR_FEATURES
    + ["City"]
)

# --------------------------------------------------
# LOAD AND COMBINE DATA
# --------------------------------------------------

def load_data():

    old_df = pd.read_csv(OLD_DATA_PATH)
    new_df = pd.read_csv(NEW_DATA_PATH)

    old_df["Date"] = pd.to_datetime(old_df["Date"])
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    # Mark incoming records
    new_df["_is_new"] = True
    old_df["_is_new"] = False

    # Incoming records take priority if City-Date overlaps
    combined = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["City", "Date"],
        keep="last"
    )

    combined = combined.sort_values(
        ["City", "Date"]
    ).reset_index(drop=True)

    return combined


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_data(df):

    df = df.copy()

    # Remove unused columns from original dataset
    df = df.drop(
        columns=["Xylene", "AQI_Bucket"],
        errors="ignore"
    )

    # Correct known Ahmedabad data-entry issue
    if {"City", "CO", "NO"}.issubset(df.columns):
        mask = (
            (df["City"] == "Ahmedabad")
            & (df["CO"] == df["NO"])
        )
        df.loc[mask, "CO"] = np.nan

    # Clip AQI to valid range
    df["AQI"] = df["AQI"].clip(0, 500)

    # Forward-fill pollutant and AQI values city-wise
    columns_to_fill = [
        col for col in POLLUTANTS + ["AQI"]
        if col in df.columns
    ]

    df[columns_to_fill] = (
        df.groupby("City")[columns_to_fill]
        .ffill(limit=3)
    )

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_features(df):

    df = df.copy()

    grouped_aqi = df.groupby("City")["AQI"]

    # Previous AQI values
    df["aqi_lag1"] = grouped_aqi.shift(1)
    df["aqi_lag2"] = grouped_aqi.shift(2)
    df["aqi_lag3"] = grouped_aqi.shift(3)

    # AQI change
    df["aqi_delta1"] = (
        df["aqi_lag1"] - df["aqi_lag2"]
    )

    # Rolling statistics
    df["aqi_roll7_mean"] = grouped_aqi.transform(
        lambda x: x.rolling(7, min_periods=3).mean()
    )

    df["aqi_roll14_mean"] = grouped_aqi.transform(
        lambda x: x.rolling(14, min_periods=5).mean()
    )

    df["aqi_roll30_mean"] = grouped_aqi.transform(
        lambda x: x.rolling(30, min_periods=7).mean()
    )

    df["aqi_roll7_std"] = grouped_aqi.transform(
        lambda x: x.rolling(7, min_periods=3).std()
    )

    # Calendar features
    df["month_sin"] = np.sin(
        2 * np.pi * df["Date"].dt.month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["Date"].dt.month / 12
    )

    df["dayofweek"] = df["Date"].dt.dayofweek

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    df["quarter"] = df["Date"].dt.quarter

    # Next available AQI is the target
    df["target_next_day_aqi"] = (
        df.groupby("City")["AQI"].shift(-1)
    )

    df["target_date"] = (
        df.groupby("City")["Date"].shift(-1)
    )

    return df


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model():

    print("Loading data...")

    df = load_data()
    df = clean_data(df)
    df = create_features(df)

    # Identify target dates belonging to incoming batch
    new_data = pd.read_csv(NEW_DATA_PATH)
    new_data["Date"] = pd.to_datetime(new_data["Date"])

    new_keys = set(
        zip(new_data["City"], new_data["Date"])
    )

    # Keep only rows whose next-day target is in new data
    evaluation_df = df[
        df.apply(
            lambda row: (
                row["City"], row["target_date"]
            ) in new_keys
            if pd.notna(row["target_date"])
            else False,
            axis=1
        )
    ].copy()

    # Confirm that target is exactly the following calendar day
    evaluation_df = evaluation_df[
        evaluation_df["target_date"]
        == evaluation_df["Date"] + pd.Timedelta(days=1)
    ]

    # Remove rows without a labelled target
    evaluation_df = evaluation_df.dropna(
        subset=["target_next_day_aqi"]
    )

    if evaluation_df.empty:
        raise ValueError(
            "No valid next-day labelled rows found in the incoming batch."
        )

    # Select the exact model input columns
    X = evaluation_df[FEATURE_COLUMNS]
    y_true = evaluation_df["target_next_day_aqi"]

    print("Loading champion model...")

    # If the notebook saved Winsorizer as __main__.Winsorizer,
    # make its class available before loading the pickle.
    try:
        from app.transformers import Winsorizer
        import __main__
        __main__.Winsorizer = Winsorizer
    except ImportError:
        pass

    champion = joblib.load(MODEL_PATH)

    print("Generating predictions...")

    y_pred = champion.predict(X)

    # Keep predictions within AQI range
    y_pred = np.clip(y_pred, 0, 500)

    # Calculate regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Save predictions
    predictions = pd.DataFrame({
        "City": evaluation_df["City"].values,
        "Date": evaluation_df["target_date"].values,
        "Actual_AQI": y_true.values,
        "Predicted_AQI": y_pred,
        "Absolute_Error": np.abs(y_true.values - y_pred)
    })

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # Save evaluation report
    report = {
        "model": "champion",
        "evaluation_type": "new_labelled_data",
        "samples_evaluated": int(len(y_true)),
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        }
    }

    with open(REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print("Evaluation completed.")
    print("Report:", REPORT_PATH)
    print("Predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    evaluate_model()