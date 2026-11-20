
import pandas as pd
import json
from pathlib import Path

# File paths
DATA_PATH = Path("data/aqi_dataset_top_3_indian_cities_50.csv")
REPORT_PATH = Path("monitoring/data_validation_report.json")

# Expected columns from our AQI dataset
REQUIRED_COLUMNS = [
    "City", "Date",
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene",
    "AQI", "AQI_Bucket"
]


def validate_data():

    print("\n===== AQI NEW DATA VALIDATION =====")

    # 1. Load dataset
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 2. Dataset size
    print("\n1. DATASET SIZE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # 3. Check required columns
    print("\n2. REQUIRED COLUMN CHECK")
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("All required columns are present.")

    # 4. Missing values
    print("\n3. MISSING VALUES")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 5. Duplicate records
    print("\n4. DUPLICATE RECORDS")
    duplicate_rows = int(df.duplicated().sum())
    print("Duplicate rows:", duplicate_rows)

    # Check duplicate city-date combinations
    duplicate_city_dates = 0
    if {"City", "Date"}.issubset(df.columns):
        duplicate_city_dates = int(
            df.duplicated(subset=["City", "Date"]).sum()
        )

    print("Duplicate City-Date records:", duplicate_city_dates)

    # 6. Data types
    print("\n5. DATA TYPES")
    print(df.dtypes)

    # 7. AQI target distribution
    print("\n6. AQI TARGET DISTRIBUTION")

    if "AQI" in df.columns:
        print("\nAQI numerical statistics:")
        print(df["AQI"].describe())

    if "AQI_Bucket" in df.columns:
        print("\nAQI category distribution:")
        print(df["AQI_Bucket"].value_counts(dropna=False))

    # 8. Basic quality checks
    quality_checks = {
        "file_exists": True,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "required_columns_present": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_city_date_records": duplicate_city_dates,
        "aqi_missing": (
            int(df["AQI"].isnull().sum())
            if "AQI" in df.columns else None
        ),
        "aqi_distribution": (
            {
                str(k): int(v)
                for k, v in df["AQI_Bucket"].value_counts(
                    dropna=False
                ).items()
            }
            if "AQI_Bucket" in df.columns else {}
        )
    }

    # 9. Save validation report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as file:
        json.dump(quality_checks, file, indent=4)

    print("\n7. VALIDATION REPORT")
    print("Report saved to:", REPORT_PATH)

    print("\n===== VALIDATION COMPLETED =====")


if __name__ == "__main__":
    validate_data()