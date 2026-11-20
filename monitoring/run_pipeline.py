
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_step(description, module):

    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at: {description}"
        )

    print(f"{description} completed successfully.")


def main():

    print("STARTING AQI MLOPS PIPELINE")

    # Step 1: Validate incoming data
    run_step(
        "STEP 1: DATA VALIDATION",
        "monitoring.prepare_data"
    )

    # Step 2: Evaluate existing champion
    run_step(
        "STEP 2: CHAMPION MODEL EVALUATION",
        "monitoring.evaluate_model"
    )

    # Step 3: Train candidate model
    run_step(
        "STEP 3: CANDIDATE MODEL TRAINING",
        "monitoring.train_candidate"
    )

    # Step 4: Compare models and promote
    run_step(
        "STEP 4: MODEL QUALITY GATE",
        "monitoring.quality_gate"
    )

    print("\n" + "=" * 60)
    print("AQI MLOPS PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
