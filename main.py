"""
Main Execution Script for NASA C-MAPSS RUL Predictor Pipeline.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Standard 26 column names for C-MAPSS dataset
COLUMN_NAMES = [
    "unit_id", "cycle", "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

# Imports matched with repository structure
from rul_predictor.data.loader import load_fd001
from rul_predictor.data.validation import validate_dataset
from rul_predictor.data.spilt import split_by_unit, validate_no_unit_overlap, prepare_xy
from rul_predictor.targets.rul import calculate_rul, cap_rul  # <-- Import directly from targets package
from rul_predictor.preprocessing.cleaner import clean_dataset
from rul_predictor.features.engineering import build_features, get_feature_columns
from rul_predictor.models.factory import create_model
from rul_predictor.models.evaluate import evaluate_reg
from rul_predictor.models.train_final import build_and_save_final_artifact


def main(
    data_path: str = "data/train_FD001.txt",
    model_name: str = "random_forest",
    rolling_window: int = 5,
    rul_cap: int = 125,
    output_path: str = "artifacts/rul_model.joblib",
) -> None:
    print("=" * 60)
    print("🚀 STARTING NASA C-MAPSS RUL PREDICTION PIPELINE")
    print("=" * 60)

    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"[❌] Dataset file not found at: {file_path.resolve()}"
        )

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from: {data_path}")
    raw_df = load_fd001(file_path)
    print(f"      Loaded dataset shape: {raw_df.shape}")

    # 2. Dataset Validation
    print("\n[2/7] Validating schema and cycles...")
    validate_dataset(raw_df)

    # 3. Target Formulation (RUL & Capping)
    print(f"\n[3/7] Calculating RUL & Capping at {rul_cap}...")
    df_rul = calculate_rul(raw_df)
    df_capped = cap_rul(df_rul, cap=rul_cap)

    # 4. Data Cleaning
    print("\n[4/7] Cleaning dataset...")
    cleaned_df, constant_cols = clean_dataset(df_capped)

    # 5. Feature Engineering
    print(f"\n[5/7] Engineering features (Window = {rolling_window})...")
    sensor_cols = [c for c in cleaned_df.columns if c.startswith("sensor_")]
    featured_df = build_features(
        cleaned_df,
        sensor_columns=sensor_cols,
        rolling_window=rolling_window,
    )
    feature_cols = get_feature_columns(featured_df)

    # 6. Model Training & Validation
    print(f"\n[6/7] Splitting data and evaluating {model_name}...")
    train_df, val_df = split_by_unit(featured_df, validation_size=0.2, random_state=42)
    validate_no_unit_overlap(train_df, val_df)

    X_train, y_train, X_val, y_val = prepare_xy(train_df, val_df, feature_cols)

    model = create_model(model_name, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_reg(model, X_val, y_val)
    print(f"\n📊 Validation Metrics ({model_name}): {metrics}")

    # 7. Final Model Training & Artifact Generation
    print("\n[7/7] Fitting final model and saving ModelArtifact...")
    X_full = featured_df[feature_cols]
    y_full = featured_df["RUL"]

    final_model = create_model(model_name, n_estimators=100, random_state=42)
    build_and_save_final_artifact(
        model=final_model,
        X_full=X_full,
        y_full=y_full,
        feature_columns=feature_cols,
        constant_columns=constant_cols,
        original_columns=COLUMN_NAMES,
        rolling_window=rolling_window,
        rul_cap=rul_cap,
        model_name=model_name,
        output_path=output_path,
    )

    print("\n" + "=" * 60)
    print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()