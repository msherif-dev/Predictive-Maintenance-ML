from typing import Any
import pandas as pd
from rul_predictor.models.artifact import ModelArtifact, save_artifact


def fit_final_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:

    model.fit(X, y)
    return model


def build_and_save_final_artifact(
    model: Any,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    feature_columns: list[str],
    constant_columns: list[str],
    original_columns: list[str],
    rolling_window: int = 5,
    rul_cap: int = 125,
    model_name: str = "bagging_dt",
    version: str = "1.0.0",
    output_path: str = "artifacts/rul_model.joblib",
) -> ModelArtifact:

    fitted_model = fit_final_model(model, X_full[feature_columns], y_full)

    metadata = {
        "project": "NASA C-MAPSS RUL Prediction",
        "task": "Remaining Useful Life regression",
        "model_name": model_name,
        "version": version,
        "target": "RUL",
        "rul_cap": rul_cap,
    }

    artifact = ModelArtifact(
        model=fitted_model,
        feature_columns=feature_columns,
        constant_columns=constant_columns,
        original_columns=original_columns,
        rolling_window=rolling_window,
        metadata=metadata,
    )


    save_artifact(artifact, output_path)
    print(f"save in : {output_path}")

    return artifact