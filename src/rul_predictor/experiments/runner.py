from dataclasses import dataclass

from rul_predictor.models.train import train_model
from rul_predictor.models.evaluate import evaluate_reg

@dataclass
class ExperimentResult:
    model_name: str
    metrics: dict
    model: object


def run_experiment(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    metrics = evaluate_reg(
        trained_model,
        X_val,
        y_val,
    )

    return ExperimentResult(
        model_name=model_name,
        metrics=metrics,
        model=trained_model,
    )


