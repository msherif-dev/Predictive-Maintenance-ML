from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_model(model_name: str, **kwargs):
    models = {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "adaboost": AdaBoostRegressor,
        "bagging": BaggingRegressor,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    return models[model_name](**kwargs)