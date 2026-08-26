import pandas as pd


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["cycle_squared"] = result["cycle"] ** 2

    return result


def add_rolling_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window: int,
) -> pd.DataFrame:
    """Add rolling mean and rolling std features for specified sensors."""

    result = df.copy()

    result = result.sort_values(["unit_id", "cycle"])

    grouped = result.groupby("unit_id", group_keys=False)

    for sensor in sensor_columns:
        # 1. Rolling Mean
        result[f"{sensor}_rolling_mean_{window}"] = grouped[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        # 2. Rolling Std 
        result[f"{sensor}_rolling_std_{window}"] = (
            grouped[sensor]
            .transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            .fillna(0.0)
        )

    return result     

# Pipeline 
def build_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    rolling_window: int,
) -> pd.DataFrame:

    result = df.copy()

    result = add_cycle_features(result)

    result = add_rolling_features(
        result,
        sensor_columns=sensor_columns,
        window=rolling_window,
    )

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:

    excluded = {
        "unit_id",
        "RUL",
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]