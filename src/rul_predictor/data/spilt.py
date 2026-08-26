import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


def split_by_unit(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["unit_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    return train_df, val_df


def validate_no_unit_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:

    train_units = set(train_df["unit_id"])
    val_units = set(val_df["unit_id"])

    overlap = train_units.intersection(val_units)

    if overlap:
        raise ValueError(
            f"Unit leakage detected: {sorted(overlap)}"
        )

def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]

    X_val = val_df[feature_columns]
    y_val = val_df["RUL"]

    return X_train, y_train, X_val, y_val