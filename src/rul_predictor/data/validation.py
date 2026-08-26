import pandas as pd 

REQUIRED_COLUMNS = {
    "unit_id",
    "cycle",
}

# Check schema
def validate_schema(df: pd.DataFrame) -> None:
    """Validate the basic dataset schema."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


# Check Cycles 
def validate_cycles(df: pd.DataFrame) -> None:
    if df["cycle"].isna().any():
        raise ValueError("cycle contains missing values")

    if (df["cycle"] <= 0).any():
        raise ValueError("cycle must be positive")


# Check unit ids 
def validate_unit_ids(df: pd.DataFrame) -> None:

    if df["unit_id"].isna().any():
        raise ValueError("unit_id contains missing values")

    if (df["unit_id"] <= 0).any():
        raise ValueError("unit_id must be positive")


# Check all 
def validate_dataset(df: pd.DataFrame) -> None:
    
    validate_schema(df)
    validate_unit_ids(df)
    validate_cycles(df)