import pandas as pd 


# Find constant_coulmns 
def find_constant_columns(df: pd.DataFrame) -> list[str]:

    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

# Remove Cnstant coulmns 
def  remove_constant_columns(
        df : pd.DataFrame , 
        constant_columns : list[str] | None = None,
):
    if constant_columns is None :
        constant_columns = find_constant_columns(df)

    cleaned = df.drop(
        columns=constant_columns,
        errors="ignore",
    ).copy()

    return cleaned, constant_columns

# remove dupliates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


# Check missing value 
def validate_no_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Missing values detected:\n{missing}"
        )


# run cleaner 
def clean_dataset(df : pd.DataFrame , 
                  constant_columns: list[str] | None = None,
):
    cleaned = df.copy()

    validate_no_missing_values(cleaned)

    cleaned = remove_duplicates(cleaned)

    cleaned, constant_columns = remove_constant_columns(
        cleaned,
        constant_columns,
    )

    return cleaned, constant_columns

