import pandas as pd 

# Calculate RUL 
def calculate_rul(df : pd.DataFrame) -> pd.DataFrame :

    result = df.copy()

    max_cycle = (
        result
        .groupby("unit_id")["cycle"]
        .transform("max")
    )

    result["RUL"] = max_cycle - result["cycle"]

    return result

def cap_rul(
    df: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:

    result = df.copy()

    result["RUL"] = result["RUL"].clip(upper=cap)

    return result