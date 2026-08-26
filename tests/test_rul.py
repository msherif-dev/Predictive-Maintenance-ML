import pandas as pd

from rul_predictor.targets.rul import calculate_rul


def test_rul_calculation():

    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })

    result = calculate_rul(df)

    assert result["RUL"].tolist() == [
        2, 1, 0,
        1, 0,
    ]