from pathlib import Path
import pandas as pd 

COLUMN_NAMES = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

def load_fd001 (path : str | Path ) -> pd.DataFrame :

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path ,
                     sep=r"\s+",
                     header=None)

    if df.shape[1] != 26 :
        raise ValueError(f"Expected 26 columns, got {df.shape[1]}")

    df.columns = COLUMN_NAMES

    return df 