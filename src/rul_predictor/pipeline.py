import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from rul_predictor.models.train_final import build_and_save_final_artifact

def run_training_pipeline(data_path: str = "data/processed/train.csv"):

    print("----------------------")
    df = pd.read_csv(data_path)

    FEATURE_COLUMNS = [col for col in df.columns if col not in ["unit_id", "time_cycle", "RUL"]]
    CONSTANT_COLUMNS = []
    ORIGINAL_COLUMNS = list(df.columns)

    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=15, 
            min_samples_leaf=3, 
            random_state=42
        ),
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        random_state=42,
    )

    print("-------------------------------------------------------")
    artifact = build_and_save_final_artifact(
        model=bagging_model,
        X_full=df,
        y_full=df["RUL"],
        feature_columns=FEATURE_COLUMNS,
        constant_columns=CONSTANT_COLUMNS,
        original_columns=ORIGINAL_COLUMNS,
        rolling_window=5,
        model_name="bagging_dt",
        version="1.0.0",
        output_path="models/rul_model.joblib",  
    )
    
    return artifact

if __name__ == "__main__":
    run_training_pipeline()