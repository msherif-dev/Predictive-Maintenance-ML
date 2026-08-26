from sklearn.metrics import(mean_absolute_error ,
                            mean_squared_error,
                            r2_score)

import numpy as np 

def evaluate_reg(model , X , y) :

    pred = model.predict(X)

    mae = mean_absolute_error(
        y,
        pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred,
        )
    )

    r2 = r2_score(
        y,
        pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }