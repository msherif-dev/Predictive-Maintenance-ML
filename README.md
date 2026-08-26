# 🛠️ Predictive Maintenance ML Engine

An end-to-end Machine Learning solution designed to predict the **Remaining Useful Life (RUL)** of turbofan engines using time-series sensor degradation data from the **NASA C-MAPSS dataset**. 

## 📌 Features
* **Dataset**: NASA C-MAPSS (`train_FD001`).
* **Feature Engineering**: Time-series rolling statistics and linear trend slopes.
* **Model**: XGBoost Regressor validated with `GroupKFold` on engine unit IDs.
