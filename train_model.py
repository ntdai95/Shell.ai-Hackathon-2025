import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.engineer import engineer_features


def train():
    train_df = pd.read_csv('data/train.csv')
    target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
    X = engineer_features(train_df.drop(columns=target_cols))
    y = train_df[target_cols]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)),
        'CatBoost': MultiOutputRegressor(CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0))
    }

    results = {}
    for name, model in models.items():
        oof_scaled = np.zeros_like(y_scaled)
        for t_idx, v_idx in kf.split(X):
            model.fit(X.iloc[t_idx], y_scaled[t_idx])
            oof_scaled[v_idx] = model.predict(X.iloc[v_idx])
        
        mape = mean_absolute_percentage_error(y, scaler_y.inverse_transform(oof_scaled))
        print(f"{name} MAPE: {mape:.2f}")
        results[name] = mape

    best_name = min(results, key=results.get)
    print(f"Best model: {best_name}")

    os.makedirs('models', exist_ok=True)
    final_model = models[best_name].fit(X, y_scaled)    
    joblib.dump(final_model, "models/model.pkl")
    joblib.dump(scaler_y, "models/scaler.pkl")
    joblib.dump(X.median(), "models/medians.pkl")


if __name__ == "__main__":
    train()