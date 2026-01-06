import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def engineer_features(df):
    df = df.copy()
    fractions = [f'Component{i}_fraction' for i in range(1, 6)]
    
    # Interaction features: Physical Mixing Rules
    for j in range(1, 11): # For each Property
        weighted_cols = []
        for i in range(1, 6): # For each Component
            prop_col = f'Component{i}_Property{j}'
            frac_col = f'Component{i}_fraction'

            # Weighted Property interaction
            w_name = f'W_P{j}_C{i}'
            df[w_name] = df[prop_col] * df[frac_col]
            weighted_cols.append(w_name)
        
        # Aggregate features for the blend
        df[f'Ideal_Blend_P{j}'] = df[weighted_cols].sum(axis=1)
        df[f'Blend_Range_P{j}'] = df[weighted_cols].max(axis=1) - df[weighted_cols].min(axis=1)

    # Statistical features of the fractions
    df['Entropy'] = -(df[fractions] * np.log(df[fractions] + 1e-9)).sum(axis=1)
    df['Max_Frac'] = df[fractions].max(axis=1)
    df['Is_Pure'] = (df['Max_Frac'] > 0.98).astype(int)
    return df


def run_pipeline():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
    X = train.drop(columns=target_cols)
    y = train[target_cols]
    X_test = test.drop(columns=['ID'])
    test_ids = test['ID']

    X = engineer_features(X)
    X_test = engineer_features(X_test)

    # CLEANING: Replace any remaining inf/nan in features (industry standard)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X.median(), inplace=True)

    # TARGET STRATEGY: Use StandardScaler instead of Log to avoid NaN/Inf errors
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6, 
            loss_function='RMSE', random_seed=42, verbose=0
        )
    }

    results = {}
    for name, model in models.items():
        oof_scaled = np.zeros_like(y_scaled)
        test_preds_scaled = np.zeros((len(X_test), len(target_cols)))
        for i in range(len(target_cols)):
            for _, (t_idx, v_idx) in enumerate(kf.split(X)):
                xt, xv = X.iloc[t_idx], X.iloc[v_idx]
                yt, _ = y_scaled[t_idx, i], y_scaled[v_idx, i]
                
                model.fit(xt, yt)
                oof_scaled[v_idx, i] = model.predict(xv)
                test_preds_scaled[:, i] += model.predict(X_test) / 5
        
        # Inverse transform to calculate real MAPE
        oof_final = scaler_y.inverse_transform(oof_scaled)
        mape = mean_absolute_percentage_error(y, oof_final)
        print(f"{name} MAPE: {mape:.6f}")
        results[name] = {'mape': mape, 'preds': scaler_y.inverse_transform(test_preds_scaled)}

    best_name = min(results, key=lambda k: results[k]['mape'])
    print(f"Best Model: {best_name}")
    
    submission = pd.DataFrame(results[best_name]['preds'], columns=target_cols)
    submission.insert(0, 'ID', test_ids)
    submission.to_csv('submission_final.csv', index=False)


if __name__ == "__main__":
    run_pipeline()