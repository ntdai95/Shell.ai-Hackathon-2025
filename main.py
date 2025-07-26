import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
import optuna
import gc
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# === CONFIG ===
N_FOLDS = 5
N_TRIALS_BASE_MODELS = 35
N_TRIALS_BLEND = 50
SEEDS = [42, 2024]
USE_ADVANCED_FEATURES = True

# === Memory Optimization ===
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int32)
        elif col_type != object:
            df[col] = df[col].astype(np.float32)

    return df

# === Load Data ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# === Feature Engineering ===
def feature_engineering(df):
    blend_cols = [f"Component{i}_fraction" for i in range(1, 6)]
    for prop in range(1, 11):
        weighted = np.zeros(len(df), dtype=np.float32)
        for comp in range(1, 6):
            weighted += df[blend_cols[comp - 1]] * df[f"Component{comp}_Property{prop}"]

        df[f"Weighted_Property{prop}"] = weighted

    for comp in range(1, 6):
        props = [f"Component{comp}_Property{p}" for p in range(1, 11)]
        df[f"Comp{comp}_Prop_Mean"] = df[props].mean(axis=1)
        df[f"Comp{comp}_Prop_Std"] = df[props].std(axis=1)

    if USE_ADVANCED_FEATURES:
        for prop in range(1, 11):
            props = [f"Component{comp}_Property{prop}" for comp in range(1, 6)]
            df[f"Prop{prop}_Mean_All"] = df[props].mean(axis=1)
            df[f"Prop{prop}_Std_All"] = df[props].std(axis=1)

        df["Total_Fraction"] = df[blend_cols].sum(axis=1)

        for i in range(1, 6):
            for j in range(i + 1, 6):
                df[f"Comp{i}_x_Comp{j}_fraction"] = df[f"Component{i}_fraction"] * df[f"Component{j}_fraction"]

        for prop in range(1, 11):
            vals = [df[f"Component{comp}_Property{prop}"] for comp in range(1, 6)]
            df[f"Prop{prop}_Max_Min_Ratio"] = np.max(vals, axis=0) / (np.min(vals, axis=0) + 1e-5)

    return df

# Apply feature engineering and memory reduction
train = feature_engineering(train)
test = feature_engineering(test)
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# === Prepare Data for Modeling ===
target_cols = [col for col in train.columns if "BlendProperty" in col]
feature_cols = [col for col in train.columns if col not in ["ID"] + target_cols]
X_full = train[feature_cols]

# === Safe log1p transform with Inf/NaN handling ===
y_full = train[target_cols].apply(np.log1p)

# Check for hidden -inf after log1p
print("Checking for inf and NaNs before CatBoost fitting...")
print("Inf counts:\n", np.isinf(y_full).sum())
print("NaN counts:\n", y_full.isnull().sum())

# Clean if needed
y_full.replace([np.inf, -np.inf], 0, inplace=True)
y_full = y_full.fillna(y_full.median())
X_test = test[feature_cols]
submission = pd.DataFrame({"ID": test["ID"]})
final_preds_all_seeds = []

# === Main Loop for Model Training and Prediction ===
for seed in tqdm(SEEDS, desc="🔄 Processing seeds"):
    preds_seed = []
    for col in tqdm(target_cols, desc=f"⚡ Processing targets for seed {seed}"):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_cb, oof_lgbm, oof_xgb, oof_y = [], [], [], []
        preds_cb = np.zeros(len(X_test), dtype=np.float32)
        preds_lgbm = np.zeros(len(X_test), dtype=np.float32)
        preds_xgb = np.zeros(len(X_test), dtype=np.float32)

        for fold, (train_idx, valid_idx) in enumerate(tqdm(kf.split(X_full), total=N_FOLDS, desc=f"🌿 Folds {col}"), 1):
            X_train, X_valid = X_full.iloc[train_idx], X_full.iloc[valid_idx]
            y_train, y_valid = y_full[col].iloc[train_idx], y_full[col].iloc[valid_idx]

            def cb_objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 1500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
                    'depth': trial.suggest_int('depth', 4, 9),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 20),
                    'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 10),
                    'random_strength': trial.suggest_uniform('random_strength', 0, 1),
                }

                model = CatBoostRegressor(**params, loss_function='MAE', verbose=0)
                model.fit(X_train, y_train, eval_set=Pool(X_valid, y_valid), early_stopping_rounds=50)
                return mean_absolute_percentage_error(y_valid, model.predict(X_valid))

            def lgbm_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 20.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 20.0),
                    'random_state': seed,
                    'verbosity': -1,
                    'metric': 'mape'
                }

                model = LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(50)])
                return mean_absolute_percentage_error(y_valid, model.predict(X_valid))

            def xgb_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
                    'max_depth': trial.suggest_int('max_depth', 4, 9),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 20.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 20.0),
                    'random_state': seed,
                    'eval_metric': 'mae'
                }

                model = XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                return mean_absolute_percentage_error(y_valid, model.predict(X_valid))

            # train best CatBoost for ensembling comparison
            study_cb = optuna.create_study(direction="minimize")
            for _ in tqdm(range(N_TRIALS_BASE_MODELS), desc="🔧 Optuna CatBoost"):
                study_cb.optimize(cb_objective, n_trials=1, show_progress_bar=False)

            model_cb = CatBoostRegressor(**study_cb.best_params, loss_function='MAE', verbose=0)
            model_cb.fit(X_train, y_train, eval_set=Pool(X_valid, y_valid), early_stopping_rounds=50)
            oof_cb.append(model_cb.predict(X_valid))
            preds_cb += model_cb.predict(X_test) / N_FOLDS

            # train best LGBM for ensembling comparison
            study_lgbm = optuna.create_study(direction="minimize")
            for _ in tqdm(range(N_TRIALS_BASE_MODELS), desc="🔧 Optuna LGBM"):
                study_lgbm.optimize(lgbm_objective, n_trials=1, show_progress_bar=False)

            model_lgbm = LGBMRegressor(**study_lgbm.best_params, random_state=seed, metric='mape', verbosity=-1)
            model_lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(50)])
            oof_lgbm.append(model_lgbm.predict(X_valid))
            preds_lgbm += model_lgbm.predict(X_test) / N_FOLDS

            # train best XGB for ensembling comparison
            study_xgb = optuna.create_study(direction="minimize")
            for _ in tqdm(range(N_TRIALS_BASE_MODELS), desc="🔧 Optuna XGB"):
                study_xgb.optimize(xgb_objective, n_trials=1, show_progress_bar=False)

            model_xgb = XGBRegressor(**study_xgb.best_params)
            model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            oof_xgb.append(model_xgb.predict(X_valid))
            preds_xgb += model_xgb.predict(X_test) / N_FOLDS

            oof_y.append(y_valid.values)
            gc.collect()

        oof_cb = np.concatenate(oof_cb)
        oof_lgbm = np.concatenate(oof_lgbm)
        oof_xgb = np.concatenate(oof_xgb)
        oof_y = np.concatenate(oof_y)

        def blend_objective(trial):
            w_cb = trial.suggest_uniform('w_cb', 0, 1)
            w_lgbm = trial.suggest_uniform('w_lgbm', 0, 1 - w_cb)
            w_xgb = 1 - w_cb - w_lgbm
            alpha = trial.suggest_loguniform('alpha', 1e-3, 10)
            X_stack = np.vstack([w_cb * oof_cb, w_lgbm * oof_lgbm, w_xgb * oof_xgb]).T
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_stack, oof_y)
            preds = ridge.predict(X_stack)
            return mean_absolute_percentage_error(oof_y, preds)
        
        # Optimize blending weights using Optuna
        study_blend = optuna.create_study(direction="minimize")
        for _ in tqdm(range(N_TRIALS_BLEND), desc="⚙️ Optuna Ridge Blend"):
            study_blend.optimize(blend_objective, n_trials=1, show_progress_bar=False)

        best_params = study_blend.best_params
        w_cb = best_params['w_cb']
        w_lgbm = best_params['w_lgbm']
        w_xgb = 1 - w_cb - w_lgbm
        alpha = best_params['alpha']

        X_stack_test = np.vstack([w_cb * preds_cb, w_lgbm * preds_lgbm, w_xgb * preds_xgb]).T
        X_stack_train = np.vstack([w_cb * oof_cb, w_lgbm * oof_lgbm, w_xgb * oof_xgb]).T

        ridge_final = Ridge(alpha=alpha)
        ridge_final.fit(X_stack_train, oof_y)
        preds_final = ridge_final.predict(X_stack_test)
        preds_seed.append(np.expm1(preds_final))

    # Stack predictions for the current seed
    preds_seed_stacked = np.vstack(preds_seed).T
    final_preds_all_seeds.append(preds_seed_stacked)
    gc.collect()

# === Final Predictions and Submission ===
final_preds = np.mean(final_preds_all_seeds, axis=0)
submission[target_cols] = final_preds
submission.to_csv("submission_blend_xgb_lgbm_cb_mape.csv", index=False)
print("\\n✅ Submission saved: submission_blend_xgb_lgbm_cb_mape.csv")