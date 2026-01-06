import numpy as np


def engineer_features(df):
    df = df.copy()
    fractions = [f'Component{i}_fraction' for i in range(1, 6)]
    for j in range(1, 11): 
        weighted_cols = []
        for i in range(1, 6):
            w_name = f'W_P{j}_C{i}'
            df[w_name] = df[f'Component{i}_Property{j}'] * df[f'Component{i}_fraction']
            weighted_cols.append(w_name)
        
        df[f'Ideal_P{j}'] = df[weighted_cols].sum(axis=1)
        df[f'Spread_P{j}'] = df[weighted_cols].max(axis=1) - df[weighted_cols].min(axis=1)

    df['Entropy'] = -(df[fractions] * np.log(df[fractions] + 1e-9)).sum(axis=1)
    df['Max_Frac'] = df[fractions].max(axis=1)
    df['Is_Pure'] = (df['Max_Frac'] > 0.98).astype(int) 
    return df