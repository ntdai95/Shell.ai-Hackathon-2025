from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import io
import numpy as np
from src.engineer import engineer_features


app = FastAPI()
MODEL = joblib.load("models/model.pkl")
SCALER = joblib.load("models/scaler.pkl")
MEDIANS = joblib.load("models/medians.pkl")


@app.post("/predict")
async def predict_to_csv(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    if 'ID' in df.columns:
        ids = df['ID'] 
    else:
        range(len(df))
    
    X = engineer_features(df.drop(columns=['ID'], errors='ignore'))
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(MEDIANS, inplace=True)
    
    preds_scaled = MODEL.predict(X)
    # Ensuring the shape is (number_of_rows, 10)
    if preds_scaled.ndim == 1: 
        preds_scaled = preds_scaled.reshape(len(df), -1)
    
    preds = SCALER.inverse_transform(preds_scaled)
    res = pd.DataFrame(preds, columns=[f'BlendProperty{i}' for i in range(1, 11)])
    res.insert(0, 'ID', ids)
    
    stream = io.StringIO()
    res.to_csv(stream, index=False)
    return StreamingResponse(io.BytesIO(stream.getvalue().encode()), media_type="text/csv")