# app.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import joblib
import json
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Mental Health Prediction API")

# -------------------------------
# LOAD TRAINED MODEL + METADATA
# -------------------------------
pipeline = joblib.load("model_pipeline.pkl")

with open("model_metadata.json") as f:
    FEATURES = json.load(f)["feature_names"]


# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return FileResponse("predict.html")


@app.post("/predict")
async def predict(request: Request):

    # Handle both form-data and JSON
    try:
        incoming = await request.form()
        data = {k.lower(): v for k, v in incoming.items()}
    except:
        incoming = await request.json()
        data = {k.lower(): v for k, v in incoming.items()}

    clean_data = {}

    # Ensure features match training metadata
    for feature in FEATURES:
        raw = data.get(feature.lower(), None)

        # Convert to float if possible
        try:
            clean_data[feature] = float(raw)
        except:
            clean_data[feature] = 0

    # Create DataFrame
    df = pd.DataFrame([clean_data], columns=FEATURES)

    # Predict
    pred = pipeline.predict(df)[0]

    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(df)[0][1])
    else:
        prob = None

    return JSONResponse({
        "prediction": pred,
        "probability": prob
    })

