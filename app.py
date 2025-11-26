# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json
import os

app = FastAPI(title="Mental Health Prediction API")

# -------------------------------
# STATIC FILES (CSS / JS / IMAGES)
# -------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# LOAD TRAINED MODEL + METADATA
# -------------------------------
pipeline = joblib.load("model_pipeline.pkl")

with open("model_metadata.json") as f:
    FEATURES = json.load(f)["feature_names"]

# -------------------------------
# ROUTES
# -------------------------------

# Serve HTML
@app.get("/")
def home():
    return FileResponse("predict.html")


# Predict route
@app.post("/predict")
async def predict(request: Request):
    try:
        # Get form data from HTML
        form = await request.form()
        data = {k.lower(): v for k, v in form.items()}

        # Ensure all required features exist
        clean_data = {}
        for feature in FEATURES:
            clean_data[feature] = data.get(feature.lower(), "")

        # Convert to DataFrame
        df = pd.DataFrame([clean_data], columns=FEATURES)

        # Make prediction
        pred = pipeline.predict(df)[0]
        prob = float(pipeline.predict_proba(df)[0][1])

        return JSONResponse({
            "prediction": pred,
            "probability": round(prob, 3)
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)
