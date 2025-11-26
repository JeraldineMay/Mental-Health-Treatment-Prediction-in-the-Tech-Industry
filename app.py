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

    # Get submitted form data
    form = await request.form()

    # Convert form keys to lowercase
    data = {k.lower(): v for k, v in form.items()}

    # Ensure all required features exist
    clean_data = {}
    for feature in FEATURES:
        clean_data[feature] = data.get(feature.lower(), "")

    # Convert to DataFrame
    df = pd.DataFrame([clean_data], columns=FEATURES)

    # Predict
    pred = pipeline.predict(df)[0]
    prob = float(pipeline.predict_proba(df)[0][1])

    return JSONResponse({
        "prediction": pred,
        "probability": round(prob, 3)
    })
