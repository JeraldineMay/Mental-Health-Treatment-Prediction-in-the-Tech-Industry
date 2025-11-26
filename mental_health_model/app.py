from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import joblib
import json
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Mental Health Prediction API")

# Load model
pipeline = joblib.load("model_pipeline.pkl")

# Load metadata
with open("model_metadata.json") as f:
    FEATURES = json.load(f)["feature_names"]

@app.get("/")
def home():
    return FileResponse("predict.html")


@app.post("/predict")
async def predict(request: Request):

    # MUST use form-data (your frontend sends multipart/form-data)
    form = await request.form()

    # Lowercase keys
    data = {k.lower(): v for k, v in form.items()}

    # Ensure all expected features exist
    clean_data = {}
    for feature in FEATURES:
        value = data.get(feature.lower(), "")

        # Convert numbers
        try:
            clean_data[feature] = float(value)
        except:
            clean_data[feature] = value

    df = pd.DataFrame([clean_data], columns=FEATURES)

    # Predict
    pred = pipeline.predict(df)[0]

    # Try probability
    try:
        prob = float(pipeline.predict_proba(df)[0][1])
    except:
        prob = None

    return JSONResponse({
        "prediction": pred,
        "probability": prob
    })
