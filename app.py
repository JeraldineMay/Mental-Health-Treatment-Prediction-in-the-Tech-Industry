from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import json

app = FastAPI(title="Mental Health Prediction API")

# -------------------------------
# LOAD MODEL + METADATA
# -------------------------------
pipeline = joblib.load("model_pipeline.pkl")

with open("model_metadata.json") as f:
    FEATURES = json.load(f)["feature_names"]

# -------------------------------
# CORS (optional but useful)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------
# STATIC FILES
# -------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return FileResponse("predict.html")


@app.post("/predict")
async def predict(request: Request):
    # Get form data
    form = await request.form()
    data = {k.lower(): v for k, v in form.items()}

    # Build DataFrame for all features
    clean_data = {}
    for feature in FEATURES:
        key = feature.lower()
        clean_data[feature] = data.get(key, "")

    df = pd.DataFrame([clean_data], columns=FEATURES)

    # Predict
    try:
        pred = pipeline.predict(df)[0]
        prob = float(pipeline.predict_proba(df)[0][1])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse({
        "prediction": pred,
        "probability": round(prob, 3)
    })
