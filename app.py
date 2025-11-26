from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json

app = FastAPI(title="Mental Health Prediction API")

# -------------------------------
# LOAD TRAINED MODEL + METADATA
# -------------------------------
pipeline = joblib.load("model_pipeline.pkl")  # your trained model pipeline

with open("model_metadata.json") as f:
    FEATURES = json.load(f)["feature_names"]  # list of all features in your model

# -------------------------------
# STATIC FILES (for CSS/images)
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
    try:
        # Get submitted form data
        form = await request.form()
        data = {k.lower(): v for k, v in form.items()}

        # Prepare clean data dictionary
        clean_data = {}
        for feature in FEATURES:
            value = data.get(feature.lower(), "")

            # Convert numeric fields to float (like age)
            if feature.lower() == "age":
                try:
                    value = float(value)
                except:
                    value = 0  # default if invalid
            clean_data[feature] = value

        # Convert to DataFrame
        df = pd.DataFrame([clean_data], columns=FEATURES)

        # Predict
        pred = pipeline.predict(df)[0]
        prob = float(pipeline.predict_proba(df)[0][1])

        return JSONResponse({
            "prediction": pred,
            "probability": round(prob, 3)
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })
