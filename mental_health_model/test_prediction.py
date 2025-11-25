import joblib
import json
import pandas as pd
from itertools import product

# Load model and metadata
MODEL_PATH = "model_pipeline.pkl"
META_PATH = "model_metadata.json"

pipeline = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]

# Define reasonable options for each categorical feature
# Adjust based on what your dataset used
categorical_options = {
    "Gender": ["Male", "Female"],
    "self_employed": ["Yes", "No"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"],
    "no_employees": ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
    "remote_work": ["Yes", "No"],
    "tech_company": ["Yes", "No"],
    "benefits": ["Yes", "No", "Don't know"],
    "care_options": ["Yes", "No", "Not sure"],
    "wellness_program": ["Yes", "No", "Don't know"],
    "seek_help": ["Yes", "No", "Don't know"],
    "anonymity": ["Yes", "No", "Don't know"],
    "leave": ["Very easy", "Somewhat easy", "Very difficult", "Don't know"],
    "mental_health_consequence": ["Yes", "No", "Maybe"],
    "phys_health_consequence": ["Yes", "No", "Maybe"],
    "coworkers": ["Yes", "No", "Some of them"],
    "supervisor": ["Yes", "No", "Some of them"],
    "mental_health_interview": ["Yes", "No", "Maybe"],
    "physical_health_interview": ["Yes", "No", "Maybe"],
    "mental_vs_physical": ["Yes", "No", "Don't know"],
    "obs_consequence": ["Yes", "No"]
}

# Define a range for numeric features
numeric_features = {
    "Age": range(18, 35)  # Typical adult age range
}

# Generate one “low-risk” candidate
candidate = {
    "Age": 23,
    "Gender": "Female",
    "self_employed": "No",
    "family_history": "No",
    "work_interfere": "Never",
    "no_employees": "1-5",
    "remote_work": "No",
    "tech_company": "No",
    "benefits": "No",
    "care_options": "No",
    "wellness_program": "No",
    "seek_help": "No",
    "anonymity": "No",
    "leave": "Very easy",
    "mental_health_consequence": "No",
    "phys_health_consequence": "No",
    "coworkers": "No",
    "supervisor": "No",
    "mental_health_interview": "No",
    "physical_health_interview": "No",
    "mental_vs_physical": "No",
    "obs_consequence": "No",
    "Country": "Canada"
}

# Predict
df = pd.DataFrame([candidate], columns=FEATURES)
prediction = pipeline.predict(df)[0]
probabilities = pipeline.predict_proba(df)[0].tolist()

print("Candidate input:")
print(json.dumps(candidate, indent=2))
print("\nPrediction:", prediction)
print("Probabilities:", probabilities)
