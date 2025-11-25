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

# Most impactful features to vary (common factors influencing mental health prediction)
key_features = {
    "family_history": ["No"],  # No family history lowers risk
    "work_interfere": ["Never", "Rarely"],  # Rare interference
    "mental_health_consequence": ["No"],  # No consequences observed
    "phys_health_consequence": ["No"],  # No consequences observed
    "seek_help": ["No"],  # Did not seek help
    "mental_health_interview": ["No"],  # No interview
}

# Fixed values for other features
fixed_values = {
    "Age": 23,
    "Gender": "Female",
    "Country": "Canada",
    "self_employed": "No",
    "no_employees": "1-5",
    "remote_work": "No",
    "tech_company": "No",
    "benefits": "No",
    "care_options": "No",
    "wellness_program": "No",
    "anonymity": "No",
    "leave": "Very easy",
    "coworkers": "No",
    "supervisor": "No",
    "physical_health_interview": "No",
    "mental_vs_physical": "No",
    "obs_consequence": "No"
}

# Generate all combinations of key features
for combo in product(*[key_features[f] for f in key_features]):
    candidate = {f: v for f, v in zip(key_features.keys(), combo)}
    candidate.update(fixed_values)
    
    df = pd.DataFrame([candidate], columns=FEATURES)
    pred = pipeline.predict(df)[0]
    
    if pred == "No":
        print("Found a candidate that predicts 'No':")
        print(json.dumps(candidate, indent=2))
        print("Prediction: No")
        break
else:
    print("No combination found that predicts 'No'. Try adjusting Age or other fixed features.")
