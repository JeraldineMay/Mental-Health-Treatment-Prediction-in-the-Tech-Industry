# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# Load dataset
df = pd.read_csv("mental_health_data.csv")

# Define target and all possible features
target = 'treatment'
possible_categorical_cols = [
    'Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
    'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
    'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview', 'physical_health_interview',
    'mental_vs_physical', 'obs_consequence', 'Country'
]
numeric_cols = ['Age']

# Keep only columns that actually exist in the dataset
categorical_cols = [col for col in possible_categorical_cols if col in df.columns]
numeric_cols = [col for col in numeric_cols if col in df.columns]

# Combine features
feature_cols = categorical_cols + numeric_cols
X = df[feature_cols]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Full pipeline with Random Forest
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200))
])

# Train model
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, "model_pipeline.pkl")

# Save feature names for FastAPI app
metadata = {"feature_names": feature_cols}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

print("Model training complete. Pipeline and metadata saved.")
