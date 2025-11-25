from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("predict.html")


@app.post("/predict")
async def predict(request: Request):

    data = await request.json()

    # ===== NORMALIZE ALL INPUTS =====
    normalized = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in data.items()}

    print("Received:", normalized)

    # Start with NO
    prediction = "no"

    # ===== RULE 1: Strong mental health indicators =====
    positive_keys = [
        "family_history",
        "mental_health_consequence",
    ]

    for key in positive_keys:
        if normalized.get(key) == "yes":
            prediction = "yes"

    # ===== RULE 2: Work interfere =====
    if normalized.get("work_interfere") in ["often"]:
        prediction = "yes"

    # ===== RULE 3: Leave difficulty =====
    if normalized.get("leave") == "very difficult":
        prediction = "yes"

    # ===== RULE 4: Physical health consequence =====
    if normalized.get("phys_health_consequence") == "yes":
        prediction = "yes"

    # ===== RULE 5: Obvious consequences =====
    if normalized.get("obs_consequence") == "yes":
        prediction = "yes"

    # Less aggressive: coworkers/supervisor removed

    # FINAL OUTPUT
    return JSONResponse({"prediction": prediction.capitalize()})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
