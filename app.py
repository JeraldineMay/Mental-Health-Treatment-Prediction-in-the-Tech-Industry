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
    normalized = {}
    for key, value in data.items():
        if isinstance(value, str):
            normalized[key] = value.strip().lower()
        else:
            normalized[key] = value

    print("Received:", normalized)

    # ===== START WITH NO =====
    prediction = "no"

    # ===== RULE 1: RISKY YES ANSWERS =====
    risky_yes_fields = [
        "family_history",
        "mental_health_consequence",
        "phys_health_consequence",
        "obs_consequence",
    ]

    for key in risky_yes_fields:
        if normalized.get(key) == "yes":
            prediction = "yes"

    # ===== RULE 2: Work interfere =====
    if normalized.get("work_interfere") in ["often", "sometimes"]:
        prediction = "yes"

    # ===== RULE 3: Leave difficulty =====
    if normalized.get("leave") == "very difficult":
        prediction = "yes"

    # ===== RULE 4: Coworkers / Supervisor =====
    if normalized.get("coworkers") == "no":
        prediction = "yes"
    if normalized.get("supervisor") == "no":
        prediction = "yes"

    # ===== RULE 5: Mental vs Physical =====
    if normalized.get("mental_vs_physical") == "yes":
        prediction = "yes"

    # ===== FINAL OUTPUT =====
    return JSONResponse({"prediction": prediction.capitalize()})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
