from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# CORS settings
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
def get_home():
    return FileResponse("predict.html")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # ========== RULE-BASED PREDICTION ==========
    # If ANY of these answers are Yes → higher risk
    risky_keys_yes = [
        "family_history",
        "mental_health_consequence",
        "phys_health_consequence",
        "obs_consequence"
    ]

    # If work interferes often/sometimes → higher risk
    work_interfere_risky = ["Often", "Sometimes"]

    # Start with "No"
    prediction = "No"

    # Rule 1: Check Yes answers
    for key in risky_keys_yes:
        if data.get(key, "").strip().lower() == "yes":
            prediction = "Yes"

    # Rule 2: Work interfere
    if data.get("work_interfere", "") in work_interfere_risky:
        prediction = "Yes"

    # Rule 3: Leave difficulty
    if data.get("leave") in ["Very difficult"]:
        prediction = "Yes"

    # Rule 4: Coworkers / supervisor support
    if data.get("coworkers") == "No" or data.get("supervisor") == "No":
        prediction = "Yes"

    # Rule 5: Age factor (optional, adjust as needed)
    try:
        age = int(data.get("age", 0))
        if age < 18 or age > 60:
            prediction = "Yes"
    except:
        pass

    return JSONResponse({"prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
