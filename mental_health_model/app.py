from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve predict.html at the root URL
@app.get("/")
def get_home():
    return FileResponse("predict.html")  # Same folder as app.py

# Example prediction endpoint
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    # Your prediction logic here
    age = int(data.get("age", 0))
    prediction = "Yes" if age < 40 else "No"
    return JSONResponse({"prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
