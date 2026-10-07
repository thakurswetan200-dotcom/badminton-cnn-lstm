from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import tempfile
import os

from src.predict import predict_video


app = FastAPI()


# -----------------------------
# Allow React frontend
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Badminton Analytics API is running"
    }


# -----------------------------
# Prediction
# -----------------------------

@app.post("/predict")
async def predict(video: UploadFile = File(...)):

    # Create temporary video file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_video:

        video_bytes = await video.read()

        temp_video.write(video_bytes)

        temp_video_path = temp_video.name

    try:

        # Run CNN-LSTM prediction
        result = predict_video(temp_video_path)

        return result

    finally:

        # Delete temporary video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)