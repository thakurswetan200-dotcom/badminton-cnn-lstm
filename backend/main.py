from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil

from src.predict import predict_video


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Badminton AI API",
    description="CNN-LSTM badminton shot recognition API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Badminton AI backend is running"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Create temporary folder
    # --------------------------------------------------------

    temp_folder = "temp"

    os.makedirs(
        temp_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create path for uploaded video
    # --------------------------------------------------------

    video_path = os.path.join(
        temp_folder,
        file.filename
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    with open(
        video_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    print(
        f"Received video: {file.filename}"
    )

    # --------------------------------------------------------
    # Run CNN-LSTM prediction
    # --------------------------------------------------------

    try:

        result = predict_video(
            video_path
        )

        print(
            "Prediction:",
            result
        )

        return result

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return {
            "error": str(e)
        }

    finally:

        # ----------------------------------------------------
        # Delete temporary video
        # ----------------------------------------------------

        if os.path.exists(video_path):

            os.remove(
                video_path
            )