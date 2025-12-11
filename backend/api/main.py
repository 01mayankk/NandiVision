# main.py
# Main FastAPI application for NandiVision Backend.

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Stage-1 = ONNX, Stage-2 = ONNX
from .stage1_predict_onnx import predict_stage1_onnx as predict_stage1
from .stage2_breed_predict_onnx import predict_breed_onnx

import io
from fastapi.datastructures import UploadFile as UP

app = FastAPI(
    title="NandiVision Backend API",
    description="API for Indian Cattle Type & Breed Classification",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "NandiVision backend is running successfully!"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Pipeline:
    1. Stage-1 (ONNX) → cow / buffalo / none
    2. If cattle detected → Stage-2 (ONNX) → breed
    """

    # Read bytes once
    bytes_data = await file.read()

    # Stage-1 file
    stage1_file = UP(filename=file.filename, file=io.BytesIO(bytes_data))

    # ---- Stage 1 ----
    stage1 = await predict_stage1(stage1_file)

    if stage1["class"] == "none":
        return {
            "type": "none",
            "type_confidence": stage1["confidence"],
            "message": "No cattle detected in image."
        }

    # Stage-2 file
    stage2_file = UP(filename=file.filename, file=io.BytesIO(bytes_data))

    # ---- Stage 2 ----
    breed = await predict_breed_onnx(stage2_file)

    return {
        "type": stage1["class"],
        "type_confidence": stage1["confidence"],
        "breed": breed["breed"],
        "breed_confidence": breed["confidence"]
    }
