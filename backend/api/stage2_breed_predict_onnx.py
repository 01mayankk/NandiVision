# stage2_breed_predict_onnx.py
# ONNX Runtime inference for breed classification.

import onnxruntime as ort # type: ignore
import torch
import numpy as np
from fastapi import UploadFile
from pathlib import Path
from .utils import preprocess_image_for_onnx

# -------------------------------------------------------
# SAFE ABSOLUTE PATHS
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # backend/

CKPT_PATH = BASE_DIR / "models" / "stage2_breed_best.pth"
MODEL_PATH = BASE_DIR / "models" / "stage2_breed.onnx"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Stage-2 checkpoint not found at: {CKPT_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Stage-2 ONNX model not found at: {MODEL_PATH}")


# -------------------------------------------------------
# Load metadata (breed names / img_size)
# -------------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
BREED_NAMES = ckpt["breed_names"]
IMG_SIZE = int(ckpt["img_size"])


# -------------------------------------------------------
# Load ONNX Runtime session
# (CPUExecutionProvider only → stable)
# -------------------------------------------------------
providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH.as_posix(), providers=providers)


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


async def predict_breed_onnx(file: UploadFile):
    """
    Output:
        {
            "breed": str,
            "confidence": float
        }
    """
    image_bytes = await file.read()

    # Preprocess → float32 numpy
    x = preprocess_image_for_onnx(image_bytes, IMG_SIZE)

    ort_inputs = {session.get_inputs()[0].name: x}
    ort_outs = session.run(None, ort_inputs)

    logits = ort_outs[0][0]
    probs = softmax(logits)

    idx = int(np.argmax(probs))

    return {
        "breed": BREED_NAMES[idx],
        "confidence": float(probs[idx])
    }
