# stage1_predict_onnx.py
# ONNX Runtime inference for Stage-1 (cow / buffalo / none).

import onnxruntime as ort # type: ignore
import numpy as np
from fastapi import UploadFile
from pathlib import Path
from .utils import preprocess_image_for_onnx
import torch

# -------------------------------------------------------
# SAFE ABSOLUTE PATHS (WORKS IN UVICORN ALWAYS)
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # backend/

CKPT = BASE_DIR / "models" / "stage1_cattle_best.pth"
ONNX_PATH = BASE_DIR / "models" / "stage1_cattle.onnx"

# --- sanity checks ---
if not CKPT.exists():
    raise FileNotFoundError(f"Stage-1 checkpoint not found at: {CKPT}")

if not ONNX_PATH.exists():
    raise FileNotFoundError(f"Stage-1 ONNX model not found at: {ONNX_PATH}")

# -------------------------------------------------------
# Load metadata (class names, img_size)
# -------------------------------------------------------
ckpt = torch.load(CKPT, map_location="cpu")
CLASS_NAMES = ckpt.get("class_names", ["cow", "buffalo", "none"])
IMG_SIZE = int(ckpt.get("img_size", 224))

# -------------------------------------------------------
# Create ONNX Runtime session (CPU only)
# -------------------------------------------------------
providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_PATH.as_posix(), providers=providers)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


async def predict_stage1_onnx(file: UploadFile):
    """
    Input: UploadFile
    Output:
        {"class": str, "confidence": float}
    """
    image_bytes = await file.read()

    # preprocess → numpy (1,3,H,W)
    x = preprocess_image_for_onnx(image_bytes, IMG_SIZE)

    inputs = {session.get_inputs()[0].name: x}
    outputs = session.run(None, inputs)

    logits = outputs[0][0]  # logits for class prediction
    probs = _softmax(logits)

    idx = int(np.argmax(probs))

    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(probs[idx])
    }
