# export_stage2_onnx.py
# Export PyTorch EfficientNet-B2 → ONNX for FastAPI GPU inference

import torch
from torchvision import models
from pathlib import Path

# -------------------------------------------
# ABSOLUTE PATHS (VERY IMPORTANT)
# -------------------------------------------
MODEL_PATH = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models/stage2_breed_best.pth")
SAVE_PATH = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models/stage2_breed.onnx")

# -------------------------------------------
# Load PyTorch checkpoint
# -------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Trained model not found at: {MODEL_PATH}")

print("Loading Stage-2 checkpoint...")
ckpt = torch.load(MODEL_PATH, map_location="cpu")

breed_names = ckpt["breed_names"]
num_classes = len(breed_names)
img_size = ckpt["img_size"]

print("Detected Breeds:", breed_names)
print("Image Size:", img_size)

# -------------------------------------------
# Build EfficientNet-B2 model
# -------------------------------------------
model = models.efficientnet_b2(weights=None)
model.classifier[1] = torch.nn.Linear(1408, num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Dummy input for ONNX conversion
dummy_input = torch.randn(1, 3, img_size, img_size)

# -------------------------------------------
# Export to ONNX
# -------------------------------------------
print("Exporting Stage-2 model to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    SAVE_PATH.as_posix(),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12
)

print("✅ Stage-2 ONNX export completed successfully!")
print("Saved at:", SAVE_PATH)
