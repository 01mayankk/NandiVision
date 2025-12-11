# export_stage1_onnx.py
# Export Stage-1 EfficientNet-B0 (cow/buffalo/none) to ONNX

import torch
from torchvision import models
from pathlib import Path

# ----------------------------------
# Correct absolute paths
# ----------------------------------
CKPT = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models/stage1_cattle_best.pth")
OUT = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models/stage1_cattle.onnx")

# Ensure checkpoint exists
if not CKPT.exists():
    raise FileNotFoundError("Stage-1 checkpoint not found at: " + str(CKPT))

# Load checkpoint
ckpt = torch.load(CKPT, map_location="cpu")
class_names = ckpt["class_names"]
num_classes = len(class_names)
img_size = ckpt["img_size"]

print("Exporting Stage-1 ONNX model...")
print("Classes:", class_names)

# Build model architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy_input = torch.randn(1, 3, img_size, img_size)

torch.onnx.export(
    model,
    dummy_input,
    OUT.as_posix(),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12
)

print("✅ Stage-1 ONNX model exported successfully!")
print("Saved at:", OUT)
