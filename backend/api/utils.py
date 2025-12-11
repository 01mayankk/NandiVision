# utils.py
# Common helper functions for preprocessing images for PyTorch & ONNX models.

from PIL import Image, UnidentifiedImageError
import io
from torchvision import transforms
import numpy as np


# -----------------------------------------------------------
# Preprocessing for PyTorch (EfficientNet)
# -----------------------------------------------------------
def get_preprocess(img_size=224):
    """
    Returns torchvision transform for PyTorch EfficientNet models.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_image_for_torch(image_bytes, preprocess, device):
    """
    Converts uploaded bytes → Torch tensor.
    Handles corrupted image errors gracefully.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid or corrupted image file.")

    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor


# -----------------------------------------------------------
# Preprocessing for ONNX Runtime
# -----------------------------------------------------------
def preprocess_image_for_onnx(image_bytes, img_size):
    """
    Convert uploaded image → normalized numpy array for ONNX.
    Output shape: (1, 3, H, W) float32
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid or corrupted image file.")

    img = img.resize((img_size, img_size), Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)           # HWC → CHW
    arr = np.expand_dims(arr, axis=0)      # Add batch dimension

    return arr.astype(np.float32)
