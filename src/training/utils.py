# utils.py
# Helper functions for dataset splitting, accuracy, etc.

import torch
from sklearn.metrics import accuracy_score
import numpy as np


# ---------------------------
# CALCULATE ACCURACY
# ---------------------------
def get_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())


# ---------------------------
# SOFTMAX (for ONNX export check)
# ---------------------------
def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
