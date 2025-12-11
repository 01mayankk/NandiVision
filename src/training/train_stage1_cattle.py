# train_stage1_cattle.py
# Stage-1 training script (EfficientNet-B0)
# Detects: cow / buffalo / none

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

from data_loader import ImageDataset, get_train_transforms, get_valid_transforms
from utils import get_accuracy


# ---------------------------------------------------------
# ABSOLUTE PATHS (VERY IMPORTANT – FIXES "n_samples=0" ISSUE)
# ---------------------------------------------------------
DATA_DIR = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/datasets/stage1")
SAVE_DIR = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models")

SAVE_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class folder names MUST match directory names EXACTLY
CLASS_NAMES = ["buffalo", "cow", "none"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15   # 15 is usually enough for Stage-1
LR = 1e-4


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
all_files = []
all_labels = []

for idx, class_name in enumerate(CLASS_NAMES):
    folder = DATA_DIR / class_name
    print(f"Loading: {folder}")  # debugging line

    # Load all images inside each folder
    for file in folder.glob("*.*"):
        all_files.append(str(file))
        all_labels.append(idx)

print("\n📊 Dataset Summary:")
print("Buffalo images :", sum(1 for l in all_labels if l == 0))
print("Cow images     :", sum(1 for l in all_labels if l == 1))
print("None images    :", sum(1 for l in all_labels if l == 2))
print("Total images   :", len(all_files))

# Safety check
if len(all_files) == 0:
    raise ValueError("❌ No images found! Check dataset paths.")

# Split dataset
train_f, val_f, train_l, val_l = train_test_split(
    all_files,
    all_labels,
    test_size=0.15,
    stratify=all_labels,
    random_state=42
)

train_ds = ImageDataset(train_f, train_l, transform=get_train_transforms(IMG_SIZE))
val_ds = ImageDataset(val_f, val_l, transform=get_valid_transforms(IMG_SIZE))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------
# MODEL SETUP — EfficientNet-B0
# ---------------------------------------------------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    train_accs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_accs.append(get_accuracy(outputs, labels))

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_acc = sum(train_accs) / len(train_accs)


    # -------------------------------
    # VALIDATION
    # -------------------------------
    model.eval()
    val_accs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_accs.append(get_accuracy(outputs, labels))

    avg_val_acc = sum(val_accs) / len(val_accs)


    print(f"\nEpoch {epoch}:")
    print(f" Train Acc = {avg_train_acc:.4f} | Val Acc = {avg_val_acc:.4f}")


    # Save best model
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(
            {
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "img_size": IMG_SIZE,
            },
            SAVE_DIR / "stage1_cattle_best.pth",
        )
        print("🔥 Saved Best Stage-1 Model!")


print("\n🎉 Training Completed!")
print("Best Validation Accuracy:", best_val_acc)
