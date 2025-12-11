# train_stage2_breed.py
# EfficientNet-B2 Breed classification (15 breeds)

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
# ABSOLUTE PATHS (IMPORTANT)
# ---------------------------------------------------------
DATA_DIR = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/datasets/stage2_breeds")
SAVE_DIR = Path("C:/Users/01may/OneDrive/Desktop/NandiVision/backend/models")

SAVE_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# LOAD BREED NAMES (Folder Names)
# ---------------------------------------------------------
BREEDS = sorted([d.name for d in DATA_DIR.iterdir()])
NUM_CLASSES = len(BREEDS)

print("\n📌 Detected Breeds:")
for b in BREEDS:
    print(" -", b)

IMG_SIZE = 256
BATCH_SIZE = 12
EPOCHS = 25
LR = 1e-4


# ---------------------------------------------------------
# LOAD ALL IMAGES + LABELS
# ---------------------------------------------------------
all_files = []
all_labels = []

print("\n📊 Loading Dataset...")
for idx, breed in enumerate(BREEDS):
    folder = DATA_DIR / breed
    count = 0
    for file in folder.glob("*.*"):
        all_files.append(str(file))
        all_labels.append(idx)
        count += 1
    print(f"{breed}: {count} images")

print("Total Images:", len(all_files))

if len(all_files) == 0:
    raise ValueError("❌ No images found in stage2_breeds folder!")


# ---------------------------------------------------------
# TRAIN / VAL SPLIT
# ---------------------------------------------------------
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
# MODEL — EfficientNet-B2
# ---------------------------------------------------------
model = models.efficientnet_b2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1408, NUM_CLASSES)  # replace classifier head
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


best_acc = 0


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

    avg_train_acc = sum(train_accs) / len(train_accs)


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_accs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_accs.append(get_accuracy(outputs, labels))

    avg_val_acc = sum(val_accs) / len(val_accs)

    print(f"\nEpoch {epoch}: TrainAcc={avg_train_acc:.4f}, ValAcc={avg_val_acc:.4f}")

    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        torch.save(
            {
                "model_state": model.state_dict(),
                "breed_names": BREEDS,
                "img_size": IMG_SIZE,
            },
            SAVE_DIR / "stage2_breed_best.pth",
        )
        print("🔥 Saved Best Stage-2 Model!")


print("\n🎉 Stage-2 Training Completed!")
print("🏆 Best Validation Accuracy:", best_acc)
