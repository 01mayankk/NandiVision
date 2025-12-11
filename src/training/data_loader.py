# data_loader.py
# Loads images for both Stage-1 and Stage-2 training
# Includes augmentation, transforms, and dataset class

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os


# ---------------------------
# TRAIN TRANSFORMS (AUGMENTED)
# ---------------------------
def get_train_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.4),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


# ---------------------------
# VALIDATION TRANSFORMS (CLEAN)
# ---------------------------
def get_valid_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


# ---------------------------
# CUSTOM DATASET CLASS
# ---------------------------
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
