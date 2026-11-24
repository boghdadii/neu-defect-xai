import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NEUDataset(Dataset):
    def __init__(self, df, transform=None, rgb=False):
        self.paths = df["filepath"].values
        self.labels = [CLASS_TO_IDX[l] for l in df["label"].values]
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB" if self.rgb else "L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_transforms(rgb=False, augment=False):
    ops = []
    if rgb:
        ops.append(transforms.Resize((224, 224)))
    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ])
    ops.append(transforms.ToTensor())
    if rgb:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(ops)

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def build_model(model_name, num_classes=6):
    if model_name == "scratch":
        return ScratchCNN(num_classes)
    elif model_name in ("resnet_frozen", "resnet_full"):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if model_name == "resnet_frozen":
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {model_name}")