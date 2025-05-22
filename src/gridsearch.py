# %%
# ==================== IMPORTS ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50, ResNet50_Weights

import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==================== SET DEVICE ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== DATA LOADING ====================
X_train = np.load('/kaggle/input/unormal/X_train 1.npy')
y_train = np.load('/kaggle/input/ml-dataset/y_train.npy', allow_pickle=True)

# Extract binary labels (benign/malignant) from column 0
y_binary_labels = y_train[:, 0]
le_bin = LabelEncoder()
y_encoded = le_bin.fit_transform(y_binary_labels)

# ==================== DATASET CLASS ====================
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== SPLIT DATA ====================
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

train_dataset = CustomImageDataset(X_train_split, y_train_split, transform=train_transform)
val_dataset_full = CustomImageDataset(X_val_split, y_val_split, transform=test_val_transform)

val_size = len(val_dataset_full) // 5
val_dataset, test_dataset = random_split(val_dataset_full, [3 * val_size, len(val_dataset_full) - 3 * val_size])

print("\n✅ Dataset sizes — Train:", len(train_dataset), "Validation:", len(val_dataset), "Test:", len(test_dataset))

# ==================== CREATE MODEL FUNCTION ====================
def create_model(dp1, dp2):
    model = resnet50(weights=None)
    model.load_state_dict(torch.load("/kaggle/input/resnet50/pytorch/default/1/resnet50-11ad3fa6 (1).pth", weights_only=True))

    model.fc = nn.Sequential(
        nn.Dropout(dp1),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(dp2),
        nn.Linear(512, 2)
    )

    for param in model.parameters():
        param.requires_grad = False
    for layer in list(model.layer4.children())[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

# ==================== TRAIN FUNCTION ====================
def train_model_grid(model, train_dataset, val_dataset, lr, bs, patience, device=device):
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_split), y=y_train_split)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6)

    model.to(device)
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(40):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = 0.0
        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_acc

# ==================== GRID SEARCH ====================
results = []

with open('/kaggle/input/param-json/params.jsonl') as f:
    params_list = [json.loads(line) for line in f]

for idx, hyperparams in enumerate(params_list):
    print(f"\n🔥 Training config {idx+1}/{len(params_list)}: {hyperparams}")
    model = create_model(hyperparams['dp1'], hyperparams['dp2'])
    val_acc = train_model_grid(
        model, train_dataset, val_dataset,
        lr=hyperparams['lr'],
        bs=hyperparams['bs'],
        patience=hyperparams['patience'],
        device=device
    )

    result = {
        'lr': hyperparams['lr'],
        'bs': hyperparams['bs'],
        'dp1': hyperparams['dp1'],
        'dp2': hyperparams['dp2'],
        'patience': hyperparams['patience'],
        'val_accuracy': val_acc
    }
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("/kaggle/working/grid_search_results.csv", index=False)
print("\n✅ Grid Search Completed!")
print(results_df.sort_values(by='val_accuracy', ascending=False))


# %%
# To list all files in /kaggle/working
import os

print(os.listdir('/kaggle/working'))

# %%
import pandas as pd

# Load the CSV
df = pd.read_csv('/kaggle/working/grid_search_results.csv')

# Print it
print(df)


