# %%
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# %%
# ---------------------
# 1. Data Preparation
# ---------------------

from collections import Counter


X_train = np.load('/kaggle/input/unormal-2/X_train 1.npy')
y_train = np.load('/kaggle/input/ml-dataset/y_train.npy', allow_pickle=True)

# Extract binary labels (benign/malignant) from column 0
y_binary_labels = y_train[:, 0]
le_bin = LabelEncoder()
y_encoded = le_bin.fit_transform(y_binary_labels)  # benign -> 0, malignant -> 1



# # Split into train/val/test
# X_train_split, X_temp, y_train_split, y_temp = train_test_split(
#     X_train, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
# )

# X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(
#     X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42


# Dataset class
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

# Transforms + Data Augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)


train_dataset = CustomImageDataset(X_train_split, y_train_split, transform=train_transform)
val_dataset_full = CustomImageDataset(X_val_split, y_val_split, transform=test_val_transform)
val_size = len(val_dataset_full) // 5
val_dataset, test_dataset = random_split(val_dataset_full, [3 * val_size, len(val_dataset_full) - 3 * val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"\nDataset sizes — Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")







from torchvision.models import resnet50, ResNet50_Weights
# ---------------------
# 2. Model Setup (ResNet50)
# ---------------------

batch = 'binary_aug_try1'

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

#Unfreeze last few conv layers
for layer in list(model.layer4.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
    

# Unfreeze the final FC layer
for param in model.fc.parameters():
    param.requires_grad = True

# Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_split), y=y_train_split)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
# ---------------------
# 3. Training Function
# ---------------------
def evaluate_model(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

def preds_labels(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler=None, epochs=5,
                device=device, save_model=True, save_model_path='best_model_resnet_3layers.pt',
                early_stopping=True, patience=5):

    model.to(device)
    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
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

        train_acc = correct / total
        val_acc = evaluate_model(model, validation_loader, device)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Validation loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
        val_losses.append(val_running_loss / len(validation_loader))


        if scheduler is not None:
            scheduler.step(val_losses[-1])

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            if save_model:
                torch.save(model.state_dict(), save_model_path)
        else:
            epochs_no_improve += 1

        if early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_acc:.4f}")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies

# ---------------------
# 4. Train & Evaluate
# ---------------------

##scheduler is optional, but it helps to reduce the learning rate when the validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    epochs=40, device=device, save_model_path=f"best_model_{batch}_resnet.pt"
)


# Predictions
all_preds, all_labels = preds_labels(model, test_loader, device=device)

# Binary Confusion Matrix
cm_bin = confusion_matrix(all_labels, all_preds, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_bin.classes_, yticklabels=le_bin.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Binary Confusion Matrix")
plt.tight_layout()
plt.savefig(f"en_{batch}_binary_confusion_matrix.pdf")
plt.close()

# Save metrics
def accuracy_metrics(cm_bin): 
    TP, TN = cm_bin[1,1], cm_bin[0,0]
    FP, FN = cm_bin[0,1], cm_bin[1,0]
    total = np.sum(cm_bin)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return pd.DataFrame({
        'Accuracy': [(TP + TN) / total],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [F1]
    })

metrics_df = accuracy_metrics(cm_bin)
metrics_df.to_csv(f"en_{batch}_metrics.csv", index=False)

# Plot training curves
def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:green')
    ax2.plot(epochs, train_accuracies, label='Train Acc', color='tab:green')
    ax2.plot(epochs, val_accuracies, label='Val Acc', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title('Training and Validation Metrics')
    plt.tight_layout()
    plt.savefig(f"{batch}_training_curve.pdf")
    plt.show()
    plt.close()

plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies)





