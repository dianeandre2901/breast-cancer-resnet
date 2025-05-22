"""
train.py
--------
Trains a fine-tuned ResNet50 model on histopathology images for binary breast cancer classification
(benign vs malignant) using transfer learning, data augmentation, and evaluation metrics.
"""

# --------------------------
# Imports and Configuration
# --------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------
# 1. Data Preparation
# --------------------------
X_train = np.load('/kaggle/input/unormal-2/X_train 1.npy')
y_train = np.load('/kaggle/input/ml-dataset/y_train.npy', allow_pickle=True)

# Encode binary labels
y_binary_labels = y_train[:, 0]  # Column 0: benign vs malignant
le_bin = LabelEncoder()
y_encoded = le_bin.fit_transform(y_binary_labels)

# Train/Validation/Test split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# Custom Dataset
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

# Transforms
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

# Build datasets and loaders
val_dataset_full = CustomImageDataset(X_val_split, y_val_split, transform=test_val_transform)
val_size = len(val_dataset_full) // 5
val_dataset, test_dataset = random_split(val_dataset_full, [3 * val_size, len(val_dataset_full) - 3 * val_size])

train_dataset = CustomImageDataset(X_train_split, y_train_split, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"\nDataset sizes — Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# --------------------------
# 2. Model Setup (ResNet50)
# --------------------------
batch = 'binary_aug_try1'
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Modify FC layer for binary classification
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last conv layers + FC head
for layer in list(model.layer4.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_split), y=y_train_split)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

# --------------------------
# 3. Training & Evaluation
# --------------------------
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def preds_labels(model, loader):
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=40,
                save_model_path='best_model.pt', early_stopping=True, patience=5):

    model.to(device)
    best_acc, epochs_no_improve = 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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
        val_acc = evaluate_model(model, val_loader)

        train_losses.append(running_loss / len(train_loader))
        val_accs.append(val_acc)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        val_losses.append(val_loss / len(val_loader))

        if scheduler:
            scheduler.step(val_losses[-1])

        print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_model_path)
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs. Best Val Acc: {best_acc:.4f}")
                break

    return train_losses, val_losses, train_accs, val_accs

# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    save_model_path=f"best_model_{batch}.pt")

# Predictions + Metrics
all_preds, all_labels = preds_labels(model, test_loader)
cm_bin = confusion_matrix(all_labels, all_preds, labels=[0, 1])

# Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_bin.classes_, yticklabels=le_bin.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"en_{batch}_confusion_matrix.pdf")
plt.close()

# Compute metrics
TP, TN = cm_bin[1,1], cm_bin[0,0]
FP, FN = cm_bin[0,1], cm_bin[1,0]
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
F1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
accuracy = (TP + TN) / np.sum(cm_bin)

metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [F1]
})
metrics_df.to_csv(f"en_{batch}_metrics.csv", index=False)

# Plot training progress
def plot_training_progress(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='green')
    ax2.plot(epochs, train_accs, label='Train Acc', color='green')
    ax2.plot(epochs, val_accs, label='Val Acc', color='red')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.legend(loc='lower right')
    plt.title('Training & Validation Progress')
    plt.tight_layout()
    plt.savefig(f"{batch}_training_curve.pdf")
    plt.show()
    plt.close()

plot_training_progress(train_losses, val_losses, train_accs, val_accs)





