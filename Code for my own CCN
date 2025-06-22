# Imports and environment setup
# ========================================
import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ========================================
# 1. Custom Dataset for Cassava Leaf Classification
# ========================================
class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Expects a CSV with columns [image_name, label], and a root directory of images.
    """
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.annotations = pd.read_csv(csv_file)  # DataFrame with columns: filename, label
        self.root_dir = root_dir                  # Directory where images are stored
        self.transform = transform                # torchvision.transforms for augmentation / normalization

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image path and label
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        return image, label

# ========================================
# 2. Prepare data transforms and DataLoaders
# ========================================
# Paths (modify as needed)
csv_train = '/content/cassava_data/train.csv'
img_dir = '/content/cassava_data/train_images'

# Train transforms: augmentations + resize + to tensor + normalize
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation transforms: resize + to tensor + normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create full dataset with train transforms initially
full_dataset = CassavaDataset(csv_file=csv_train, root_dir=img_dir, transform=train_transform)

# Split into train and validation subsets (90% train, 10% val)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# For validation, we want to apply val_transform instead of train_transform.
class WrappedSubset(Dataset):
    """
    Wrap a Subset to apply a different transform (e.g., validation transforms).
    """
    def __init__(self, subset, base_dataset: CassavaDataset, transform):
        self.subset = subset
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        img_name = self.base_dataset.annotations.iloc[orig_idx, 0]
        label = int(self.base_dataset.annotations.iloc[orig_idx, 1])
        img_path = os.path.join(self.base_dataset.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# Create train and val datasets/loaders
train_dataset = train_subset  # uses train_transform via full_dataset
val_dataset = WrappedSubset(val_subset, full_dataset, val_transform)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 40

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Device: {device}, Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# ========================================
# 3. Compute class weights for imbalanced data
# ========================================
# Extract labels from the training subset to compute class counts
train_labels = []
for _, label in train_dataset:
    train_labels.append(label)
label_series = pd.Series(train_labels)
class_counts = label_series.value_counts().sort_index().values.astype(float)  # array of counts per class
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
class_weights = class_weights.to(device)
print(f"Class counts: {class_counts}, Class weights: {class_weights}")

# ========================================
# 4. Define Inception block and custom CNN model
# ========================================
class InceptionBlock(nn.Module):
    """
    Inception block with four branches:
      1) 1x1 convolution
      2) 1x1 convolution -> 3x3 convolution
      3) 1x1 convolution -> 5x5 convolution
      4) 3x3 maxpool -> 1x1 convolution
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super().__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 1x1 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)

class CustomInceptionCNN(nn.Module):
    """
    Custom CNN with multiple Inception blocks and convolutional layers.
    """
    def __init__(self, input_channels: int, hidden_units: int, num_classes: int):
        super().__init__()
        # Initial conv block
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        # Inception block 1
        self.inception1 = InceptionBlock(in_channels=hidden_units,
                                         out_1x1=32, red_3x3=32, out_3x3=64,
                                         red_5x5=32, out_5x5=16, pool_proj=16)
        # Following conv blocks
        self.layer2 = nn.Sequential(
            nn.Conv2d(32+64+16+16, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        # Inception block 2
        self.inception2 = InceptionBlock(in_channels=hidden_units,
                                         out_1x1=32, red_3x3=32, out_3x3=64,
                                         red_5x5=32, out_5x5=16, pool_proj=16)
        self.layer4 = nn.Sequential(
            nn.Conv2d(32+64+16+16, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        # Inception block 3
        self.inception3 = InceptionBlock(in_channels=hidden_units,
                                         out_1x1=32, red_3x3=32, out_3x3=64,
                                         red_5x5=32, out_5x5=16, pool_proj=16)
        self.layer5 = nn.Sequential(
            nn.Conv2d(32+64+16+16, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        # Classifier
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.inception1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.inception2(x)
        x = self.layer4(x)
        x = self.inception3(x)
        x = self.layer5(x)
        # Global average pooling to 1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========================================
# 5. Initialize model, loss, optimizer, scheduler
# ========================================
input_channels = 3
hidden_units = 128
num_classes = len(class_counts)  # e.g., 5 for Cassava

model = CustomInceptionCNN(input_channels, hidden_units, num_classes).to(device)

# Loss with class weights for imbalance
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.001)

# LR scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# ========================================
# 6. Training and validation loop
# ========================================
epochs = 12
best_val_f1 = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1s = []
val_f1s = []

for epoch in range(1, epochs + 1):
    # ---- Training phase ----
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if batch_idx % 300 == 0:
            seen = batch_idx * images.size(0)
            total = len(train_loader.dataset)
            print(f"Epoch {epoch}, processed {seen}/{total} samples")

    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='macro')

    # ---- Validation phase ----
    model.eval()
    val_running_loss = 0.0
    val_labels = []
    val_preds = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_running_loss += loss.item()

            preds = outputs.softmax(dim=1).argmax(dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    # Logging
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save model if improved
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model updated!")

    # Step LR scheduler
    scheduler.step()

    # Record metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

# ========================================
# 7. Plot training curves
# ========================================
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(18, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accuracies, label='Val Accuracy', marker='o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# F1 Score plot
plt.subplot(1, 3, 3)
plt.plot(epochs_range, train_f1s, label='Train F1 Score', marker='o')
plt.plot(epochs_range, val_f1s, label='Val F1 Score', marker='o')
plt.title('F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
