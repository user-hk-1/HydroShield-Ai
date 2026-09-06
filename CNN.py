import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Ensure UTF-8 output on Windows with immediate flushing
sys.stdout.reconfigure(encoding='utf-8')

# --- Configurations ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 12     # Warmup classification head
EPOCHS_STAGE2 = 15     # Fine-tune upper conv layers
TRAIN_DIR = 'data/Train'
MODEL_SAVE_PATH = 'rapid_model.pth'

# Detect GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_and_train_model():
    print(f"🚀 Initializing PyTorch training engine on device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"🔥 Active GPU Accelerator: {torch.cuda.get_device_name(0)}", flush=True)

    # Data Transforms (Augmentation + Normalization)
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Separate dataset objects for train and validation transforms
    train_dataset_full = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=TRAIN_DIR, transform=val_transform)
    
    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    print(f"Detected Classes: {class_names} ({num_classes} classes)", flush=True)

    # Generate deterministic 80/20 train/val indices
    num_images = len(train_dataset_full)
    indices = list(range(num_images))
    torch.manual_seed(123)
    permuted_indices = torch.randperm(num_images).tolist()
    
    train_size = int(0.8 * num_images)
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load Pre-trained MobileNetV2 Architecture
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze base feature extractor layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head for custom classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_stage1 = optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_acc = 0.0

    print("\n🔥 STAGE 1: Training Classification Head on GPU (12 Epochs)...", flush=True)
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_stage1.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        train_acc = (train_correct / len(train_indices)) * 100
        val_acc = (val_correct / len(val_indices)) * 100
        print(f"Epoch {epoch:02d}/{EPOCHS_STAGE1:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n⚡ STAGE 2: Unfreezing Upper Conv Layers for GPU Fine-Tuning...", flush=True)
    # Unfreeze top feature layers (blocks 10+)
    for param in model.features[10:].parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_stage2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        train_acc = (train_correct / len(train_indices)) * 100
        val_acc = (val_correct / len(val_indices)) * 100
        print(f"Epoch {epoch:02d}/{EPOCHS_STAGE2:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n✅ Training & Fine-Tuning Complete! Best model saved as '{MODEL_SAVE_PATH}'.", flush=True)
    print(f"🏆 Peak Validation Accuracy on GPU: {best_val_acc:.2f}%", flush=True)

if __name__ == "__main__":
    build_and_train_model()