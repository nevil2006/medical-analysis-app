import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import clip
import pandas as pd
from PIL import Image

# -------------------
# Paths
# -------------------
train_path = r"C:\Users\nevil\Desktop\VLM RESEARCH\TRAIN_RESIZED"
valid_path = r"C:\Users\nevil\Desktop\VLM RESEARCH\VALD_RESIZED"
test_path  = r"C:\Users\nevil\Desktop\VLM RESEARCH\TEST_RESIZED"

train_csv = r"C:\Users\nevil\Desktop\VLM RESEARCH\train.csv"
valid_csv = r"C:\Users\nevil\Desktop\VLM RESEARCH\valid.csv"
test_csv  = r"C:\Users\nevil\Desktop\VLM RESEARCH\test.csv"

model_save_path = r"C:\Users\nevil\Desktop\VLM RESEARCH\best_clip_model.pth"

# -------------------
# Device
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------
# Load CLIP
# -------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze CLIP visual encoder by default
for param in clip_model.visual.parameters():
    param.requires_grad = False

# -------------------
# Datasets & Loaders
# -------------------
train_data = datasets.ImageFolder(train_path, transform=preprocess)
valid_data = datasets.ImageFolder(valid_path, transform=preprocess)
test_data  = datasets.ImageFolder(test_path, transform=preprocess)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = train_data.classes
print("Subtype classes:", class_names)

# -------------------
# Mapping subtypes -> main types
# -------------------
benign_subtypes = ["ADENOSIS", "Fibro adenoma", "Phyllodes-tumor", "Tubular Adenoma"]
malignant_subtypes = ["Ductal_Carcinoma", "Lobular_Carcinoma", "Mucinous_Carcinoma", "Papillary_carcinoma"]

def get_main_type(subtype):
    if subtype in benign_subtypes:
        return "benign"
    elif subtype in malignant_subtypes:
        return "malignant"
    else:
        return "unknown"

# -------------------
# Classifier Head
# -------------------
class ClipClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

classifier = ClipClassifier(clip_model.visual.output_dim, len(class_names)).to(device)

# -------------------
# Class weighting for imbalanced dataset
# -------------------
label_counts = [len([lbl for _, lbl in train_data.samples if lbl == i]) for i in range(len(class_names))]
class_weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in label_counts], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

# -------------------
# Feature Extraction
# -------------------
def extract_features(images):
    with torch.no_grad():  # keep CLIP frozen
        features = clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float()  # ensure float32
    return features

# -------------------
# Evaluation + CSV Export
# -------------------
def evaluate(loader, split="Test", save_csv=None):
    classifier.eval()
    correct, total = 0, 0
    records = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            features = extract_features(images)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                true_subtype = class_names[labels[i].item()]
                pred_subtype = class_names[preds[i].item()]
                main_type = get_main_type(pred_subtype)

                img_path = loader.dataset.samples[batch_idx*loader.batch_size + i][0]

                records.append({
                    "filename": os.path.basename(img_path),
                    "true_subtype": true_subtype,
                    "pred_subtype": pred_subtype,
                    "pred_main_type": main_type
                })

    acc = 100 * correct / total
    print(f"{split} Accuracy: {acc:.2f}%")

    if save_csv:
        pd.DataFrame(records).to_csv(save_csv, index=False)
        print(f"Saved predictions to {save_csv}")

    return acc

# -------------------
# Training Loop with Early Stopping & Model Save
# -------------------
def train_model(epochs=50, patience=10):
    best_acc = 0
    counter = 0

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            features = extract_features(images)
            outputs = classifier(features)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        val_acc = evaluate(valid_loader, "Validation")

        # Check for improvement
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(classifier.state_dict(), model_save_path)
            print(f"Validation accuracy improved. Model saved to {model_save_path}")
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

# -------------------
# Train the model
# -------------------
train_model()

# -------------------
# Load & Final Evaluations & CSV Save
# -------------------
classifier.load_state_dict(torch.load(model_save_path))
evaluate(train_loader, "Train", train_csv)
evaluate(valid_loader, "Validation", valid_csv)
evaluate(test_loader, "Test", test_csv)

# -------------------
# Predict Single Image
# -------------------
def predict_image(image_path):
    classifier.eval()
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = extract_features(image)
        outputs = classifier(features)
        _, pred = torch.max(outputs, 1)
        pred_subtype = class_names[pred.item()]
        pred_main_type = get_main_type(pred_subtype)
    print(f"Image: {image_path}")
    print(f"Predicted Subtype: {pred_subtype}")
    print(f"Predicted Main Type: {pred_main_type}")
    return pred_subtype, pred_main_type

# -------------------
# Example usage
# -------------------
test_image_path = input("Enter path of the image to predict: ")
predict_image(test_image_path)
