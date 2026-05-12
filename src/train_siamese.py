import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\data\signatures"
MODEL_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\models\siamese_model.pth"
IMG_SIZE = (128, 128)
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0001

# ── Load Data ─────────────────────────────────────────────────────────────
def load_data(data_path):
    data = {}  # signer_id -> {"genuine": [], "forged": []}
    for signer_folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, signer_folder)
        if not os.path.isdir(folder_path):
            continue
        signer_id = signer_folder
        data[signer_id] = {"genuine": [], "forged": []}
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            if filename.startswith("original"):
                data[signer_id]["genuine"].append(img)
            elif filename.startswith("forgeries"):
                data[signer_id]["forged"].append(img)
    return data

# ── Build Pairs ───────────────────────────────────────────────────────────
def build_pairs(data, signers):
    pairs, labels = [], []
    signer_list = [s for s in signers if len(data[s]["genuine"]) >= 2]

    for signer in signer_list:
        genuines = data[signer]["genuine"]
        forged = data[signer]["forged"]

        # Positive pairs — two genuine from same signer
        for img1, img2 in combinations(genuines, 2):
            pairs.append((img1, img2))
            labels.append(1)

        # Negative pairs — genuine vs forged (same signer)
        for g, f in zip(genuines, forged):
            pairs.append((g, f))
            labels.append(0)

    # Negative pairs — genuine from different signers
    for i, s1 in enumerate(signer_list):
        for s2 in signer_list[i+1:i+3]:
            g1 = random.choice(data[s1]["genuine"])
            g2 = random.choice(data[s2]["genuine"])
            pairs.append((g1, g2))
            labels.append(0)

    return pairs, labels

# ── Dataset ───────────────────────────────────────────────────────────────
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        img1 = torch.tensor(img1).unsqueeze(0)
        img2 = torch.tensor(img2).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

# ── Model ─────────────────────────────────────────────────────────────────
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, 64)  # 64-dim embedding
        )

    def forward(self, x):
        return self.features(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingNet()

    def forward(self, img1, img2):
        e1 = self.embedding(img1)
        e2 = self.embedding(img2)
        return e1, e2

# ── Contrastive Loss ──────────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        dist = torch.nn.functional.pairwise_distance(e1, e2)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()

# ── Train ─────────────────────────────────────────────────────────────────
print("Loading data...")
data = load_data(DATA_PATH)
signers = list(data.keys())

# Split signers
random.seed(42)
random.shuffle(signers)
split = int(0.8 * len(signers))
train_signers = signers[:split]
test_signers = signers[split:]

print("Building pairs...")
train_pairs, train_labels = build_pairs(data, train_signers)
test_pairs, test_labels = build_pairs(data, test_signers)
print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")

train_dataset = SiameseDataset(train_pairs, train_labels)
test_dataset = SiameseDataset(test_pairs, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SiameseNet().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        e1, e2 = model(img1, img2)
        loss = criterion(e1, e2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────
model.eval()
correct = 0
total = 0
THRESHOLD = 0.5

with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2 = img1.to(device), img2.to(device)
        e1, e2 = model(img1, img2)
        dist = torch.nn.functional.pairwise_distance(e1, e2)
        pred = (dist < THRESHOLD).float()
        correct += (pred == label.to(device)).sum().item()
        total += label.size(0)

print(f"\nTest Accuracy: {correct/total*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved.")