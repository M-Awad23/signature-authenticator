import cv2
import numpy as np
import torch
import torch.nn as nn
import sys

IMG_SIZE = (128, 128)
MODEL_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\models\siamese_model.pth"
THRESHOLD = 0.5

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.features(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingNet()

    def forward(self, img1, img2):
        return self.embedding(img1), self.embedding(img2)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not load {path}")
        sys.exit(1)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

def predict(img1_path, img2_path):
    model = SiameseNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    with torch.no_grad():
        e1, e2 = model(img1, img2)
        dist = torch.nn.functional.pairwise_distance(e1, e2).item()
    
    print(f"Raw distance: {dist:.6f}")

    match = dist < THRESHOLD
    result = "✅ MATCH — Same signer" if match else "❌ NO MATCH — Different signer or forged"
    print(f"Distance: {dist:.4f} (threshold: {THRESHOLD})")
    print(f"Result: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python demo_siamese.py <reference_image> <query_image>")
    else:
        predict(sys.argv[1], sys.argv[2])