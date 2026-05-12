import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__, template_folder='../templates')

IMG_SIZE = (128, 128)
MODEL_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\models\siamese_model.pth"
DB_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\models\embeddings.json"
THRESHOLD = 0.5

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

# Load model once at startup
model = SiameseNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ── Helpers ───────────────────────────────────────────────────────────────
def preprocess(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

def get_embedding(img_tensor):
    with torch.no_grad():
        return model.embedding(img_tensor).squeeze().numpy().tolist()

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enroll", methods=["POST"])
def enroll():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    files = request.files.getlist("signatures")
    if len(files) < 2:
        return jsonify({"error": "at least 2 reference signatures required"}), 400

    embeddings = []
    for file in files:
        img_tensor = preprocess(file)
        emb = get_embedding(img_tensor)
        embeddings.append(emb)

    db = load_db()
    db[user_id] = embeddings
    save_db(db)

    return jsonify({
        "message": f"User '{user_id}' enrolled successfully",
        "signatures_stored": len(embeddings)
    })

@app.route("/verify", methods=["POST"])
def verify():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    file = request.files.get("signature")
    if not file:
        return jsonify({"error": "signature image required"}), 400

    db = load_db()
    if user_id not in db:
        return jsonify({"error": f"User '{user_id}' not enrolled"}), 404

    query_tensor = preprocess(file)
    query_emb = torch.tensor(get_embedding(query_tensor))

    distances = []
    for ref_emb in db[user_id]:
        ref_tensor = torch.tensor(ref_emb)
        dist = torch.nn.functional.pairwise_distance(
            query_emb.unsqueeze(0),
            ref_tensor.unsqueeze(0)
        ).item()
        distances.append(dist)

    avg_distance = sum(distances) / len(distances)
    match = avg_distance < THRESHOLD
    similarity = max(0, 100 - (avg_distance / THRESHOLD) * 100)

    return jsonify({
        "user_id": user_id,
        "match": match,
        "similarity": round(similarity, 2),
        "avg_distance": round(avg_distance, 4),
        "threshold": THRESHOLD,
        "result": "GENUINE" if match else "FORGED"
    })

@app.route("/users", methods=["GET"])
def users():
    db = load_db()
    return jsonify({"enrolled_users": list(db.keys())})

if __name__ == "__main__":
    app.run(debug=True, port=5000)