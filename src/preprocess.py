import os
import cv2
import numpy as np

DATA_PATH = r"C:\Users\ADMIN\Desktop\Projects\signature-auth\data\signatures"
IMG_SIZE = (128, 128)
def load_images(data_path):
    images = []
    labels = []

    for signer_folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, signer_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)

            if filename.startswith("original"):
                label = 1  # genuine
            elif filename.startswith("forgeries"):
                label = 0  # forged
            else:
                continue

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = preprocess(img)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def preprocess(img):
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    # Otsu thresholding — converts to binary
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Normalize to 0-1
    img = img / 255.0
    return img

if __name__ == "__main__":
    images, labels = load_images(DATA_PATH)
    print(f"Loaded {len(images)} images")
    print(f"Genuine: {sum(labels)} | Forged: {len(labels) - sum(labels)}")

    from features import extract_hog_features
    features = extract_hog_features(images)
    print(f"Feature vector shape: {features.shape}")