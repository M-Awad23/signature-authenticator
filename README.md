# Signature Authenticator

Identity-aware handwritten signature verification using a Siamese CNN.

## Overview

This project implements offline signature verification — given a query signature and a set of reference signatures from an enrolled user, the system determines whether the query belongs to the same person.

Unlike simple classifiers that label signatures as "genuine" or "forged" in isolation, this system compares signatures against a specific enrolled identity, making it resistant to cross-identity spoofing.

## Results

| Model | Accuracy |
|-------|----------|
| HOG + SVM (baseline) | 84.47% |
| CNN (standalone) | 99.24%* |
| Siamese CNN (identity-aware) | 88.19% |

*The standalone CNN result is inflated due to the nature of the task — the Siamese model is the correct formulation.

## Architecture

- **Embedding network**: 3-layer CNN → 64-dimensional signature embedding
- **Loss function**: Contrastive loss with margin = 1.0
- **Training split**: By signer (no signer appears in both train and test)
- **Dataset**: [CEDAR Signature Database](http://www.cedar.buffalo.edu/NIJ/data/) — 55 signers, 24 signatures each (12 genuine + 12 forged)

## Project Structure
signature-auth/
├── src/
│   ├── preprocess.py       # data loading and preprocessing
│   ├── features.py         # HOG feature extraction (baseline)
│   ├── train_siamese.py    # Siamese CNN training
│   ├── demo_siamese.py     # CLI demo
│   └── app.py              # Flask REST API
└── templates/
└── index.html          # web interface

## API

**POST /enroll** — register a user with reference signatures  
**POST /verify** — verify a query signature against an enrolled user  
**GET /users** — list all enrolled users

## Setup

Install dependencies:

    pip install torch torchvision opencv-python scikit-learn scikit-image flask numpy

Download the CEDAR dataset and place it under `data/signatures/`.

Train the model:

    cd src && python train_siamese.py

Run the API:

    python app.py

Open `http://127.0.0.1:5000` in your browser.

## Tech Stack

Python · PyTorch · OpenCV · scikit-learn · Flask