# Signature Authenticator

Identity-aware handwritten signature verification using a Siamese CNN.

## Overview

Most signature classifiers predict genuine vs. forged in isolation — without knowing whose signature it's supposed to be. This project takes a different approach: given a query signature and a registered user, it asks **"does this signature belong to this person?"**

Built with PyTorch and served via a Flask REST API with a web interface.

## How it works

1. **Enroll** a user with 2+ reference signatures
2. **Verify** a query signature against their stored embeddings
3. The model computes the distance between embeddings — small distance = same person

## Model

- Architecture: Siamese CNN with contrastive loss
- Dataset: CEDAR (55 signers, 24 signatures each — 12 genuine, 12 forged)
- Split: by signer — test signers never seen during training
- Accuracy: **88.19%** on unseen signers

## Stack

- PyTorch — model training
- OpenCV + scikit-image — preprocessing
- Flask — REST API
- Vanilla HTML/CSS/JS — web interface

## API