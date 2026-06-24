# Face Recognition using Siamese Neural Networks

A deep learning based face recognition system developed using Siamese Neural Networks and PyTorch for image similarity learning and identity verification.

## Overview

Traditional face classification systems require retraining whenever a new identity is added. Siamese Networks overcome this limitation by learning similarity between image pairs rather than classifying identities directly.

This project implements a Siamese Neural Network that learns facial embeddings and determines whether two face images belong to the same person.

---

## Objectives

- Learn facial similarity using deep neural networks.
- Perform identity verification rather than classification.
- Build a scalable face recognition system.
- Explore metric learning using Siamese Networks.

---

## Architecture

The system consists of:

1. Twin Convolutional Neural Networks
2. Shared weights between both branches
3. Feature embedding generation
4. L1 distance computation
5. Binary similarity prediction

```text
Face Image A
       │
       ▼
Embedding Network
       │
       ▼
Feature Vector A
                \
                 > L1 Distance → Fully Connected Layer → Similarity Score
                /
Feature Vector B
       ▲
       │
Embedding Network
       ▲
       │
Face Image B
```

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- PIL
- Torchvision

---

## Dataset Preparation

The dataset contains:

- Anchor Images
- Positive Images (same person)
- Negative Images (different person)

Image preprocessing:

- Resize to 100 × 100
- Tensor conversion
- Normalization

---

## Model Details

### Embedding Network

A convolutional neural network is used to generate feature embeddings for face images.

### Similarity Learning

Similarity is computed using:

```python
torch.abs(embedding1 - embedding2)
```

which acts as an L1 distance layer.

### Classification Layer

The distance vector is passed through:

```python
Linear(4096, 1)
```

to predict whether two faces belong to the same identity.

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Framework | PyTorch |
| Epochs | 30 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Image Size | 100 × 100 |

---

## Evaluation Metrics

The model was evaluated using:

- Precision
- Recall
- Validation Loss

Model checkpoints were automatically saved whenever validation loss improved.

---

## Project Structure

```text
Face-Recognition-Siamese-CNN/

README.md

requirements.txt

Face Recognition(Pytorch).ipynb
```

---

## Installation

```bash
git clone https://github.com/MokshGujar/Face-Recognition-Siamese-CNN.git

cd Face-Recognition-Siamese-CNN

pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Face Recognition(Pytorch).ipynb
```

Run all cells to:

- Load dataset
- Train Siamese Network
- Evaluate performance
- Save best model

---

## Applications

- Face Verification
- Attendance Systems
- Access Control
- Authentication Systems
- Security Applications

---

## Future Improvements

- Triplet Loss implementation
- FaceNet architecture
- ArcFace embeddings
- Real-time webcam inference
- Deployment with FastAPI

---

## Author

### Moksh Gujar

B.Sc. Data Science  
M.Sc. Data Science (Pursuing)

Areas of Interest:

- Machine Learning
- Deep Learning
- Computer Vision
- Natural Language Processing
- Generative AI

Portfolio:

https://moksh-gujar-portfolio.netlify.app
