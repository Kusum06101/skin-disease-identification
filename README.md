# Skin Disease Identification using Deep Learning
Live Demo: https://huggingface.co/spaces/KusumC/skin-disease-website


A web-based deep learning application that classifies 31 skin diseases from medical images and provides Ayurvedic treatment recommendations.

---

## Overview

This project combines computer vision and healthcare AI to assist early detection of skin diseases. Users upload an image, and the system predicts the disease and suggests holistic Ayurvedic remedies.

---

## Features

- Multi-class CNN-based skin disease classification
- 31 disease categories
- 95.57% validation accuracy
- Real-time web interface using FastAPI
- Ayurvedic herbal, dietary, and lifestyle recommendations
- Robust preprocessing and augmentation pipeline

---

## Tech Stack

- Python
- PyTorch
- FastAPI
- Computer Vision
- Deep Learning

---

## Project Structure

```
skin-disease-identification/
│
├── app/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Demo Screenshots

### Homepage
![Homepage](results/screenshots/homepage.png)

### Prediction Example 1
![Prediction 1](results/screenshots/prediction1.png)

### Prediction Example 2
![Prediction 2](results/screenshots/prediction2.png)

### Prediction Example 3
![Prediction 3](results/screenshots/prediction3.png)

### System Workflow
![Workflow](results/screenshots/workflow.png)

---

## How to Run the Project

```
git clone https://github.com/YOUR-USERNAME/skin-disease-identification.git
cd skin-disease-identification
pip install -r requirements.txt
python app/main.py
```

Open your browser and go to:

```
http://127.0.0.1:8000
```

---

## Results

Model achieved **95.57% validation accuracy** across diverse skin tones and lighting conditions.

---

## Authors

Kusum C and research team

---

## Future Work

- Mobile deployment
- Explainable AI integration
- Larger dataset expansion
