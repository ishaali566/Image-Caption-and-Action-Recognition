# Image Caption and Action Recognition (DLA3 Project)

## Project Overview
This project implements a **hybrid deep learning pipeline** for:

1. **Image Captioning** – Generating human-readable captions for images.  
2. **Action Recognition** – Detecting and classifying human actions in images.

The project is designed for research and academic purposes and is structured for easy experimentation.  
> Note: This repository contains **code only**. Datasets and pre-trained models are **not included** to keep the repo lightweight.

### Module Details

1. **Image Captioning Module**  
   - Uses **InceptionV3 (CNN)** for visual feature extraction.  
   - LSTM networks generate captions from extracted features.  
   - Trained on **Flickr8k** dataset (8,000 images, 5 captions each).  

2. **Action Recognition Module**  
   - Uses **MobileNetV2 (CNN)** with a 3-layer dense classifier.  
   - Detects **40 human actions** using the **Stanford 40 Actions dataset**.  
   - Achieves **85-88% accuracy**, with **95-97% top-3 accuracy**.

---

## Technical Implementation

- Backend: **Flask (Python)** providing a REST API for inference.  
- Frontend: Simple web interface for **image upload** and **real-time predictions**.  
- Deep Learning Frameworks: **TensorFlow/Keras** with transfer learning from ImageNet-pretrained networks.  

---

Architecture Highlights

Caption Model: InceptionV3 + LSTM

Visual feature vectors: 2048 dimensions

LSTM units: 512

Action Model: MobileNetV2 + Dense classifier (3 layers)

Evaluation Metrics:

Image Captioning: BLEU/METEOR/ROUGE-L scores → overall accuracy ~45%

Action Recognition: Accuracy ~76%
---

## Directory Structure
DLA3/
├── backend/
│ ├── app.py # Flask backend
│ └── requirements.txt # Dependencies
├── frontend/
│ ├── index.html # Web interface
│ ├── script.js
│ └── style.css
├── models/
│ ├── action_recognition_model.py
│ ├── image_caption_model.py
│ └── saved_models/ # NOT included in repo
├── main.py # Pipeline runner
├── train_action_model.py
├── train_caption_model.py
└── README.md


---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/ishaali566/Image-Caption-and-Action-Recognition.git
cd Image-Caption-and-Action-Recognition

python -m venv .venv
source .venv/Scripts/activate     # Windows
# OR
source .venv/bin/activate         # Linux / Mac

pip install -r backend/requirements.txt

python backend/app.py

Open browser at: http://127.0.0.1:5000/

Upload an image to get caption + action predictions.

Train Models

Action Recognition

python train_action_model.py


Image Captioning

python train_caption_model.py


Make sure datasets are downloaded and available locally.
Pre-trained models are not included in this repository.

