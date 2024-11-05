# Facial Feature-Based OSA Classification System

## Overview
This project implements a deep learning-based classification system for detecting Obstructive Sleep Apnea (OSA) using facial landmarks and features. The system utilizes computer vision techniques to extract facial features and a neural network to classify subjects as either OSA or normal cases.

## Features
* Facial landmark detection using dlib
* Neck contour estimation
* Binary classification (OSA vs Normal)
* Real-time visualization of facial features
* Confidence score prediction
* Interactive Jupyter notebook implementation

## Prerequisites
* opencv-python>=4.5.0
* dlib>=19.22.0
* tensorflow>=2.5.0
* numpy>=1.19.5
* scikit-learn>=0.24.0
* matplotlib>=3.3.0

## Installation
1. Clone the repository:
```
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Download the shape predictor file:
* Download `shape_predictor_68_face_landmarks.dat` from dlib's official website
* Place it in the project root directory

## Project Structure
```
├── data/
│   ├── OSA/
│   └── normal/
├── models/
│   └── osa_classifier.h5
├── notebooks/
│   └── OSA_Classification.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Usage
1. Data Preparation:
   * Organize your dataset into OSA and normal categories
   * Place images in respective folders under `data/`

2. Training the Model:
   * Open `notebooks/OSA_Classification.ipynb`
   * Follow the step-by-step instructions for:
     * Feature extraction
     * Model training
     * Evaluation

3. Making Predictions:
```python
from src.feature_extraction import process_image

image_path = 'path/to/your/image.jpg'
result_image, prediction = process_image(image_path)
```

## Model Architecture
* Input: 140-dimensional vector (68 facial landmarks + 2 neck points)
* Hidden layers: 128 neurons (ReLU) → Dropout(0.3) → 64 neurons (ReLU) → Dropout(0.3)
* Output: Single neuron (Sigmoid) for binary classification

## Results Visualization
The system provides visual feedback including:
* Facial landmark points (green)
* Key facial feature points (red)
* Neck contour line (blue)
* Classification result with confidence score


## Acknowledgments
* Facial landmark detection using dlib
* Deep learning implementation using Pipeline and XGboost



## References
1. [Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
2. [Face landmark detection guide](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker))
