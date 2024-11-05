# FacialOSA
This project implements a deep learning-based classification system for detecting Obstructive Sleep Apnea (OSA) using facial landmarks and features. The system utilizes computer vision techniques to extract facial features and a neural network to classify subjects as either OSA or normal cases.
## Features
- Facial landmark detection using dlib
- Neck contour estimation
- Binary classification (OSA vs Normal)
- Real-time visualization of facial features
- Confidence score prediction
- Interactive Jupyter notebook implementation

## Prerequisites
python
# Required Python packages
opencv-python>=4.5.0
dlib>=19.22.0
tensorflow>=2.5.0
numpy>=1.19.5
scikit-learn>=0.24.0
matplotlib>=3.3.0
Installation

Clone the repository:

bashCopygit clone [your-repository-url]
cd [repository-name]

Install required packages:

bashCopypip install -r requirements.txt

Download the shape predictor file:


Download shape_predictor_68_face_landmarks.dat from dlib's official website
Place it in the project root directory
