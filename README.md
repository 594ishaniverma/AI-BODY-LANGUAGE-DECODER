# 🧍 AI Body Language Decoder

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-orange.svg)](https://google.github.io/mediapipe/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-yellow.svg)](https://scikit-learn.org/)

## 📖 Overview
The AI Body Language Decoder is a real-time behavioral classification system designed to interpret human postures, gestures, and emotional states. By capturing live video feeds, the system extracts complex spatial coordinate data and feeds it into a trained machine learning model to instantly classify the user's body language.

## ✨ Key Features
* **Comprehensive Tracking:** Simultaneously tracks over 500 3D landmarks across the pose, face, and both hands using MediaPipe Holistic.
* **Real-Time Classification:** Processes live webcam frames with extremely low latency to provide instant behavioral feedback.
* **Custom ML Pipeline:** Features a fully customizable data collection and training pipeline using Pandas and Scikit-Learn, allowing users to train the model on entirely new body language states.

## 🛠️ Technology Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe Holistic
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest / Ridge Classifier)

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/594ishaniverma/AI-BODY-LANGUAGE-DECODER.git](https://github.com/594ishaniverma/AI-BODY-LANGUAGE-DECODER.git)
cd AI-BODY-LANGUAGE-DECODER