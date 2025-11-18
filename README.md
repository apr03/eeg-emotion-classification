# EEG Emotion Classification - Machine Learning on Brain Signals

A complete end-to-end project that classifies **human emotions from EEG brainwave signals** using signal processing, feature extraction, and machine learning.  
This repository includes the processing pipeline, visualizations, MATLAB analysis, and plots used to interpret EEG signals.

---

## 🔧 Tech Stack

- **Python:** NumPy, SciPy, Matplotlib, scikit-learn  
- **MATLAB:** Signal processing & spectral analysis  
- **Machine Learning Models:** SVM, Logistic Regression, Random Forest  
- **Feature Extraction:** Statistical + Frequency domain  
- **Visualization:** Confusion matrix, t-SNE, PSD, correlation heatmaps  

---

## 🚀 Features

### 📥 EEG Data Handling
- Supports multi-channel EEG datasets (DEAP/DREAMER compatible)
- Preprocessing pipeline for filtering and noise reduction
- MATLAB script (`analysis.m`) for deep signal visualization

---

### 🧹 Preprocessing Pipeline
- Band-pass filtering  
- Normalization  
- Artifact/Noise reduction  
- Epoch segmentation  
- Label mapping  

---

### 📊 Feature Engineering

Extracts both **time-domain** and **frequency-domain** features:

- Mean, variance, RMS  
- Alpha/Beta/Theta/Delta band power  
- Power Spectral Density (PSD)  
- Statistical descriptors  
- Channel correlation analysis  

---

### 🤖 Machine Learning Models

Implements:

- Training and testing of models  
- Performance metrics: Accuracy, F1-score  
- Confusion matrix generation  
- Feature importance and ranking  

Models can be extended easily.

---

### 🔍 Visualization & Analysis

Includes multiple visual outputs:

- Confusion Matrix  
- t-SNE Visualization  
- Power Spectral Density (PSD) graphs  
- Correlation Heatmaps  
- Emotion distribution pie charts  
- Significant feature contribution plots  

---

### 📘 MATLAB Support

`analysis.m` provides:

- Frequency band visualization  
- Cross-verification of Python outputs  
- Spectral and statistical analysis  
- Electrode-wise comparison  

---

## 🗂 Repository Structure
```bash
├── main.py # Python pipeline
├── analysis.m # MATLAB EEG analysis
├── check_matlab.py # Cross-check MATLAB output
├── requirements.txt # Dependencies
├── *.png # Generated visualizations
└── README.md # Documentation
```

## ▶️ How to Run

### 1. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 2. Add EEG Dataset
- Create a ``` data/ ``` folder and place your EEG dataset inside. 
- Modify ``` main.py ``` to ensure correct data path.


### 3. Run the Pipeline
```bash
python main.py
```

## 🎯 Results

The project provides insights into:

- EEG signal variation across emotions  
- Dominant frequency bands responsible for emotional states  
- Model performance and confusion matrix  
- Feature-space clustering using t-SNE  
- Channel-wise contribution to prediction
