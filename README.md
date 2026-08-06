# American Sign Language Translator (ASLT)

A real-time American Sign Language (ASL) fingerspelling recognition application built with **PyTorch**, **MediaPipe**, **OpenCV**, and **PySide6**. The system captures live webcam input, extracts hand landmarks using MediaPipe, classifies ASL letters with a neural network, and displays predictions through an interactive desktop interface.

---

## Features

- Real-time webcam capture
- MediaPipe hand landmark detection
- Wrist-centered, scale-normalized landmark preprocessing
- PyTorch neural network for ASL letter classification
- Live confidence scoring
- Top-2 prediction display
- Word construction interface
- Backspace support for correcting predicted letters
- Desktop GUI built with PySide6
- GPU acceleration (CUDA supported)

---

## Demo

### Live Webcam Recognition

- Detects a single hand in real time
- Predicts ASL fingerspelled letters
- Displays:
  - Current prediction
  - Alternative prediction
  - Confidence scores
  - Constructed word

---

## Project Structure

```text
ASLT/
│
├── gui.py                 # Main application entry point
├── server.py              # Model loading and inference
├── model.py               # Neural network architecture
├── mpipeline.py           # MediaPipe landmark extraction
├── config.py              # Configuration settings
├── class_names.py         # Label mappings
├── style.qss              # Qt stylesheet
├── checkpoint.pth         # Trained model
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Webcam
      │
      ▼
OpenCV Video Capture
      │
      ▼
MediaPipe Hands
      │
      ▼
21 Hand Landmarks
      │
      ▼
Landmark Normalization
(Wrist-centered + Scale Normalized)
      │
      ▼
PyTorch Classifier
      │
      ▼
Top-2 Predictions
      │
      ▼
PySide6 Desktop Interface
```

---

## Landmark Preprocessing

Instead of classifying raw images, the model operates directly on **MediaPipe hand landmarks**.

Each hand consists of:

- 21 landmarks
- x, y, z coordinates

Resulting in a **63-dimensional feature vector**.

To improve robustness, landmarks are:

- centered relative to the wrist
- normalized by hand size
- independent of camera position
- more resistant to user-to-user variation

---

## Technologies

- Python
- PyTorch
- MediaPipe
- OpenCV
- PySide6 (Qt)
- NumPy

---

## Installation

Clone the repository

```bash
git clone https://github.com/DukeOrji/ASLT.git
cd ASLT
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python gui.py
```

---

## Controls

| Key / Button | Action |
|--------------|--------|
| **Confirm** | Capture current hand sign |
| **Backspace** | Remove last predicted letter |

---

## Model

The classifier predicts one of the **26 ASL alphabet letters** and SPACE.

Input:

- 63 normalized landmark values

Output:

- Predicted letter
- Prediction confidence
- Second-best prediction

---

## Current Capabilities

- ASL alphabet recognition
- Live webcam inference
- Confidence estimation
- Interactive GUI
- Word construction
- Prediction correction

---

## Future Improvements

- Continuous recognition without manual confirmation
- Automatic word segmentation
- Sentence construction
- Language model for autocorrection
- Dynamic gesture recognition (J and Z)
- Custom dataset collection interface
- Model quantization for faster inference
- Export to ONNX / TensorRT
- Mobile deployment

---
