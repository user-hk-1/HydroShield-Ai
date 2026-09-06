# 🌊 HydroShield-AI: Intelligent Municipal Flood & Road Surface Hazard Triage System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![GPU Accelerated](https://img.shields.io/badge/NVIDIA_GPU-GTX_1650-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end, AI-powered municipal emergency triage platform that automatically classifies road surface hazards (submerged potholes, flash waterlogging) from citizen reports, augments vision classifications with real-time weather metrics and HTML5 GPS coordinates, and dispatches automated emergency alerts to municipal channels via Telegram.

---

## 📌 Key Features

- 🧠 **PyTorch GPU Deep Learning**: Fine-tuned **MobileNetV2** Convolutional Neural Network trained with PyTorch CUDA acceleration on an NVIDIA GeForce GTX 1650 GPU.
- 🎯 **High Accuracy Hazard Recognition**: Achieves **98.32% overall dataset accuracy** (467/475 correct) and **100.00% test accuracy** across 3 classes:
  1. `Clear Road`
  2. `Submerged Pothole Hazard`
  3. `Waterlogged Safe Road`
- 🤖 **Autonomous Municipal Agent Engine (`agent.py`)**: Evaluates vision predictions alongside live OpenMeteo rainfall/wind metrics and Nominatim reverse-geocoded street addresses to calculate a multi-factor risk score.
- 📡 **Automated Telegram Emergency Broadcasting**: Instantly posts structured Markdown alerts to municipal emergency channels, complete with OpenStreetMap live navigation links, confidence scores, and media previews.
- 💻 **Interactive Glassmorphism Dashboard (`app.py`)**: Built with Streamlit, supporting live browser HTML5 GPS detection, camera capture, file upload, and real-time triage inspection.

---

## 📊 Model Performance & Confusion Matrix

### Test Dataset (9 Validation Images) — **100.00% Accuracy**

| Actual \ Predicted | `Clear Road` | `Submerged Pothole` | `Waterlogged Road` | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **`Clear Road`** | **3** | 0 | 0 | **100.0%** |
| **`Submerged Pothole Hazard`** | 0 | **3** | 0 | **100.0%** |
| **`Waterlogged Road`** | 0 | 0 | **3** | **100.0%** |

---

### Full Dataset (475 Images) — **98.32% Accuracy**

| Actual \ Predicted | `Clear Road` | `Submerged Pothole` | `Waterlogged Road` | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **`Clear Road`** (155) | **154** | 0 | 1 | **99.4%** |
| **`Submerged Pothole Hazard`** (157) | 1 | **154** | 2 | **98.1%** |
| **`Waterlogged Road`** (163) | 0 | 4 | **159** | **97.5%** |

---

## 🏗️ System Architecture

```
[ Citizen Photo / Live Capture ] + [ Browser HTML5 GPS ]
                      │
                      ▼
    [ PyTorch MobileNetV2 Vision Engine (GTX 1650 GPU) ]
                      │ (Hazard Label + Confidence %)
                      ▼
      [ Autonomous Municipal Reasoning Agent ] ◄─── [ OpenMeteo Weather API ]
                      │ (Multi-Factor Risk Triage)
                      ▼
  [ Interactive Streamlit Web Command Center ] ──► [ Telegram Emergency Channel Dispatch ]
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/user-hk-1/HydroShield-Ai.git
cd HydroShield-Ai
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_channel_chat_id_here
```

### 4. Train the Model (Optional)
To retrain the PyTorch GPU vision classifier:
```bash
python CNN.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
HydroShield-Ai/
├── CNN.py                       # PyTorch MobileNetV2 GPU Training Script
├── agent.py                     # Autonomous Municipal Reasoning & Triage Agent
├── app.py                       # Streamlit Web Application Dashboard
├── rapid_model.pth              # Fine-tuned PyTorch Model Weights (GPU/CPU)
├── bg.jpg                       # Background UI Asset
├── requirements.txt             # Python Dependencies
├── .gitignore                   # Ignored files (.env, .venv, data, caches)
└── README.md                    # Project Documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
