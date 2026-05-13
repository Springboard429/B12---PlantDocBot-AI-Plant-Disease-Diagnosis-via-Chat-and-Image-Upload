# 🌿 PlantDocBot — AI Plant Disease Diagnosis via Chat and Image Upload

> An intelligent, dual-modal plant disease diagnosis system powered by a custom **SimpleCNN** for image analysis and a fine-tuned **DistilBERT** model for natural language symptom descriptions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (React + Vite)](#frontend-react--vite)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Models](#models)
- [Notebooks](#notebooks)
- [Environment Variables](#environment-variables)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🌱 Overview

**PlantDocBot** is an AI-powered plant disease diagnosis platform that allows users to:
1. **Upload a leaf image** — a custom Convolutional Neural Network (SimpleCNN) classifies the disease from visual features.
2. **Describe symptoms in natural language** — a fine-tuned DistilBERT transformer model predicts the disease from textual input.
3. **Combine both inputs** — the system processes both image and text simultaneously and returns unified predictions.

The application is designed to assist farmers, agronomists, and plant enthusiasts in quickly identifying plant diseases without laboratory testing.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image Diagnosis** | Upload a plant leaf image → CNN predicts disease with confidence score |
| 💬 **Chat / Text Diagnosis** | Describe symptoms in plain English → NLP model returns disease prediction |
| 🔀 **Combined Diagnosis** | Submit both image and text simultaneously for dual-model predictions |
| 📊 **Confidence Scores** | Each prediction comes with a percentage confidence score |
| 🏷️ **38 Disease Classes** | Trained across 38 plant disease categories from the PlantVillage dataset |
| ⚡ **REST API** | FastAPI backend with clean JSON endpoints |
| 🎨 **Responsive UI** | Modern React + TypeScript frontend with confidence visualization |
| ❤️ **Health Check** | `/health` endpoint for monitoring API and model status |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────┐
│                  React + Vite Frontend                  │
│         (TypeScript · Responsive UI · Axios)            │
└────────────────────────┬──────────────────────────────┘
                         │ HTTP REST (JSON / multipart)
                         ▼
┌───────────────────────────────────────────────────────┐
│               FastAPI Backend (Python)                  │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  /predict/image  │    │   /predict/text           │  │
│  │  SimpleCNN Model │    │   DistilBERT Model        │  │
│  │  (PyTorch .pth)  │    │   (HuggingFace Transformers│  │
│  └──────────────────┘    └──────────────────────────┘  │
│           └──────────────────────┘                      │
│                /predict/combined                        │
└───────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │      Models/         │
              │  SimpleCNN .pth      │
              │  DistilBERT folder   │
              │  label_encoder.pkl   │
              └──────────────────────┘
```

---

## 📁 Project Structure

```
PlantDocBot/
│
├── Backend/
│   └── main.py                         # FastAPI app — all API endpoints
│
├── Frontend/
│   ├── src/
│   │   ├── App.tsx                     # Main React application component
│   │   ├── App.css                     # Application styles
│   │   ├── main.tsx                    # React entry point
│   │   └── index.css                   # Global CSS
│   ├── index.html                      # HTML shell
│   ├── package.json                    # Node dependencies
│   ├── vite.config.ts                  # Vite build configuration
│   ├── tsconfig.json                   # TypeScript configuration
│   ├── .env.example                    # Environment variable template
│   └── README.md                       # Frontend-specific setup guide
│
├── Models/
│   ├── best_simplecnn_plant_disease.pth  # Trained CNN model weights (~98 MB)
│   ├── distilbert_plant_disease_model/   # Fine-tuned DistilBERT model files
│   └── label_encoder.pkl                 # Sklearn LabelEncoder for class mapping
│
├── Notebooks/
│   ├── plantdoc1.ipynb                 # Data exploration & preprocessing
│   ├── plantdoc2.ipynb                 # CNN training pipeline
│   ├── plantdoc3.ipynb                 # DistilBERT fine-tuning
│   ├── plantdoc5.ipynb                 # Evaluation & metrics
│   ├── cnn_inference.py                # Standalone CNN inference script
│   └── text_inference.py               # Standalone text inference script
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python backend dependencies
└── README.md                           # This file
```

---

## 🛠️ Prerequisites

### Backend
- Python **3.9+**
- pip
- CUDA-capable GPU *(optional but recommended for faster inference)*

### Frontend
- Node.js **20+**
- npm **9+**

---

## ⚙️ Installation & Setup

### Backend (FastAPI)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Springboard429/B12---PlantDocBot-AI-Plant-Disease-Diagnosis-via-Chat-and-Image-Upload.git
   cd B12---PlantDocBot-AI-Plant-Disease-Diagnosis-via-Chat-and-Image-Upload
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place model files** in the `Models/` directory:
   - `best_simplecnn_plant_disease.pth`
   - `distilbert_plant_disease_model/` (folder with tokenizer & weights)
   - `label_encoder.pkl`

5. **Update model paths** in `Backend/main.py` (and inference scripts) to point to your local `Models/` directory.

---

### Frontend (React + Vite)

1. **Navigate to the Frontend folder:**
   ```bash
   cd Frontend
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Configure the API base URL:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```

---

## 🚀 Running the Application

### Step 1 — Start the Backend API

From the project root (with the virtual environment active):
```bash
cd Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://127.0.0.1:8000**  
Interactive API docs: **http://127.0.0.1:8000/docs**

### Step 2 — Start the Frontend

In a new terminal:
```bash
cd Frontend
npm run dev
```

The UI will be available at: **http://localhost:5173**

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API health and model load status |
| `GET` | `/classes` | List all 38 disease class names |
| `POST` | `/predict/image` | Predict disease from uploaded leaf image |
| `POST` | `/predict/text` | Predict disease from symptom description text |
| `POST` | `/predict/combined` | Predict using both image and text inputs |

### `/predict/image`
- **Request**: `multipart/form-data` with field `file` (image file)
- **Response**:
  ```json
  {
    "disease": "Tomato___Early_blight",
    "confidence": 94.72,
    "model": "cnn-model"
  }
  ```

### `/predict/text`
- **Request**: `application/x-www-form-urlencoded` with field `text`
- **Response**:
  ```json
  {
    "disease": "Tomato___Early_blight",
    "confidence": 87.5,
    "model": "text-model"
  }
  ```

### `/predict/combined`
- **Request**: `multipart/form-data` with optional `file` and optional `text`
- **Response**:
  ```json
  {
    "image_prediction": { "disease": "...", "confidence": 94.72, "model": "cnn-model" },
    "text_prediction":  { "disease": "...", "confidence": 87.5,  "model": "text-model" }
  }
  ```

---

## 🤖 Models

### SimpleCNN — Image Classification
- **Architecture**: 3-block CNN (Conv2d → BatchNorm → ReLU → MaxPool) + fully connected classifier
- **Input**: 224×224 RGB leaf images
- **Output**: 38 disease class probabilities (softmax)
- **Training Dataset**: PlantVillage dataset
- **File**: `Models/best_simplecnn_plant_disease.pth`

### DistilBERT — Text Classification
- **Architecture**: Fine-tuned `distilbert-base-uncased` for sequence classification
- **Input**: Natural language symptom description (max 128 tokens)
- **Output**: Disease label decoded via `LabelEncoder`
- **File**: `Models/distilbert_plant_disease_model/`

### Label Encoder
- **File**: `Models/label_encoder.pkl`
- Sklearn `LabelEncoder` serialized with `joblib`, maps integer predictions → disease class names.

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `plantdoc1.ipynb` | Dataset exploration, class distribution analysis, image visualization |
| `plantdoc2.ipynb` | SimpleCNN model training and validation |
| `plantdoc3.ipynb` | DistilBERT fine-tuning on plant disease text corpus |
| `plantdoc5.ipynb` | Model evaluation, confusion matrix, accuracy/F1 metrics |
| `cnn_inference.py` | Standalone script to run CNN predictions on new images |
| `text_inference.py` | Standalone script to run text-based disease predictions |

---

## 🔐 Environment Variables

### Frontend (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI backend base URL |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, TypeScript, Vite 8 |
| **Backend** | Python, FastAPI, Uvicorn |
| **Image Model** | PyTorch, torchvision, SimpleCNN |
| **Text Model** | HuggingFace Transformers, DistilBERT |
| **ML Utilities** | scikit-learn, joblib, Pillow |
| **Data / Training** | Jupyter Notebooks, Google Colab |

---

## 👥 Team

**Project**: B12 — PlantDocBot  
**Track**: Springboard 429  
**Domain**: AI / Deep Learning / Computer Vision / NLP

---

> 🌿 *Empowering farmers and plant lovers with AI-driven disease diagnosis — one leaf at a time.*
