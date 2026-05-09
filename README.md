# 🌿 PlantDocBot

AI-powered plant disease detection system using:

- CNN for image classification
- DistilBERT for symptom-based disease prediction
- FastAPI backend
- React frontend

---

# Features

✅ Leaf image disease detection  
✅ Symptom text analysis  
✅ Top-3 disease predictions  
✅ Confidence scores  
✅ FastAPI REST API  
✅ React frontend UI  

---

# Dataset Classes

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato_Early_blight
- Potato_Late_blight
- Potato_healthy
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato_Target_Spot
- Tomato_Tomato_YellowLeaf_Curl_Virus
- Tomato_Tomato_mosaic_virus
- Tomato_healthy

---

# Tech Stack

## Backend
- FastAPI
- PyTorch
- Transformers
- DistilBERT

## Frontend
- React.js

---

# Project Structure

```plaintext
plant_doc_project/
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── models/
│
├── frontend/
│
├── distilbert_model/
│
├── cnn_model/
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone project

```bash
git clone <your-repository-url>
cd plant_doc_project
```

---

## Create virtual environment

```bash
python -m venv .venv
```

---

## Activate environment

### Windows

```powershell
.venv\Scripts\activate
```

---

## Install requirements

```bash
pip install -r requirements.txt
```

---

# Run Backend

```bash
python -m uvicorn api.main:app --reload
```

Backend URL:

```plaintext
http://127.0.0.1:8000
```

Swagger Docs:

```plaintext
http://127.0.0.1:8000/docs
```

---

# Run Frontend

```bash
cd frontend
npm install
npm start
```

Frontend URL:

```plaintext
http://localhost:3000
```

---

# API Endpoints

## Image Prediction

```http
POST /predict/image
```

## Text Prediction

```http
POST /predict/text
```

## Combined Prediction

```http
POST /predict/combined
```

---

# Example Text Prediction

Input:

```plaintext
yellow spots on tomato leaves
```

Output:

```json
{
  "disease": "Tomato_Leaf_Mold",
  "confidence": 49.2
}
```

---

# Models

## CNN
Used for plant leaf image classification.

## DistilBERT
Used for symptom-based disease prediction.

---

# Author

PlantDocBot Project