# PlantDocBot - AI Plant Disease Diagnosis

An AI-powered system for detecting plant diseases using image and text inputs.

This project uses:
- CNN model for image-based disease prediction
- NLP model for symptom-based disease prediction
- React frontend with FastAPI backend

---

# Project Structure

```bash
project-root/
│
├── models/
│   ├── cnn_best_92_plant_disease_model.pth
│   └── text_model/
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_dataset_split.ipynb
│   ├── 03_preprocessing_and_loader.ipynb
│   ├── 04_modeltraining.ipynb
│   └── 05_text_model_training.ipynb
│
├── plantdoc-frontend/
│   ├── public/
│   └── src/
│       ├── assets/
│       └── pages/
│           ├── Dashboard.jsx
│           └── Landing.jsx
│
├── src/
│   ├── inference.py
│   └── text_inference.py
│
├── main.py
├── document_converter.py
├── requirements.txt
└── README.md
---

# Features

## Image Prediction
- Upload plant leaf image
- Detect disease using CNN model
- Shows confidence score

## Text Prediction
- Enter symptoms as text
- Predict disease using NLP model

## Combined Prediction
- Uses both image and text together
- Improves prediction accuracy

---

# Backend Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend server:

```bash
uvicorn main:app --reload
```

Backend runs at:

```bash
http://127.0.0.1:8000
```

---

# Frontend Setup

Move to frontend folder:

```bash
cd plantdoc-frontend
```

Install dependencies:

```bash
npm install
```

Run frontend:

```bash
npm run dev
```

Frontend runs at:

```bash
http://localhost:5173
```

---

# API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | / | API status |
| POST | /predict-image | Image prediction |
| POST | /predict-text | Text prediction |
| POST | /convert-pdf | PDF conversion |

---

# Technologies Used

Frontend:
- React.js
- Vite
- JavaScript
- CSS

Backend:
- FastAPI
- Python
- Uvicorn

Machine Learning:
- PyTorch
- Transformers
- CNN
- NLP

---

# Working Flow

1. User uploads image or enters symptoms
2. Frontend sends request to backend
3. Backend processes data
4. Model predicts disease
5. Result shown in frontend

---

# Important Notes

- node_modules is not uploaded
- .venv is not uploaded
- Keep models inside models folder
- Do not change folder structure

---

# Future Improvements

- Mobile application
- Real-time detection
- Cloud deployment
- Better UI improvements

---

# License

MIT License
