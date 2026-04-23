# PlantDocBot – AI Plant Disease Detection and Assistant

PlantDocBot is an AI-powered application designed to detect plant diseases from images and provide intelligent insights using machine learning and natural language processing. It integrates computer vision and NLP models within a full-stack web application to deliver an interactive plant disease prediction assistant.

---

## Features

- Plant disease detection using image classification models  
- Text-based assistance using a fine-tuned DistilBERT model  
- Backend API built with FastAPI  
- Frontend built using React and Vite  
- End-to-end ML workflow including training, evaluation, and inference  
- Jupyter notebooks for model development and experimentation  

---

## Project Structure
B12---PlantDocBot-AI-Plant-Disease-Diagnosis-via-Chat-and-Image-Upload/
│
├── backend/
│   ├── models/              # Trained models
│   ├── app.py               # Backend API (FastAPI)
│   └── __pycache__/
│
├── frontend/
│   ├── src/                
│   ├── node_modules/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig*.json
│   └── vite.config.ts     
│
├── notebooks/
│   ├── 01_image_dataset_eda.ipynb
│   ├── 02_image_dataset_preprocessing.ipynb
│   ├── 03_model_training_and_comparison.ipynb
│   ├── 04_model_inference.ipynb
│   ├── 05_text_dataset.ipynb
│   ├── 06_fine_tune_DistilBERT.ipynb
│   └── 07_text_model_inference.ipynb
│
├── venv/                 
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

---

## Tech Stack

### Backend
- Python
- FastAPI
- Machine Learning models

### Frontend
- React
- Vite
- TypeScript

### AI/ML
- Image classification models (CNN-based)
- NLP model: DistilBERT (fine-tuned)
- Jupyter Notebooks for training and experimentation

---

## Notebooks Overview

The notebooks directory contains all workflows related to model development:

- 01_image_dataset_eda: Exploratory data analysis on image dataset  
- 02_image_dataset_preprocessing: Image preprocessing and cleaning  
- 03_model_training_and_comparison: Training and evaluating image models  
- 04_model_inference: Running inference on trained image models  
- 05_text_dataset: Text dataset preparation  
- 06_fine_tune_DistilBERT: Fine-tuning the model  
- 07_text_model_inference: Text model inference  

---

## Installation and Setup

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Springboard429/B12---PlantDocBot-AI-Plant-Disease-Diagnosis-via-Chat-and-Image-Upload.git
cd B12---PlantDocBot-AI-Plant-Disease-Diagnosis-via-Chat-and-Image-Upload
git checkout Malavika
```

---

### 2. Backend Setup

```bash
cd backend
python -m venv venv
```

Activate virtual environment:

- On Linux/macOS:
```bash
source venv/bin/activate
```

- On Windows:
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

---

### 3. Run Backend

```bash
uvicorn app:app --reload
```

---

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Usage

1. Start the FastAPI backend server  
2. Start the React frontend  
3. Open the application in your browser  
4. Upload a plant image to detect diseases  
5. Or enter the symptoms to detect diseases   

---

## Future Improvements

- Cloud deployment (AWS, GCP, or similar)  
- Mobile-friendly interface  
- Multi-language support  
- Expansion of plant disease categories  
- Enhanced conversational assistant  

---

## License

This project is licensed under the terms specified in the LICENSE file.

---