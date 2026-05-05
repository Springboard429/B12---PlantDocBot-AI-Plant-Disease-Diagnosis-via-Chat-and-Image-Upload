# PlantDoc AI Dashboard

A robust web application for predicting plant diseases from images of leaves and natural language symptoms. The system operates locally with a FastAPI backend and a React (Vite) frontend.

## Project Structure

```text
plantdisease/
|-- backend/
|   |-- main.py              # FastAPI scalable backend service
|   |-- models_utils.py      # Contains logic for model inference (PyTorch & HuggingFace)
|
|-- models/                  # Stores trained weights and model data
|   |-- best_model.pth       # Trained CNN weights for predicting 38 image classes
|   |-- fine_tuned_model/    # Trained NLP configuration arrays
|   |-- checkpoints/
|
|-- data/                    # Dataset artifacts
|   |-- dataset.csv
|
|-- frontend/                # React / Vite SPA Dashboard
|   |-- src/App.jsx          
|
|-- notebooks/               # Data exploration scripts 
|   |-- analysis.ipynb, etc.
```

## Setup & Running

**1. Start the Backend API**
```bash
# Optional: Setup virtual environment & activate
pip install -r requirements.txt
python backend/main.py
```
> The API will be available at `http://localhost:8000`.

**2. Start the Frontend Dashboard**
```bash
cd frontend
npm install
npm run dev
```
> Access the modern dashboard at `http://localhost:5173`.
