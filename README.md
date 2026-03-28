# PlantDoc

Plant disease detection system with a FastAPI backend and a React + Vite frontend.

## Setup

### 1. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Project Layout

- `api/`: FastAPI backend and prediction routes
- `assets/models/`: Runtime ML model assets
- `data/`: Processed tabular data files
- `dataset/`: Dataset splits used for training/validation
- `docs/`: Technical specifications
- `frontend/`: React + Vite + TypeScript web app
- `notebooks/`: Training and experimentation notebooks
- `scripts/`: Utility and local inference scripts

## Model Assets (Required)

The backend loads models from `assets/models/` using fixed paths in the codebase:

- Image model path used by `api/services/image_model.py`:
	- `assets/models/simplecnn/best_simplecnn_plant_disease.pth`

- Text model path used by `api/services/text_model.py`:
	- `assets/models/distilbert_plantdisease_model/`

### Expected directory structure

```text
assets/
└── models/
		├── simplecnn/
		│   └── best_simplecnn_plant_disease.pth
		└── distilbert_plantdisease_model/
				├── config.json
				├── model.safetensors
				├── tokenizer.json
				└── tokenizer_config.json
```

If the files are not in these locations, the API will fail to load models at startup.

## Run Backend

```bash
uvicorn api.main:app --reload
```

## Run Frontend

```bash
cd frontend
cp .env.example .env
npm run dev
```
