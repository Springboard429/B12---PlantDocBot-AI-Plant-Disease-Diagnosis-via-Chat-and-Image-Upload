```markdown
# 🌿 PlantDocBot - AI Plant Disease Diagnosis

An AI-powered system for detecting plant diseases using:
- 📷 Image-based CNN model
- 💬 Text-based NLP model
- 🌐 Interactive web interface

---

## 📁 Project Structure

```

│
├── api/                         # Backend (FastAPI)

│   ├── routers/                # API endpoints

│   │   └── predict.py          # Handles prediction requests

│   ├── services/               # Core logic for models

│   │   ├── image_model.py      # Image model inference logic

│   │   └── text_model.py       # Text model inference logic

│   ├── main.py                 # FastAPI app entry point

│   └── models.py               # Request/response schemas

│

├── frontend/                   # Frontend (React + Vite)

│   ├── node_modules/           # Installed dependencies (ignored in Git)

│   ├── public/                 # Static assets

│   │   ├── favicon             # Website icon

│   │   └── icons               # UI icons

│   │

│   ├── src/                    # Main frontend source code

│   │   ├── App.tsx             # Main UI component

│   │   ├── App.css             # App styling

│   │   ├── index.css           # Global styles

│   │   └── main.tsx            # React entry point

│   │

│   ├── .env.example            # Example environment variables

│   ├── .gitignore

│   ├── eslint.config.js        # ESLint configuration

│   ├── index.html              # Root HTML file

│   ├── package.json            # Project dependencies

│   ├── package-lock.json

│   ├── README.md               # Frontend documentation

│   ├── tsconfig.app.json

│   ├── tsconfig.json

│   ├── tsconfig.node.json

│   └── vite.config.ts          # Vite config

│

├── models/                     # Trained AI models (Download from Google Drive)

│   ├── image_model/

│   │   └── best_cnn_plantdoc.pth   # Download from Drive

│   │

│   └── text_model/

│       └── plant_disease_modelc/    # Download entire folder from Drive

│           ├── classes.json

│           ├── config.json

│           ├── model.safetensors

│           ├── plantvillage.parquet

│           ├── tokenizer.json

│           ├── tokenizer_config.json

│           └── training_args.bin

│

├── notebooks/                  # Jupyter notebooks

│   ├── plantdoc_eda.ipynb      # Exploratory Data analysis

│   ├── preprocessing.ipynb     # Data cleaning, preprocessing, and augmentation

│   ├── training.ipynb          # Training CNN model for image classification

│   └── training_text_model.ipynb   # Training NLP model for symptom-based

│

├── docs/                       # Additional documentation

├── venv/                       # Virtual environment (not uploaded)

├── .gitattributes

├── .gitignore

└── requirements.txt            # Python dependencies

```

---

## ⚠️ Model Download (IMPORTANT)

Due to GitHub file size limits, trained models are not included in this repository.

👉 Download ALL models from Google Drive:  
**[Download Models (Image + Text)](https://drive.google.com/drive/folders/1OCFm9VF3sxWvgVjDkqnzT3Mmj6ma5xKK?usp=sharing)**

### After downloading:

Place files exactly like this:

```

models/

├── image_model/

│   └── best_cnn_plantdoc.pth

│

└── text_model/

    └── plant_disease_modelc/
    
        ├── classes.json
   
        ├── config.json
       
        ├── model.safetensors
        
        ├── plantvillage.parquet
        
        ├── tokenizer.json
        
        ├── tokenizer_config.json
        
        └── training_args.bin

````

⚠️ Both image and text models MUST be placed correctly or the backend will fail.

---

## ⚙️ Backend Setup (FastAPI)

```bash
cd plant_disease2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
````

Backend runs at:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 💻 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
[http://localhost:5173](http://localhost:5173)

---

## 🔗 How the System Works

1. User uploads image or enters symptoms
2. Frontend sends request to backend
3. Backend processes:

   * Image → CNN model
   * Text → NLP model
4. Result is displayed in UI

---

## ⚠️ Important Notes

* `node_modules/` is not uploaded
* `venv/` is not uploaded
* Models must be downloaded manually
* Folder structure must not be changed

---

## 🚀 Future Enhancements

* Mobile app
* Real-time camera detection
* Cloud deployment

---

## 📜 License

MIT License

```
```
