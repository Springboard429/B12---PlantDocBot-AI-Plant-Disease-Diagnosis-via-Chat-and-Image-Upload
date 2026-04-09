from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from document_converter import convert_document
from src.inference import predict_image
from src.text_inference import predict_disease
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Document Converter API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/convert-pdf")
async def convert_pdf_to_markdown(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        convert_document(tmp_path, "./converted_docs")

        return JSONResponse(content={
            "filename": file.filename,
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        prediction, confidence = predict_image(tmp_path)

        return {
            "disease": prediction,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
class TextInput(BaseModel):
    text: str


@app.post("/predict-text")
async def predict_text_api(data: TextInput):

    try:
        result = predict_disease(data.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))