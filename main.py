from inference import predict_image
from fastapi import UploadFile, File, Form
from text_inference import predict_disease
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from document_converter import convert_document
app = FastAPI(title="Document Converter API")
@app.post("/convert-pdf")
async def convert_pdf(File: UploadFile = File(...)):

    if not File.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            content = await File.read()
            tmp_file.write(content)

            tmp_file_path = tmp_file.name

        convert_document(tmp_file_path, output_dir="./converted_docs")

        return JSONResponse(content={
            "filename": File.filename,
            "status": "success"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ✅ ADD THIS AT VERY BOTTOM (OUTSIDE FUNCTION)
@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    result = predict_disease(text)
    return {"prediction": result}
import shutil

@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_path)

    return {"prediction": result}