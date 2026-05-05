# main.py
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models_utils import get_text_predictions, get_image_prediction

app = FastAPI(title="PlantDoc API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str = "My tomato leaf has black spots"


@app.get("/health")
async def health_check():
    """Checking for APi status."""
    return {"status": "healthy"}

@app.post("/predict/image")
async def inference_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        image_bytes = await file.read()
        class_name, confidence, class_idx = get_image_prediction(image_bytes)
        return JSONResponse(content={
            "predicted_class": class_name, 
            "class_id": class_idx, 
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/text")
async def inference_text(payload: TextRequest):
    """
    Expects JSON input: {"text": "My tomato leaf has black spots"}
    """
    text = payload.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    # Call the Hugging Face text classification model
    prediction = get_text_predictions(text)
    
    return {
        "input_text": text, 
        "predicted_disease": prediction.get("label"),
        "confidence": prediction.get("score")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)