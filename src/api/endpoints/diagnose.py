"""
Diagnose Endpoint - Quick Diagnosis Only
"""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image

from ..models.schemas import DiagnosisResponse
from ...models.classifier import load_model
from ...utils.helpers import get_device, preprocess_image

router = APIRouter()
device = get_device()
model = load_model(device)


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_image(file: UploadFile = File(...)):
    """
    Quick diagnosis only (no heatmap, no treatment).
    Returns: Prediction and Confidence only.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        probs, pred_class, confidence = model.predict(tensor)
        
        return DiagnosisResponse(
            prediction=pred_class,
            confidence=float(confidence),
            probabilities={
                "Fractura": float(probs[0]),
                "Metal": float(probs[1]),
                "Texto": float(probs[2])
            }
        )
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
