"""
Treatment Endpoint - Clinical Description Only
"""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image

from ...models.classifier import load_model
from ...utils.helpers import get_device, preprocess_image, get_treatment

router = APIRouter()
device = get_device()
model = load_model(device)


@router.post("/treatment")
async def get_treatment_endpoint(file: UploadFile = File(...)):
    """
    Generate clinical description based on X-ray findings.
    Returns: Caption + Treatment text only.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        probs, pred_class, confidence = model.predict(tensor)
        
        detected_classes = [model.class_names[i] for i, p in enumerate(probs) if p >= 0.5]
        caption = get_caption(detected_classes)
        treatment = get_treatment(detected_classes, probs)
        
        return {
            "prediction": pred_class,
            "confidence": float(confidence),
            "caption": caption,
            "treatment": treatment
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
