"""
Explain Endpoint - Heatmap + Explanation Only
"""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import numpy as np

from ...models.classifier import load_model
from ...models.gradcam import generate_heatmap
from ...utils.helpers import get_device, preprocess_image

router = APIRouter()
device = get_device()
model = load_model(device)


@router.post("/explain")
async def explain_image(file: UploadFile = File(...)):
    """
    Generate Grad-CAM heatmap and explanation for an X-ray.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        probs, pred_class, confidence = model.predict(tensor)
        
        heatmap = generate_heatmap(model, tensor, np.argmax(probs))
        heatmap_b64 = base64.b64encode(heatmap).decode('utf-8')
        
        return {
            "prediction": pred_class,
            "confidence": float(confidence),
            "heatmap": heatmap_b64,
            "probabilities": {
                "Fractura": float(probs[0]),
                "Metal": float(probs[1]),
                "Texto": float(probs[2])
            }
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
