"""
Analyze Endpoint - Full Analysis with Prediction + Heatmap + Caption + Treatment
"""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import numpy as np

from ..models.schemas import AnalysisResponse
from ...models.classifier import load_model
from ...models.gradcam import generate_heatmap
from ...utils.helpers import get_device, preprocess_image, get_caption, get_treatment

router = APIRouter()
device = get_device()
model = load_model(device)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Full analysis of a wrist X-ray image.
    Returns: Prediction, Confidence, Heatmap, Caption, Treatment
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get prediction
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        probs, pred_class, confidence = model.predict(tensor)
        
        # Generate heatmap
        heatmap = generate_heatmap(model, tensor, np.argmax(probs))
        heatmap_b64 = base64.b64encode(heatmap).decode('utf-8')
        
        # Generate outputs
        detected_classes = [model.class_names[i] for i, p in enumerate(probs) if p >= 0.5]
        caption = get_caption(detected_classes)
        treatment = get_treatment(detected_classes, probs)
        
        return AnalysisResponse(
            prediction=pred_class,
            confidence=float(confidence),
            detected_classes=detected_classes,
            caption=caption,
            treatment=treatment,
            heatmap=heatmap_b64
        )
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
