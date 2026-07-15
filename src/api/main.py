"""
FastAPI Backend for Wrist Fracture Detection
Interactive Chat + Clean Formatting
"""

import sys
import os
import io
import base64
import re
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import ollama

# --- INITIALIZE APP ---
app = FastAPI(title="Wrist Fracture Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SET DEVICE ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# --- LOAD RESNET MODEL ---
print("📦 Loading ResNet model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('models/checkpoints/baseline_resnet50.pth', map_location='cpu'))
model = model.to(device)
model.eval()
print("✅ ResNet model loaded!")

# --- PREPROCESSING TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --- CLASS NAMES ---
class_names = ['Fractura', 'Metal', 'Texto']
threshold = 0.5

# Store last analysis for chat context
last_analysis = {
    "detected_classes": [],
    "probs": [],
    "image_name": "",
    "caption": ""
}

# --- HELPER: Clean Formatting ---
def clean_text(text):
    """Remove markdown symbols like *, #, -, etc."""
    # Remove markdown headers and bullet points
    text = re.sub(r'[*#_-]{1,3}\s*', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# --- HELPER FUNCTIONS ---
def get_caption(detected_classes):
    if len(detected_classes) == 0:
        return "Normal: No fracture, no metal"
    elif "Fractura" in detected_classes and "Metal" in detected_classes:
        return "Fracture + Metal detected"
    elif "Fractura" in detected_classes:
        return "Fracture detected"
    elif "Metal" in detected_classes:
        return "Metal hardware detected"
    elif "Texto" in detected_classes:
        return "Text annotations present"
    else:
        return "Abnormalities detected"

def get_treatment(detected_classes, probs):
    if len(detected_classes) == 0:
        diagnosis = "a normal wrist X-ray"
    else:
        diagnosis = f"a wrist X-ray showing {', '.join(detected_classes)}"
    
    prompt = f"""Write a short clinical description for a medical student about {diagnosis}.

Include:
- What this finding typically indicates
- Common clinical observations
- Typical follow-up considerations

Educational summary only, not medical advice."""

    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return clean_text(response['message']['content'])
    except Exception as e:
        return f"Error: {e}"

def generate_heatmap(image_tensor, pred_idx, rgb_img):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    def target_fn(output):
        return output[pred_idx]
    grayscale_cam = cam(input_tensor=image_tensor, targets=[target_fn])[0, :]
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# --- API: Serve HTML ---
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("src/api/static/index.html", "r") as f:
        return f.read()

# --- API: Analyze Image ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        detected_classes = []
        for i, name in enumerate(class_names):
            if probs[i] >= threshold:
                detected_classes.append(name)
        
        pred_idx = np.argmax(probs)
        pred_name = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])
        
        caption = get_caption(detected_classes)
        treatment = get_treatment(detected_classes, probs)
        
        # Store for chat context
        last_analysis["detected_classes"] = detected_classes
        last_analysis["probs"] = probs.tolist()
        last_analysis["image_name"] = file.filename
        last_analysis["caption"] = caption
        
        # Heatmap
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        if "Fractura" in detected_classes or "Metal" in detected_classes:
            class_for_heatmap = 0 if "Fractura" in detected_classes else 1
            heatmap = generate_heatmap(image_tensor, class_for_heatmap, rgb_img)
            heatmap_img = Image.fromarray(heatmap)
            draw = ImageDraw.Draw(heatmap_img)
            text = f"Pred: {pred_name} ({pred_conf:.2f}) | {caption}"
            draw.rectangle([(0, 200-25), (224, 200+15)], fill=(0, 0, 0, 180))
            draw.text((10, 200-20), text, fill=(255, 255, 255))
            heatmap = np.array(heatmap_img)
        else:
            heatmap = (rgb_img * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap)
            draw = ImageDraw.Draw(heatmap_img)
            draw.text((10, 200-20), caption, fill=(255, 255, 255))
            heatmap = np.array(heatmap_img)
        
        buffered = io.BytesIO()
        Image.fromarray(heatmap).save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "prediction": pred_name,
            "confidence": pred_conf,
            "detected_classes": detected_classes,
            "caption": caption,
            "treatment": treatment,
            "heatmap": heatmap_b64,
            "image_name": file.filename
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- API: Chat with User ---
@app.post("/chat")
async def chat(request: dict):
    """Handle follow-up questions about the X-ray."""
    user_message = request.get("message", "")
    
    detected = last_analysis["detected_classes"]
    probs = last_analysis["probs"]
    caption = last_analysis["caption"]
    
    if not detected:
        return {"response": "Please upload and analyze an X-ray first."}
    
    context = f"""
The patient's wrist X-ray shows: {caption}
Confidence scores: Fractura={probs[0]:.2f}, Metal={probs[1]:.2f}, Texto={probs[2]:.2f}

The user asks: {user_message}

Provide a clear, helpful educational response. Use simple language. Do not use markdown (*, #, -).
"""

    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': context}]
        )
        return {"response": clean_text(response['message']['content'])}
    except Exception as e:
        return {"response": f"Error: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)