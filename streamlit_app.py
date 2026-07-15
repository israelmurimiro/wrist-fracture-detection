cd '/Users/israelm/Desktop/DATA SCIENCE/1. NOW/4.DEEP-LEARNING-2/wrist-fracture-detection'
cat > streamlit_app_simple.py << 'EOF'
"""
WristVision AI - Simplified Streamlit App
Wrist Fracture Detection with Grad-CAM Explainability
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- GRAD-CAM IMPORT WITH FALLBACK ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="WristVision AI",
    page_icon="🦴",
    layout="wide"
)

# --- DEVICE ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMAGE_SIZE = 224
CLASS_NAMES = ['Fractura', 'Metal', 'Texto']

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = Path("models/checkpoints/baseline_resnet50.pth")
    
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    return transform(image).unsqueeze(0).to(device)

def predict(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    cls = CLASS_NAMES[idx]
    conf = float(probs[idx])
    return probs, idx, cls, conf

def generate_heatmap(model, tensor, target_idx):
    if not GRAD_CAM_AVAILABLE:
        return None
    target_layer = model.layer4[-1]
    def target_fn(out):
        return out[target_idx]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[target_fn])[0]
    return heatmap

def overlay_heatmap(image_arr, heatmap, intensity=0.6):
    if heatmap is None:
        return (image_arr * 255).astype(np.uint8)
    overlay = show_cam_on_image(image_arr, heatmap, use_rgb=True).astype(np.float32) / 255.0
    blended = (1 - intensity) * image_arr + intensity * overlay
    return np.clip(blended * 255, 0, 255).astype(np.uint8)

def get_caption(detected):
    if not detected:
        return "Normal: No fracture, no metal"
    if "Fractura" in detected and "Metal" in detected:
        return "Fracture + Metal detected"
    if "Fractura" in detected:
        return "Fracture detected"
    if "Metal" in detected:
        return "Metal hardware detected"
    if "Texto" in detected:
        return "Text annotations present"
    return "Abnormalities detected"

# --- UI ---
st.title("🦴 WristVision AI")
st.caption("Wrist Fracture Detection with Explainable AI")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
    intensity = st.slider("Heatmap intensity", 0.10, 0.90, 0.60, 0.05)
    st.divider()
    st.caption(f"Device: {device}")
    st.caption("Architecture: ResNet-50")
    st.warning("Educational research only. Not for diagnosis.")

# Upload
uploaded = st.file_uploader("Upload a wrist X-ray", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert('RGB')
        tensor = preprocess_image(image)
        
        # Predict
        probs, idx, pred_class, conf = predict(model, tensor)
        detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
        caption = get_caption(detected)
        
        # Heatmap
        heatmap = generate_heatmap(model, tensor, idx) if GRAD_CAM_AVAILABLE else None
        img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
        overlay = overlay_heatmap(img_array, heatmap, intensity)
        
        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original X-ray")
            st.image(image, width="stretch")
        
        with col2:
            st.subheader("Grad-CAM Heatmap")
            st.image(overlay, width="stretch")
        
        # Results
        st.divider()
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", pred_class)
        col2.metric("Confidence", f"{conf:.1%}")
        col3.metric("Threshold", f"{threshold:.0%}")
        
        st.write("**Class Probabilities:**")
        prob_cols = st.columns(3)
        for i, name in enumerate(CLASS_NAMES):
            with prob_cols[i]:
                st.metric(name, f"{probs[i]:.1%}")
        
        st.info(f"**Caption:** {caption}")
        
        # Technical details
        with st.expander("Technical details"):
            st.write(f"**Predicted class:** {pred_class}")
            st.write(f"**Confidence:** {conf:.6f}")
            st.write(f"**Detected:** {', '.join(detected) if detected else 'None'}")
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👆 Upload a wrist X-ray to begin.")

st.divider()
st.caption("WristVision AI · Educational Research Use Only")
EOF
