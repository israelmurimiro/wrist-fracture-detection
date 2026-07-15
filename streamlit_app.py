"""
WristVision AI - Streamlit App
Wrist Fracture Detection with Grad-CAM Explainability & AI Clinical Descriptions
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50
import ollama

# --- GRAD-CAM IMPORT WITH FALLBACK ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    try:
        from grad_cam import GradCAM
        from grad_cam.utils.image import show_cam_on_image
        GRAD_CAM_AVAILABLE = True
    except ImportError:
        GRAD_CAM_AVAILABLE = False


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WristVision AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/checkpoints/baseline_resnet50.pth")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGE_SIZE = 224
CLASS_NAMES = ['Fractura', 'Metal', 'Texto']

# Performance metrics
TEST_AUC = 0.971
TEST_SENSITIVITY = 0.951
TEST_SPECIFICITY = 0.925
TEST_PRECISION = 0.913
TEST_ACCURACY = 0.976

OLLAMA_MODEL = "llama3.2:3b"

# Session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = None


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #030817;
            --surface: #08172d;
            --surface-2: #0b203e;
            --line: rgba(66, 170, 255, 0.22);
            --blue: #168cff;
            --cyan: #35d8ff;
            --text: #f7fbff;
            --muted: #a9bdd6;
        }
        .stApp {
            background:
                radial-gradient(circle at 78% 4%, rgba(0, 126, 255, 0.16), transparent 24%),
                radial-gradient(circle at 18% 18%, rgba(0, 80, 190, 0.11), transparent 28%),
                linear-gradient(160deg, #020714 0%, #061226 52%, #020714 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020a18 0%, #06152a 100%);
            border-right: 1px solid var(--line);
        }
        .block-container { max-width: 1460px; padding-top: 1.55rem; padding-bottom: 4rem; }
        [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3, [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
        }
        p, label, .stCaption { color: var(--muted); }
        div[data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: linear-gradient(145deg, rgba(11, 37, 72, 0.95), rgba(4, 18, 39, 0.96));
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        }
        [data-testid="stMetricLabel"] { color: #7acbff !important; font-weight: 700; }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 800;
            font-size: 1.85rem !important;
            line-height: 1.15 !important;
        }
        [data-testid="stFileUploader"] {
            padding: 1rem;
            border: 1px dashed rgba(66, 190, 255, 0.48);
            border-radius: 18px;
            background: rgba(9, 32, 63, 0.94);
        }
        [data-testid="stFileUploader"]:hover { border-color: var(--cyan); }
        [data-testid="stTabs"] {
            padding: 0.35rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(4, 18, 40, 0.72);
        }
        [data-testid="stTabs"] button {
            border-radius: 10px;
            color: #9fb4ce;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: white;
            background: rgba(17, 109, 210, 0.36);
        }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(53, 178, 255, 0.44);
            border-radius: 10px;
            background: linear-gradient(115deg, #0d72df, #084a9e);
            color: white;
            font-weight: 700;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--cyan);
            color: white;
            box-shadow: 0 8px 24px rgba(22, 139, 255, 0.25);
        }
        hr { border-color: rgba(56, 171, 255, 0.14); }

        /* --- Custom tag styling --- */
        .app-tag {
            text-align: right;
            font-size: 14px;
            color: #7acbff;
            margin-top: -6px;
            margin-bottom: 10px;
            font-weight: 400;
            letter-spacing: 0.5px;
            opacity: 0.8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING & HELPERS
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def prepare_image(image):
    rgb = image.convert("RGB")
    resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(resized).astype(np.float32) / 255.0
    tensor = image_transform(rgb).unsqueeze(0).to(DEVICE)
    return rgb, resized, arr, tensor


def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        probs = torch.sigmoid(out).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    cls = CLASS_NAMES[idx]
    conf = float(probs[idx])
    return probs, idx, cls, conf


def generate_gradcam(model, tensor, target_idx):
    if not GRAD_CAM_AVAILABLE:
        return None
    def target_fn(out):
        return out[target_idx]
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[target_fn])[0]
    return heatmap


def overlay_heatmap(image_arr, heatmap, intensity):
    if heatmap is None:
        return (image_arr * 255).astype(np.uint8)
    overlay = show_cam_on_image(image_arr, heatmap, use_rgb=True).astype(np.float32) / 255.0
    blended = (1 - intensity) * image_arr + intensity * overlay
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def generate_clinical_description(detected_classes, probs):
    if not detected_classes:
        diagnosis = "a normal wrist X-ray"
    else:
        diagnosis = f"a wrist X-ray showing {', '.join(detected_classes)}"

    prompt = f"""You are a medical student studying orthopedics. Write a short clinical description of {diagnosis}.

Include:
- What this finding typically indicates in clinical practice
- Common clinical observations associated with this finding
- Typical clinical follow-up considerations

Write this as an educational summary for a student, not as medical advice."""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ Ollama error: {e}. Please ensure Ollama is running (`ollama serve`)."


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


def add_history(filename, pred_class, confidence, threshold, caption, clinical):
    st.session_state.analysis_history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Prediction": pred_class,
        "Confidence": round(confidence, 6),
        "Displayed Confidence": f"{confidence:.1%}",
        "Threshold": round(threshold, 2),
        "Caption": caption,
        "Clinical Description": clinical[:100] + "..." if len(clinical) > 100 else clinical
    })


def build_report(filename, dims, pred_class, confidence, threshold, caption, clinical, probs):
    return f"""
WRISTVISION AI RESEARCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE
Filename: {filename}
Original dimensions: {dims[0]} x {dims[1]} px
Model input: {IMAGE_SIZE} x {IMAGE_SIZE} px

MODEL OUTPUT
Prediction: {pred_class}
Confidence: {confidence:.4f} ({confidence:.1%})
Threshold: {threshold:.2f}
Caption: {caption}

Probabilities:
  Fractura: {probs[0]:.4f}
  Metal: {probs[1]:.4f}
  Texto: {probs[2]:.4f}

CLINICAL DESCRIPTION (from AI)
{clinical}

MODEL
Architecture: ResNet-50
Device: {DEVICE}

TEST PERFORMANCE
ROC-AUC: {TEST_AUC:.4f}
Sensitivity: {TEST_SENSITIVITY:.4f}
Specificity: {TEST_SPECIFICITY:.4f}
Precision: {TEST_PRECISION:.4f}
Accuracy: {TEST_ACCURACY:.4f}

LIMITATIONS
Educational prototype only. Not for clinical use.
"""


def render_processing():
    progress = st.progress(0)
    status = st.empty()
    steps = [("Preprocessing", 20), ("Running inference", 58), ("Computing Grad-CAM", 84), ("Preparing dashboard", 100)]
    for text, val in steps:
        status.caption(f"✓ {text}")
        progress.progress(val)
        time.sleep(0.08)
    status.empty()
    progress.empty()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🦴 WristVision AI")
    st.caption("Wrist X-Ray Research Dashboard")
    st.divider()

    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
    intensity = st.slider("Heatmap intensity", 0.10, 0.90, 0.60, 0.05)
    show_details = st.toggle("Show technical details", True)

    st.divider()
    st.subheader("Model performance")
    st.metric("ROC-AUC", f"{TEST_AUC:.3f}")
    st.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
    st.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
    st.metric("Precision", f"{TEST_PRECISION:.2%}")

    st.divider()
    st.caption(f"Device: {DEVICE}")
    st.caption("Architecture: ResNet-50")
    st.caption(f"Input: {IMAGE_SIZE}×{IMAGE_SIZE}")
    st.warning("Educational research only. Not for diagnosis.")


# ============================================================
# HEADER WITH TAG
# ============================================================

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.caption("🔵 LIVE RESEARCH PROTOTYPE")
    st.title("🦴 WristVision AI")
    
    # --- TAG: by ISRAEL MURIMIRO (right-aligned, below title) ---
    st.markdown(
        '<div class="app-tag">by ISRAEL MURIMIRO</div>',
        unsafe_allow_html=True,
    )

    st.caption("BIOMEDICAL COMPUTER-VISION RESEARCH DASHBOARD")
    st.subheader("See what an AI model notices in a wrist X-ray.")
    st.write(
        "ResNet-50 classifier with Grad-CAM attention, adjustable thresholds, "
        "and AI-generated clinical descriptions."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.info("ResNet-50")
    col2.info("PyTorch")
    col3.info("Grad-CAM")
    col4.info("Ollama")

with right:
    with st.container(border=True):
        st.subheader("Held-out test performance")
        a, b = st.columns(2)
        c, d = st.columns(2)
        a.metric("ROC-AUC", f"{TEST_AUC:.3f}")
        b.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
        c.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
        d.metric("Precision", f"{TEST_PRECISION:.2%}")

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.header("Analyze a wrist X-ray")
uploaded = st.file_uploader("Drag and drop a wrist X-ray", type=["png", "jpg", "jpeg"], width="stretch")

if uploaded is None:
    st.info("Upload a wrist X-ray above to begin.")
    with st.expander("Project overview"):
        st.write("This app uses ResNet-50 to detect fractures, metal hardware, and text annotations.")
    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

try:
    pil_img = Image.open(uploaded)
    orig_size = pil_img.size

    _, resized, arr, tensor = prepare_image(pil_img)
    key = (uploaded.name, uploaded.size, threshold, intensity)
    if key != st.session_state.last_file_key:
        render_processing()
        st.session_state.last_file_key = key

    probs, idx, pred_class, conf = predict(model, tensor)
    detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]

    caption = get_caption(detected)

    with st.spinner("Generating AI clinical description..."):
        clinical = generate_clinical_description(detected, probs)

    heatmap = generate_gradcam(model, tensor, idx) if GRAD_CAM_AVAILABLE else None
    overlay = overlay_heatmap(arr, heatmap, intensity)

    add_history(uploaded.name, pred_class, conf, threshold, caption, clinical)

    st.success("Analysis complete")
    st.header("Model result")

    # Result summary
    res_left, res_right = st.columns([0.85, 1.45], gap="large")
    with res_left:
        with st.container(border=True):
            st.caption("TOP PREDICTION")
            st.title(pred_class)
            st.subheader(f"{conf:.1%} confidence")
            st.progress(conf)
            st.caption("Model output only—not a calibrated probability.")

    with res_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Classification", pred_class)
        m2.metric("Confidence", f"{conf:.1%}")
        m3.metric("Threshold", f"{threshold:.0%}")

        for name, p in zip(CLASS_NAMES, probs):
            st.metric(name, f"{p:.1%}")

        if conf >= threshold:
            st.info(f"Prediction: {pred_class} with {conf:.1%} confidence (threshold: {threshold:.0%}).")
        else:
            st.success(f"Confidence {conf:.1%} below threshold {threshold:.0%}.")

    # Tabs
    tabs = st.tabs(["Image comparison", "Attention map", "AI Summary", "Clinical Description", "Explanation", "Performance"])

    with tabs[0]:
        col_orig, col_heat = st.columns(2)
        with col_orig:
            st.subheader("Original X-ray")
            st.image(resized, caption=uploaded.name, width="stretch")
        with col_heat:
            st.subheader("Grad-CAM overlay")
            st.image(overlay, caption="Model-attention visualization", width="stretch")

    with tabs[1]:
        st.image(overlay, caption="Warmer regions contributed more strongly.", width="stretch")
        st.warning("Grad-CAM is not a lesion outline; it may respond to artifacts or text.")

    with tabs[2]:
        with st.container(border=True):
            st.subheader("AI Research Summary")
            st.write(f"**Prediction:** {pred_class}")
            st.write(f"**Confidence:** {conf:.1%}")
            st.write(f"**Caption:** {caption}")
            st.divider()
            st.write("**Class Probabilities:**")
            for name, p in zip(CLASS_NAMES, probs):
                st.write(f"- {name}: {p:.1%}")

    with tabs[3]:
        st.subheader("💬 AI-Generated Clinical Description (Ollama)")
        with st.container(border=True):
            st.write(clinical)
        st.caption("This description is generated by a local LLM (Ollama) and is for educational purposes only.")

    with tabs[4]:
        st.subheader("How to read the output")
        st.write(f"The model predicted **{pred_class}** with {conf:.1%} confidence.")
        st.write(f"At threshold {threshold:.0%}, classification is {'positive' if conf >= threshold else 'negative'}.")
        st.subheader("Threshold matters")
        st.write("Lower thresholds increase sensitivity; higher thresholds reduce false positives.")
        st.subheader("Limitations")
        st.write("This output is not a clinical diagnosis.")

    with tabs[5]:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.metric("Test Accuracy", f"{TEST_ACCURACY:.2%}")
            st.write("Per-class F1-scores: ~0.90+")
        with col2:
            st.metric("ROC-AUC", f"{TEST_AUC:.4f}")
            st.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
            st.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
            st.metric("Precision", f"{TEST_PRECISION:.2%}")

    if show_details:
        with st.expander("Technical details"):
            a, b = st.columns(2)
            a.write(f"**Filename:** `{uploaded.name}`")
            a.write(f"**Original dimensions:** `{orig_size[0]} × {orig_size[1]}`")
            a.write(f"**Model input:** `{IMAGE_SIZE}×{IMAGE_SIZE}`")
            b.write(f"**Device:** `{DEVICE}`")
            b.write(f"**Predicted class:** `{pred_class}`")
            b.write(f"**Confidence:** `{conf:.6f}`")
            b.write(f"**Detected:** `{', '.join(detected) if detected else 'None'}`")

    # Download report
    report = build_report(uploaded.name, orig_size, pred_class, conf, threshold, caption, clinical, probs)
    st.download_button("Download research report", data=report, file_name="wristvision_report.txt", mime="text/plain", width="stretch")

except Exception as e:
    st.error(f"Error: {e}")


# ============================================================
# HISTORY
# ============================================================

st.divider()
st.header("Analysis history")
if not st.session_state.analysis_history:
    st.info("No analyses yet.")
else:
    df = pd.DataFrame(st.session_state.analysis_history)
    hist1, hist2, hist3, hist4 = st.columns(4)
    hist1.metric("Total analyses", len(df))
    hist2.metric("Avg confidence", f"{df['Confidence'].mean():.1%}")
    hist3.metric("High (≥50%)", int((df['Confidence'] >= 0.5).sum()))
    hist4.metric("Max confidence", f"{df['Confidence'].max():.1%}")

    st.dataframe(df[["Time", "Filename", "Prediction", "Displayed Confidence", "Caption"]], width="stretch", hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    dcol, ccol = st.columns(2)
    with dcol:
        st.download_button("Download history", data=csv, file_name="history.csv", mime="text/csv", width="stretch")
    with ccol:
        if st.button("Clear history", width="stretch"):
            st.session_state.analysis_history = []
            st.rerun()

st.divider()
st.caption("WristVision AI · Educational Research Use Only")