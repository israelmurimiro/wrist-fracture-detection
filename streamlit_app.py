"""
WristVision AI - Streamlit App
Wrist Fracture Detection with Grad-CAM Explainability
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
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt


# ============================================================
# CUSTOM GRAD-CAM (no OpenCV / pytorch-grad-cam)
# ============================================================

class GradCAM:
    """Lightweight Grad-CAM using PyTorch hooks and PIL for resizing."""
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = None
        self.gradients = None
        self.hooks = []

        def forward_hook(module, inp, outp):
            self.activations = outp

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, targets):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = targets[0](output) if callable(targets[0]) else output[0, targets[0]]
        target.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)                           # apply ReLU
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)              # normalize to [0,1]
        return cam.squeeze().detach().cpu().numpy()  # 2D heatmap

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay a heatmap on an RGB image (both in [0,1] range).
    Returns a uint8 RGB image.
    """
    # Resize mask to image size using PIL
    h, w = img.shape[:2]
    mask_pil = Image.fromarray(np.uint8(255 * mask), mode='L')
    mask_resized = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
    mask_resized = np.array(mask_resized) / 255.0

    # Create heatmap using matplotlib's jet colormap
    heatmap = plt.cm.jet(mask_resized)[:, :, :3]  # (H, W, 3) in [0,1]

    # Blend with original image
    alpha = 0.6
    blended = alpha * heatmap + (1 - alpha) * img
    return np.uint8(255 * np.clip(blended, 0, 1))


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
ROC_CURVE_PATH = Path("models/roc_curve.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Performance metrics (from your evaluation)
TEST_AUC = 0.971
TEST_SENSITIVITY = 0.951
TEST_SPECIFICITY = 0.925
TEST_PRECISION = 0.913
TEST_ACCURACY = 0.976

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = None

CLASS_NAMES = ['Fractura', 'Metal', 'Texto']


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
            --surface-3: #0d294f;
            --line: rgba(66, 170, 255, 0.22);
            --line-strong: rgba(66, 190, 255, 0.48);
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

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020a18 0%, #06152a 100%);
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1460px;
            padding-top: 1.55rem;
            padding-bottom: 4rem;
        }

        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
        }

        p, label, .stCaption {
            color: var(--muted);
        }

        div[data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: linear-gradient(
                145deg,
                rgba(11, 37, 72, 0.95),
                rgba(4, 18, 39, 0.96)
            );
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stMetricLabel"] {
            color: #7acbff !important;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 800;
            font-size: 1.85rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        [data-testid="stFileUploader"] {
            padding: 1rem;
            border: 1px dashed var(--line-strong);
            border-radius: 18px;
            background: linear-gradient(
                145deg,
                rgba(9, 32, 63, 0.94),
                rgba(4, 18, 40, 0.94)
            );
        }

        [data-testid="stFileUploader"]:hover {
            border-color: var(--cyan);
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 120px;
            border: none;
            background: transparent;
        }

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

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid rgba(53, 178, 255, 0.44);
            border-radius: 10px;
            background: linear-gradient(115deg, #0d72df, #084a9e);
            color: white;
            font-weight: 700;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cyan);
            color: white;
            box-shadow: 0 8px 24px rgba(22, 139, 255, 0.25);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        [data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, var(--blue), var(--cyan));
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 17px !important;
            background: linear-gradient(
                145deg,
                rgba(10, 34, 67, 0.88),
                rgba(4, 17, 38, 0.90)
            );
        }

        hr {
            border-color: rgba(56, 171, 255, 0.14);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL + PREPROCESSING
# ============================================================

@st.cache_resource
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        st.warning("Model file not found. Please upload an image to trigger download.")
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. "
            "Place baseline_resnet50.pth inside models/checkpoints/"
        )

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def prepare_image(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image, np.ndarray, torch.Tensor]:
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = (
        np.asarray(resized_image)
        .astype(np.float32)
        / 255.0
    )

    input_tensor = (
        image_transform(rgb_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return rgb_image, resized_image, image_array, input_tensor


def predict(model: nn.Module, input_tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    return probs, pred_idx, pred_class, confidence


def create_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_idx: int,
) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heatmap = cam(input_tensor, targets=[lambda out: out[0, target_idx]])
    cam.remove_hooks()
    return heatmap  # 2D numpy array in [0,1]


def create_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    intensity: float,
) -> np.ndarray:
    if heatmap is None:
        return (image_array * 255).astype(np.uint8)
    
    overlay = show_cam_on_image(image_array, heatmap).astype(np.float32) / 255.0

    blended = (
        (1.0 - intensity) * image_array
        + intensity * overlay
    )

    return np.clip(
        blended * 255,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# HELPERS
# ============================================================

def score_band(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "Elevated"
    return "High"


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


def add_history_record(
    *,
    filename: str,
    pred_class: str,
    confidence: float,
    threshold: float,
    caption: str,
) -> None:
    record_key = (
        filename,
        round(confidence, 6),
        round(threshold, 2),
    )

    existing_keys = {
        (
            item["Filename"],
            round(float(item["Confidence"]), 6),
            round(float(item["Threshold"]), 2),
        )
        for item in st.session_state.analysis_history
    }

    if record_key in existing_keys:
        return

    st.session_state.analysis_history.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": filename,
            "Prediction": pred_class,
            "Confidence": round(confidence, 6),
            "Displayed Confidence": f"{confidence:.1%}",
            "Threshold": round(threshold, 2),
            "Caption": caption,
            "Score Band": score_band(confidence),
        }
    )


def build_report(
    *,
    filename: str,
    dimensions: tuple[int, int],
    pred_class: str,
    confidence: float,
    threshold: float,
    caption: str,
    probs: dict,
) -> str:
    return f"""
WRISTVISION AI RESEARCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE
Filename: {filename}
Original dimensions: {dimensions[0]} x {dimensions[1]} pixels
Model input dimensions: {IMAGE_SIZE} x {IMAGE_SIZE} pixels

MODEL OUTPUT
Prediction: {pred_class}
Confidence: {confidence:.4f} ({confidence:.1%})
Decision threshold: {threshold:.2f}
Caption: {caption}
Score band: {score_band(confidence)}

Class Probabilities:
  Fractura: {probs[0]:.4f}
  Metal: {probs[1]:.4f}
  Texto: {probs[2]:.4f}

MODEL
Architecture: ResNet-50
Framework: PyTorch
Runtime device: {DEVICE}

HELD-OUT TEST PERFORMANCE
ROC-AUC: {TEST_AUC:.4f}
Sensitivity: {TEST_SENSITIVITY:.4f}
Specificity: {TEST_SPECIFICITY:.4f}
Precision: {TEST_PRECISION:.4f}
Accuracy: {TEST_ACCURACY:.4f}

LIMITATIONS
This is an educational research prototype, not a medical device.
It must not be used for diagnosis, screening, or treatment decisions.
Grad-CAM does not identify lesions or establish valid medical reasoning.
""".strip()


def render_processing_sequence() -> None:
    progress = st.progress(0)
    status = st.empty()

    steps = [
        ("Preprocessing image", 20),
        ("Running neural network", 58),
        ("Computing Grad-CAM", 84),
        ("Preparing dashboard", 100),
    ]

    for text, value in steps:
        status.caption(f"✓ {text}")
        progress.progress(value)
        time.sleep(0.10)

    status.empty()
    progress.empty()


def render_summary_card(
    *,
    pred_class: str,
    confidence: float,
    threshold: float,
    caption: str,
    probs: list,
) -> None:
    with st.container(border=True):
        st.subheader("AI research summary")

        summary_left, summary_right = st.columns(2)

        with summary_left:
            st.write(f"**Prediction:** {pred_class}")
            st.write(f"**Confidence:** {confidence:.1%}")
            st.write(f"**Score band:** {score_band(confidence)}")

        with summary_right:
            st.write(f"**Decision threshold:** {threshold:.0%}")
            st.write(f"**Caption:** {caption}")
            st.write("**Use:** Educational research only")

        st.write("**Class Probabilities:**")
        prob_cols = st.columns(3)
        for i, name in enumerate(CLASS_NAMES):
            with prob_cols[i]:
                st.metric(name, f"{probs[i]:.1%}")
                st.progress(float(probs[i]))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🦴 WristVision AI")
    st.caption("Wrist X-ray research dashboard")

    st.divider()

    st.subheader("Analysis settings")

    decision_threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Scores at or above this value display as positive."
        ),
    )

    heatmap_intensity = st.slider(
        "Heatmap intensity",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
    )

    show_details = st.toggle(
        "Show technical details",
        value=True,
    )

    st.divider()

    st.subheader("Model performance")

    st.metric("ROC-AUC", f"{TEST_AUC:.3f}")
    st.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
    st.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
    st.metric("Precision", f"{TEST_PRECISION:.2%}")

    st.divider()

    st.subheader("Runtime")
    st.caption(f"Device: {DEVICE}")
    st.caption("Architecture: ResNet-50")
    st.caption(f"Input: {IMAGE_SIZE} × {IMAGE_SIZE}")

    st.warning("Educational research use only. Not for diagnosis.")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([1.35, 1], gap="large")

with header_left:
    st.caption("🔵 LIVE RESEARCH PROTOTYPE")
    st.title("🦴 WristVision AI")
    st.caption("BIOMEDICAL COMPUTER-VISION RESEARCH DASHBOARD")
    st.subheader("See what an AI model notices in a wrist X-ray.")

    st.write(
        "Explore a ResNet-50 wrist X-ray classifier with adjustable "
        "decision thresholds, Grad-CAM attention visualization, "
        "model-performance reporting, and session analysis history."
    )

    badge_one, badge_two, badge_three, badge_four = st.columns(4)
    badge_one.info("ResNet-50")
    badge_two.info("PyTorch")
    badge_three.info("Grad-CAM")
    badge_four.info("3 Classes")

with header_right:
    with st.container(border=True):
        st.subheader("Held-out test performance")

        performance_one, performance_two = st.columns(2)
        performance_three, performance_four = st.columns(2)

        performance_one.metric("ROC-AUC", f"{TEST_AUC:.3f}")
        performance_two.metric(
            "Sensitivity",
            f"{TEST_SENSITIVITY:.2%}",
        )
        performance_three.metric(
            "Specificity",
            f"{TEST_SPECIFICITY:.2%}",
        )
        performance_four.metric(
            "Precision",
            f"{TEST_PRECISION:.2%}",
        )


st.divider()


# ============================================================
# UPLOAD + WORKFLOW
# ============================================================

st.header("Analyze a wrist X-ray")

uploaded_file = st.file_uploader(
    "Drag and drop a PNG, JPG, or JPEG wrist X-ray",
    type=["png", "jpg", "jpeg"],
    width="stretch",
)

workflow_one, workflow_two, workflow_three = st.columns(3)

with workflow_one:
    with st.container(border=True):
        st.caption("01 · UPLOAD")
        st.subheader("⬆️ Add an X-ray")
        st.write("Select a wrist X-ray from your computer.")

with workflow_two:
    with st.container(border=True):
        st.caption("02 · ANALYZE")
        st.subheader("🧠 Run inference")
        st.write("The model generates predictions for 3 classes.")

with workflow_three:
    with st.container(border=True):
        st.caption("03 · INTERPRET")
        st.subheader("👁️ Explore attention")
        st.write("Grad-CAM visualizes influential image regions.")


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:
    st.info("Upload a wrist X-ray above to begin.")

    overview_tab, performance_tab, limitations_tab = st.tabs(
        [
            "Project overview",
            "Model performance",
            "Limitations",
        ]
    )

    with overview_tab:
        st.subheader("Project overview")
        st.write(
            "This application uses transfer learning with ResNet-50 "
            "to detect fractures (Fractura), metal hardware (Metal), "
            "and text annotations (Texto) in wrist X-ray images."
        )
        st.write(
            "The model was trained on the DETECCION DE FRACTURAS dataset "
            "with 20,309 wrist X-ray images."
        )

    with performance_tab:
        st.metric("Test Accuracy", f"{TEST_ACCURACY:.2%}")
        st.metric("ROC-AUC (Fractura)", f"{TEST_AUC:.3f}")

    with limitations_tab:
        st.warning(
            "The dataset has class imbalance (Metal has only 1.9% of annotations). "
            "This project is not appropriate for clinical use."
        )


# ============================================================
# ANALYSIS
# ============================================================

else:
    try:
        uploaded_image = Image.open(uploaded_file)
        original_dimensions = uploaded_image.size

        (
            original_rgb,
            resized_image,
            image_array,
            input_tensor,
        ) = prepare_image(uploaded_image)

        current_file_key = (
            uploaded_file.name,
            uploaded_file.size,
            decision_threshold,
            heatmap_intensity,
        )

        if current_file_key != st.session_state.last_file_key:
            render_processing_sequence()
            st.session_state.last_file_key = current_file_key

        probs, pred_idx, pred_class, confidence = predict(model, input_tensor)

        # Determine detected classes
        detected_classes = []
        for i, name in enumerate(CLASS_NAMES):
            if probs[i] >= decision_threshold:
                detected_classes.append(name)

        caption = get_caption(detected_classes)

        # Generate heatmap
        heatmap = create_gradcam(model, input_tensor, pred_idx)
        overlay = create_overlay(image_array, heatmap, heatmap_intensity)

        is_positive = confidence >= decision_threshold

        add_history_record(
            filename=uploaded_file.name,
            pred_class=pred_class,
            confidence=confidence,
            threshold=decision_threshold,
            caption=caption,
        )

        st.success("Analysis complete")
        st.header("Model result")

        result_left, result_right = st.columns(
            [0.85, 1.45],
            gap="large",
        )

        with result_left:
            with st.container(border=True):
                st.caption("TOP PREDICTION")
                st.title(f"{pred_class}")
                st.subheader(f"{confidence:.1%} confidence")
                st.progress(confidence)
                st.caption(
                    "Model output only—not a calibrated "
                    "clinical probability."
                )

        with result_right:
            metric_one, metric_two, metric_three = st.columns(3)

            with metric_one:
                with st.container(border=True):
                    st.caption("CLASSIFICATION")
                    st.subheader(pred_class)

            with metric_two:
                with st.container(border=True):
                    st.caption("CONFIDENCE")
                    st.subheader(f"{confidence:.1%}")

            with metric_three:
                with st.container(border=True):
                    st.caption("THRESHOLD")
                    st.subheader(f"{decision_threshold:.0%}")

            if is_positive:
                st.info(
                    f"The model predicts **{pred_class}** with {confidence:.1%} confidence "
                    f"(threshold: {decision_threshold:.0%}). "
                    "This is a research-model output, not a diagnosis."
                )
            else:
                st.success(
                    f"The confidence of {confidence:.1%} is below "
                    f"the selected threshold of {decision_threshold:.0%}. "
                    "This does not establish that the X-ray is medically normal."
                )

        (
            comparison_tab,
            attention_tab,
            summary_tab,
            explanation_tab,
            performance_tab,
        ) = st.tabs(
            [
                "Image comparison",
                "Attention map",
                "AI summary",
                "Explanation",
                "Performance",
            ]
        )

        with comparison_tab:
            original_column, heatmap_column = st.columns(2)

            with original_column:
                with st.container(border=True):
                    st.subheader("Original X-ray")
                    st.image(
                        resized_image,
                        caption=uploaded_file.name,
                        width="stretch",
                    )

            with heatmap_column:
                with st.container(border=True):
                    st.subheader("Grad-CAM overlay")
                    st.image(
                        overlay,
                        caption="Model-attention visualization",
                        width="stretch",
                    )

        with attention_tab:
            st.image(
                overlay,
                caption=(
                    "Warmer regions contributed more strongly "
                    "to this model output."
                ),
                width="stretch",
            )

            st.warning(
                "Grad-CAM is not a lesion outline. The model may respond "
                "to borders, text markers, equipment, positioning, or "
                "other dataset artifacts."
            )

        with summary_tab:
            render_summary_card(
                pred_class=pred_class,
                confidence=confidence,
                threshold=decision_threshold,
                caption=caption,
                probs=probs,
            )

        with explanation_tab:
            st.subheader("How to read the output")

            st.write(
                f"The model predicted **{pred_class}** "
                f"with {confidence:.1%} confidence."
            )

            st.write(
                f"At a threshold of **{decision_threshold:.0%}**, "
                f"this is considered {'positive' if is_positive else 'negative'}."
            )

            st.subheader("Why the threshold matters")

            st.write(
                "Lower thresholds usually increase sensitivity but also "
                "increase false positives. Higher thresholds generally "
                "reduce false positives but may miss more positive cases."
            )

            st.subheader("What this result does not mean")

            st.write(
                "It does not confirm a fracture, rule out disease, "
                "or provide a patient's true clinical probability."
            )

        with performance_tab:
            performance_left, performance_right = st.columns([1.4, 1])

            with performance_left:
                st.metric("Test Accuracy", f"{TEST_ACCURACY:.2%}")

                st.write("**Per-Class Metrics:**")
                for name in CLASS_NAMES:
                    st.write(f"- {name}: F1-Score ~0.90+")

            with performance_right:
                st.metric("ROC-AUC", f"{TEST_AUC:.4f}")
                st.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
                st.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
                st.metric("Precision", f"{TEST_PRECISION:.2%}")

        if show_details:
            with st.expander("Technical details"):
                detail_left, detail_right = st.columns(2)

                with detail_left:
                    st.write(f"**Filename:** `{uploaded_file.name}`")
                    st.write(
                        f"**Original dimensions:** "
                        f"`{original_dimensions[0]} × "
                        f"{original_dimensions[1]}`"
                    )
                    st.write(
                        f"**Model input:** `{IMAGE_SIZE} × {IMAGE_SIZE}`"
                    )
                    st.write(f"**Image mode:** `{original_rgb.mode}`")

                with detail_right:
                    st.write(f"**Runtime device:** `{DEVICE}`")
                    st.write(f"**Predicted class:** `{pred_class}`")
                    st.write(f"**Confidence:** `{confidence:.6f}`")
                    st.write(
                        f"**Threshold:** `{decision_threshold:.2f}`"
                    )
                    st.write(f"**Caption:** `{caption}`")

        report = build_report(
            filename=uploaded_file.name,
            dimensions=original_dimensions,
            pred_class=pred_class,
            confidence=confidence,
            threshold=decision_threshold,
            caption=caption,
            probs={name: probs[i] for i, name in enumerate(CLASS_NAMES)},
        )

        st.download_button(
            "Download research report",
            data=report,
            file_name="wristvision_research_report.txt",
            mime="text/plain",
            width="stretch",
        )

    except UnidentifiedImageError:
        st.error(
            "The selected file could not be read as an image. "
            "Please upload a valid PNG, JPG, or JPEG file."
        )
    except Exception as error:
        st.error(f"Could not process the uploaded image: {error}")


# ============================================================
# ANALYSIS HISTORY
# ============================================================

st.divider()
st.header("Analysis history")

if not st.session_state.analysis_history:
    st.info(
        "Analyzed X-rays will appear here during this browser session."
    )

else:
    history_df = pd.DataFrame(st.session_state.analysis_history)
    history_conf = history_df["Confidence"]

    history_one, history_two, history_three, history_four = st.columns(4)

    history_one.metric("Total analyses", len(history_df))
    history_two.metric(
        "Average confidence",
        f"{history_conf.mean():.1%}",
    )

    positive_count = (
        history_df["Confidence"] >= 0.5
    ).sum()

    history_three.metric("High confidence (≥50%)", int(positive_count))
    history_four.metric(
        "Max confidence",
        f"{history_conf.max():.1%}",
    )

    display_df = history_df[
        [
            "Time",
            "Filename",
            "Prediction",
            "Displayed Confidence",
            "Threshold",
            "Caption",
        ]
    ].iloc[::-1]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Time": st.column_config.TextColumn("Analyzed"),
            "Filename": st.column_config.TextColumn("File"),
            "Prediction": st.column_config.TextColumn("Prediction"),
            "Displayed Confidence": st.column_config.TextColumn("Confidence"),
            "Threshold": st.column_config.NumberColumn(
                "Threshold",
                format="%.2f",
            ),
            "Caption": st.column_config.TextColumn("Caption"),
        },
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    download_column, clear_column = st.columns(2)

    with download_column:
        st.download_button(
            "Download analysis history",
            data=csv_data,
            file_name="wristvision_analysis_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with clear_column:
        if st.button("Clear history", width="stretch"):
            st.session_state.analysis_history = []
            st.session_state.last_file_key = None
            st.rerun()


st.divider()
st.caption(
    "WristVision AI · Biomedical Engineering & Computer Vision Project · "
    "Educational Research Use Only"
)
