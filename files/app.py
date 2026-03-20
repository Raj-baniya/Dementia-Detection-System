"""
NeuroScan AI - Dementia Detection via MRI
Streamlit Application
Model: EfficientNet-B3 | 4-Class Classification
"""

import streamlit as st
import numpy as np
import os
import json
from PIL import Image, ImageOps
import io
import time
import base64
from pathlib import Path

# ─── Page Config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Dementia Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports that may fail gracefully ────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
INPUT_SIZE   = (300, 300)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATE_PATHS = [
    Path(r"F:\Dementia-MiniProject\final_model\content\drive\MyDrive\DementiaProject\checkpoints_efficientnet\final_model.pth"),
    Path(r"F:\Dementia-MiniProject\best_model\content\drive\MyDrive\DementiaProject\checkpoints_efficientnet\best_model.pth"),
]
HISTORY_JSON_PATH = PROJECT_ROOT / "training_history.json"
HISTORY_IMG_PATH = PROJECT_ROOT / "training_history.png"
HISTORY_IMG_ALT_PATH = PROJECT_ROOT / "training_history (1).png"
CONFUSION_IMG_PATH = PROJECT_ROOT / "confusion_matrix.png"
MODEL_INFO_PATH = PROJECT_ROOT / "model_info.txt"

RAW_TO_DISPLAY_CLASS = {
    "NonDemented": "Non Demented",
    "VeryMildDemented": "Very Mild Demented",
    "MildDemented": "Mild Demented",
    "ModerateDemented": "Moderate Demented",
}

CLASS_COLORS = {
    "Non Demented":       "#10B981",
    "Very Mild Demented": "#F59E0B",
    "Mild Demented":      "#F97316",
    "Moderate Demented":  "#EF4444",
}

CLASS_INFO = {
    "Non Demented": {
        "icon": "✅",
        "description": "No detectable signs of dementia-related atrophy observed in this scan.",
        "recommendation": "Routine monitoring recommended. Maintain healthy lifestyle.",
    },
    "Very Mild Demented": {
        "icon": "⚠️",
        "description": "Early structural changes detected, often pre-symptomatic.",
        "recommendation": "Early intervention and cognitive assessments are advised.",
    },
    "Mild Demented": {
        "icon": "🔶",
        "description": "Noticeable atrophy patterns consistent with mild cognitive impairment.",
        "recommendation": "Neurologist consultation strongly recommended.",
    },
    "Moderate Demented": {
        "icon": "🔴",
        "description": "Significant atrophy and marked cognitive decline indicators present.",
        "recommendation": "Immediate specialist consultation and care planning required.",
    },
}

def display_class_name(raw_name: str) -> str:
    return RAW_TO_DISPLAY_CLASS.get(raw_name, raw_name)


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    if not HISTORY_JSON_PATH.exists():
        return {}
    try:
        with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_model_info_text() -> str:
    if not MODEL_INFO_PATH.exists():
        return ""
    return MODEL_INFO_PATH.read_text(encoding="utf-8")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

      :root {
        --bg-primary:   #0A0E1A;
        --bg-card:      #111827;
        --bg-card2:     #1A2235;
        --accent-blue:  #3B82F6;
        --accent-cyan:  #06B6D4;
        --accent-green: #10B981;
        --text-primary: #F1F5F9;
        --text-muted:   #94A3B8;
        --border:       rgba(255,255,255,0.08);
        --shadow:       0 8px 32px rgba(0,0,0,0.4);
      }

      .stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }

      /* Sidebar */
      [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
      }

      /* Cards */
      .ns-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: var(--shadow);
      }
      .ns-card-accent {
        background: linear-gradient(135deg, #1A2235 0%, #0F172A 100%);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
      }

      /* Metric tiles */
      .metric-tile {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
      }
      .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
      }
      .metric-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
      }

      /* Result badge */
      .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 14px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 12px 0;
      }

      /* Confidence bar */
      .conf-bar-wrap { margin: 6px 0; }
      .conf-bar-label {
        display: flex;
        justify-content: space-between;
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-bottom: 4px;
      }
      .conf-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
      }
      .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease;
      }

      /* Section headings */
      .section-heading {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        border-left: 3px solid var(--accent-blue);
        padding-left: 12px;
        margin: 20px 0 14px 0;
      }

      /* Disclaimer */
      .disclaimer-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 16px 20px;
        color: #FCA5A5;
        font-size: 0.84rem;
        line-height: 1.6;
        margin-top: 12px;
      }

      /* Hero header */
      .hero {
        text-align: center;
        padding: 40px 20px 30px;
      }
      .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
      }
      .hero p {
        color: var(--text-muted);
        font-size: 1rem;
        max-width: 600px;
        margin: 0 auto;
      }

      /* Pill tag */
      .pill {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        color: #93C5FD;
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
      }

      /* Upload zone */
      [data-testid="stFileUploadDropzone"] {
        background: rgba(59,130,246,0.05) !important;
        border: 2px dashed rgba(59,130,246,0.4) !important;
        border-radius: 16px !important;
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
      }
      .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        border-radius: 8px;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.2) !important;
        color: #93C5FD !important;
      }

      /* Remove default Streamlit padding */
      .block-container { padding-top: 1rem !important; }

      /* Table styling */
      .stDataFrame { border-radius: 12px; overflow: hidden; }

      /* Success / warning / error */
      .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px !important;
      }

      h1, h2, h3, h4 { color: var(--text-primary) !important; }
      p, li { color: var(--text-muted); }

      /* Scrollbar */
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: var(--bg-primary); }
      ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load real PyTorch EfficientNet-B3 checkpoint. Returns (bundle, status_message)."""
    if not TORCH_AVAILABLE:
        return None, "PyTorch/timm not installed. Run: pip install torch torchvision timm"

    model_path = next((p for p in MODEL_CANDIDATE_PATHS if p.exists()), None)
    if model_path is None:
        return None, "Real model checkpoint not found at configured .pth path(s)."

    class DementiaEfficientNetB3(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            self.head = nn.Sequential(
                nn.Linear(1536, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(128, 4),
            )

        def forward(self, x):
            feats = self.backbone(x)
            return self.head(feats)

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu")
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            return None, "Unsupported checkpoint format. Expected a state_dict under key 'model'."

        model = DementiaEfficientNetB3()
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        raw_classes = checkpoint.get(
            "classes",
            ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
        )
        display_classes = [display_class_name(c) for c in raw_classes]
        bundle = {
            "model": model,
            "classes": display_classes,
            "val_acc": checkpoint.get("val_acc"),
            "val_auroc": checkpoint.get("val_auroc"),
            "path": str(model_path),
        }
        status = f"✅ Real model loaded: {model_path.name}"
        return bundle, status
    except Exception as e:
        return None, f"Error loading .pth model: {e}"


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image):
    """Preprocess image for PyTorch EfficientNet-B3 inference."""
    img = pil_image.convert("RGB")
    img = img.resize(INPUT_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


# ─── Inference ────────────────────────────────────────────────────────────────
def build_tta_views(pil_image: Image.Image, num_views: int = 8):
    """Build deterministic TTA views (flip/rotate/crop variants)."""
    base = pil_image.convert("RGB")
    views = [base]

    # Horizontal mirror for symmetry robustness on MRI slices.
    views.append(ImageOps.mirror(base))

    # Mild rotations to reduce orientation sensitivity.
    views.append(base.rotate(-8, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))
    views.append(base.rotate(8, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))

    # Center crop + resize to stabilize border/background effects.
    w, h = base.size
    crop_margin_w = max(int(w * 0.04), 1)
    crop_margin_h = max(int(h * 0.04), 1)
    crop = base.crop((crop_margin_w, crop_margin_h, w - crop_margin_w, h - crop_margin_h)).resize(base.size, Image.LANCZOS)
    views.append(crop)

    # Additional mild transforms for higher TTA budgets.
    views.append(ImageOps.mirror(crop))
    views.append(base.rotate(-5, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))
    views.append(base.rotate(5, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))
    views.append(base.rotate(-12, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))
    views.append(base.rotate(12, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0)))
    views.append(ImageOps.flip(base))
    views.append(ImageOps.mirror(base.rotate(6, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))))

    return views[:max(1, num_views)]


def run_inference(model_bundle, pil_image: Image.Image, tta_views: int = 8, temperature: float = 0.9):
    """
    Run model inference and return (predicted_class, confidence, probabilities_dict).
    Applies temperature scaling (T=0.5) to sharpen predictions.
    """
    model = model_bundle["model"]
    views = build_tta_views(pil_image, num_views=tta_views)

    with torch.no_grad():
        logits_list = []
        for view in views:
            inputs = preprocess_image(view)
            logits_list.append(model(inputs))

        mean_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)

        # Temperature scaling: T<1 sharpens, T>1 smooths (better calibration typically near 1).
        temp = max(float(temperature), 1e-3)
        calibrated_logits = mean_logits / temp
        probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()[0]

    class_idx = int(np.argmax(probs))
    classes = model_bundle.get("classes", CLASS_NAMES)
    pred_class = classes[class_idx] if class_idx < len(classes) else CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])
    return pred_class, confidence


# ─── GradCAM ─────────────────────────────────────────────────────────────────
def generate_gradcam(model_bundle, pil_image: Image.Image, class_idx: int):
    """Generate GradCAM heatmap. Returns numpy array or None."""
    if not TORCH_AVAILABLE or model_bundle is None:
        return None
    try:
        model = model_bundle["model"]
        conv_output = []
        gradients = []

        def fwd_hook(_module, _inp, out):
            conv_output.append(out)

        def bwd_hook(_module, _gin, gout):
            gradients.append(gout[0])

        target_layer = model.backbone.conv_head
        hook1 = target_layer.register_forward_hook(fwd_hook)
        hook2 = target_layer.register_full_backward_hook(bwd_hook)

        inputs = preprocess_image(pil_image)
        outputs = model(inputs)
        score = outputs[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        acts = conv_output[0][0].detach().cpu().numpy()
        grads = gradients[0][0].detach().cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * acts, axis=0)

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        if CV2_AVAILABLE:
            cam = cv2.resize(cam, INPUT_SIZE)
        else:
            cam = np.array(Image.fromarray(cam).resize(INPUT_SIZE))
        hook1.remove()
        hook2.remove()
        return cam
    except Exception:
        return None


def plot_training_history(history: dict):
    """Plotly chart from real training history json."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Loss", "Accuracy", "AUROC", "Learning Rate"),
        vertical_spacing=0.18,
    )

    if history.get("train_loss"):
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#06B6D4")), row=1, col=1)
    if history.get("val_loss"):
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#EF4444")), row=1, col=1)
    if history.get("train_acc"):
        fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], mode="lines", name="Train Acc", line=dict(color="#10B981")), row=1, col=2)
    if history.get("val_acc"):
        fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], mode="lines", name="Val Acc", line=dict(color="#3B82F6")), row=1, col=2)
    if history.get("val_auroc"):
        fig.add_trace(go.Scatter(x=epochs, y=history["val_auroc"], mode="lines", name="Val AUROC", line=dict(color="#F59E0B")), row=2, col=1)
    if history.get("lr"):
        fig.add_trace(go.Scatter(x=epochs, y=history["lr"], mode="lines", name="LR", line=dict(color="#A78BFA")), row=2, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(color="#94A3B8")),
        font=dict(family="Inter"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748B"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748B"))
    return fig


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def plot_probability_bars(prob_dict: dict):
    """Plotly horizontal bar chart for class probabilities."""
    names  = list(prob_dict.keys())
    values = [v * 100 for v in prob_dict.values()]
    colors = [CLASS_COLORS[n] for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=colors, opacity=0.9),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            ticksuffix="%",
            tickfont=dict(color="#64748B"),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#94A3B8", size=12),
            autorange="reversed",
        ),
        font=dict(family="Inter"),
        showlegend=False,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    """Plotly annotated confusion matrix heatmap."""
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annotations = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cm[i,j]}</b><br><span style='font-size:10px'>{cm_norm[i,j]*100:.1f}%</span>",
                showarrow=False,
                font=dict(color="white" if cm_norm[i,j] > 0.4 else "#94A3B8", size=11),
                xref="x", yref="y",
            ))

    short_names = ["Non\nDemented", "Very Mild", "Mild", "Moderate"]
    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=short_names,
        y=short_names,
        colorscale=[[0, "#0F172A"], [0.5, "#1E3A5F"], [1, "#3B82F6"]],
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#64748B"),
            tickformat=".0%",
            len=0.85,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        annotations=annotations,
        xaxis=dict(title=dict(text="Predicted", font=dict(color="#64748B")), tickfont=dict(color="#94A3B8"), showgrid=False),
        yaxis=dict(title=dict(text="Actual", font=dict(color="#64748B")), tickfont=dict(color="#94A3B8"), autorange="reversed"),
        title=dict(text="Confusion Matrix — Validation Set (n=3,614)", font=dict(color="#94A3B8", size=13), x=0.5),
    )
    return fig


def plot_model_comparison():
    """Plotly grouped bar chart comparing models."""
    models  = list(MODEL_COMPARISON.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#3B82F6", "#06B6D4", "#10B981", "#F59E0B"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
