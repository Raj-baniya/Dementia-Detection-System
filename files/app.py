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
