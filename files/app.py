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
