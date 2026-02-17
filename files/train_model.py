"""
train_model.py
==============
Training script for NeuroScan AI — EfficientNet-B3 Dementia Classifier.

Dataset:  Kaggle Alzheimer's MRI Dataset (Augmented)
          https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
Classes:  Non Demented | Very Mild Demented | Mild Demented | Moderate Demented
Target:   97%+ validation accuracy with 90-98% prediction confidence

Usage:
    python train_model.py --data_dir ./data --epochs 50 --batch_size 32

Output:
    model/efficientnet_b3_dementia.h5
    model/training_history.json
    model/class_indices.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 for dementia classification")
    parser.add_argument("--data_dir",   type=str, default="./data",   help="Root data directory (train/ and val/ subdirs)")
    parser.add_argument("--model_dir",  type=str, default="./model",  help="Output directory for saved model")
    parser.add_argument("--epochs",     type=int, default=50,         help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32,         help="Batch size")
    parser.add_argument("--img_size",   type=int, default=300,        help="Input image size (default 300 for B3)")
    parser.add_argument("--lr",         type=float, default=1e-4,     help="Initial learning rate")
    parser.add_argument("--warmup",     type=int, default=20,         help="Warm-up epochs before fine-tuning")
    parser.add_argument("--unfreeze",   type=int, default=30,         help="Number of top layers to unfreeze for fine-tuning")
    return parser.parse_args()


def build_model(num_classes: int = 4, img_size: int = 300, dropout: float = 0.3):
    """
