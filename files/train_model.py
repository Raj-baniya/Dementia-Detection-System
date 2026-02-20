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
    Build EfficientNet-B3 transfer learning model.

    Architecture:
        EfficientNet-B3 (ImageNet weights, frozen) →
        GlobalAveragePooling2D →
        BatchNormalization →
        Dense(256, relu) →
        Dropout(0.3) →
        Dense(4, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base_model.trainable = False   # freeze for warm-up phase

    inputs  = tf.keras.Input(shape=(img_size, img_size, 3))
    x       = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    x       = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB3_Dementia")
    return model, base_model


def unfreeze_top_layers(base_model, n_layers: int = 30):
    """Unfreeze the top n layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    print(f"  Fine-tuning: {sum(1 for l in base_model.layers if l.trainable)} / "
          f"{len(base_model.layers)} base layers trainable")


def make_generators(data_dir: str, img_size: int, batch_size: int):
    """Create train and validation data generators with augmentation."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
