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
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
        validation_split=0.2,       # use this split if val/ subdir not present
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if not os.path.exists(val_path):
        # Fallback: use validation_split from train dir
        print("  val/ directory not found — using 20% validation split from train/")
        train_data = train_gen.flow_from_directory(
            train_path, target_size=(img_size, img_size),
            batch_size=batch_size, class_mode="categorical",
            subset="training", shuffle=True, seed=42,
        )
        val_data = train_gen.flow_from_directory(
            train_path, target_size=(img_size, img_size),
            batch_size=batch_size, class_mode="categorical",
            subset="validation", shuffle=False, seed=42,
        )
    else:
        train_data = train_gen.flow_from_directory(
            train_path, target_size=(img_size, img_size),
            batch_size=batch_size, class_mode="categorical",
            shuffle=True, seed=42,
        )
        val_data = val_gen.flow_from_directory(
            val_path, target_size=(img_size, img_size),
            batch_size=batch_size, class_mode="categorical",
            shuffle=False,
        )

    return train_data, val_data


def train(args):
    import tensorflow as tf

    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 60)
    print("  NeuroScan AI — EfficientNet-B3 Training")
    print("=" * 60)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Image size : {args.img_size}×{args.img_size}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    # ── Data generators ──────────────────────────────────────────────────────
    print("\n[1/4] Preparing data generators...")
    train_data, val_data = make_generators(args.data_dir, args.img_size, args.batch_size)

    class_indices = train_data.class_indices
    with open(os.path.join(args.model_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)
    print(f"  Classes found: {class_indices}")

    # Compute class weights to handle imbalance
    n_samples = np.array([v for v in train_data.classes])
    class_counts = np.bincount(n_samples)
    total = class_counts.sum()
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    print(f"  Class weights: {class_weights}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[2/4] Building EfficientNet-B3 model...")
    model, base_model = build_model(num_classes=len(class_indices), img_size=args.img_size)
    model.summary(print_fn=lambda x: None)   # suppress verbose output
    print(f"  Total parameters: {model.count_params():,}")

    # ── Phase 1: Warm-up (frozen base) ────────────────────────────────────────
    print(f"\n[3/4] Phase 1: Warm-up training ({args.warmup} epochs, base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_warmup = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "best_warmup.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1,
        ),
    ]

    history_warmup = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.warmup,
        class_weight=class_weights,
        callbacks=callbacks_warmup,
        verbose=1,
    )

    # ── Phase 2: Fine-tuning (unfreeze top layers) ────────────────────────────
    print(f"\n[4/4] Phase 2: Fine-tuning (top {args.unfreeze} layers unfrozen)...")
    unfreeze_top_layers(base_model, args.unfreeze)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),   # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    remaining_epochs = args.epochs - args.warmup
    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "efficientnet_b3_dementia.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=1e-8, verbose=1,
        ),
    ]

    history_finetune = model.fit(
        train_data,
        validation_data=val_data,
        epochs=remaining_epochs,
        initial_epoch=args.warmup,
        class_weight=class_weights,
        callbacks=callbacks_finetune,
        verbose=1,
    )

    # ── Save full training history ────────────────────────────────────────────
    combined_history = {}
    for key in history_warmup.history:
        combined_history[key] = (
            history_warmup.history[key] + history_finetune.history.get(key, [])
        )
    with open(os.path.join(args.model_dir, "training_history.json"), "w") as f:
        json.dump(combined_history, f, indent=2)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation on Validation Set")
    print("=" * 60)
    loss, acc = model.evaluate(val_data, verbose=0)
    print(f"  Validation Loss     : {loss:.4f}")
    print(f"  Validation Accuracy : {acc*100:.2f}%")

    # Per-class report
    val_data.reset()
    y_true = val_data.classes
    y_pred_probs = model.predict(val_data, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        idx_to_class = {v: k for k, v in class_indices.items()}
        target_names = [idx_to_class[i] for i in range(len(class_indices))]
        print("\n  Per-Class Classification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print("  Confusion Matrix:")
        print(cm)
        np.save(os.path.join(args.model_dir, "confusion_matrix.npy"), cm)
    except ImportError:
        print("  (Install scikit-learn for detailed classification report)")

    # Save final model again (in case checkpoint didn't fire on last epoch)
    final_path = os.path.join(args.model_dir, "efficientnet_b3_dementia.h5")
    if not os.path.exists(final_path):
        model.save(final_path)
    print(f"\n  ✅ Model saved to: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    args = get_args()
    train(args)
