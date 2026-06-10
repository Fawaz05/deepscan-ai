"""
model_training.py
-----------------
Fine-tuning improvements for the MobileNetV2-based deepfake detector.
Loads from the .keras directory (config.json + model.weights.h5),
unfreezes the last 30 layers, applies advanced augmentation, and
trains with callbacks.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  Custom augmentation layers
# ─────────────────────────────────────────────

class GaussianNoise(layers.Layer):
    """Inject Gaussian noise during training for noise robustness."""
    def __init__(self, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs), stddev=self.stddev)
            return inputs + noise
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"stddev": self.stddev})
        return cfg


def build_augmentation_pipeline():
    """Advanced augmentation: rotation, zoom, brightness, contrast, shear, flip + noise."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        GaussianNoise(stddev=0.02),
    ], name="augmentation")


# ─────────────────────────────────────────────
# 2.  Model loader
# ─────────────────────────────────────────────

def load_model_from_keras_dir(model_dir: str) -> keras.Model:
    """
    Load a Keras 3 model stored as a directory containing
    config.json and model.weights.h5.
    """
    config_path  = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.weights.h5")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"model.weights.h5 not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model = keras.models.model_from_json(json.dumps(config))
    model.load_weights(weights_path)
    print(f"[✓] Model loaded from {model_dir}")
    return model


# ─────────────────────────────────────────────
# 3.  Fine-tuning setup
# ─────────────────────────────────────────────

def prepare_for_finetuning(
    model: keras.Model,
    unfreeze_last_n: int = 30,
    learning_rate: float = 1e-5,
) -> keras.Model:
    """
    Unfreeze the last `unfreeze_last_n` layers of MobileNetV2 backbone
    and recompile with a low learning rate.
    """
    # Locate the MobileNetV2 sub-model (index 1 in Sequential)
    backbone = model.layers[1]  # Functional (mobilenetv2_1.00_224)

    # First freeze all backbone layers
    backbone.trainable = True
    total = len(backbone.layers)
    freeze_until = total - unfreeze_last_n
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"[✓] MobileNetV2 backbone: {trainable_count}/{total} layers trainable")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model


# ─────────────────────────────────────────────
# 4.  Dataset helpers
# ─────────────────────────────────────────────

def build_dataset(
    data_dir: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    augment: bool = True,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Build tf.data pipelines from a directory with real/ and fake/ subdirs."""
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    # MobileNetV2 preprocessing
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    aug_pipeline = build_augmentation_pipeline()

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess_train(x, y):
        x = preprocess(tf.cast(x, tf.float32))
        x = aug_pipeline(x, training=True)
        return x, y

    def preprocess_val(x, y):
        x = preprocess(tf.cast(x, tf.float32))
        return x, y

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess_val,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds


def compute_weights(data_dir: str) -> dict:
    """Compute class weights to handle class imbalance."""
    import pathlib
    real_count = len(list(pathlib.Path(data_dir).glob("real/*")))
    fake_count = len(list(pathlib.Path(data_dir).glob("fake/*")))
    labels = [0] * real_count + [1] * fake_count
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(labels))
    return {0: weights[0], 1: weights[1]}


# ─────────────────────────────────────────────
# 5.  Training
# ─────────────────────────────────────────────

def train(
    model_dir: str,
    data_dir: str,
    output_dir: str = "models/finetuned",
    epochs: int = 30,
    batch_size: int = 32,
    unfreeze_last_n: int = 30,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load & prepare
    model = load_model_from_keras_dir(model_dir)
    model = prepare_for_finetuning(model, unfreeze_last_n=unfreeze_last_n)
    model.summary()

    # Data
    train_ds, val_ds = build_dataset(data_dir, batch_size=batch_size)
    class_weights = compute_weights(data_dir)
    print(f"[✓] Class weights: {class_weights}")

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True, mode="max"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.weights.h5"),
            monitor="val_auc", save_best_only=True, save_weights_only=True, mode="max"
        ),
        callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs")),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        class_weight=class_weights,
    )

    # Evaluate
    _evaluate(model, val_ds, output_dir)

    # Save final model
    final_path = os.path.join(output_dir, "model_finetuned_final.keras")
    model.save(final_path)
    print(f"[✓] Final model saved to {final_path}")

    _plot_history(history, output_dir)
    return model


def _evaluate(model: keras.Model, val_ds, output_dir: str):
    """Detailed evaluation with precision, recall, F1, confusion matrix."""
    y_true, y_pred_prob = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_pred_prob.extend(preds.flatten().tolist())
        y_true.extend(y.numpy().flatten().tolist())

    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"[✓] Confusion matrix saved.")


def _plot_history(history, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history.history["auc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("AUC"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    print(f"[✓] Training history plot saved.")


if __name__ == "__main__":
    train(
        model_dir="models/model_finetuned_keras",
        data_dir="data",         # expects data/real/ and data/fake/
        output_dir="models/finetuned",
        epochs=30,
        batch_size=32,
        unfreeze_last_n=30,
    )
