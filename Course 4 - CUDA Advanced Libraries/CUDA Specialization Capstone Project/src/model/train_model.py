import os
import sys
import argparse

# Point TF to cuDNN libs installed via pip
site_packages = os.path.join(sys.prefix, "lib", "python3.8", "site-packages")
cudnn_lib = os.path.join(site_packages, "nvidia", "cudnn", "lib")

if os.path.exists(cudnn_lib):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = cudnn_lib + ":" + current_ld
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Application")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "src", "model", "planetary_detector.keras")


def verify_gpu():
    """Verify and log GPU + cuDNN availability. Exits if no GPU found."""
    print("=" * 60)
    print("GPU / cuDNN VERIFICATION")
    print("=" * 60)

    # TensorFlow build info
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Built with CUDA    : {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU     : {tf.test.is_built_with_gpu_support()}")

    # cuDNN version (via TF's internal binding)
    if hasattr(tf.sysconfig, "get_build_info"):
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA version       : {build_info.get('cuda_version', 'N/A')}")
        print(f"cuDNN version      : {build_info.get('cudnn_version', 'N/A')}")

    # Physical GPU devices
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Physical GPUs      : {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name} (type: {gpu.device_type})")
        tf.config.experimental.set_memory_growth(gpu, True)
        details = tf.config.experimental.get_device_details(gpu)
        if details:
            print(f"      Device name  : {details.get('device_name', 'N/A')}")
            print(f"      Compute cap  : {details.get('compute_capability', 'N/A')}")

    # Confirm cuDNN is loadable by running a small convolution on GPU
    if gpus:
        try:
            with tf.device("/GPU:0"):
                x = tf.random.normal([1, 8, 8, 1])
                conv = tf.keras.layers.Conv2D(1, 3, padding="same")
                _ = conv(x)
            print("cuDNN conv test    : PASSED (Conv2D executed on GPU)")
        except Exception as e:
            print(f"cuDNN conv test    : FAILED ({e})")

    print("=" * 60)

    if not gpus:
        print("\nERROR: No GPU detected. This project requires a CUDA-capable "
              "GPU with cuDNN. Exiting.")
        sys.exit(1)

    return True


# ---------------------------------------------------------------------------
# Data loading — walks Positive/Negative subdirectories
# ---------------------------------------------------------------------------
def load_image(path, label):
    """Read, decode, and preprocess a single image."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


def gather_paths_and_labels(root_dir):
    """Recursively collect (path, label) pairs.

    Everything under <root>/Positive/* → label 1
    Everything under <root>/Negative/* → label 0
    """
    paths, labels = [], []
    for label_name, label_val in [("Positive", 1), ("Negative", 0)]:
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(dirpath, fname))
                    labels.append(label_val)
    return paths, labels


def build_dataset(paths, labels, shuffle=True, augment=False):
    """Create a tf.data.Dataset from file paths and labels."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def augment_image(img, label):
    """Apply random augmentations for training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


# ---------------------------------------------------------------------------
# Model — lightweight CNN for binary classification
# ---------------------------------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    verify_gpu()

    # Enable device placement logging so output proves GPU is used
    tf.debugging.set_log_device_placement(True)
    print("\nDevice placement logging enabled — ops will show GPU:0 in output.\n")

    # Gather training data
    paths, labels = gather_paths_and_labels(TRAINING_DIR)
    print(f"Total training images: {len(paths)}")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Positive: {n_pos}  |  Negative: {n_neg}")

    if len(paths) == 0:
        print("ERROR: No images found. Check TRAINING_DIR:", TRAINING_DIR)
        sys.exit(1)

    # Compute class weights to handle imbalance
    total = len(labels)
    class_weight = {
        0: total / (2.0 * max(n_neg, 1)),
        1: total / (2.0 * max(n_pos, 1)),
    }
    print(f"  Class weights: {class_weight}\n")

    # Shuffle and split into train / validation
    combined = list(zip(paths, labels))
    np.random.seed(SEED)
    np.random.shuffle(combined)
    split_idx = int(len(combined) * (1 - VALIDATION_SPLIT))
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    train_ds = build_dataset(list(train_paths), list(train_labels), shuffle=True, augment=True)
    val_ds = build_dataset(list(val_paths), list(val_labels), shuffle=False, augment=False)

    # Build model on GPU — Conv2D layers will use cuDNN kernels automatically
    with tf.device("/GPU:0"):
        model = build_model()
    model.summary()

    # Callbacks
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    # Train on GPU — cuDNN accelerates Conv2D, BatchNorm, and pooling ops
    print("\nStarting GPU-accelerated training (cuDNN backend)...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=cb,
    )

    # Disable verbose device logging for save/eval
    tf.debugging.set_log_device_placement(False)

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Final evaluation on validation set
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation loss: {val_loss:.4f}  |  Validation accuracy: {val_acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def predict(image_path, model_path=MODEL_SAVE_PATH):
    """Load model and predict on a single image. Returns (label, confidence)."""
    model = tf.keras.models.load_model(model_path)
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    img = tf.expand_dims(img, 0)

    prob = model.predict(img)[0][0]
    label = "Planetary Object" if prob >= 0.5 else "Not Planetary Object"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    print(f"{image_path}: {label} ({confidence:.2%})")
    return label, float(confidence)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planetary Object Binary Classifier")
    parser.add_argument("--predict", type=str, default=None,
                        help="Path to an image to classify (skips training)")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        train()