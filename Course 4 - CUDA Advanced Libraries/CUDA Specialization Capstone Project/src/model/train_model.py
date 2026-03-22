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

# Suppress noisy PNG iCCP warnings and TF info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
SEED = 42
TARGET_NEGATIVE_RATIO = 0.6

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
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = preprocess_image_tensor(img)
    return img, label


def preprocess_image_tensor(img):
    """Normalize and resize while preserving aspect ratio via padding."""
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(
        img,
        target_height=IMG_SIZE[0],
        target_width=IMG_SIZE[1],
        antialias=True,
    )
    return img


def is_valid_image(filepath):
    """Check file magic bytes to verify it's actually JPEG, PNG, GIF, or BMP."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
        if len(header) < 4:
            return False
        # JPEG: FF D8 FF
        if header[:3] == b'\xff\xd8\xff':
            return True
        # PNG: 89 50 4E 47
        if header[:4] == b'\x89PNG':
            return True
        # GIF: GIF87a or GIF89a
        if header[:3] == b'GIF':
            return True
        # BMP: BM
        if header[:2] == b'BM':
            return True
        return False
    except (IOError, OSError):
        return False


def gather_paths_and_labels(root_dir):
    """Recursively collect (path, label) pairs.

    Everything under <root>/Positive/* → label 1
    Everything under <root>/Negative/* → label 0
    Validates each file's actual format (not just extension).
    """
    paths, labels = [], []
    skipped = []
    for label_name, label_val in [("Positive", 1), ("Negative", 0)]:
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                    fpath = os.path.join(dirpath, fname)
                    if is_valid_image(fpath):
                        paths.append(fpath)
                        labels.append(label_val)
                    else:
                        skipped.append(fpath)
    if skipped:
        print(f"  WARNING: Skipped {len(skipped)} file(s) with invalid format:")
        for s in skipped[:10]:
            print(f"    - {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
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


def augment_negative_image(img, label):
    """Apply moderate augmentation to hard-negative examples."""
    pad_y = IMG_SIZE[0] // 20
    pad_x = IMG_SIZE[1] // 20
    img = tf.image.resize_with_crop_or_pad(
        img,
        target_height=IMG_SIZE[0] + (2 * pad_y),
        target_width=IMG_SIZE[1] + (2 * pad_x),
    )
    img = tf.image.random_crop(img, size=[IMG_SIZE[0], IMG_SIZE[1], 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def split_train_validation(paths, labels):
    """Split positives and negatives separately to preserve class balance."""
    positive_paths = [path for path, label in zip(paths, labels) if label == 1]
    negative_paths = [path for path, label in zip(paths, labels) if label == 0]

    rng = np.random.default_rng(SEED)
    rng.shuffle(positive_paths)
    rng.shuffle(negative_paths)

    pos_split = int(len(positive_paths) * (1 - VALIDATION_SPLIT))
    neg_split = int(len(negative_paths) * (1 - VALIDATION_SPLIT))

    train_paths = positive_paths[:pos_split] + negative_paths[:neg_split]
    train_labels = [1] * pos_split + [0] * neg_split
    val_paths = positive_paths[pos_split:] + negative_paths[neg_split:]
    val_labels = [1] * (len(positive_paths) - pos_split) + [0] * (len(negative_paths) - neg_split)

    combined_train = list(zip(train_paths, train_labels))
    combined_val = list(zip(val_paths, val_labels))
    rng.shuffle(combined_train)
    rng.shuffle(combined_val)

    return combined_train, combined_val


def build_balanced_train_dataset(paths, labels):
    """Moderately oversample negatives with augmentation to reduce false positives."""
    positive_paths = [path for path, label in zip(paths, labels) if label == 1]
    negative_paths = [path for path, label in zip(paths, labels) if label == 0]

    if not positive_paths or not negative_paths:
        return build_dataset(paths, labels, shuffle=True, augment=True)

    target_negative_count = max(
        len(negative_paths),
        int(np.ceil(len(positive_paths) * TARGET_NEGATIVE_RATIO)),
    )
    negative_repeat = int(np.ceil(target_negative_count / len(negative_paths)))

    positive_ds = tf.data.Dataset.from_tensor_slices(
        (positive_paths, [1] * len(positive_paths))
    )
    positive_ds = positive_ds.shuffle(
        len(positive_paths), seed=SEED, reshuffle_each_iteration=True
    )
    positive_ds = positive_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    positive_ds = positive_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    negative_ds = tf.data.Dataset.from_tensor_slices(
        (negative_paths, [0] * len(negative_paths))
    )
    negative_ds = negative_ds.shuffle(
        len(negative_paths), seed=SEED, reshuffle_each_iteration=True
    )
    negative_ds = negative_ds.repeat(negative_repeat)
    negative_ds = negative_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    negative_ds = negative_ds.map(
        augment_negative_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    negative_ds = negative_ds.take(target_negative_count)

    total_samples = len(positive_paths) + target_negative_count
    train_ds = positive_ds.concatenate(negative_ds)
    train_ds = train_ds.shuffle(total_samples, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(
        "Using adjusted training batches with "
        f"{len(positive_paths)} positives and {target_negative_count} augmented negatives"
    )
    return train_ds


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
        layers.Dense(64, activation="relu"),
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

    print()

    # Gather training data
    paths, labels = gather_paths_and_labels(TRAINING_DIR)
    print(f"Total training images: {len(paths)}")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Positive: {n_pos}  |  Negative: {n_neg}")

    if len(paths) == 0:
        print("ERROR: No images found. Check TRAINING_DIR:", TRAINING_DIR)
        sys.exit(1)

    # Stratified train / validation split keeps both classes represented.
    train_data, val_data = split_train_validation(paths, labels)

    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    train_ds = build_balanced_train_dataset(list(train_paths), list(train_labels))
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
        callbacks=cb,
    )

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
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = preprocess_image_tensor(img)
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