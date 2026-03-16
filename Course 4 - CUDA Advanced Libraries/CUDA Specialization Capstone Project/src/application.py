"""
Planetary Object Detection Application

Loads a trained model and classifies every image in data/Test/ as either
"Planetary Object" or "Not Planetary Object" — simulating an incoming
telescope image stream where labels are unknown.

Usage:
    python src/application/application.py
    python src/application/application.py --test-dir path/to/images
    python src/application/application.py --model path/to/model.keras
"""

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

import tensorflow as tf

IMG_SIZE = (128, 128)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "model", "planetary_detector.keras")
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "Test")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "model", "train_model.py")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def collect_images(directory):
    """Recursively find all image files under a directory."""
    image_paths = []
    for dirpath, _, filenames in os.walk(directory):
        for fname in sorted(filenames):
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths


def classify_image(model, image_path):
    """Run inference on a single image. Returns (label, confidence)."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    img = tf.expand_dims(img, 0)

    prob = model.predict(img, verbose=0)[0][0]
    if prob >= 0.5:
        return "Planetary Object", float(prob)
    else:
        return "Not Planetary Object", float(1.0 - prob)


def run(test_dir, model_path):
    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Running inference on: {'GPU' if gpus else 'CPU'}\n")

    # Load model — train first if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Running training script: {TRAIN_SCRIPT}\n")
        import subprocess
        result = subprocess.run([sys.executable, TRAIN_SCRIPT])
        if result.returncode != 0:
            print("ERROR: Training failed. Cannot proceed.")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"ERROR: Training completed but model not found at {model_path}")
            sys.exit(1)

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Collect images
    image_paths = collect_images(test_dir)
    if not image_paths:
        print(f"ERROR: No images found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) in {test_dir}\n")
    print(f"{'Image':<60} {'Classification':<25} {'Confidence':>10}")
    print("-" * 97)

    # Classify each image
    results = []
    for path in image_paths:
        label, confidence = classify_image(model, path)
        display_name = os.path.relpath(path, test_dir)
        print(f"{display_name:<60} {label:<25} {confidence:>9.2%}")
        results.append((display_name, label, confidence))

    # Summary
    n_planetary = sum(1 for _, lbl, _ in results if lbl == "Planetary Object")
    n_not = len(results) - n_planetary
    print("-" * 97)
    print(f"\nSummary: {n_planetary} planetary object(s), {n_not} non-planetary, "
          f"{len(results)} total")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify telescope images as planetary objects or not"
    )
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Directory of images to classify")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained .keras model")
    args = parser.parse_args()

    run(args.test_dir, args.model)
