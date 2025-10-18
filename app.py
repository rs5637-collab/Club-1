import os
import sys
import zipfile
import random
import shutil
import subprocess
from pathlib import Path

import torch
import pandas as pd
import yaml

# -----------------------------
# 🔧 SAFE OPENCV INSTALL (FIXES libGL.so.1 ERROR)
# -----------------------------
def install_headless_opencv():
    """Ensure only opencv-python-headless is installed (no GUI deps)."""
    try:
        import cv2
        # If already imported, we assume it's working
        return
    except ImportError:
        pass

    # Uninstall any existing OpenCV variants
    for pkg in ["opencv-python", "opencv-contrib-python", "opencv-python-headless"]:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
        except:
            pass

    # Install headless version cleanly
    print("Installing opencv-python-headless (no GUI)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "opencv-python-headless==4.10.0.84"
    ])

# Run the safe install
install_headless_opencv()
import cv2  # Now safe to import

# -----------------------------
# Rest of your dependencies
# -----------------------------
def ensure_installed(pkg, alias=None):
    try:
        __import__(alias or pkg)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])

ensure_installed("PyYAML", "yaml")
ensure_installed("pandas")
ensure_installed("ultralytics")

# Now import YOLO — it will work
from ultralytics import YOLO

# -----------------------------
# Your original logic (unchanged)
# -----------------------------
BASE = "/workspaces/Club-1"
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

CLASSES = ["fall"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for split in ["train", "validation"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(DATA, split, sub), exist_ok=True)

zip_files = [f for f in os.listdir(BASE) if f.endswith(".zip")]
if not zip_files:
    sys.exit("No dataset zip found in base folder.")

zip_path = os.path.join(BASE, zip_files[0])
extract_path = os.path.join(BASE, "_temp_extract")
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)
print("Dataset extracted.")

imgs, lbls = {}, {}
for root, _, files in os.walk(extract_path):
    for f in files:
        stem, ext = Path(f).stem, Path(f).suffix.lower()
        p = os.path.join(root, f)
        if ext in [".jpg", ".jpeg", ".png"]:
            imgs[stem] = p
        elif ext == ".txt" and f.lower() != "classes.txt":
            lbls[stem] = p

pairs = [(imgs[k], lbls[k]) for k in imgs if k in lbls]
if not pairs:
    sys.exit("Couldn’t find any matching image-label pairs.")

print(f"Found {len(pairs)} usable samples.")

random.shuffle(pairs)
cut = int(0.8 * len(pairs))
train_set, val_set = pairs[:cut], pairs[cut:]

def cp(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, Path(src).name))

for img, lbl in train_set:
    cp(img, os.path.join(DATA, "train/images"))
    cp(lbl, os.path.join(DATA, "train/labels"))

for img, lbl in val_set:
    cp(img, os.path.join(DATA, "validation/images"))
    cp(lbl, os.path.join(DATA, "validation/labels"))

data_yaml = {
    "train": os.path.abspath(os.path.join(DATA, "train/images")),
    "val": os.path.abspath(os.path.join(DATA, "validation/images")),
    "nc": len(CLASSES),
    "names": CLASSES
}
yaml_path = os.path.join(BASE, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)
print("data.yaml ready.")

print("Starting YOLO training...")
model = YOLO("yolov8n.pt")

model.train(
    data=yaml_path,
    epochs=10,
    imgsz=224,
    batch=4,
    project=os.path.join(BASE, "runs"),
    name="train_fast",
    exist_ok=True,
    patience=5,
    workers=0,
    device=DEVICE,
    hsv_h=0, hsv_s=0, hsv_v=0,
    degrees=0, translate=0, scale=0,
    shear=0, fliplr=0, mosaic=0, mixup=0
)

run_dir = os.path.join(BASE, "runs", "train_fast")
best_weight = os.path.join(run_dir, "weights", "best.pt")
results_csv = os.path.join(run_dir, "results.csv")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    final = df.iloc[-1]
    print("\nFinal validation metrics:")
    print(f"Precision  : {final.get('metrics/precision(B)', float('nan')):.3f}")
    print(f"Recall     : {final.get('metrics/recall(B)', float('nan')):.3f}")
    print(f"mAP@0.5    : {final.get('metrics/mAP50(B)', float('nan')):.3f}")
    print(f"mAP@0.5:0.95: {final.get('metrics/mAP50-95(B)', float('nan')):.3f}")

out_zip = os.path.join(BASE, "fall_detection_model_fast.zip")
with zipfile.ZipFile(out_zip, "w") as zf:
    if os.path.exists(best_weight):
        zf.write(best_weight, "best.pt")
    zf.write(yaml_path, "data.yaml")
    plot_img = os.path.join(run_dir, "results.png")
    if os.path.exists(plot_img):
        zf.write(plot_img, "training_results.png")

print(f"\nEverything done. Zipped output -> {out_zip}")