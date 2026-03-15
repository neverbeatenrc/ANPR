"""
train.py — Phase 0: Custom YOLOv8 License Plate Detector Training
==================================================================
This script fine-tunes a YOLOv8 'nano' base model on a custom license plate
dataset to produce a specialised plate localisation model.

The trained weights (best.pt) are used in pipeline.py as the second-stage
detector to precisely localise license plates within vehicle bounding boxes.

How to Run
----------
  # 1. Prepare your dataset in standard YOLO format:
  #
  #    dataset/
  #    ├── images/
  #    │   ├── train/   <- training images
  #    │   └── val/     <- validation images
  #    └── labels/
  #        ├── train/   <- YOLO .txt annotation files
  #        └── val/
  #
  # 2. Create a dataset.yaml file (see template below) or point DATA_YAML
  #    to your existing one.
  #
  # 3. Run:
  #    python train.py
  #
  # Output artifacts (saved to runs/detect/license_plate_detector/):
  #   - weights/best.pt           <- Best checkpoint (use in pipeline.py)
  #   - weights/last.pt           <- Final epoch checkpoint
  #   - results.csv               <- Per-epoch loss and mAP metrics
  #   - confusion_matrix.png      <- Needed for academic report
  #   - confusion_matrix_normalized.png
  #   - P_curve.png, R_curve.png, PR_curve.png, F1_curve.png
  #   - results.png               <- Combined loss/mAP training curves
  #   - val_batch*_pred.png       <- Sample validation predictions

# ============================================================
# Example dataset.yaml content (save as data/license_plates.yaml):
# ============================================================
#
# path: ./dataset          # Root path of the dataset (relative to dataset.yaml)
# train: images/train
# val:   images/val
#
# nc: 1                    # Number of classes (1 = 'license_plate')
# names: ['license_plate']
# ============================================================

# Recommended Dataset Source:
#   Roboflow Universe — search for "license plate" datasets:
#   https://universe.roboflow.com/
#   Download in "YOLOv8 PyTorch" format, which produces the correct directory
#   structure and a pre-configured data.yaml file.
"""

import argparse
import torch
from ultralytics import YOLO
from pathlib import Path

# ==============================================================================
# Training Configuration
# ==============================================================================

# Path to the dataset config YAML file.
# Edit this to point to your actual dataset.yaml location.
DATA_YAML = 'data/license_plates/data.yaml'

# Base pre-trained YOLOv8 model to fine-tune from.
# yolov8n.pt — Nano (fastest, lightest — recommended for 4GB VRAM)
# yolov8s.pt — Small (better accuracy, still fits in 4GB with batch=8)
BASE_MODEL = 'yolov8n.pt'
DEFAULT_FINETUNE_WEIGHTS = 'runs/detect/license_plate_detector/weights/best.pt'

# ---- GPU / VRAM Optimisation for NVIDIA RTX 2050 (4GB) ----
# batch=8:    Safe for 4GB VRAM with imgsz=640. Increase to 16 if you have headroom.
# imgsz=640:  Standard YOLOv8 input size. Gives best accuracy/speed balance.
# amp=True:   Automatic Mixed Precision (FP16). Halves VRAM usage, speeds training.
# workers=4:  Number of DataLoader workers. Reduce to 2 if CPU is a bottleneck.
# cache=True: Cache images in RAM for faster epochs (only if you have >8GB RAM).
BATCH_SIZE   = 8      # Reduce to 4 if you hit CUDA OOM errors
IMAGE_SIZE   = 640    # Standard YOLO input resolution
EPOCHS       = 10     # Reduced to 10 to demonstrate end-to-end completion in reasonable time
AMP          = True   # Automatic Mixed Precision — critical for 4GB VRAM
WORKERS      = 4      # DataLoader parallel workers
CACHE_IMAGES = False  # Set True if RAM >= 16GB for faster training

# Project output directory and run name
# Results will be saved to: PROJECT_DIR / RUN_NAME /
PROJECT_DIR = 'runs/detect'
RUN_NAME    = 'license_plate_detector'

# --- Optimiser Hyperparameters ---
# lr0=0.01:   Initial learning rate (YOLOv8 default, good starting point)
# lrf=0.01:   Final learning rate = lr0 * lrf
# momentum=0.937, weight_decay=0.0005: SGD defaults, generally robust
# warmup_epochs=3: Let the LR warm up before full training to stabilise
LEARNING_RATE   = 0.01
LR_FINAL_FACTOR = 0.01
MOMENTUM        = 0.937
WEIGHT_DECAY    = 0.0005
WARMUP_EPOCHS   = 3

# --- Augmentation (helpful for small license plate datasets) ---
# These are YOLOv8 augmentation hyperparameters.
# Increase augmentation if your dataset is small (<500 images).
AUGMENT_DEGREES  = 0.0    # No rotation (plates are upright)
AUGMENT_SHEAR    = 5.0    # Slight shearing to simulate camera angle
AUGMENT_MOSAIC   = 1.0    # Mosaic augmentation (strong, helps small datasets)
AUGMENT_MIXUP    = 0.0    # Mixup — not ideal for ANPR, disable


def check_gpu():
    """
    Check CUDA availability and print GPU info.
    Exits with a warning if no GPU is found (training will be very slow on CPU).
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] Detected: {device_name}")
        print(f"[GPU] VRAM:     {vram_gb:.1f} GB")
        print(f"[GPU] AMP (FP16) is {'ENABLED' if AMP else 'DISABLED'}\n")
        return 0  # CUDA device index
    else:
        print("[WARNING] No CUDA GPU detected. Training on CPU will be very slow.")
        print("          Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
        return 'cpu'


def validate_dataset():
    """
    Basic sanity check — ensure the DATA_YAML file exists before starting.
    Prints a helpful error message with instructions if not found.
    """
    yaml_path = Path(DATA_YAML)
    if not yaml_path.exists():
        print(f"\n[ERROR] Dataset YAML not found: {yaml_path.resolve()}")
        print("        Create it with the following content:\n")
        print("        path: ./dataset")
        print("        train: images/train")
        print("        val:   images/val")
        print("        nc: 1")
        print("        names: ['license_plate']\n")
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    print(f"[DATA]  Dataset config found: {yaml_path.resolve()}")


def train(weights: str | None = None):
    """
    Main training entry point.

    Flow:
      1. Check GPU availability.
      2. Validate dataset config.
      3. Load the base YOLOv8 model.
      4. Call model.train() with VRAM-optimised hyperparameters.
      5. Run model.val() to generate final validation metrics and plots.
      6. Print the path to the best weights for use in pipeline.py.
    """
    device = check_gpu()
    validate_dataset()

    # --- Pick training start point ---
    if weights:
        start_weights = Path(weights)
    else:
        preferred = Path(DEFAULT_FINETUNE_WEIGHTS)
        start_weights = preferred if preferred.exists() else Path(BASE_MODEL)

    print(f"[MODEL] Loading start weights: {start_weights}")
    model = YOLO(str(start_weights))

    # --- Start Training ---
    print("\n" + "="*60)
    print("  YOLOv8 License Plate Detector — Training Run")
    print("="*60)
    print(f"  Dataset:     {DATA_YAML}")
    print(f"  Start from:  {start_weights}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  Image size:  {IMAGE_SIZE}")
    print(f"  AMP (FP16):  {AMP}")
    print(f"  Device:      {device}")
    print("="*60 + "\n")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        amp=AMP,                     # FP16 mixed precision — halves VRAM usage
        device=device,
        workers=WORKERS,
        cache=CACHE_IMAGES,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,               # Resume / overwrite existing run directory

        # --- Optimiser ---
        lr0=LEARNING_RATE,
        lrf=LR_FINAL_FACTOR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,

        # --- Augmentation ---
        degrees=AUGMENT_DEGREES,     # No rotation for license plates
        shear=AUGMENT_SHEAR,         # Slight shear for angle variation
        mosaic=AUGMENT_MOSAIC,       # Mosaic helps with small datasets
        mixup=AUGMENT_MIXUP,

        # --- Output / Logging ---
        save=True,                   # Save best.pt and last.pt
        save_period=-1,              # Only save best (-1 = end of training)
        plots=True,                  # Generate confusion matrix + loss curves
        verbose=True,
    )

    # --- Post-Training Validation ---
    # This generates the final mAP@0.5 and mAP@0.5:0.95 scores and saves
    # validation prediction images (val_batch*_pred.png) — useful for reports.
    print("\n[EVAL] Running final validation pass to generate report metrics...")
    val_metrics = model.val(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        plots=True,                  # Saves confusion_matrix.png, PR_curve.png, etc.
        save_json=False,
        project=PROJECT_DIR,
        name=RUN_NAME + '_val',
        exist_ok=True,
    )

    # --- Summary ---
    best_weights = Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'
    print("\n" + "="*60)
    print("  Training Complete!")
    print(f"  Best weights saved to: {best_weights.resolve()}")
    print(f"  mAP@0.5:     {val_metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:{val_metrics.box.map:.4f}")
    print("="*60)
    print("\n  To use in pipeline.py, set:")
    print(f"  PLATE_MODEL_PATH = '{best_weights}'")
    print()

    return str(best_weights)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/fine-tune YOLO plate detector')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Optional checkpoint path to fine-tune from (default: auto-detect previous best.pt)',
    )
    args = parser.parse_args()

    trained_weights_path = train(weights=args.weights)
