# -*- coding: utf-8 -*-
# configs_utils.py
import os
import shutil
import random
import gc
import torch
from PIL import Image
import requests

# -------------------------- Global Configuration Parameters (Centralized Management, Easy to Modify) --------------------------
# Path Configuration
IMAGE_DIR = "image"  # Original image folder
KIND_FILE = "kind.txt"  # Category file path
PSEUDO_LABEL_DIR = "pseudo_labels"  # Pseudo label save directory
DATASET_ROOT = "datasets"  # Dataset root directory
YOLO_WORLD_RESULTS_DIR = "runs/results/yolo_world"  # YOLO World prediction results save directory
YOLOV8N_RESULTS_DIR = "runs/results/yolov8n"  # YOLOv8n prediction results save directory
TRAIN_LOG_DIR = "runs/detect/yolov8n_finetune_lworld"  # Training log directory
EVALUATION_DIR = "runs/evaluation"  # Evaluation results save directory

# Model and Training Configuration
CONF_THRESH = 0.15  # Pseudo label confidence threshold
TRAIN_RATIO = 0.7  # Train/validation set split ratio
EPOCHS = 20  # Maximum fine-tuning epochs
BATCH_SIZE = 8  # Memory optimization: batch size = 1
DEVICE = "cpu"  # Running device (CPU)
IOU_THRESH = 0.4  # IOU threshold for NMS

# -------------------------- Common Utility Functions --------------------------
def load_classes(kind_file_path):
    """Load category list (common function, called by multiple modules)"""
    if not os.path.exists(kind_file_path):
        raise FileNotFoundError(f"Category file {kind_file_path} does not exist, please check the path")
    with open(kind_file_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    assert len(classes) > 0, "No valid categories in kind.txt, please check the file"
    print(f"Successfully loaded {len(classes)} categories: {classes}")
    return classes

def validate_label_format(label_path, num_classes):
    """Validate whether a single label file format is correct according to YOLO standards (common function)"""
    try:
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            return False
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for line in lines:
            parts = line.split()
            # Must be 5 values (category ID + 4 normalized coordinates)
            if len(parts) != 5:
                return False
            # Category ID must be a valid integer
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= num_classes:
                return False
            # Coordinates must be floats in the range [0, 1], width and height cannot be 0
            coords = list(map(float, parts[1:5]))
            if not all(0 <= x <= 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
                return False
        return True
    except Exception:
        return False

def fix_label_file(label_path, num_classes):
    """Fix single label file format (common function)"""
    try:
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        fixed_lines = []
        for line in lines:
            parts = line.split()
            # Handle 6-value format (with confidence): remove the last value
            if len(parts) == 6:
                parts = parts[:5]
            # Validate 5-value format
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < num_classes:
                        coords = list(map(float, parts[1:5]))
                        # Limit coordinate range to [0, 1], ensure width and height are not 0
                        coords = [max(0.0001, min(0.9999, x)) for x in coords]
                        if coords[2] > 0 and coords[3] > 0:
                            fixed_lines.append(
                                f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
                            )
                except:
                    continue
        # Write back fixed labels
        with open(label_path, "w") as f:
            if fixed_lines:
                f.write("\n".join(fixed_lines) + "\n")
                return True
            else:
                f.write("")
                return False
    except Exception as e:
        print(f"Failed to fix label file {label_path}: {e}")
        return False

def create_dirs(dir_paths):
    """Batch create directories (common function)"""
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Created/confirmed directories: {dir_paths}")

def get_valid_image_paths(image_dir):
    """Get all valid image paths (common function)"""
    valid_ext = (".jpg", ".png", ".jpeg", ".bmp")
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(valid_ext)
    ]
    assert len(image_paths) > 0, f"No valid images in {image_dir} folder, please check the path"
    print(f"Found {len(image_paths)} valid images")
    return image_paths

# -------------------------- Disable SSL Verification (Solve Network Problems, Common Configuration) --------------------------
requests.packages.urllib3.disable_warnings()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -------------------------- Global Variable Initialization --------------------------
# Load category list
CLASSES = load_classes(KIND_FILE)
NUM_CLASSES = len(CLASSES)
