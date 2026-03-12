# -*- coding: utf-8 -*-
# pseudo_label_generator.py
from ultralytics import YOLO
import os
import gc
import torch
from config_utils import (
    IMAGE_DIR, KIND_FILE, PSEUDO_LABEL_DIR, YOLO_WORLD_RESULTS_DIR,
    CONF_THRESH, IOU_THRESH, DEVICE, load_classes, create_dirs, get_valid_image_paths
)

def load_yolo_world_model(classes):
    # Load YOLO World model (adapted for 8.3.x version)
    # Prioritize using local small model, otherwise use large model
    model_candidates = ["yolov8n-world.pt", "yolov8l-world.pt"]
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    if not model_path:
        raise FileNotFoundError("Local YOLO World model not found, please manually download yolov8n-world.pt or yolov8l-world.pt to current directory")
    
    print(f"Loading local model: {model_path}")
    model = YOLO(model_path)
    # 8.3.x version exclusive interface: set custom classes
    model.set_classes(classes)
    print(f"Set {len(classes)} detection classes")
    return model

def generate_pseudo_labels():
    # Core logic: generate YOLO format pseudo labels
    print("\n" + "="*50)
    print("Start executing YOLO World pseudo label generation process")
    print("="*50)
    
    # 1. Initialization (load classes, create directories)
    CLASSES = load_classes(KIND_FILE)
    NUM_CLASSES = len(CLASSES)
    create_dirs([PSEUDO_LABEL_DIR, YOLO_WORLD_RESULTS_DIR])
    
    # 2. Load model
    model = load_yolo_world_model(CLASSES)
    
    # 3. Get valid image paths
    image_paths = get_valid_image_paths(IMAGE_DIR)
    
    # 4. Predict one by one (memory optimization)
    valid_sample_count = 0
    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {img_idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            # Single image prediction, reduce memory usage
            result = model(
                img_path,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                save=False,
                device=DEVICE,
                verbose=False
            )[0]  # Take single result (single image)
            
            # Save prediction result image
            img_name = os.path.basename(img_path)
            pred_save_path = os.path.join(YOLO_WORLD_RESULTS_DIR, img_name)
            result.save(pred_save_path)
            print(f"   - Prediction image saved: {pred_save_path}")
            
            # Generate pseudo label
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(PSEUDO_LABEL_DIR, label_name)
            
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                with open(label_path, "w") as f:
                    f.write("")
                print(f"   - No valid detection boxes, generate empty label")
                continue
            
            # Write valid labels (YOLO format)
            with open(label_path, "w", encoding="utf-8") as f:
                valid_boxes = 0
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if 0 <= cls_id < NUM_CLASSES:
                        xc, yc, w, h = box.xywhn[0].tolist()
                        if all(0 <= x <= 1 for x in [xc, yc, w, h]) and w > 0 and h > 0:
                            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                            valid_boxes += 1
            
            if valid_boxes > 0:
                valid_sample_count += 1
                print(f"   - Generated valid labels: {valid_boxes} detection boxes")
            else:
                print(f"   - No valid labels (filtered invalid boxes)")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   Failed to process image: {e}")
            continue
    
    # 5. Result summary
    print("\n" + "="*50)
    print(f"Pseudo label generation completed!")
    print(f"   - Total images: {len(image_paths)}")
    print(f"   - Valid pseudo label samples: {valid_sample_count}")
    print(f"   - Pseudo label save path: {PSEUDO_LABEL_DIR}")
    print(f"   - Prediction image save path: {YOLO_WORLD_RESULTS_DIR}")
    print("="*50)
    return valid_sample_count

if __name__ == "__main__":
    # When running independently, only generate pseudo labels
    generate_pseudo_labels()
