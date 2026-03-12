# -*- coding: utf-8 -*-
# trainer.py
from ultralytics import YOLO
import os
import shutil
import random
from config_utils import (
    IMAGE_DIR, PSEUDO_LABEL_DIR, DATASET_ROOT, CLASSES, NUM_CLASSES,
    TRAIN_RATIO, EPOCHS, BATCH_SIZE, DEVICE, YOLOV8N_RESULTS_DIR,
    validate_label_format, create_dirs
)

def organize_dataset():
    # Convert pseudo labels to YOLOv8 training dataset structure
    print("\nStart organizing dataset...")
    
    # Create dataset directory structure
    dirs_to_create = [
        os.path.join(DATASET_ROOT, "images", "train"),
        os.path.join(DATASET_ROOT, "images", "val"),
        os.path.join(DATASET_ROOT, "labels", "train"),
        os.path.join(DATASET_ROOT, "labels", "val")
    ]
    create_dirs(dirs_to_create)
    
    # Collect all valid pseudo label images
    label_files = [f for f in os.listdir(PSEUDO_LABEL_DIR) if f.endswith(".txt")]
    valid_img_names = []
    
    for label_file in label_files:
        try:
            with open(os.path.join(PSEUDO_LABEL_DIR, label_file), "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Strictly validate label format to ensure it meets YOLOv8 requirements
            valid_labels = []
            for line in lines:
                parts = line.split()
                # YOLO format must be exactly 5 values, no more no less
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        # Ensure category ID is within valid range
                        if 0 <= cls_id < NUM_CLASSES:
                            coords = list(map(float, parts[1:5]))
                            # Check if coordinate range is valid and width/height are not 0
                            if all(0 <= x <= 1 for x in coords) and coords[2] > 0 and coords[3] > 0:
                                # Reformat coordinates to ensure consistent precision
                                valid_labels.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                    except ValueError:
                        continue
            # Handle old format labels that may contain confidence (convert to standard format)
            if not valid_labels and len(lines) > 0:
                for line in lines:
                    parts = line.split()
                    # If there are 6 values, try removing the last confidence value
                    if len(parts) == 6:
                        try:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < NUM_CLASSES:
                                coords = list(map(float, parts[1:5]))
                                if all(0 <= x <= 1 for x in coords) and coords[2] > 0 and coords[3] > 0:
                                    valid_labels.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Failed to read label file: {e}")
            continue
                  
        if len(valid_labels) > 0:  # Ensure there are valid label lines
                    # Rewrite label file, keeping only valid labels
                    with open(os.path.join(PSEUDO_LABEL_DIR, label_file), "w") as f:
                        f.write("\n".join(valid_labels) + "\n")
                    
                    # Try different image formats
                    img_name_base = os.path.splitext(label_file)[0]
                    found = False
                    for ext in ["jpg", "png", "jpeg", "bmp"]:
                        img_name = f"{img_name_base}.{ext}"
                        if os.path.exists(os.path.join(IMAGE_DIR, img_name)):
                            valid_img_names.append((img_name_base, ext))
                            found = True
                            break
                    if not found:
                        print(f"Warning: Image file corresponding to label file {label_file} does not exist")
        # No additional except block needed here, keep original processing logic
    
    # Check valid sample count
    if len(valid_img_names) == 0:
        print("Warning: No valid pseudo label samples! Please lower CONF_THRESH to a lower value")
        return None
    elif len(valid_img_names) < 5:
        print(f"Warning: Only {len(valid_img_names)} valid pseudo label samples! Too few samples may affect training results")
    
    # Split train/validation sets - ensure at least 1 sample in validation set
    random.shuffle(valid_img_names)
    if len(valid_img_names) == 1:
        train_samples = valid_img_names
        val_samples = valid_img_names.copy()  # When there's only one sample, use it for both training and validation
    else:
        # Ensure at least 1 sample in validation set
        val_size = max(1, int(len(valid_img_names) * (1 - TRAIN_RATIO)))
        train_size = len(valid_img_names) - val_size
        train_samples = valid_img_names[:train_size]
        val_samples = valid_img_names[train_size:]
    
    print(f"Split into training set: {len(train_samples)} images, validation set: {len(val_samples)} images")
    
    # Copy files
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_name, ext in samples:
            try:
                # Copy image
                src_img = os.path.join(IMAGE_DIR, f"{img_name}.{ext}")
                dst_img = os.path.join(DATASET_ROOT, "images", split, f"{img_name}.{ext}")
                shutil.copyfile(src_img, dst_img)
                # Copy label
                src_label = os.path.join(PSEUDO_LABEL_DIR, f"{img_name}.txt")
                dst_label = os.path.join(DATASET_ROOT, "labels", split, f"{img_name}.txt")
                shutil.copyfile(src_label, dst_label)
            except Exception as e:
                print(f"Error copying file {img_name}.{ext}: {e}")
    
    # Generate data.yaml
    path = os.path.abspath(DATASET_ROOT).replace('\\', '/')
    # Manually construct names string to avoid backslash issues in f-strings
    names_str = '[' + ', '.join([f'"{c}"' for c in CLASSES]) + ']'
    data_yaml = f"""path: {path}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {names_str}"""
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml.strip())
    
    # Verify generated yaml file
    print(f"Dataset organization completed! data.yaml path: {yaml_path}")
    print(f"Final valid samples: {len(valid_img_names)}")

    print(f"Dataset organization completed! data.yaml path: {yaml_path}")
    return yaml_path

def fix_all_labels():
    # Fix all label files to ensure correct format
    print("\nFixing label file formats...")
    fixed_count = 0
    
    for split in ["train", "val"]:
        label_dir = os.path.join(DATASET_ROOT, "labels", split)
        if not os.path.exists(label_dir):
            continue
            
        for label_file in os.listdir(label_dir):
            if label_file.endswith(".txt"):
                label_path = os.path.join(label_dir, label_file)
                if not validate_label_format(label_path, NUM_CLASSES):
                    # Try to fix the label file
                    try:
                        with open(label_path, "r") as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                        fixed_lines = []
                        for line in lines:
                            parts = line.split()
                            # Handle 6-value format that may contain confidence
                            if len(parts) == 6:
                                parts = parts[:5]  # Remove confidence
                            
                            if len(parts) == 5:
                                try:
                                    cls_id = int(parts[0])
                                    if 0 <= cls_id < NUM_CLASSES:
                                        coords = list(map(float, parts[1:5]))
                                        # Limit coordinate range
                                        coords = [max(0, min(1, x)) for x in coords]
                                        # Ensure width and height are not 0
                                        if coords[2] > 0 and coords[3] > 0:
                                            fixed_lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                                except:
                                    continue
                        
                        # Write back fixed labels
                        with open(label_path, "w") as f:
                            if fixed_lines:
                                f.write("\n".join(fixed_lines) + "\n")
                                fixed_count += 1
                            else:
                                # No valid labels, clear the file
                                f.write("")
                    except Exception as e:
                        print(f"Failed to fix label file {label_file}: {e}")
    
    print(f"Fix completed! Processed {fixed_count} label files")
    return fixed_count

def test_yolov8n_prediction():
    # Use pre-trained YOLOv8n model for prediction and save results
    print("\nExecuting YOLOv8n pre-trained model prediction test...")
    
    # Create YOLOv8n prediction results save directory
    create_dirs([YOLOV8N_RESULTS_DIR])
    
    try:
        # Load pre-trained YOLOv8n model
        model = YOLO("yolov8n.pt")
        print("Successfully loaded pre-trained YOLOv8n model")
        
        # Get some test images
        test_img_paths = []
        for f in os.listdir(IMAGE_DIR):
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                test_img_paths.append(os.path.join(IMAGE_DIR, f))
                if len(test_img_paths) >= 10:  # Only test 10 images
                    break
        
        if test_img_paths:
            print(f"Start predicting {len(test_img_paths)} images...")
            
            # Perform prediction
            results = model(test_img_paths, conf=0.5, device=DEVICE, save=False)
            
            # Manually save prediction results
            for result in results:
                img_name = os.path.basename(result.path)
                save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
                result.save(save_path)
                print(f"Prediction result saved: {save_path}")
            
            print(f"YOLOv8n prediction results successfully saved to: {YOLOV8N_RESULTS_DIR}")
        else:
            print("Warning: No test images found")
            
    except Exception as e:
        print(f"Error executing YOLOv8n prediction test: {e}")

def fine_tune_yolov8n(data_yaml):
    # Fine-tune YOLOv8n model using pseudo labels
    print("\nStart fine-tuning YOLOv8n with high-quality pseudo labels...")
    
    # Create YOLOv8n prediction results save directory
    create_dirs([YOLOV8N_RESULTS_DIR])
    print(f"YOLOv8n prediction results will be saved to: {YOLOV8N_RESULTS_DIR}")
    
    # First fix all label files
    fix_all_labels()
    
    # Verify training set labels
    train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
    valid_train_count = 0
    for label_file in os.listdir(train_label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(train_label_dir, label_file)
            if os.path.getsize(label_path) > 0 and validate_label_format(label_path, NUM_CLASSES):
                valid_train_count += 1
    
    print(f"Verified training set: {valid_train_count} valid samples")
    
    # Only train if there are enough samples
    if valid_train_count < 3:
        print(f"??  Insufficient valid training samples: {valid_train_count} (need at least 3)")
        return None
    
    try:
        model = YOLO("yolov8n.pt")

        # Optimized training parameters for 8.3.x version, for small sample size
        train_results = model.train(
            data=data_yaml,
            epochs=min(8, EPOCHS),  # Reduce epochs for small sample size to avoid overfitting
            batch=BATCH_SIZE,  # Use smaller batch size
            device=DEVICE,
            lr0=0.0005,  # Smaller learning rate
            lrf=0.01,
            weight_decay=0.001,
            save=True,
            project="runs/detect",
            name="yolov8n_finetune_lworld",
            exist_ok=True,
            plots=True,
            patience=2,  # Early stopping to avoid overfitting
            workers=0  # Avoid multi-threading issues
        )

        if hasattr(model, 'best') and model.best and os.path.exists(model.best):
            print(f"\nFine-tuning completed! Best model path: {model.best}")
            return model.best
        else:
            print("? Fine-tuning completed but best model not found")
            return None
    except Exception as e:
        print(f"? Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # When running independently, execute dataset organization and model fine-tuning
    from config_utils import load_classes, KIND_FILE
    
    # Organize dataset
    data_yaml_path = organize_dataset()
    
    # Check if there are enough samples for fine-tuning
    if data_yaml_path is not None:
        # Count actual training sample count
        train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
        train_labels = [f for f in os.listdir(train_label_dir) if f.endswith(".txt")]
        
        # Strictly check validity of each label file
        valid_train_count = 0
        for label_file in train_labels:
            label_path = os.path.join(train_label_dir, label_file)
            if validate_label_format(label_path, NUM_CLASSES):
                valid_train_count += 1
    
        print(f"Strictly verified valid training samples: {valid_train_count}")
        
        # Only perform fine-tuning when there are enough samples
        if valid_train_count >= 3:  # Increase requirements to ensure enough training samples
            # Fine-tune YOLOv8n
            best_model = fine_tune_yolov8n(data_yaml_path)
        else:
            print("??  Insufficient training sample count, skipping model fine-tuning stage")
            print("Suggestion: Try further reducing CONF_THRESH or prepare more annotated data")
    
    # Execute YOLOv8n prediction test
    test_yolov8n_prediction()
