# -*- coding: utf-8 -*-
# evaluator.py
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from ultralytics.utils.metrics import ConfusionMatrix
from config_utils import (
    DATASET_ROOT, CLASSES, NUM_CLASSES, DEVICE, YOLOV8N_RESULTS_DIR,
    EVALUATION_DIR, create_dirs
)

def visualize_evaluation(best_model_path, data_yaml):
    # Use fine-tuned model for evaluation and generate visualization results
    print("\nStart visualization evaluation...")
    model = YOLO(best_model_path)
    results = model.val(data=data_yaml, device=DEVICE, save_json=True)

    # Create evaluation results save directory
    create_dirs([EVALUATION_DIR, os.path.join(EVALUATION_DIR, "pred_examples")])

    # Only plot PR curves for classes that were actually detected
    detected_classes = []
    for i, cls in enumerate(CLASSES):
        if i < len(results.box.ap):  # Check if this class has AP data
            detected_classes.append((i, cls))
    
    # PR curve (display all detected classes in one plot)
    if detected_classes:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        for i, cls in detected_classes:
            ax.plot(results.box.r[i], results.box.p[i], label=f"{cls} (AP={results.box.ap[i]:.2f})")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title("PR Curves for Detected Classes", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(EVALUATION_DIR, "pr_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("No PR curves generated: No classes were detected")

    # Prediction examples
    val_img_dir = os.path.join(DATASET_ROOT, "images", "val")
    val_img_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)[:15]]
    pred_results = model(val_img_paths, conf=0.5, device=DEVICE, save=False)
    
    # Manually save prediction result images to YOLOv8n results directory
    for result in pred_results:
        img_name = os.path.basename(result.path)
        
        # Save to YOLOv8n results directory
        yolo8n_save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
        result.save(yolo8n_save_path)
        
        # Also save to evaluation directory as backup
        eval_save_path = os.path.join(EVALUATION_DIR, "pred_examples", img_name)
        result.save(eval_save_path)

    # Output core metrics
    print("\n=== Core Evaluation Metrics ===")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Inference speed: {results.speed['inference']:.2f} FPS")
    print("\n=== Top5 High AP Categories ===")
    ap_sorted = sorted(zip(CLASSES, results.box.ap), key=lambda x: x[1], reverse=True)[:5]
    for cls, ap in ap_sorted:
        print(f"  {cls:<25} AP: {ap:.4f}")
    print("\n=== Categories Needing Optimization (AP<0.3) ===")
    low_ap = [(cls, ap) for cls, ap in zip(CLASSES, results.box.ap) if ap < 0.3]
    if low_ap:
        for cls, ap in low_ap:
            print(f"  {cls:<25} AP: {ap:.4f} (suggest adding real annotations)")
    else:
        print("  All categories AP>=0.3, pseudo label quality is excellent!")

    print(f"\nEvaluation results save path: {EVALUATION_DIR}")
    print(f"YOLOv8n prediction results save path: {YOLOV8N_RESULTS_DIR}")

if __name__ == "__main__":
    # When running independently, load best model and perform evaluation
    from config_utils import load_classes, KIND_FILE
    
    # Load categories
    CLASSES = load_classes(KIND_FILE)
    NUM_CLASSES = len(CLASSES)
    
    # Best model path (default value, can be modified according to actual situation)
    best_model_path = "runs/detect/yolov8n_finetune_lworld/weights/best.pt"
    data_yaml = os.path.join(DATASET_ROOT, "data.yaml")
    
    # Check if files exist
    if os.path.exists(best_model_path) and os.path.exists(data_yaml):
        visualize_evaluation(best_model_path, data_yaml)
    else:
        print(f"Error: Model file {best_model_path} or data configuration file {data_yaml} does not exist!")
        print("Please first run labeller_new.py to generate pseudo labels, then run trainer_new.py for model fine-tuning")
