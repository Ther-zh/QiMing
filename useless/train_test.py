# model_train_evaluator.py
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import os
import shutil
import random
import matplotlib.pyplot as plt
from configs_utils import (
    IMAGE_DIR, PSEUDO_LABEL_DIR, DATASET_ROOT, YOLOV8N_RESULTS_DIR,
    TRAIN_RATIO, EPOCHS, BATCH_SIZE, DEVICE, KIND_FILE, TRAIN_LOG_DIR,
    EVALUATION_DIR, load_classes, validate_label_format, fix_label_file,
    create_dirs, get_valid_image_paths
)

def organize_dataset(classes):
    """整理数据集为YOLOv8训练格式，生成data.yaml"""
    print("\n" + "="*50)
    print("? 开始整理数据集（适配YOLOv8）")
    print("="*50)
    
    NUM_CLASSES = len(classes)
    # 1. 创建数据集目录结构
    dataset_dirs = [
        os.path.join(DATASET_ROOT, "images", "train"),
        os.path.join(DATASET_ROOT, "images", "val"),
        os.path.join(DATASET_ROOT, "labels", "train"),
        os.path.join(DATASET_ROOT, "labels", "val")
    ]
    create_dirs(dataset_dirs)
    
    # 2. 收集有效伪标签样本（标签+对应图片）
    valid_samples = []  # 格式：(img_name_base, img_ext)
    label_files = [f for f in os.listdir(PSEUDO_LABEL_DIR) if f.endswith(".txt")]
    
    for label_file in label_files:
        label_path = os.path.join(PSEUDO_LABEL_DIR, label_file)
        # 修复标签格式
        fix_label_file(label_path, NUM_CLASSES)
        # 验证修复后的标签
        if not validate_label_format(label_path, NUM_CLASSES):
            continue
        
        # 查找对应图片
        img_name_base = os.path.splitext(label_file)[0]
        img_ext = None
        for ext in ["jpg", "png", "jpeg", "bmp"]:
            if os.path.exists(os.path.join(IMAGE_DIR, f"{img_name_base}.{ext}")):
                img_ext = ext
                break
        if img_ext:
            valid_samples.append((img_name_base, img_ext))
        else:
            print(f"??  标签文件 {label_file} 无对应图片，跳过")
    
    # 3. 校验有效样本数量
    if len(valid_samples) == 0:
        print("? 无有效伪标签样本，无法整理数据集")
        return None
    elif len(valid_samples) < 5:
        print(f"??  有效样本仅 {len(valid_samples)} 张，可能影响训练效果")
    
    # 4. 划分训练集/验证集（确保验证集至少1个样本）
    random.shuffle(valid_samples)
    if len(valid_samples) == 1:
        train_samples = valid_samples
        val_samples = valid_samples.copy()
    else:
        val_size = max(1, int(len(valid_samples) * (1 - TRAIN_RATIO)))
        train_size = len(valid_samples) - val_size
        train_samples = valid_samples[:train_size]
        val_samples = valid_samples[train_size:]
    
    print(f"? 样本划分：训练集 {len(train_samples)} 张，验证集 {len(val_samples)} 张")
    
    # 5. 复制图片和标签到对应目录
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_name_base, img_ext in samples:
            try:
                # 复制图片
                src_img = os.path.join(IMAGE_DIR, f"{img_name_base}.{img_ext}")
                dst_img = os.path.join(DATASET_ROOT, "images", split, f"{img_name_base}.{img_ext}")
                shutil.copyfile(src_img, dst_img)
                # 复制标签
                src_label = os.path.join(PSEUDO_LABEL_DIR, f"{img_name_base}.txt")
                dst_label = os.path.join(DATASET_ROOT, "labels", split, f"{img_name_base}.txt")
                shutil.copyfile(src_label, dst_label)
            except Exception as e:
                print(f"??  复制文件 {img_name_base}.{img_ext} 失败：{e}")
    
    # 6. 生成data.yaml
    data_yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    path = os.path.abspath(DATASET_ROOT).replace("\\", "/")
    names_str = '[' + ', '.join([f'"{c}"' for c in classes]) + ']'
    data_yaml_content = f"""path: {path}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {names_str}"""
    
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content.strip())
    
    print(f"? 数据集整理完成！")
    print(f"   - data.yaml 路径：{data_yaml_path}")
    print(f"   - 有效样本总数：{len(valid_samples)}")
    print("="*50)
    return data_yaml_path

def fine_tune_yolov8n(data_yaml_path, classes):
    """微调YOLOv8n模型（适配少量伪标签样本）"""
    print("\n" + "="*50)
    print("? 开始微调 YOLOv8n 模型")
    print("="*50)
    
    NUM_CLASSES = len(classes)
    # 1. 修复所有标签（二次保障）
    fixed_count = 0
    for split in ["train", "val"]:
        label_dir = os.path.join(DATASET_ROOT, "labels", split)
        for label_file in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_file)
            if fix_label_file(label_path, NUM_CLASSES):
                fixed_count += 1
    print(f"? 修复标签文件：{fixed_count} 个")
    
    # 2. 验证训练集有效样本数
    train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
    valid_train_count = 0
    for label_file in os.listdir(train_label_dir):
        label_path = os.path.join(train_label_dir, label_file)
        if validate_label_format(label_path, NUM_CLASSES):
            valid_train_count += 1
    
    print(f"? 有效训练样本数：{valid_train_count}")
    if valid_train_count < 3:
        print("? 有效训练样本不足3个，跳过微调")
        return None
    
    # 3. 加载模型并微调
    model = YOLO("yolov8n.pt")
    print("? 加载预训练 YOLOv8n 模型")
    
    try:
        train_results = model.train(
            data=data_yaml_path,
            epochs=min(8, EPOCHS),  # 限制最大轮次，避免过拟合
            batch=BATCH_SIZE,
            device=DEVICE,
            lr0=0.0005,  # 减小学习率
            lrf=0.01,
            weight_decay=0.001,
            save=True,
            project="runs/detect",
            name="yolov8n_finetune_lworld",
            exist_ok=True,
            plots=True,
            patience=2,  # 早停机制
            workers=0  # 避免多线程问题
        )
        
        # 返回最佳模型路径
        best_model_path = model.best if hasattr(model, 'best') and os.path.exists(model.best) else None
        if best_model_path:
            print(f"? 微调完成！最佳模型路径：{best_model_path}")
        else:
            print("? 微调完成但未找到最佳模型")
        print("="*50)
        return best_model_path
    except Exception as e:
        print(f"? 微调失败：{e}")
        import traceback
        traceback.print_exc()
        print("="*50)
        return None

def visualize_evaluation(best_model_path, data_yaml_path, classes):
    """可视化评估最佳模型（混淆矩阵、PR曲线、预测示例）"""
    print("\n" + "="*50)
    print("? 开始可视化评估最佳模型")
    print("="*50)
    
    NUM_CLASSES = len(classes)
    create_dirs([EVALUATION_DIR, os.path.join(EVALUATION_DIR, "pred_examples")])
    
    # 1. 加载模型并评估验证集
    model = YOLO(best_model_path)
    results = model.val(data=data_yaml_path, device=DEVICE, save_json=True, verbose=False)
    
    # 2. 绘制混淆矩阵
    cm = ConfusionMatrix(nc=NUM_CLASSES, names=classes)
    cm.process_preds(results.preds, results.targets)
    cm.plot(save_dir=EVALUATION_DIR, names=classes, figsize=(16, 14))
    print("? 混淆矩阵已保存到：", EVALUATION_DIR)
    
    # 3. 绘制PR曲线（分两列显示24类）
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    # 前12类
    for i, cls in enumerate(classes[:12]):
        axes[0].plot(results.box.r[i], results.box.p[i], label=f"{cls} (AP={results.box.ap[i]:.2f})")
    axes[0].set_xlabel("Recall", fontsize=10)
    axes[0].set_ylabel("Precision", fontsize=10)
    axes[0].set_title("PR Curves (Classes 1-12)", fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)
    # 后12类
    for i, cls in enumerate(classes[12:]):
        axes[1].plot(results.box.r[12 + i], results.box.p[12 + i], label=f"{cls} (AP={results.box.ap[12 + i]:.2f})")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("PR Curves (Classes 13-24)", fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    pr_curve_path = os.path.join(EVALUATION_DIR, "pr_curve.png")
    plt.savefig(pr_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("? PR曲线已保存到：", pr_curve_path)
    
    # 4. 保存预测示例
    val_img_dir = os.path.join(DATASET_ROOT, "images", "val")
    val_img_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)[:15]]
    pred_results = model(val_img_paths, conf=0.5, device=DEVICE, save=False)
    
    for result in pred_results:
        img_name = os.path.basename(result.path)
        # 保存到YOLOv8n结果目录
        yolov8n_save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
        result.save(yolov8n_save_path)
        # 保存到评估目录
        eval_save_path = os.path.join(EVALUATION_DIR, "pred_examples", img_name)
        result.save(eval_save_path)
    print("? 预测示例已保存到：", os.path.join(EVALUATION_DIR, "pred_examples"))
    
    # 5. 输出核心指标
    print("\n" + "="*30)
    print("? 核心评估指标")
    print("="*30)
    print(f"mAP@0.5:        {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95:   {results.box.map:.4f}")
    print(f"推理速度:       {results.speed['inference']:.2f} FPS")
    
    print("\n? Top5 高AP类别")
    ap_sorted = sorted(zip(classes, results.box.ap), key=lambda x: x[1], reverse=True)[:5]
    for cls, ap in ap_sorted:
        print(f"  {cls:<25} AP: {ap:.4f}")
    
    print("\n??  需优化类别（AP<0.3）")
    low_ap = [(cls, ap) for cls, ap in zip(classes, results.box.ap) if ap < 0.3]
    if low_ap:
        for cls, ap in low_ap:
            print(f"  {cls:<25} AP: {ap:.4f}（建议补充真实标注）")
    else:
        print("  所有类别 AP≥0.3，伪标签质量优秀！")
    print("="*50)

def test_yolov8n_pretrained():
    """用预训练YOLOv8n模型测试原始图片（独立功能）"""
    print("\n" + "="*50)
    print("? 执行预训练 YOLOv8n 预测测试")
    print("="*50)
    
    create_dirs([YOLOV8N_RESULTS_DIR])
    try:
        model = YOLO("yolov8n.pt")
        print("? 加载预训练 YOLOv8n 模型")
        
        # 获取测试图片（最多10张）
        test_img_paths = get_valid_image_paths(IMAGE_DIR)[:10]
        print(f"? 开始预测 {len(test_img_paths)} 张图片")
        
        # 预测并保存结果
        pred_results = model(test_img_paths, conf=0.5, device=DEVICE, save=False)
        for result in pred_results:
            img_name = os.path.basename(result.path)
            save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
            result.save(save_path)
            print(f"   - 已保存：{save_path}")
        
        print(f"? 预训练模型预测完成！结果路径：{YOLOV8N_RESULTS_DIR}")
    except Exception as e:
        print(f"? 预训练模型预测失败：{e}")
    print("="*50)

def main():
    """主流程：数据集整理→模型微调→评估→预训练测试"""
    print("="*60)
    print("? YOLOv8n 伪标签训练与评估全流程")
    print("="*60)
    
    try:
        # 1. 加载类别
        CLASSES = load_classes(KIND_FILE)
        NUM_CLASSES = len(CLASSES)
        
        # 2. 整理数据集
        data_yaml_path = organize_dataset(CLASSES)
        if not data_yaml_path:
            print("? 数据集整理失败，终止流程")
            return
        
        # 3. 微调YOLOv8n
        best_model_path = fine_tune_yolov8n(data_yaml_path, CLASSES)
        if not best_model_path:
            print("? 模型微调失败，跳过评估")
            test_yolov8n_pretrained()
            return
        
        # 4. 可视化评估
        visualize_evaluation(best_model_path, data_yaml_path, CLASSES)
        
        # 5. 预训练模型测试（可选，独立功能）
        test_yolov8n_pretrained()
        
        # 6. 全流程汇总
        print("\n" + "="*60)
        print("? 全流程执行完成！")
        print("="*60)
        valid_labels_count = len([
            f for f in os.listdir(PSEUDO_LABEL_DIR)
            if f.endswith('.txt') and validate_label_format(os.path.join(PSEUDO_LABEL_DIR, f), NUM_CLASSES)
        ])
        print(f"? 核心输出路径：")
        print(f"   - 伪标签：{PSEUDO_LABEL_DIR}（有效样本：{valid_labels_count} 张）")
        print(f"   - 数据集：{DATASET_ROOT}")
        print(f"   - 训练日志：{TRAIN_LOG_DIR}")
        print(f"   - 评估报告：{EVALUATION_DIR}")
        print(f"   - 预测结果：{YOLOV8N_RESULTS_DIR}")
        print("="*60)
    
    except Exception as e:
        print(f"\n? 流程执行失败：{e}")
        import traceback
        traceback.print_exc()
        print("="*60)

if __name__ == "__main__":
    # 独立运行时，执行“数据集整理→训练→评估→预训练测试”
    main()