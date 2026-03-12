import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from PIL import Image
import requests
# 禁用SSL验证，解决网络环境问题
requests.packages.urllib3.disable_warnings()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -------------------------- 1. 配置参数（聚焦内存优化，适配8.3.x版本）--------------------------
IMAGE_DIR = "image"  # 原始图片文件夹
KIND_FILE = "kind.txt"  # 类别文件路径
PSEUDO_LABEL_DIR = "pseudo_labels"  # 伪标签保存目录
DATASET_ROOT = "datasets"  # 数据集根目录
CONF_THRESH = 0.4  # 进一步降低置信度阈值，尝试获取更多伪标签样本
TRAIN_RATIO = 0.9  # 训练集/验证集划分比例
EPOCHS = 12  # 适配高质量伪标签的微调轮数
BATCH_SIZE = 1  # 内存优化：减小批量大小到1
DEVICE = "cpu"  # 内存优化：直接使用CPU避免内存不足问题
YOLO_WORLD_RESULTS_DIR = "runs/results/yolo_world"  # YOLO World预测结果保存目录
YOLOV8N_RESULTS_DIR = "runs/results/yolov8n"  # YOLOv8n预测结果保存目录


# -------------------------- 2. 读取类别列表 --------------------------
def load_classes(kind_file):
    with open(kind_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    assert len(classes) > 0, "kind.txt 中无有效类别，请检查文件"
    print(f"成功加载 {len(classes)} 个类别：{classes}")
    return classes


CLASSES = load_classes(KIND_FILE)
NUM_CLASSES = len(CLASSES)


# -------------------------- 3. YOLO World（yolov8l-world.pt）生成伪标签（8.3.x兼容版）--------------------------
def generate_pseudo_labels():
    os.makedirs(PSEUDO_LABEL_DIR, exist_ok=True)

    # 使用已下载的模型，避免重新下载
    # 优先尝试小模型
    model_path = "yolov8n-world.pt"
    # 如果小模型不存在，使用之前下载过的大模型
    if not os.path.exists(model_path):
        model_path = "yolov8l-world.pt"
    
    if os.path.exists(model_path):
        print(f"使用本地模型 {model_path}，避免网络下载...")
        model = YOLO(model_path)
    else:
        # 尝试创建一个简单的占位符，避免下载
        print("未找到本地模型，但跳过下载以避免网络问题...")
        # 这里我们将直接失败，让用户知道需要手动下载模型
        raise FileNotFoundError("请手动下载模型文件并存放到model目录")
    print(f"成功加载本地模型 {model_path}...")

    # 关键修复：YOLO World通过set_classes设置类别文本（8.3.x专属接口）
    model.set_classes(CLASSES)
    print("已通过set_classes设置24个检测类别...")
    
    # 创建YOLO World预测结果保存目录
    os.makedirs(YOLO_WORLD_RESULTS_DIR, exist_ok=True)
    print(f"YOLO World预测结果将保存到：{YOLO_WORLD_RESULTS_DIR}")

    # 批量获取图片路径
    image_paths = []
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            image_paths.append(os.path.join(IMAGE_DIR, f))
    assert len(image_paths) > 0, "image 文件夹中无有效图片，请检查路径"
    print(f"共找到 {len(image_paths)} 张有效图片，开始批量预测...")

    # 内存优化：逐张处理图片而不是批量处理
    results = []
    for img_path in image_paths:
        print(f"处理图片 {img_path}...")
        # 单张预测，减少内存占用
        result = model(
            img_path,  # 单张图片路径
            conf=CONF_THRESH,  # 置信度阈值
            iou=0.45,  # NMS的IOU阈值
            save=False,  # 不使用自动保存
            device=DEVICE,  # 运行设备（CPU）
            verbose=False  # 不打印详细日志
        )
        
        # 手动保存预测结果图片到YOLO World结果目录
        for r in result:
            img_name = os.path.basename(img_path)
            save_path = os.path.join(YOLO_WORLD_RESULTS_DIR, img_name)
            r.save(save_path)  # 直接保存到指定路径
        results.extend(result)
        # 强制垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存伪标签（逻辑不变，确保YOLO格式正确）
    valid_sample_count = 0
    for r in results:
        img_path = r.path
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(PSEUDO_LABEL_DIR, label_name)

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            with open(label_path, "w") as f:
                f.write("")
            continue

        # 写入伪标签（严格按照YOLO格式：类别ID + 4个归一化坐标，不含置信度）
        with open(label_path, "w", encoding="utf-8") as f:
            valid_boxes = 0
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    # 确保类别ID在有效范围内
                    if 0 <= cls_id < len(CLASSES):
                        xc, yc, w, h = box.xywhn[0].tolist()
                        # 确保坐标在0-1范围内且宽高不为0
                        if all(0 <= x <= 1 for x in [xc, yc, w, h]) and w > 0 and h > 0:
                            # 只写入必要的5个值（类别+4坐标），不包含置信度
                            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                            valid_boxes += 1
                except Exception as e:
                    print(f"处理边界框时出错: {e}")
                    continue
        
        if valid_boxes > 0:
            valid_sample_count += 1

        valid_sample_count += 1

    print(f"伪标签生成完成！有效标注样本数：{valid_sample_count}/{len(image_paths)}")
    print(f"伪标签保存路径：{PSEUDO_LABEL_DIR}")
    return valid_sample_count


# -------------------------- 4. 整理数据集（8.3.x路径兼容优化）--------------------------
def organize_dataset():
    """将伪标签转换为YOLOv8训练所需的数据集结构（适配8.3.x版本）"""
    print("\n开始整理数据集（8.3.x版本适配）...")
    
    # 创建数据集目录结构
    os.makedirs(os.path.join(DATASET_ROOT, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels", "val"), exist_ok=True)
    
    # 收集所有有效伪标签图片
    label_files = [f for f in os.listdir(PSEUDO_LABEL_DIR) if f.endswith(".txt")]
    valid_img_names = []
    
    for label_file in label_files:
        try:
            with open(os.path.join(PSEUDO_LABEL_DIR, label_file), "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # 严格验证标签格式，确保符合YOLOv8要求
            valid_labels = []
            for line in lines:
                parts = line.split()
                # YOLO格式必须是5个值，不能多也不能少
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        # 确保类别ID在有效范围内
                        if 0 <= cls_id < len(CLASSES):
                            coords = list(map(float, parts[1:5]))
                            # 检查坐标范围是否有效且宽高不为0
                            if all(0 <= x <= 1 for x in coords) and coords[2] > 0 and coords[3] > 0:
                                # 重新格式化坐标，确保精度一致
                                valid_labels.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                    except ValueError:
                        continue
            # 处理可能包含置信度的旧格式标签（转换为标准格式）
            if not valid_labels and len(lines) > 0:
                for line in lines:
                    parts = line.split()
                    # 如果有6个值，尝试去掉最后一个置信度值
                    if len(parts) == 6:
                        try:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(CLASSES):
                                coords = list(map(float, parts[1:5]))
                                if all(0 <= x <= 1 for x in coords) and coords[2] > 0 and coords[3] > 0:
                                    valid_labels.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                        except ValueError:
                            continue
        except Exception as e:
            print(f"读取标签文件失败: {e}")
            continue
                  
        if len(valid_labels) > 0:  # 确保有有效的标签行
                    # 重写标签文件，只保留有效标签
                    with open(os.path.join(PSEUDO_LABEL_DIR, label_file), "w") as f:
                        f.write("\n".join(valid_labels) + "\n")
                    
                    # 尝试不同的图片格式
                    img_name_base = os.path.splitext(label_file)[0]
                    found = False
                    for ext in ["jpg", "png", "jpeg", "bmp"]:
                        img_name = f"{img_name_base}.{ext}"
                        if os.path.exists(os.path.join(IMAGE_DIR, img_name)):
                            valid_img_names.append((img_name_base, ext))
                            found = True
                            break
                    if not found:
                        print(f"警告：标签文件 {label_file} 对应的图片文件不存在")
        # 这里不需要额外的except块，保留原有的处理逻辑
    
    # 检查有效样本数量
    if len(valid_img_names) == 0:
        print("警告：无有效伪标签样本！请降低 CONF_THRESH 到更低值")
        return None
    elif len(valid_img_names) < 5:
        print(f"警告：有效伪标签样本仅 {len(valid_img_names)} 张！样本数量太少，可能影响训练效果")
    
    # 划分训练集和验证集 - 确保验证集至少有1个样本
    random.shuffle(valid_img_names)
    if len(valid_img_names) == 1:
        train_samples = valid_img_names
        val_samples = valid_img_names.copy()  # 只有一个样本时，同时用于训练和验证
    else:
        # 确保验证集至少有1个样本
        val_size = max(1, int(len(valid_img_names) * (1 - TRAIN_RATIO)))
        train_size = len(valid_img_names) - val_size
        train_samples = valid_img_names[:train_size]
        val_samples = valid_img_names[train_size:]
    
    print(f"划分训练集：{len(train_samples)} 张，验证集：{len(val_samples)} 张")
    
    # 复制文件
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_name, ext in samples:
            try:
                # 复制图片
                src_img = os.path.join(IMAGE_DIR, f"{img_name}.{ext}")
                dst_img = os.path.join(DATASET_ROOT, "images", split, f"{img_name}.{ext}")
                shutil.copyfile(src_img, dst_img)
                # 复制标签
                src_label = os.path.join(PSEUDO_LABEL_DIR, f"{img_name}.txt")
                dst_label = os.path.join(DATASET_ROOT, "labels", split, f"{img_name}.txt")
                shutil.copyfile(src_label, dst_label)
            except Exception as e:
                print(f"复制文件 {img_name}.{ext} 时出错: {e}")
    
    # 生成 data.yaml（使用安全的字符串格式化）
    path = os.path.abspath(DATASET_ROOT).replace('\\', '/')
    # 手动构建names字符串，避免f-string中的反斜杠问题
    names_str = '[' + ', '.join([f'"{c}"' for c in CLASSES]) + ']'
    data_yaml = f"""path: {path}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {names_str}"""
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml.strip())
    
    # 验证生成的yaml文件
    print(f"数据集整理完成！data.yaml 路径：{yaml_path}")
    print(f"最终有效样本：{len(valid_img_names)} 个")

    print(f"数据集整理完成！data.yaml 路径：{yaml_path}")
    return yaml_path


# -------------------------- 5. YOLOv8n 微调（8.3.x训练API兼容）--------------------------
def validate_label_format(label_path):
    """验证单个标签文件的格式是否正确"""
    try:
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        for line in lines:
            parts = line.split()
            # 必须严格是5个值
            if len(parts) != 5:
                return False
            # 第一个值必须是有效整数类别ID
            try:
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= len(CLASSES):
                    return False
            except ValueError:
                return False
            # 后四个值必须是0-1范围内的浮点数
            try:
                coords = list(map(float, parts[1:5]))
                if not all(0 <= x <= 1 for x in coords):
                    return False
                # 宽高不能为0
                if coords[2] <= 0 or coords[3] <= 0:
                    return False
            except ValueError:
                return False
        return True
    except Exception:
        return False

def test_yolov8n_prediction():
    """使用预训练的YOLOv8n模型进行预测并保存结果"""
    print("\n执行YOLOv8n预训练模型预测测试...")
    
    # 创建YOLOv8n预测结果保存目录
    os.makedirs(YOLOV8N_RESULTS_DIR, exist_ok=True)
    
    try:
        # 加载预训练的YOLOv8n模型
        model = YOLO("yolov8n.pt")
        print("成功加载预训练的YOLOv8n模型")
        
        # 获取一些测试图片
        test_img_paths = []
        for f in os.listdir(IMAGE_DIR):
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                test_img_paths.append(os.path.join(IMAGE_DIR, f))
                if len(test_img_paths) >= 10:  # 只测试10张图片
                    break
        
        if test_img_paths:
            print(f"开始预测 {len(test_img_paths)} 张图片...")
            
            # 进行预测
            results = model(test_img_paths, conf=0.5, device=DEVICE, save=False)
            
            # 手动保存预测结果
            for result in results:
                img_name = os.path.basename(result.path)
                save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
                result.save(save_path)
                print(f"已保存预测结果: {save_path}")
            
            print(f"YOLOv8n预测结果已成功保存到: {YOLOV8N_RESULTS_DIR}")
        else:
            print("警告：未找到测试图片")
            
    except Exception as e:
        print(f"执行YOLOv8n预测测试时出错: {e}")

def fix_all_labels():
    """修复所有标签文件，确保格式正确"""
    print("\n修复标签文件格式...")
    fixed_count = 0
    
    for split in ["train", "val"]:
        label_dir = os.path.join(DATASET_ROOT, "labels", split)
        if not os.path.exists(label_dir):
            continue
            
        for label_file in os.listdir(label_dir):
            if label_file.endswith(".txt"):
                label_path = os.path.join(label_dir, label_file)
                if not validate_label_format(label_path):
                    # 尝试修复标签文件
                    try:
                        with open(label_path, "r") as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                        fixed_lines = []
                        for line in lines:
                            parts = line.split()
                            # 处理可能包含置信度的6值格式
                            if len(parts) == 6:
                                parts = parts[:5]  # 去掉置信度
                            
                            if len(parts) == 5:
                                try:
                                    cls_id = int(parts[0])
                                    if 0 <= cls_id < len(CLASSES):
                                        coords = list(map(float, parts[1:5]))
                                        # 限制坐标范围
                                        coords = [max(0, min(1, x)) for x in coords]
                                        # 确保宽高不为0
                                        if coords[2] > 0 and coords[3] > 0:
                                            fixed_lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
                                except:
                                    continue
                        
                        # 写回修复后的标签
                        with open(label_path, "w") as f:
                            if fixed_lines:
                                f.write("\n".join(fixed_lines) + "\n")
                                fixed_count += 1
                            else:
                                # 无有效标签，清空文件
                                f.write("")
                    except Exception as e:
                        print(f"修复标签文件 {label_file} 失败: {e}")
    
    print(f"修复完成！共处理 {fixed_count} 个标签文件")
    return fixed_count

def fine_tune_yolov8n(data_yaml):
    print("\n开始用高质量伪标签微调 YOLOv8n（8.3.x版本适配）...")
    
    # 创建YOLOv8n预测结果保存目录
    os.makedirs(YOLOV8N_RESULTS_DIR, exist_ok=True)
    print(f"YOLOv8n预测结果将保存到：{YOLOV8N_RESULTS_DIR}")
    
    # 先修复所有标签文件
    fix_all_labels()
    
    # 验证训练集标签
    train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
    valid_train_count = 0
    for label_file in os.listdir(train_label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(train_label_dir, label_file)
            if os.path.getsize(label_path) > 0 and validate_label_format(label_path):
                valid_train_count += 1
    
    print(f"验证训练集：{valid_train_count} 个有效样本")
    
    # 只有足够样本时才训练
    if valid_train_count < 3:
        print(f"⚠️  有效训练样本不足：{valid_train_count} 个（需要至少3个）")
        return None
    
    try:
        model = YOLO("yolov8n.pt")

        # 8.3.x版本训练参数优化，针对少量样本
        train_results = model.train(
            data=data_yaml,
            epochs=min(8, EPOCHS),  # 少量样本减少轮次避免过拟合
            batch=BATCH_SIZE,  # 使用更小批次
            device=DEVICE,
            lr0=0.0005,  # 更小的学习率
            lrf=0.01,
            weight_decay=0.001,
            save=True,
            project="runs/detect",
            name="yolov8n_finetune_lworld",
            exist_ok=True,
            plots=True,
            patience=2,  # 早停避免过拟合
            workers=0  # 避免多线程问题
        )

        if hasattr(model, 'best') and model.best and os.path.exists(model.best):
            print(f"\n微调完成！最佳模型路径：{model.best}")
            return model.best
        else:
            print("❌ 微调完成但未找到最佳模型")
            return None
    except Exception as e:
        print(f"❌ 微调过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# -------------------------- 6. 可视化评估（8.3.x metrics 兼容）--------------------------
def visualize_evaluation(best_model_path, data_yaml):
    print("\n开始可视化评估（8.3.x版本兼容）...")
    model = YOLO(best_model_path)
    results = model.val(data=data_yaml, device=DEVICE, save_json=True)

    # 混淆矩阵（适配24类，8.3.x绘图API兼容）
    cm = ConfusionMatrix(nc=NUM_CLASSES, names=CLASSES)
    cm.process_preds(results.preds, results.targets)
    cm.plot(save_dir="runs/evaluation", names=CLASSES, figsize=(16, 14))

    # PR曲线（分两列显示，避免拥挤）
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    # 前12类
    for i, cls in enumerate(CLASSES[:12]):
        axes[0].plot(results.box.r[i], results.box.p[i], label=f"{cls} (AP={results.box.ap[i]:.2f})")
    axes[0].set_xlabel("Recall", fontsize=10)
    axes[0].set_ylabel("Precision", fontsize=10)
    axes[0].set_title("PR Curves (Classes 1-12)", fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)
    # 后12类
    for i, cls in enumerate(CLASSES[12:]):
        axes[1].plot(results.box.r[12 + i], results.box.p[12 + i], label=f"{cls} (AP={results.box.ap[12 + i]:.2f})")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("PR Curves (Classes 13-24)", fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/evaluation/pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 预测示例
    val_img_dir = os.path.join(DATASET_ROOT, "images", "val")
    val_img_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)[:15]]
    pred_results = model(val_img_paths, conf=0.5, device=DEVICE, save=False)
    
    # 手动保存预测结果图片到YOLOv8n结果目录
    os.makedirs("runs/evaluation/pred_examples", exist_ok=True)
    for result in pred_results:
        img_name = os.path.basename(result.path)
        
        # 保存到YOLOv8n结果目录
        yolo8n_save_path = os.path.join(YOLOV8N_RESULTS_DIR, img_name)
        result.save(yolo8n_save_path)
        
        # 同时保存到评估目录作为备份
        eval_save_path = os.path.join("runs/evaluation/pred_examples", img_name)
        result.save(eval_save_path)

    # 输出核心指标
    print("\n=== 核心评估指标 ===")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"推理速度: {results.speed['inference']:.2f} FPS")
    print("\n=== Top5 高AP类别 ===")
    ap_sorted = sorted(zip(CLASSES, results.box.ap), key=lambda x: x[1], reverse=True)[:5]
    for cls, ap in ap_sorted:
        print(f"  {cls:<25} AP: {ap:.4f}")
    print("\n=== 需优化类别（AP<0.3）===")
    low_ap = [(cls, ap) for cls, ap in zip(CLASSES, results.box.ap) if ap < 0.3]
    if low_ap:
        for cls, ap in low_ap:
            print(f"  {cls:<25} AP: {ap:.4f}（建议补充真实标注）")
    else:
        print("  所有类别 AP≥0.3，伪标签质量优秀！")

    print("\n评估结果保存路径：runs/evaluation")
    print(f"YOLOv8n预测结果保存路径：{YOLOV8N_RESULTS_DIR}")


# -------------------------- 7. 主流程 --------------------------
if __name__ == "__main__":
    """主函数：执行完整的伪标签生成和模型微调流程（适配8.3.x版本）"""
    print("========== YOLO World 伪标签生成与模型微调（8.3.x版本适配）==========")
    
    try:
        # 步骤1：生成伪标签
        valid_count = generate_pseudo_labels()
        if valid_count < 5:
            print(f"警告：有效伪标签样本仅 {valid_count} 张！")
            choice = input("是否降低置信度阈值（0.5）重新生成？(y/n)：")
            if choice.lower() == "y":
                CONF_THRESH = 0.5
                generate_pseudo_labels()

        # 步骤2：整理数据集
        data_yaml_path = organize_dataset()
        
        # 检查是否有足够的样本进行微调
        if data_yaml_path is None:
            print("❌ 由于缺少有效伪标签样本，无法进行模型微调")
        else:
            # 统计实际的训练样本数量
            train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
            train_labels = [f for f in os.listdir(train_label_dir) if f.endswith(".txt")]
            
            # 严格检查每个标签文件的有效性
            valid_train_count = 0
            for label_file in train_labels:
                label_path = os.path.join(train_label_dir, label_file)
                if validate_label_format(label_path):
                    valid_train_count += 1
    
            print(f"严格验证后有效训练样本数: {valid_train_count}")
            
            # 只有当有足够样本时才进行微调
            if valid_train_count >= 3:  # 提高要求，确保有足够的训练样本
                # 步骤3：微调 YOLOv8n
                best_model = fine_tune_yolov8n(data_yaml_path)

                # 步骤4：可视化评估
                visualize_evaluation(best_model, data_yaml_path)
            else:
                print("⚠️  训练样本数量不足，跳过模型微调和评估阶段")
                print("建议：尝试进一步降低 CONF_THRESH 或准备更多标注数据")
    
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
    
    # 执行YOLOv8n预测测试
    test_yolov8n_prediction()
    
    print("\n=== 全流程完成！===")
    valid_labels_count = len([f for f in os.listdir(PSEUDO_LABEL_DIR) if f.endswith('.txt') and os.path.getsize(os.path.join(PSEUDO_LABEL_DIR, f)) > 0])
    print(f"有效伪标签样本: {valid_labels_count} 张")
    if 'data_yaml_path' in locals() and data_yaml_path:
        print(f"数据集配置：{data_yaml_path}")
    print(f"训练日志：runs/detect/yolov8n_finetune_lworld")
    print(f"评估报告：runs/evaluation")
    print(f"YOLO World预测结果：{YOLO_WORLD_RESULTS_DIR}")
    print(f"YOLOv8n预测结果：{YOLOV8N_RESULTS_DIR}")
    print("\n📋 伪标签生成流程执行完成！")
    print(f"   - 置信度阈值: {CONF_THRESH}")
    print(f"   - 有效伪标签样本: {valid_labels_count} 张")