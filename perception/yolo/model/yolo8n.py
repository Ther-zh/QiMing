from ultralytics import YOLO
import os
from pathlib import Path
import cv2

def yolo_batch_detect(
    model_path: str = "yolov8n.pt",  # YOLO模型路径（默认使用预训练模型）
    input_folder: str = "input_images",  # 输入图片文件夹路径
    output_folder: str = "output_annotated",  # 输出标注图片文件夹路径
    save_txt: bool = True,  # 是否保存检测结果文本文件（类别、坐标、置信度）
    conf_threshold: float = 0.25,  # 置信度阈值（过滤低置信度检测结果）
    iou_threshold: float = 0.45  # NMS的IOU阈值
):
    """
    使用YOLOv8批量检测文件夹中的图片，并保存标注结果
    
    Args:
        model_path: YOLO模型文件路径（本地文件或官方预训练模型名）
        input_folder: 存放待检测图片的文件夹路径
        output_folder: 存放标注后图片的文件夹路径
        save_txt: 是否保存检测结果到txt文件
        conf_threshold: 检测置信度阈值
        iou_threshold: 非极大值抑制的IOU阈值
    """
    # 1. 验证输入文件夹是否存在
    input_dir = Path(input_folder)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_folder}")
    
    # 2. 创建输出文件夹（如果不存在）
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)  # parents=True支持多级文件夹创建
    if save_txt:
        txt_output_dir = output_dir / "labels"  # 文本结果单独存放在labels子文件夹
        txt_output_dir.mkdir(exist_ok=True)
    
    # 3. 加载YOLO模型（自动下载预训练模型yolov8n.pt，如果本地没有）
    print(f"正在加载模型：{model_path}")
    model = YOLO(model_path)
    print("模型加载完成！开始批量检测...")
    
    # 4. 获取输入文件夹中所有支持的图片文件（可扩展其他格式）
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"警告：输入文件夹 {input_folder} 中未找到支持的图片文件！")
        return
    
    # 5. 批量处理每张图片
    for idx, img_file in enumerate(image_files, 1):
        img_name = img_file.name  # 图片文件名（含后缀）
        print(f"正在处理第 {idx}/{len(image_files)} 张图片：{img_name}")
        
        # 6. 执行检测（返回Results对象）
        results = model(
            str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,  # 不自动保存，手动控制保存路径
            verbose=False  # 关闭单张图片的检测日志
        )
        
        # 7. 提取检测结果并绘制标注（使用YOLO内置的plot方法）
        result = results[0]  # 单张图片的检测结果
        annotated_img = result.plot()  # 绘制标注框、类别、置信度的图片（BGR格式）
        
        # 8. 保存标注后的图片（保持原文件名）
        output_img_path = output_dir / img_name
        cv2.imwrite(str(output_img_path), annotated_img)
        print(f"标注图片已保存：{output_img_path}")
        
        # 9. 可选：保存检测结果到txt文件（YOLO格式：class_id x1 y1 x2 y2 confidence）
        if save_txt:
            txt_filename = img_file.stem + ".txt"  # 文本文件名（与图片同名，后缀为txt）
            txt_output_path = txt_output_dir / txt_filename
            
            with open(txt_output_path, "w", encoding="utf-8") as f:
                # 提取检测结果：boxes（边界框）、names（类别名称映射）
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])  # 类别ID
                        class_name = result.names[class_id]  # 类别名称
                        conf = float(box.conf[0])  # 置信度
                        # 边界框坐标（xyxy格式：x1 y1 x2 y2，像素坐标）
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        # 写入格式：类别名称 置信度 x1 y1 x2 y2（方便人类阅读）
                        f.write(f"{class_name} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
            print(f"检测结果文本已保存：{txt_output_path}")
    
    print("\n批量检测完成！所有结果已保存到：", output_dir)

if __name__ == "__main__":
    # -------------------------- 请根据需求修改以下参数 --------------------------
    INPUT_FOLDER = "images"    # 待检测图片的文件夹路径（相对路径或绝对路径）
    OUTPUT_FOLDER = "output_annotated"  # 标注结果保存的文件夹路径
    MODEL_PATH = "yolov8n.pt"        # 模型路径（本地文件或官方预训练模型名）
    SAVE_TXT = True                  # 是否保存检测结果文本文件
    CONF_THRESHOLD = 0.25            # 置信度阈值（可调整，如0.3过滤更多低置信度结果）
    # ----------------------------------------------------------------------------
    
    # 执行批量检测
    yolo_batch_detect(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_txt=SAVE_TXT,
        conf_threshold=CONF_THRESHOLD
    )