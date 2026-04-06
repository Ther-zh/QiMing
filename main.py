import time
import threading
import signal
import cv2
import numpy as np
from typing import Dict, Any, List

# 导入工具模块
from utils.logger import logger
from utils.config_loader import config_loader
from utils.message_queue import message_queue

# 导入核心模块
from core.input_thread import InputThread
from core.asr_thread import ASRThread
from core.vision_thread import VisionThread
from core.inference_thread import InferenceThread
from core.resource_manager import ResourceManager

# 导入模拟模块
from simulation.debug_viewer import DebugViewer

class BlindGuideSystem:
    def __init__(self):
        """
        初始化导盲系统
        """
        # 加载配置
        self.config = config_loader.get_config()
        self.risk_rules = config_loader.get_risk_rules()
        
        # 初始化资源管理器（用于GPU显存监控）
        self.resource_manager = ResourceManager()
        
        # 初始化线程（与主系统共享 ResourceManager，避免重复 NVML 与资源状态分裂）
        self.input_thread = InputThread()
        self.asr_thread = ASRThread()
        self.vision_thread = VisionThread()
        self.inference_thread = InferenceThread(resource_manager=self.resource_manager)
        
        # 初始化调试查看器
        self.debug_viewer = DebugViewer()
        
        # 系统状态
        self.running = False
        
        # 视频写入器
        self.video_writer = None
        self.depth_video_writer = None

        # 结果记录文件
        self.result_file = None
    

    
    def start(self):
        """
        启动系统
        """
        logger.info("导盲系统启动中...")
        
        # 打印初始资源状态
        logger.info("\n" + self.resource_manager.get_memory_summary())
        
        # 打印配置信息
        logger.info(f"系统模式: {self.config.get('system', {}).get('mode')}")
        logger.info(f"输入模式: {self.config.get('system', {}).get('input_mode', 'simulated')}")
        logger.info(f"YOLO模型: {self.config.get('models', {}).get('yolo', {}).get('model_path')}")
        logger.info(f"VDA模型: {self.config.get('models', {}).get('vda', {}).get('model_path')}")
        logger.info(f"ASR模型: {self.config.get('models', {}).get('asr', {}).get('model_path')}")
        _llm = self.config.get("models", {}).get("llm", {}) or {}
        logger.info(f"LLM模型: {_llm.get('model_path')}")
        logger.info(
            f"LLM Ollama: 多模态={_llm.get('model_name')}, 纯文本={_llm.get('model_name_text') or _llm.get('model_name')}"
        )
        if self.config.get('system', {}).get('input_mode') == 'simulated':
            logger.info(f"视频路径: {self.config.get('simulation', {}).get('video_paths', {}).get('camera1')}")
        
        # 初始化视频写入器
        self._init_video_writer()
        
        # 初始化结果文件
        self._init_result_file()
        
        # 打印资源状态（初始化后）
        self.resource_manager.print_gpu_memory("初始化完成")
        
        # 创建消息队列
        message_queue.create_queue("audio")
        # 有界队列：YOLO+VDA 耗时长时阻塞输入线程，避免 vision 队列堆积占满内存
        message_queue.create_queue("vision", maxsize=2)
        message_queue.create_queue("inference", maxsize=8)
        message_queue.create_queue("control")
        message_queue.create_queue("record", maxsize=4)
        
        # 启动调试查看器（如果开启且有图形界面）
        debug_enabled = self.config.get("system", {}).get("debug", False)
        if debug_enabled:
            try:
                # 检测是否有图形界面
                import os
                if os.environ.get('DISPLAY') or os.name == 'nt':
                    self.debug_viewer.start()
                    debug_thread = threading.Thread(
                        target=self.debug_viewer.show,
                        daemon=True
                    )
                    debug_thread.start()
                else:
                    logger.warning("没有图形界面，禁用调试查看器")
                    # 强制禁用调试模式
                    self.config["system"]["debug"] = False
            except Exception as e:
                logger.warning(f"启动调试查看器失败: {e}")
                # 强制禁用调试模式
                self.config["system"]["debug"] = False
        
        # 启动线程
        self.input_thread.start()
        self.asr_thread.start()
        self.vision_thread.start()
        self.inference_thread.start()
        
        self.running = True
        logger.info("导盲系统启动完成")
        
        # 启动主循环
        self._main_loop()
    
    def _init_video_writer(self):
        """
        初始化视频写入器
        """
        try:
            # 获取视频路径
            video_path = self.config.get("simulation", {}).get("video_paths", {}).get("camera1")
            if video_path:
                # 获取视频信息
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                self.sample_interval = int(
                    self.config.get("system", {}).get("vision_sample_interval", 25)
                )
                self.processing_fps = self.video_fps / self.sample_interval
                
                logger.info(f"视频帧率: {self.video_fps}, 处理帧率: {self.processing_fps:.2f}, 采样间隔: {self.sample_interval}")
                
                # 初始化视频写入器，使用更兼容的设置
                output_path = "output/output_video.avi"
                # 尝试不同的编码器
                fourcc_options = ["MJPG", "XVID", "DIVX"]
                self.video_writer = None
                self.depth_video_writer = None

                # 使用摄像头配置中的分辨率
                cam_width, cam_height = self.config.get("cameras", {}).get("camera1", {}).get(
                    "resolution", [640, 480]
                )
                logger.info(f"使用摄像头分辨率: {cam_width}x{cam_height}")

                import os

                os.makedirs(
                    os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                    exist_ok=True,
                )
                exec_cfg = self.config.get("execution", {})
                save_main = exec_cfg.get("save_video_output", True)
                save_depth = exec_cfg.get("save_depth_colormap_video", True)
                depth_path = "output/output_depth_colormap.avi"

                for fourcc_code in fourcc_options:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                        main_ok = not save_main
                        depth_ok = not save_depth
                        if save_main:
                            vw = cv2.VideoWriter(
                                output_path,
                                fourcc,
                                self.processing_fps,
                                (cam_width, cam_height),
                            )
                            if vw.isOpened():
                                self.video_writer = vw
                                main_ok = True
                                logger.info(
                                    f"视频写入器已初始化: {output_path}, 编码器: {fourcc_code}, "
                                    f"帧率: {self.processing_fps:.2f}, 分辨率: {cam_width}x{cam_height}"
                                )
                            else:
                                vw.release()
                                logger.warning(
                                    f"主视频 VideoWriter 无法打开: {output_path} ({fourcc_code})"
                                )
                        if save_depth and main_ok:
                            dw = cv2.VideoWriter(
                                depth_path,
                                fourcc,
                                self.processing_fps,
                                (cam_width, cam_height),
                            )
                            if dw.isOpened():
                                self.depth_video_writer = dw
                                depth_ok = True
                                logger.info(
                                    f"深度伪彩色写入器: {depth_path}, 编码器: {fourcc_code}, "
                                    f"帧率: {self.processing_fps:.2f}"
                                )
                            else:
                                dw.release()
                                logger.warning(
                                    f"深度视频 VideoWriter 无法打开: {depth_path} ({fourcc_code})"
                                )
                                if self.video_writer:
                                    self.video_writer.release()
                                    self.video_writer = None
                        if main_ok and depth_ok:
                            break
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        if self.depth_video_writer:
                            self.depth_video_writer.release()
                            self.depth_video_writer = None
                    except Exception as e:
                        logger.warning(f"尝试编码器 {fourcc_code} 失败: {e}")
                        continue
        except Exception as e:
            logger.warning(f"初始化视频写入器失败: {e}")
            self.video_writer = None
            self.depth_video_writer = None
    
    def _init_result_file(self):
        """
        初始化结果文件
        """
        try:
            # 确保output目录存在
            import os
            os.makedirs("output", exist_ok=True)
            
            self.result_file = open("output/system_results.txt", "w", encoding="utf-8")
            self.result_file.write("# 导盲系统测试结果\n\n")
            self.result_file.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.result_file.write(f"视频路径: {self.config.get('simulation', {}).get('video_paths', {}).get('camera1')}\n\n")
            self.result_file.flush()  # 立即写入
            logger.info("结果文件已初始化，输出路径: output/system_results.txt")
            logger.info(f"结果文件对象: {self.result_file}")
        except Exception as e:
            logger.warning(f"初始化结果文件失败: {e}")
            self.result_file = None
    
    def stop(self):
        """
        停止系统
        """
        logger.info("导盲系统停止中...")
        
        # 打印最终资源状态
        logger.info("\n" + self.resource_manager.get_memory_summary())
        
        self.running = False
        
        # 停止线程
        if hasattr(self, 'input_thread'):
            self.input_thread.stop()
        if hasattr(self, 'asr_thread'):
            self.asr_thread.stop()
        if hasattr(self, 'vision_thread'):
            self.vision_thread.stop()
        if hasattr(self, 'inference_thread'):
            self.inference_thread.stop()
        
        # 等待线程结束
        if hasattr(self, 'input_thread') and self.input_thread.is_alive():
            self.input_thread.join(timeout=2)
        if hasattr(self, 'asr_thread') and self.asr_thread.is_alive():
            self.asr_thread.join(timeout=2)
        if hasattr(self, 'vision_thread') and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2)
        
        # 停止调试查看器
        try:
            if self.config.get("system", {}).get("debug", False):
                self.debug_viewer.stop()
        except Exception as e:
            logger.warning(f"停止调试查看器失败: {e}")

        # 排空录制队列，避免停止前最后一帧未写入
        while True:
            rec = message_queue.receive_message("record", block=False)
            if not rec:
                break
            if self.video_writer is not None and rec.get("main") is not None:
                self.video_writer.write(rec["main"])
            if self.depth_video_writer is not None and rec.get("depth") is not None:
                self.depth_video_writer.write(rec["depth"])
        
        # 关闭视频写入器
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info("主视频写入器已关闭")
            except Exception as e:
                logger.warning(f"关闭主视频写入器失败: {e}")
        if self.depth_video_writer:
            try:
                self.depth_video_writer.release()
                logger.info("深度伪彩色视频写入器已关闭")
            except Exception as e:
                logger.warning(f"关闭深度视频写入器失败: {e}")
        
        # 关闭结果文件
        if self.result_file:
            try:
                # 获取ASR和LLM结果
                asr_results = []
                llm_results = []
                if hasattr(self, 'inference_thread'):
                    results = self.inference_thread.get_results()
                    asr_results = results.get("asr_results", [])
                    llm_results = results.get("llm_results", [])
                
                logger.info(f"准备写入结果文件，ASR结果数量: {len(asr_results)}, LLM结果数量: {len(llm_results)}")
                logger.info(f"结果文件对象: {self.result_file}")
                
                # 写入ASR和LLM结果
                self.result_file.write("\n# ASR识别结果\n")
                for i, result in enumerate(asr_results):
                    logger.info(f"写入ASR结果 {i+1}: {result}")
                    self.result_file.write(f"{i+1}. {result}\n")
                
                self.result_file.write("\n# LLM回复结果\n")
                for i, result in enumerate(llm_results):
                    logger.info(f"写入LLM结果 {i+1}: {result}")
                    self.result_file.write(f"{i+1}. {result}\n")
                
                self.result_file.flush()  # 立即写入
                self.result_file.close()
                logger.info("结果文件已关闭")
            except Exception as e:
                logger.warning(f"关闭结果文件失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("导盲系统已停止")
    
    def _draw_targets(self, frame, tracked_targets):
        """
        在帧上绘制目标信息
        
        Args:
            frame: 输入帧
            tracked_targets: 跟踪目标列表
        """
        for target in tracked_targets:
            # 绘制边界框
            x1, y1, x2, y2 = target.get('bbox', [0, 0, 0, 0])
            class_name = target.get('class_name', 'unknown')
            distance = target.get('distance', 0)
            speed = target.get('speed', 0)
            
            # 计算边界框中心
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 绘制边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制目标信息
            info_text = f"{class_name} - 距离: {distance:.1f}m - 速度: {speed:.1f}m/s"
            cv2.putText(frame, info_text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def _main_loop(self):
        """
        主循环
        """
        while self.running:
            try:
                # 检查控制消息
                control_message = message_queue.receive_message("control", block=False)
                if control_message:
                    if control_message.get("type") == "stop" or control_message.get("type") == "video_ended":
                        logger.info("收到停止消息，等待5秒让所有线程处理完所有消息...")
                        for _ in range(50):
                            while True:
                                rec = message_queue.receive_message("record", block=False)
                                if not rec:
                                    break
                                if self.video_writer is not None and rec.get("main") is not None:
                                    self.video_writer.write(rec["main"])
                                if self.depth_video_writer is not None and rec.get("depth") is not None:
                                    self.depth_video_writer.write(rec["depth"])
                            time.sleep(0.1)
                        logger.info("等待完毕，现在停止系统")
                        self.stop()
                        break

                # 消费推理线程发来的可视化帧
                while True:
                    rec = message_queue.receive_message("record", block=False)
                    if not rec:
                        break
                    if self.video_writer is not None and rec.get("main") is not None:
                        self.video_writer.write(rec["main"])
                    if self.depth_video_writer is not None and rec.get("depth") is not None:
                        self.depth_video_writer.write(rec["depth"])
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                time.sleep(0.5)
        

    
    def signal_handler(self, sig, frame):
        """
        信号处理函数
        """
        logger.info("收到停止信号，正在停止系统...")
        self.stop()
        import os
        os._exit(0)

def main():
    """
    主函数
    """
    import faulthandler
    faulthandler.enable(all_threads=True)

    # 创建系统实例
    system = BlindGuideSystem()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        # 启动系统
        system.start()
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止系统...")
        system.stop()
    # 不再在finally中调用stop()，因为start()方法内部会在视频结束时调用stop()
    # Jetson/FunASR/PyTorch 在解释器自然退出时偶发 native abort（EXIT 134）；业务已正常 stop 后直接退出进程。
    import os
    os._exit(0)

if __name__ == "__main__":
    main()
