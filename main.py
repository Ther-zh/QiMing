import time
import threading
import signal
import sys
import cv2
import numpy as np
from typing import Dict, Any, List

# 导入工具模块
from utils.logger import logger
from utils.config_loader import config_loader

# 导入核心模块
from core.resource_manager import ResourceManager
from core.realtime_scheduler import RealtimeScheduler
from core.complex_scene_scheduler import ComplexSceneScheduler

# 导入感知模块
from perception.yolo.yolo_detector import YoloDetector
from perception.yolo.mock_yolo import MockYoloDetector
from perception.vda.vda_depth import VDADepthEstimator
from perception.vda.mock_vda import MockVDADepthEstimator
from perception.asr.funasr_asr import FunASRRecognizer
from perception.asr.mock_asr import MockFunASRRecognizer
from perception.llm.qwen_multimodal import QwenMultimodal
from perception.llm.mock_llm import MockQwenMultimodal

# 导入融合模块
from fusion.frame_sync import FrameSync
from fusion.depth_fusion import DepthFusion
from fusion.target_tracker import TargetTracker
from fusion.metadata_wrapper import MetadataWrapper

# 导入执行模块
from execution.broadcast_scheduler import BroadcastScheduler
from execution.tts_engine import TTSEngine

# 导入模拟模块
from simulation.camera_simulator import CameraSimulator
from simulation.debug_viewer import DebugViewer

class BlindGuideSystem:
    def __init__(self):
        """
        初始化导盲系统
        """
        # 加载配置
        self.config = config_loader.get_config()
        self.risk_rules = config_loader.get_risk_rules()
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        
        # 初始化模拟模块
        self.camera_simulator = CameraSimulator()
        self.debug_viewer = DebugViewer()
        
        # 初始化感知模块
        self.yolo = self._init_yolo()
        self.vda = self._init_vda()
        self.asr = self._init_asr()
        self.llm = self._init_llm()
        
        # 初始化融合模块
        self.frame_sync = FrameSync()
        self.depth_fusion = DepthFusion()
        self.target_tracker = TargetTracker()
        self.metadata_wrapper = MetadataWrapper()
        
        # 初始化核心调度模块
        self.realtime_scheduler = RealtimeScheduler()
        self.complex_scene_scheduler = ComplexSceneScheduler(self.resource_manager)
        
        # 初始化执行模块
        self.broadcast_scheduler = BroadcastScheduler()
        self.tts_engine = TTSEngine(self.config.get("execution", {}))
        
        # 系统状态
        self.running = False
        
        # 视频写入器
        self.video_writer = None
        
        # 结果记录文件
        self.result_file = None
        
        # 帧计数器和采样频率
        self.frame_count = 0
        self.sample_interval = 5  # 每5帧采样一次，降低处理频率以匹配推理速度
        self.video_fps = 30  # 默认视频帧率
        self.processing_fps = 5  # 目标处理帧率
        self.last_process_time = time.time()
        
        # 存储ASR和LLM结果
        self.asr_results = []
        self.llm_results = []
    
    def _init_yolo(self):
        """
        初始化YOLO模块
        """
        yolo_config = self.config.get("models", {}).get("yolo", {})
        if yolo_config.get("type", "mock") == "real":
            return YoloDetector(yolo_config)
        else:
            return MockYoloDetector(yolo_config)
    
    def _init_vda(self):
        """
        初始化VDA模块
        """
        vda_config = self.config.get("models", {}).get("vda", {})
        if vda_config.get("type", "mock") == "real":
            return VDADepthEstimator(vda_config)
        else:
            return MockVDADepthEstimator(vda_config)
    
    def _init_asr(self):
        """
        初始化ASR模块
        """
        asr_config = self.config.get("models", {}).get("asr", {})
        if asr_config.get("type", "mock") == "real":
            return FunASRRecognizer(asr_config)
        else:
            return MockFunASRRecognizer(asr_config)
    
    def _init_llm(self):
        """
        初始化LLM模块
        """
        llm_config = self.config.get("models", {}).get("llm", {})
        if llm_config.get("type", "mock") == "real":
            return QwenMultimodal(llm_config)
        else:
            return MockQwenMultimodal(llm_config)
    
    def start(self):
        """
        启动系统
        """
        logger.info("导盲系统启动中...")
        
        # 打印配置信息
        logger.info(f"系统模式: {self.config.get('system', {}).get('mode')}")
        logger.info(f"YOLO模型: {self.config.get('models', {}).get('yolo', {}).get('model_path')}")
        logger.info(f"VDA模型: {self.config.get('models', {}).get('vda', {}).get('model_path')}")
        logger.info(f"ASR模型: {self.config.get('models', {}).get('asr', {}).get('model_path')}")
        logger.info(f"LLM模型: {self.config.get('models', {}).get('llm', {}).get('model_path')}")
        logger.info(f"视频路径: {self.config.get('simulation', {}).get('video_paths', {}).get('camera1')}")
        
        # 初始化视频写入器
        self._init_video_writer()
        
        # 初始化结果文件
        self._init_result_file()
        
        # 启动资源管理器
        self.resource_manager.start()
        
        # 启动摄像头模拟
        self.camera_simulator.start()
        
        # 启动帧同步
        self.frame_sync.start()
        
        # 启动实时安全调度
        self.realtime_scheduler.start()
        
        # 启动复杂场景调度
        self.complex_scene_scheduler.start()
        
        # 启动语音播报调度
        self.broadcast_scheduler.start()
        
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
                
                # 固定每25帧处理一次
                self.sample_interval = 25
                self.processing_fps = self.video_fps / self.sample_interval
                
                logger.info(f"视频帧率: {self.video_fps}, 处理帧率: {self.processing_fps:.2f}, 采样间隔: {self.sample_interval}")
                
                # 初始化视频写入器，使用更兼容的设置
                output_path = "output/output_video.avi"
                # 尝试不同的编码器
                fourcc_options = ['MJPG', 'XVID', 'DIVX']
                self.video_writer = None
                
                # 使用摄像头配置中的分辨率
                cam_width, cam_height = self.config.get("cameras", {}).get("camera1", {}).get("resolution", [640, 480])
                logger.info(f"使用摄像头分辨率: {cam_width}x{cam_height}")
                
                for fourcc_code in fourcc_options:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                        # 确保目录存在
                        import os
                        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.processing_fps, (cam_width, cam_height))
                        logger.info(f"视频写入器已初始化，输出路径: {output_path}, 编码器: {fourcc_code}, 帧率: {self.processing_fps:.2f}, 分辨率: {cam_width}x{cam_height}")
                        break
                    except Exception as e:
                        logger.warning(f"尝试编码器 {fourcc_code} 失败: {e}")
                        continue
        except Exception as e:
            logger.warning(f"初始化视频写入器失败: {e}")
            self.video_writer = None
    
    def _init_result_file(self):
        """
        初始化结果文件
        """
        try:
            self.result_file = open("output/system_results.txt", "w", encoding="utf-8")
            self.result_file.write("# 导盲系统测试结果\n\n")
            self.result_file.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.result_file.write(f"视频路径: {self.config.get('simulation', {}).get('video_paths', {}).get('camera1')}\n\n")
            logger.info("结果文件已初始化，输出路径: output/system_results.txt")
        except Exception as e:
            logger.warning(f"初始化结果文件失败: {e}")
            self.result_file = None
    
    def stop(self):
        """
        停止系统
        """
        logger.info("导盲系统停止中...")
        
        self.running = False
        
        # 停止各个模块
        self.broadcast_scheduler.stop()
        self.complex_scene_scheduler.stop()
        self.realtime_scheduler.stop()
        self.frame_sync.stop()
        self.camera_simulator.stop()
        self.resource_manager.stop()
        
        # 停止调试查看器
        try:
            if self.config.get("system", {}).get("debug", False):
                self.debug_viewer.stop()
        except Exception as e:
            logger.warning(f"停止调试查看器失败: {e}")
        
        # 关闭视频写入器
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info("视频写入器已关闭")
            except Exception as e:
                logger.warning(f"关闭视频写入器失败: {e}")
        
        # 关闭结果文件
        if self.result_file:
            try:
                # 写入ASR和LLM结果
                self.result_file.write("\n# ASR识别结果\n")
                for i, result in enumerate(self.asr_results):
                    self.result_file.write(f"{i+1}. {result}\n")
                
                self.result_file.write("\n# LLM回复结果\n")
                for i, result in enumerate(self.llm_results):
                    self.result_file.write(f"{i+1}. {result}\n")
                
                self.result_file.close()
                logger.info("结果文件已关闭")
            except Exception as e:
                logger.warning(f"关闭结果文件失败: {e}")
        
        # 释放模型资源
        if hasattr(self, 'yolo'):
            self.yolo.release()
        if hasattr(self, 'vda'):
            self.vda.release()
        if hasattr(self, 'asr'):
            self.asr.release()
        if hasattr(self, 'llm'):
            self.llm.release()
        if hasattr(self, 'tts_engine'):
            self.tts_engine.release()
        
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
        video_ended = False
        
        while self.running and not video_ended:
            try:
                # 动态计算处理时间
                current_time = time.time()
                elapsed_time = current_time - self.last_process_time
                target_interval = 1.0 / self.processing_fps
                
                # 检查视频是否播放完毕
                main_camera = "camera1"  # 主摄
                if self.camera_simulator.is_video_ended(main_camera):
                    logger.info("视频播放完毕，停止系统")
                    video_ended = True
                    continue
                
                # 获取摄像头帧
                frames = self.camera_simulator.get_all_frames()
                
                # 处理主摄画面
                if main_camera in frames:
                    frame, timestamp = frames[main_camera]
                    if frame is not None:
                        # 帧计数器
                        self.frame_count += 1
                        
                        # 获取真实音频数据
                        audio_data, audio_timestamp = self.camera_simulator.get_audio("camera1")
                        if audio_data is not None:
                            # 执行ASR识别
                            logger.debug(f"[Main] 处理音频数据，长度: {len(audio_data)}")
                            wake_detected, asr_text = self.asr.inference(audio_data)
                            logger.debug(f"[ASR] 识别结果: {asr_text}")
                        else:
                            # 如果没有音频数据，使用空数据
                            import numpy as np
                            audio_data = np.array([])
                            wake_detected, asr_text = self.asr.inference(audio_data)
                            logger.debug(f"[ASR] 识别结果 (无音频): {asr_text}")
                        
                        # 记录ASR结果
                        if asr_text:
                            self.asr_results.append(asr_text)
                            logger.info(f"[Main] 已记录ASR结果: {asr_text}")
                        
                        # 帧采样
                        if self.frame_count % self.sample_interval == 0:
                            # 申请资源
                            if self.resource_manager.request_resources("inference", priority=0):
                                try:
                                    # 添加帧到同步模块
                                    self.frame_sync.add_frame(frame, timestamp, 0)
                                    
                                    # 执行YOLO检测
                                    yolo_results = self.yolo.inference(frame)
                                    self.frame_sync.add_yolo_result(yolo_results, timestamp)
                                    
                                    # 执行VDA深度估计
                                    depth_map = self.vda.inference(frame)
                                    self.frame_sync.add_vda_result(depth_map, timestamp)
                                    
                                    # 获取同步数据
                                    sync_data = self.frame_sync.get_sync_data()
                                    if sync_data:
                                        sync_frame, sync_yolo, sync_depth, sync_timestamp, sync_camera_id = sync_data
                                        
                                        # 计算目标距离
                                        targets_with_distance = self.depth_fusion.calculate_target_distances(sync_yolo, sync_depth)
                                        
                                        # 跟踪目标并计算速度
                                        tracked_targets = self.target_tracker.track_targets(targets_with_distance, sync_timestamp)
                                        
                                        # 封装元数据
                                        metadata = self.metadata_wrapper.wrap_metadata(
                                            sync_frame,
                                            tracked_targets,
                                            sync_timestamp,
                                            sync_camera_id
                                        )
                                        
                                        # 处理实时安全调度
                                        self.realtime_scheduler.process_metadata(metadata)
                                        
                                        # 检查是否有告警
                                        alert = self.realtime_scheduler.get_alert()
                                        if alert:
                                            # 高优先级处理危险警报
                                            if self.resource_manager.request_resources("alert", priority=10):
                                                try:
                                                    # 添加到语音播报队列
                                                    self.broadcast_scheduler.add_message(
                                                        alert.get("message"),
                                                        priority=1 if alert.get("level") == "level1" else 2,
                                                        alert_type=alert.get("level")
                                                    )
                                                    logger.info(f"[Alert] 危险警报: {alert.get('message')}")
                                                finally:
                                                    self.resource_manager.release_resources("alert")
                                        
                                        # 检查是否触发复杂场景
                                        if self.realtime_scheduler.is_complex_scene_triggered():
                                            # 中优先级处理复杂场景
                                            if self.resource_manager.request_resources("llm", priority=5):
                                                try:
                                                    # 处理复杂场景
                                                    response = self.complex_scene_scheduler.process_complex_scene(
                                                        sync_frame,
                                                        metadata,
                                                        "请分析当前场景并提供导航建议"
                                                    )
                                                    if response:
                                                        self.broadcast_scheduler.add_message(
                                                            response,
                                                            priority=3,
                                                            alert_type="complex_scene"
                                                        )
                                                        # 记录LLM结果
                                                        self.llm_results.append(response)
                                                finally:
                                                    self.resource_manager.release_resources("llm")
                                            # 重置触发信号
                                            self.realtime_scheduler.reset_complex_scene_trigger()
                                        
                                        # 更新调试画面
                                        if self.config.get("system", {}).get("debug", False):
                                            # 确定危险等级
                                            risk_level = "level4"  # 默认安全
                                            if alert:
                                                risk_level = alert.get("level")
                                            
                                            self.debug_viewer.update_frame(
                                                sync_frame,
                                                tracked_targets,
                                                sync_depth,
                                                risk_level
                                            )
                                        
                                        # 如果检测到唤醒词，调用LLM处理
                                        if wake_detected:
                                            # 中优先级处理唤醒词
                                            if self.resource_manager.request_resources("llm", priority=5):
                                                try:
                                                    logger.info(f"[Main] 检测到唤醒词，调用LLM处理...")
                                                    # 将NumPy数组转换为PIL Image
                                                    from PIL import Image
                                                    sync_image = Image.fromarray(sync_frame)
                                                    # 调用复杂场景调度器处理
                                                    response = self.complex_scene_scheduler.handle_wake_word(
                                                        asr_text,
                                                        sync_image,
                                                        metadata
                                                    )
                                                    if response:
                                                        logger.info(f"[LLM] 回复: {response}")
                                                        # 添加到语音播报队列
                                                        self.broadcast_scheduler.add_message(
                                                            response,
                                                            priority=3,
                                                            alert_type="wake_word"
                                                        )
                                                        # 记录LLM结果
                                                        self.llm_results.append(response)
                                                        logger.info(f"[Main] 已记录LLM结果: {response}")
                                                finally:
                                                    self.resource_manager.release_resources("llm")
                                        
                                        # 绘制目标信息
                                        output_frame = self._draw_targets(sync_frame.copy(), tracked_targets)
                                        
                                        # 写入视频文件
                                        if self.video_writer:
                                            self.video_writer.write(output_frame)
                                finally:
                                    # 释放资源
                                    self.resource_manager.release_resources("inference")
                    else:
                        # 没有帧数据，摄像头可能还在初始化，继续等待
                        logger.debug("摄像头还在初始化，继续等待...")
                        time.sleep(0.1)
                    
                    # 动态控制帧率
                    processing_time = time.time() - current_time
                    sleep_time = max(0, target_interval - processing_time)
                    time.sleep(sleep_time)
                    
                    # 更新处理时间
                    self.last_process_time = current_time + processing_time + sleep_time
                    
                    # 定期更新心跳，避免超时
                    self.resource_manager.update_heartbeat("main")
                    
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                # 出错时也要更新心跳
                self.resource_manager.update_heartbeat("main")
                time.sleep(0.5)
        
        # 视频播放完毕，停止系统
        if video_ended:
            logger.info("视频处理完成，停止系统")
            self.stop()
        

    
    def signal_handler(self, sig, frame):
        """
        信号处理函数
        """
        logger.info("收到停止信号，正在停止系统...")
        self.stop()
        sys.exit(0)

def main():
    """
    主函数
    """
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
    finally:
        # 停止系统
        system.stop()

if __name__ == "__main__":
    main()
