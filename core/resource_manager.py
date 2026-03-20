import threading
import time
import psutil
from typing import Dict, Any, Optional
from collections import defaultdict

from utils.logger import logger
from utils.config_loader import config_loader

try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class ResourceManager:
    def __init__(self):
        """
        初始化资源管理器
        """
        self.config = config_loader.get_config()
        self.resources = defaultdict(bool)  # 资源使用状态
        self.heartbeat_times = defaultdict(float)  # 模块心跳时间
        self.lock = threading.Lock()
        
        self.nvml_initialized = False
        self.gpu_handle = None
        self.gpu_name = None
        self._init_nvml()
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.running = False
    
    def _init_nvml(self):
        """初始化NVML进行GPU监控"""
        if NVML_AVAILABLE:
            try:
                nvmlInit()
                gpu_count = nvmlDeviceGetCount()
                if gpu_count > 0:
                    self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
                    self.gpu_name = nvmlDeviceGetName(self.gpu_handle)
                    self.nvml_initialized = True
                    logger.info(f"GPU监控已初始化: {self.gpu_name}")
            except Exception as e:
                logger.warning(f"NVML初始化失败: {e}")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        获取GPU显存信息
        
        Returns:
            包含total, used, free, usage_percent的字典（单位：GB）
        """
        result = {"total": 0, "used": 0, "free": 0, "usage_percent": 0, "available": False}
        
        if self.nvml_initialized and self.gpu_handle:
            try:
                info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                result["total"] = info.total / 1024**3
                result["used"] = info.used / 1024**3
                result["free"] = info.free / 1024**3
                result["usage_percent"] = (info.used / info.total) * 100 if info.total > 0 else 0
                result["available"] = True
            except Exception as e:
                logger.debug(f"获取GPU显存信息失败: {e}")
        
        return result
    
    def print_gpu_memory(self, stage: str = ""):
        """
        打印GPU显存使用情况
        
        Args:
            stage: 当前阶段描述
        """
        gpu_info = self.get_gpu_memory_info()
        if gpu_info["available"]:
            prefix = f"[{stage}] " if stage else ""
            logger.info(f"{prefix}GPU显存: 总计={gpu_info['total']:.2f}GB, "
                       f"使用={gpu_info['used']:.2f}GB ({gpu_info['usage_percent']:.1f}%), "
                       f"可用={gpu_info['free']:.2f}GB")
        else:
            logger.debug("GPU显存监控不可用")
    
    def get_memory_summary(self) -> str:
        """
        获取内存使用摘要字符串
        
        Returns:
            格式化的内存使用摘要
        """
        lines = []
        lines.append("=" * 50)
        lines.append("系统资源使用情况")
        lines.append("=" * 50)
        
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        lines.append(f"CPU使用率: {cpu_usage:.1f}%")
        lines.append(f"内存使用: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")
        
        gpu_info = self.get_gpu_memory_info()
        if gpu_info["available"]:
            lines.append(f"GPU型号: {self.gpu_name}")
            lines.append(f"GPU显存: {gpu_info['used']:.2f}GB / {gpu_info['total']:.2f}GB ({gpu_info['usage_percent']:.1f}%)")
        else:
            lines.append("GPU显存: 不可用")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def start(self):
        """
        启动资源管理器
        """
        self.running = True
        self.monitor_thread.start()
        logger.info("资源管理器已启动")
    
    def stop(self):
        """
        停止资源管理器
        """
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
        logger.info("资源管理器已停止")
    
    def request_resources(self, resource_type: str, priority: int = 0) -> bool:
        """
        申请资源
        
        Args:
            resource_type: 资源类型
            priority: 优先级，值越高优先级越高
            
        Returns:
            是否申请成功
        """
        # 获取verbose配置
        try:
            verbose = self.config.get("system", {}).get("verbose", True)
        except Exception:
            verbose = True
        
        with self.lock:
            # 对于危险警报等紧急任务，即使资源被占用也允许申请
            if priority > 5:
                if verbose:
                    logger.info(f"高优先级任务 {resource_type} 强制申请资源")
                # 对于最高优先级的危险警报，直接返回成功
                if priority > 8:
                    return True
            
            # 检查资源是否可用
            if not self.resources[resource_type]:
                # 检查系统资源
                if self._check_system_resources():
                    self.resources[resource_type] = True
                    if verbose:
                        logger.info(f"资源 {resource_type} 申请成功")
                    return True
                else:
                    # 对于高优先级任务，即使系统资源不足也允许申请
                    if priority > 3:
                        if verbose:
                            logger.warning(f"系统资源不足，但高优先级任务 {resource_type} 强制申请资源")
                        self.resources[resource_type] = True
                        return True
                    else:
                        if verbose:
                            logger.warning(f"系统资源不足，无法申请 {resource_type}")
                        return False
            else:
                # 对于高优先级任务，尝试抢占资源
                if priority > 5:
                    if verbose:
                        logger.info(f"高优先级任务 {resource_type} 抢占资源")
                    return True
                else:
                    if verbose:
                        logger.warning(f"资源 {resource_type} 已被占用")
                    return False
    
    def release_resources(self, resource_type: str):
        """
        释放资源
        
        Args:
            resource_type: 资源类型
        """
        # 获取verbose配置
        try:
            verbose = self.config.get("system", {}).get("verbose", True)
        except Exception:
            verbose = True
        
        with self.lock:
            if self.resources[resource_type]:
                self.resources[resource_type] = False
                if verbose:
                    logger.info(f"资源 {resource_type} 已释放")
    
    def update_heartbeat(self, module_name: str):
        """
        更新模块心跳
        
        Args:
            module_name: 模块名称
        """
        with self.lock:
            self.heartbeat_times[module_name] = time.time()
    
    def _check_system_resources(self) -> bool:
        """
        检查系统资源
        
        Returns:
            资源是否充足
        """
        resource_config = self.config.get("resources", {})
        
        # 检查CPU使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu_threshold = resource_config.get("cpu_threshold", 80)
        if cpu_usage > cpu_threshold:
            logger.warning(f"CPU使用率过高: {cpu_usage}%")
            return False
        
        # 检查内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_threshold = resource_config.get("memory_threshold", 80)
        if memory_usage > memory_threshold:
            logger.warning(f"内存使用率过高: {memory_usage}%")
            return False
        
        # 检查GPU内存使用率（如果有GPU）
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                gpu_usage = (gpu_memory / gpu_total) * 100
                gpu_threshold = resource_config.get("gpu_memory_threshold", 90)
                if gpu_usage > gpu_threshold:
                    logger.warning(f"GPU内存使用率过高: {gpu_usage}%")
                    return False
        except ImportError:
            pass
        
        return True
    
    def _monitor_resources(self):
        """
        监控系统资源和模块心跳
        """
        monitor_interval = 5  # 每5秒输出一次资源状态
        last_monitor_time = time.time()
        
        while self.running:
            # 检查系统资源
            self._check_system_resources()
            
            # 检查模块心跳
            self._check_heartbeats()
            
            # 检查资源使用情况
            self._check_resource_usage()
            
            # 定期输出GPU显存状态
            current_time = time.time()
            if current_time - last_monitor_time >= monitor_interval:
                self.print_gpu_memory("资源监控")
                last_monitor_time = current_time
            
            time.sleep(1)  # 每秒检查一次
    
    def _check_heartbeats(self):
        """
        检查模块心跳
        """
        current_time = time.time()
        with self.lock:
            for module_name, last_time in list(self.heartbeat_times.items()):
                if current_time - last_time > 0.5:  # 超过500ms无心跳
                    logger.warning(f"模块 {module_name} 心跳超时")
                    # 这里可以添加自动重启模块的逻辑
    
    def _check_resource_usage(self):
        """
        检查资源使用情况
        """
        with self.lock:
            for resource_type, in_use in self.resources.items():
                if in_use:
                    logger.debug(f"资源 {resource_type} 正在使用中")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        获取资源状态
        
        Returns:
            资源状态字典
        """
        with self.lock:
            # 获取系统资源状态
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 获取GPU状态（如果有）
            gpu_usage = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                    gpu_usage = (gpu_memory / gpu_total) * 100
            except ImportError:
                pass
            
            return {
                "system": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "gpu_usage": gpu_usage
                },
                "resources": dict(self.resources),
                "heartbeats": dict(self.heartbeat_times)
            }
