import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name: str, log_dir: str = "logs"):
        """
        初始化日志配置
        
        Args:
            name: 日志名称
            log_dir: 日志存储目录
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建file handler
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            
            # 创建console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
            # 设置formatter
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """
        输出DEBUG级别的日志
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """
        输出INFO级别的日志
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        输出WARNING级别的日志
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        输出ERROR级别的日志
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """
        输出CRITICAL级别的日志
        """
        self.logger.critical(message)

# 创建全局日志实例
logger = Logger("blind_guide_system")
