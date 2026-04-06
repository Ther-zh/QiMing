import queue
import threading
from typing import Any, Dict, Optional

class MessageQueue:
    """
    消息队列类，用于模块间通信
    """
    
    def __init__(self):
        self.queues = {}
        self.lock = threading.RLock()
    
    def create_queue(self, name: str, maxsize: int = 0):
        """
        创建一个新的消息队列

        Args:
            name: 队列名称
            maxsize: 最大长度；>0 时满则 put 阻塞，用于对生产者反压（如 vision 避免积压大图）
        """
        with self.lock:
            if name not in self.queues:
                self.queues[name] = queue.Queue(maxsize=maxsize)

    def send_message(self, queue_name: str, message: Any):
        """
        发送消息到指定队列

        Args:
            queue_name: 队列名称
            message: 消息内容
        """
        with self.lock:
            q = self.queues.get(queue_name)
        if q is not None:
            q.put(message)
    
    def receive_message(self, queue_name: str, block: bool = True, timeout: Optional[float] = None) -> Any:
        """
        从指定队列接收消息
        
        Args:
            queue_name: 队列名称
            block: 是否阻塞
            timeout: 超时时间
            
        Returns:
            消息内容，如果队列为空且非阻塞则返回None
        """
        with self.lock:
            if queue_name not in self.queues:
                return None
        
        try:
            return self.queues[queue_name].get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self, queue_name: str) -> int:
        """
        获取队列大小
        
        Args:
            queue_name: 队列名称
            
        Returns:
            队列大小
        """
        with self.lock:
            if queue_name in self.queues:
                return self.queues[queue_name].qsize()
            return 0
    
    def clear_queue(self, queue_name: str):
        """
        清空队列
        
        Args:
            queue_name: 队列名称
        """
        with self.lock:
            if queue_name in self.queues:
                while not self.queues[queue_name].empty():
                    try:
                        self.queues[queue_name].get_nowait()
                    except queue.Empty:
                        break

# 创建全局消息队列实例
message_queue = MessageQueue()