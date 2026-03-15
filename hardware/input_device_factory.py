from hardware.input_device import InputDevice
from hardware.real_input_device import RealInputDevice
from hardware.simulated_input_device import SimulatedInputDevice
from utils.config_loader import config_loader
from utils.logger import logger

class InputDeviceFactory:
    """
    输入设备工厂类
    """
    
    @staticmethod
    def create_input_device() -> InputDevice:
        """
        创建输入设备实例
        
        Returns:
            InputDevice: 输入设备实例
        """
        config = config_loader.get_config()
        input_mode = config.get("system", {}).get("input_mode", "simulated")
        
        if input_mode == "real":
            logger.info("创建真实输入设备实例")
            return RealInputDevice()
        else:
            logger.info("创建模拟输入设备实例")
            return SimulatedInputDevice()
