import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self.risk_rules: Dict[str, Any] = {}
    
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        加载主配置文件
        
        Args:
            config_file: 主配置文件名
            
        Returns:
            配置字典
        """
        config_path = os.path.join(self.config_dir, config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def load_risk_rules(self, rules_file: str = "risk_rules.yaml") -> Dict[str, Any]:
        """
        加载危险分级规则配置文件
        
        Args:
            rules_file: 规则配置文件名
            
        Returns:
            规则配置字典
        """
        rules_path = os.path.join(self.config_dir, rules_file)
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"规则配置文件不存在: {rules_path}")
        
        with open(rules_path, "r", encoding="utf-8") as f:
            self.risk_rules = yaml.safe_load(f)
        
        return self.risk_rules
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取主配置
        
        Returns:
            配置字典
        """
        if not self.config:
            self.load_config()
        return self.config
    
    def get_risk_rules(self) -> Dict[str, Any]:
        """
        获取危险分级规则
        
        Returns:
            规则配置字典
        """
        if not self.risk_rules:
            self.load_risk_rules()
        return self.risk_rules

# 创建全局配置加载器实例
config_loader = ConfigLoader()
