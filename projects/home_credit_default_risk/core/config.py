"""
配置管理模块

提供统一的配置管理功能，支持YAML配置文件的加载、验证和管理。

作者：Augment Agent
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from .base import Singleton


class ConfigManager(metaclass=Singleton):
    """
    配置管理器
    
    使用单例模式确保全局配置的一致性
    """
    
    def __init__(self):
        """初始化配置管理器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._configs = {}
        self._config_paths = {}
        self._default_config_dir = Path(__file__).parent.parent / "config"
        
    def load_config(self, config_path: str, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            config_name: 配置名称，如果不提供则使用文件名
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_name is None:
            config_name = config_path.stem
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            self._configs[config_name] = config
            self._config_paths[config_name] = str(config_path)
            
            self.logger.info(f"成功加载配置: {config_name} from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def load_all_configs(self, config_dir: Optional[str] = None) -> Dict[str, Dict]:
        """
        加载目录下的所有配置文件
        
        Args:
            config_dir: 配置目录路径，默认使用项目配置目录
            
        Returns:
            所有配置的字典
        """
        if config_dir is None:
            config_dir = self._default_config_dir
        else:
            config_dir = Path(config_dir)
            
        if not config_dir.exists():
            self.logger.warning(f"配置目录不存在: {config_dir}")
            return {}
        
        configs = {}
        for config_file in config_dir.glob("*.yaml"):
            try:
                config_name = config_file.stem
                config = self.load_config(config_file, config_name)
                configs[config_name] = config
            except Exception as e:
                self.logger.error(f"加载配置文件失败 {config_file}: {e}")
                
        return configs
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        获取指定配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置字典
        """
        if config_name not in self._configs:
            # 尝试从默认目录加载
            default_path = self._default_config_dir / f"{config_name}.yaml"
            if default_path.exists():
                return self.load_config(default_path, config_name)
            else:
                raise KeyError(f"配置不存在: {config_name}")
        
        return self._configs[config_name].copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get_config("data_config")
    
    def get_feature_config(self) -> Dict[str, Any]:
        """获取特征工程配置"""
        return self.get_config("feature_config")
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_config("model_config")
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """获取流水线配置"""
        return self.get_config("pipeline_config")
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config_name: 配置名称
            updates: 更新内容
        """
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")
        
        self._configs[config_name].update(updates)
        self.logger.info(f"更新配置: {config_name}")
    
    def save_config(self, config_name: str, save_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            config_name: 配置名称
            save_path: 保存路径，默认覆盖原文件
        """
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")
        
        if save_path is None:
            save_path = self._config_paths.get(config_name)
            if save_path is None:
                save_path = self._default_config_dir / f"{config_name}.yaml"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._configs[config_name], f, 
                         default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"保存配置: {config_name} to {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            raise
    
    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """
        验证配置格式
        
        Args:
            config_name: 配置名称
            schema: 验证模式
            
        Returns:
            验证是否通过
        """
        # 简单的配置验证，可以扩展为更复杂的验证逻辑
        config = self.get_config(config_name)
        
        for key, expected_type in schema.items():
            if key not in config:
                self.logger.error(f"配置缺少必需字段: {key}")
                return False
            
            if not isinstance(config[key], expected_type):
                self.logger.error(f"配置字段类型错误: {key}, 期望 {expected_type}, 实际 {type(config[key])}")
                return False
        
        return True
    
    def get_nested_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config_name: 配置名称
            key_path: 键路径，用点分隔，如 "model.xgboost.max_depth"
            default: 默认值
            
        Returns:
            配置值
        """
        config = self.get_config(config_name)
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_nested_value(self, config_name: str, key_path: str, value: Any):
        """
        设置嵌套配置值
        
        Args:
            config_name: 配置名称
            key_path: 键路径，用点分隔
            value: 配置值
        """
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")
        
        config = self._configs[config_name]
        keys = key_path.split('.')
        
        # 导航到最后一级的父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最终值
        config[keys[-1]] = value
        self.logger.info(f"设置配置值: {config_name}.{key_path} = {value}")
    
    def list_configs(self) -> List[str]:
        """列出所有已加载的配置"""
        return list(self._configs.keys())
    
    def clear_configs(self):
        """清空所有配置"""
        self._configs.clear()
        self._config_paths.clear()
        self.logger.info("清空所有配置")
    
    def get_config_info(self) -> Dict[str, str]:
        """获取配置信息"""
        return {
            config_name: self._config_paths.get(config_name, "内存中")
            for config_name in self._configs.keys()
        }
