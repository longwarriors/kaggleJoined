"""
Home Credit Default Risk - 核心模块

提供项目的基础设施，包括：
- 基础抽象类
- 配置管理
- 日志管理
- 通用工具函数

作者：Augment Agent
"""

from .base import BaseProcessor, BaseModel, BaseEvaluator
from .config import ConfigManager
from .logger import LoggerManager
from .utils import *

__all__ = [
    'BaseProcessor',
    'BaseModel', 
    'BaseEvaluator',
    'ConfigManager',
    'LoggerManager'
]
