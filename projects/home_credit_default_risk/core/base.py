"""
基础抽象类模块

定义项目中所有组件的基础接口和抽象类，确保一致的设计模式。

作者：Augment Agent
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime


class BaseProcessor(ABC):
    """
    所有数据处理器的基类
    
    提供通用的处理器接口和基础功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化处理器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.is_fitted = False
        self.metadata = {}
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseProcessor':
        """
        拟合处理器
        
        Args:
            data: 训练数据
            **kwargs: 其他参数
            
        Returns:
            self: 返回自身以支持链式调用
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        转换数据
        
        Args:
            data: 待转换数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        拟合并转换数据
        
        Args:
            data: 数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def get_metadata(self) -> Dict:
        """获取处理器元数据"""
        return self.metadata.copy()
    
    def _validate_fitted(self):
        """验证处理器是否已拟合"""
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} 尚未拟合，请先调用fit方法")
    
    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BaseModel(ABC):
    """
    所有模型的基类
    
    提供通用的模型接口和基础功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化模型
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.model = None
        self.training_history = {}
        self.feature_names = []
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 其他参数
            
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测概率（如果支持）
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
            
        Returns:
            预测概率
        """
        raise NotImplementedError("该模型不支持概率预测")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取特征重要性"""
        return None
    
    def get_training_history(self) -> Dict:
        """获取训练历史"""
        return self.training_history.copy()
    
    def _validate_trained(self):
        """验证模型是否已训练"""
        if not self.is_trained:
            raise ValueError(f"{self.__class__.__name__} 尚未训练，请先调用train方法")
    
    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BaseEvaluator(ABC):
    """
    所有评估器的基类
    
    提供通用的评估接口和基础功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.evaluation_results = {}
        
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict:
        """
        评估预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        pass
    
    def get_evaluation_results(self) -> Dict:
        """获取评估结果"""
        return self.evaluation_results.copy()
    
    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BasePipeline(ABC):
    """
    所有流水线的基类
    
    提供通用的流水线接口和基础功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化流水线
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.steps = []
        self.results = {}
        
    @abstractmethod
    def run(self, **kwargs) -> Dict:
        """
        运行流水线
        
        Args:
            **kwargs: 其他参数
            
        Returns:
            运行结果字典
        """
        pass
    
    def add_step(self, name: str, processor: Union[BaseProcessor, BaseModel]):
        """
        添加流水线步骤
        
        Args:
            name: 步骤名称
            processor: 处理器或模型
        """
        self.steps.append((name, processor))
        self._log_info(f"添加步骤: {name}")
    
    def get_results(self) -> Dict:
        """获取流水线结果"""
        return self.results.copy()
    
    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class Singleton(type):
    """单例模式元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
