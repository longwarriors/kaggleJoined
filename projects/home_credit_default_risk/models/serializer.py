"""
模型序列化模块

负责模型的保存、加载和版本管理。

作者：Augment Agent
"""

import os
import pickle
import json
import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from ..core.base import BaseProcessor
from ..core.logger import LoggerManager


class ModelSerializer(BaseProcessor):
    """
    模型序列化器
    
    负责模型的保存、加载和版本管理，支持多种序列化格式。
    """
    
    def __init__(self, config: Dict = None, logger: Any = None):
        """初始化模型序列化器"""
        super().__init__(config, logger)
        
        # 设置默认配置
        self.default_config = {
            'output': {
                'base_dir': './outputs',
                'subdirs': {
                    'models': 'models',
                    'metadata': 'metadata'
                },
                'formats': {
                    'model': 'pickle',  # 'pickle', 'joblib', 'onnx'
                    'metadata': 'json'
                },
                'naming': {
                    'use_timestamp': True,
                    'timestamp_format': '%Y%m%d_%H%M%S'
                }
            }
        }
        
        # 合并配置
        if config is None:
            self.config = self.default_config
        else:
            self.config = config
            if 'output' not in self.config:
                self.config['output'] = self.default_config['output']
        
        # 设置日志
        if logger is None:
            logger_manager = LoggerManager()
            self.logger = logger_manager.setup_logger('model_serializer')
        else:
            self.logger = logger
        
        # 创建输出目录
        self._create_output_dirs()
    
    def fit(self, data: Any = None, **kwargs) -> 'ModelSerializer':
        """拟合方法（仅为兼容BaseProcessor接口）"""
        self.is_fitted = True
        return self
    
    def transform(self, data: Any = None, **kwargs) -> Any:
        """转换方法（仅为兼容BaseProcessor接口）"""
        self._validate_fitted()
        return data
    
    def save_model(self, 
                  model: Any, 
                  model_name: str, 
                  model_type: str, 
                  metadata: Dict = None, 
                  version: str = None,
                  feature_names: List[str] = None) -> str:
        """
        保存模型及其元数据
        
        Args:
            model: 要保存的模型对象
            model_name: 模型名称
            model_type: 模型类型 (如 'lightgbm', 'xgboost', 'sklearn', 'ensemble')
            metadata: 模型元数据 (如训练参数、性能指标等)
            version: 模型版本 (如果为None，则自动生成)
            feature_names: 特征名称列表
            
        Returns:
            保存的模型路径
        """
        # 验证输入
        if model is None:
            raise ValueError("模型对象不能为空")
        
        # 生成版本号和时间戳
        timestamp = datetime.datetime.now().strftime(
            self.config['output']['naming']['timestamp_format']
        )
        
        if version is None:
            version = f"v{timestamp}"
        
        # 构建模型文件名
        model_filename = f"{model_name}_{model_type}_{version}"
        
        # 获取模型保存目录
        models_dir = self._get_models_dir()
        
        # 确定序列化格式
        format_type = self.config['output']['formats']['model']
        
        # 保存模型
        model_path = os.path.join(models_dir, f"{model_filename}.{format_type}")
        
        try:
            if format_type == 'pickle':
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            elif format_type == 'joblib':
                import joblib
                joblib.dump(model, model_path)
            elif format_type == 'onnx':
                # 需要根据模型类型实现ONNX转换
                self._save_as_onnx(model, model_path, model_type)
            else:
                raise ValueError(f"不支持的序列化格式: {format_type}")
            
            self.logger.info(f"模型已保存: {model_path}")
            
            # 准备元数据
            if metadata is None:
                metadata = {}
            
            # 添加基本元数据
            metadata.update({
                'model_name': model_name,
                'model_type': model_type,
                'version': version,
                'timestamp': timestamp,
                'format': format_type,
                'path': model_path,
                'feature_names': feature_names
            })
            
            # 添加模型指纹
            metadata['model_fingerprint'] = self._calculate_model_fingerprint(model_path)
            
            # 保存元数据
            metadata_path = self._save_metadata(metadata, model_filename)
            self.logger.info(f"模型元数据已保存: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            raise
    
    def load_model(self, 
                  model_path: str = None, 
                  model_name: str = None, 
                  model_type: str = None, 
                  version: str = None,
                  latest: bool = False) -> Tuple[Any, Dict]:
        """
        加载模型及其元数据
        
        Args:
            model_path: 模型文件路径 (优先)
            model_name: 模型名称 (与model_type和version一起使用)
            model_type: 模型类型
            version: 模型版本
            latest: 是否加载最新版本 (当model_path为None时有效)
            
        Returns:
            (model, metadata): 加载的模型对象和元数据
        """
        # 确定模型路径
        if model_path is None:
            if latest:
                model_path, metadata = self._find_latest_model(model_name, model_type)
                if model_path is None:
                    raise FileNotFoundError(f"未找到模型: {model_name}, {model_type}")
            else:
                if model_name is None or model_type is None:
                    raise ValueError("必须提供model_name和model_type，或者提供model_path")
                
                # 构建模型文件名
                if version is None:
                    # 查找最新版本
                    model_path, metadata = self._find_latest_model(model_name, model_type)
                    if model_path is None:
                        raise FileNotFoundError(f"未找到模型: {model_name}, {model_type}")
                else:
                    # 使用指定版本
                    format_type = self.config['output']['formats']['model']
                    model_filename = f"{model_name}_{model_type}_{version}"
                    models_dir = self._get_models_dir()
                    model_path = os.path.join(models_dir, f"{model_filename}.{format_type}")
                    
                    # 加载元数据
                    metadata = self._load_metadata(model_filename)
        else:
            # 从路径中提取文件名
            model_filename = os.path.basename(model_path).split('.')[0]
            metadata = self._load_metadata(model_filename)
        
        # 验证模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 验证模型指纹
        if metadata and 'model_fingerprint' in metadata:
            current_fingerprint = self._calculate_model_fingerprint(model_path)
            if current_fingerprint != metadata['model_fingerprint']:
                self.logger.warning(f"模型指纹不匹配，模型可能已被修改: {model_path}")
        
        # 确定序列化格式
        format_type = os.path.splitext(model_path)[1][1:]
        
        # 加载模型
        try:
            if format_type == 'pickle':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif format_type == 'joblib':
                import joblib
                model = joblib.load(model_path)
            elif format_type == 'onnx':
                model = self._load_from_onnx(model_path)
            else:
                raise ValueError(f"不支持的序列化格式: {format_type}")
            
            self.logger.info(f"模型已加载: {model_path}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def list_available_models(self, model_name: str = None, model_type: str = None) -> List[Dict]:
        """
        列出可用的模型
        
        Args:
            model_name: 过滤特定模型名称 (可选)
            model_type: 过滤特定模型类型 (可选)
            
        Returns:
            模型元数据列表
        """
        models_dir = self._get_models_dir()
        metadata_dir = self._get_metadata_dir()
        
        # 获取所有元数据文件
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        
        # 加载所有元数据
        all_metadata = []
        for metadata_file in metadata_files:
            try:
                with open(os.path.join(metadata_dir, metadata_file), 'r') as f:
                    metadata = json.load(f)
                
                # 过滤条件
                if model_name is not None and metadata.get('model_name') != model_name:
                    continue
                
                if model_type is not None and metadata.get('model_type') != model_type:
                    continue
                
                # 验证模型文件存在
                model_path = metadata.get('path')
                if model_path and os.path.exists(model_path):
                    metadata['exists'] = True
                else:
                    metadata['exists'] = False
                
                all_metadata.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"加载元数据失败: {metadata_file}, 错误: {e}")
        
        # 按时间戳排序
        all_metadata.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_metadata
    
    def delete_model(self, model_path: str = None, model_name: str = None, 
                    model_type: str = None, version: str = None) -> bool:
        """
        删除模型及其元数据
        
        Args:
            model_path: 模型文件路径 (优先)
            model_name: 模型名称
            model_type: 模型类型
            version: 模型版本
            
        Returns:
            是否成功删除
        """
        # 确定模型路径
        if model_path is None:
            if model_name is None or model_type is None or version is None:
                raise ValueError("必须提供model_path，或者提供model_name、model_type和version")
            
            # 构建模型文件名
            format_type = self.config['output']['formats']['model']
            model_filename = f"{model_name}_{model_type}_{version}"
            models_dir = self._get_models_dir()
            model_path = os.path.join(models_dir, f"{model_filename}.{format_type}")
        else:
            # 从路径中提取文件名
            model_filename = os.path.basename(model_path).split('.')[0]
        
        # 删除模型文件
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                self.logger.info(f"模型文件已删除: {model_path}")
            else:
                self.logger.warning(f"模型文件不存在: {model_path}")
                return False
            
            # 删除元数据文件
            metadata_path = os.path.join(
                self._get_metadata_dir(), 
                f"{model_filename}.json"
            )
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                self.logger.info(f"元数据文件已删除: {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除模型失败: {e}")
            return False
    
    def _create_output_dirs(self):
        """创建输出目录"""
        base_dir = self.config['output']['base_dir']
        models_dir = os.path.join(base_dir, self.config['output']['subdirs']['models'])
        metadata_dir = os.path.join(base_dir, self.config['output']['subdirs']['metadata'])
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
    
    def _get_models_dir(self) -> str:
        """获取模型保存目录"""
        return os.path.join(
            self.config['output']['base_dir'],
            self.config['output']['subdirs']['models']
        )
    
    def _get_metadata_dir(self) -> str:
        """获取元数据保存目录"""
        return os.path.join(
            self.config['output']['base_dir'],
            self.config['output']['subdirs']['metadata']
        )
    
    def _save_metadata(self, metadata: Dict, model_filename: str) -> str:
        """保存模型元数据"""
        metadata_dir = self._get_metadata_dir()
        metadata_path = os.path.join(metadata_dir, f"{model_filename}.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def _load_metadata(self, model_filename: str) -> Dict:
        """加载模型元数据"""
        metadata_dir = self._get_metadata_dir()
        metadata_path = os.path.join(metadata_dir, f"{model_filename}.json")
        
        if not os.path.exists(metadata_path):
            self.logger.warning(f"元数据文件不存在: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _find_latest_model(self, model_name: str, model_type: str) -> Tuple[Optional[str], Dict]:
        """查找最新版本的模型"""
        all_metadata = self.list_available_models(model_name, model_type)
        
        if not all_metadata:
            return None, {}
        
        # 获取最新的元数据
        latest_metadata = all_metadata[0]
        
        # 检查模型文件是否存在
        model_path = latest_metadata.get('path')
        if model_path and os.path.exists(model_path):
            return model_path, latest_metadata
        else:
            self.logger.warning(f"最新模型文件不存在: {model_path}")
            return None, {}
    
    def _calculate_model_fingerprint(self, model_path: str) -> str:
        """计算模型文件的指纹"""
        hasher = hashlib.md5()
        with open(model_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def _save_as_onnx(self, model: Any, model_path: str, model_type: str):
        """将模型保存为ONNX格式"""
        try:
            import onnx
            import onnxmltools
            
            if model_type == 'lightgbm':
                onnx_model = onnxmltools.convert_lightgbm(model)
            elif model_type == 'xgboost':
                onnx_model = onnxmltools.convert_xgboost(model)
            elif model_type == 'sklearn':
                onnx_model = onnxmltools.convert_sklearn(model)
            else:
                raise ValueError(f"不支持转换为ONNX的模型类型: {model_type}")
            
            onnx.save_model(onnx_model, model_path)
            
        except ImportError:
            self.logger.error("未安装ONNX相关库，无法保存为ONNX格式")
            raise
    
    def _load_from_onnx(self, model_path: str) -> Any:
        """从ONNX格式加载模型"""
        try:
            import onnx
            import onnxruntime
            
            # 加载ONNX模型
            onnx_model = onnx.load(model_path)
            
            # 创建推理会话
            session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
            
            # 返回包装后的会话
            return ONNXModelWrapper(session)
            
        except ImportError:
            self.logger.error("未安装ONNX相关库，无法加载ONNX格式模型")
            raise


class ONNXModelWrapper:
    """ONNX模型包装器，提供与其他模型兼容的接口"""
    
    def __init__(self, session):
        self.session = session
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, X):
        """预测方法"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.session.run([self.output_name], {self.input_name: X.astype(np.float32)})[0]
