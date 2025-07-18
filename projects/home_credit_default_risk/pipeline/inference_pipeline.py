"""
推理管线模块

负责加载训练好的模型进行预测推理。

作者：Augment Agent
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np

from ..core.base import BasePipeline
from ..core.config import ConfigManager
from ..core.logger import LoggerManager

from ..data.loader import DataLoader
from ..features.builder import FeatureBuilder
from ..models.serializer import ModelSerializer


class InferencePipeline(BasePipeline):
    """
    推理管线
    
    负责加载训练好的模型和特征工程器，对新数据进行预测。
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Any = None):
        """初始化推理管线"""
        # 初始化配置和日志
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_config('config.yaml')
        
        if logger is None:
            logger_manager = LoggerManager()
            logger = logger_manager.setup_pipeline_logger("inference_pipeline")
        
        super().__init__(config, logger)
        
        # 初始化组件
        self.data_loader = DataLoader(config, logger)
        self.feature_builder = FeatureBuilder(config, logger)
        self.model_serializer = ModelSerializer(config, logger)
        
        # 模型和特征工程器缓存
        self.loaded_models = {}
        self.loaded_feature_builder = None
        self.feature_names = None
        
        # 性能统计
        self.inference_stats = {
            'total_predictions': 0,
            'total_time': 0.0,
            'avg_time_per_prediction': 0.0
        }
    
    def load_models(self, 
                   model_configs: List[Dict] = None,
                   model_dir: str = None,
                   latest: bool = True) -> Dict[str, Any]:
        """
        加载训练好的模型
        
        Args:
            model_configs: 模型配置列表，格式：
                [{'name': 'lgb_1', 'type': 'lightgbm', 'version': 'v20240101_120000'}]
            model_dir: 模型目录路径
            latest: 是否加载最新版本
            
        Returns:
            加载的模型字典
        """
        self._log_info("开始加载模型...")
        
        if model_configs is None:
            # 自动发现模型
            model_configs = self._discover_models(model_dir)
        
        loaded_count = 0
        for model_config in model_configs:
            try:
                model_name = model_config['name']
                model_type = model_config['type']
                version = model_config.get('version')
                
                # 加载模型
                model, metadata = self.model_serializer.load_model(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    latest=latest
                )
                
                # 缓存模型
                self.loaded_models[model_name] = {
                    'model': model,
                    'metadata': metadata,
                    'type': model_type
                }
                
                # 保存特征名称（假设所有模型使用相同的特征）
                if self.feature_names is None and 'feature_names' in metadata:
                    self.feature_names = metadata['feature_names']
                
                loaded_count += 1
                self._log_info(f"模型已加载: {model_name} ({model_type})")
                
            except Exception as e:
                self._log_error(f"加载模型失败: {model_config}, 错误: {e}")
        
        self._log_info(f"共加载 {loaded_count} 个模型")
        return self.loaded_models
    
    def load_feature_builder(self, feature_builder_path: str = None) -> FeatureBuilder:
        """
        加载特征工程器
        
        Args:
            feature_builder_path: 特征工程器路径
            
        Returns:
            加载的特征工程器
        """
        if feature_builder_path is None:
            # 使用默认路径
            feature_builder_path = os.path.join(
                self.config.get('output', {}).get('base_dir', './outputs'),
                'feature_builder.pkl'
            )
        
        try:
            if os.path.exists(feature_builder_path):
                import pickle
                with open(feature_builder_path, 'rb') as f:
                    self.loaded_feature_builder = pickle.load(f)
                self._log_info(f"特征工程器已加载: {feature_builder_path}")
            else:
                # 如果没有保存的特征工程器，使用新的实例
                self.loaded_feature_builder = self.feature_builder
                self._log_warning("未找到保存的特征工程器，使用新实例")
                
        except Exception as e:
            self._log_error(f"加载特征工程器失败: {e}")
            self.loaded_feature_builder = self.feature_builder
        
        return self.loaded_feature_builder
    
    def predict(self, 
               input_data: Union[pd.DataFrame, str, Dict],
               model_names: List[str] = None,
               ensemble_method: str = 'average',
               return_probabilities: bool = True) -> Dict:
        """
        对输入数据进行预测
        
        Args:
            input_data: 输入数据，可以是DataFrame、文件路径或数据字典
            model_names: 要使用的模型名称列表，如果为None则使用所有加载的模型
            ensemble_method: 集成方法 ('average', 'weighted', 'voting')
            return_probabilities: 是否返回概率值
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        try:
            # 1. 数据预处理
            self._log_info("开始数据预处理...")
            processed_data = self._preprocess_input_data(input_data)
            
            # 2. 特征工程
            self._log_info("开始特征工程...")
            features_df = self._apply_feature_engineering(processed_data)
            
            # 3. 模型预测
            self._log_info("开始模型预测...")
            predictions = self._predict_with_models(
                features_df, 
                model_names, 
                return_probabilities
            )
            
            # 4. 集成预测结果
            self._log_info("开始集成预测结果...")
            ensemble_prediction = self._ensemble_predictions(
                predictions, 
                ensemble_method
            )
            
            # 5. 后处理
            final_results = self._postprocess_predictions(
                ensemble_prediction,
                processed_data,
                predictions
            )
            
            # 更新性能统计
            inference_time = time.time() - start_time
            self._update_inference_stats(len(processed_data), inference_time)
            
            self._log_info(f"预测完成，耗时: {inference_time:.2f}秒")
            
            return final_results
            
        except Exception as e:
            self._log_error(f"预测失败: {e}")
            raise
    
    def predict_single(self, 
                      sample_data: Dict,
                      model_names: List[str] = None) -> Dict:
        """
        对单个样本进行预测
        
        Args:
            sample_data: 单个样本数据字典
            model_names: 要使用的模型名称列表
            
        Returns:
            预测结果
        """
        # 转换为DataFrame
        df = pd.DataFrame([sample_data])
        
        # 调用批量预测
        results = self.predict(df, model_names)
        
        # 返回第一个结果
        return {
            'prediction': results['ensemble_prediction'][0],
            'probability': results['ensemble_probability'][0] if 'ensemble_probability' in results else None,
            'individual_predictions': {
                name: pred[0] for name, pred in results['individual_predictions'].items()
            }
        }
    
    def batch_predict(self, 
                     input_file: str,
                     output_file: str,
                     batch_size: int = 1000,
                     model_names: List[str] = None) -> Dict:
        """
        批量预测
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            batch_size: 批处理大小
            model_names: 要使用的模型名称列表
            
        Returns:
            批量预测统计信息
        """
        self._log_info(f"开始批量预测: {input_file} -> {output_file}")
        
        # 读取输入数据
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.parquet'):
            data = pd.read_parquet(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {input_file}")
        
        total_samples = len(data)
        processed_samples = 0
        all_results = []
        
        # 分批处理
        for i in range(0, total_samples, batch_size):
            batch_data = data.iloc[i:i+batch_size]
            
            # 预测
            batch_results = self.predict(batch_data, model_names)
            
            # 收集结果
            batch_df = pd.DataFrame({
                'prediction': batch_results['ensemble_prediction'],
                'probability': batch_results.get('ensemble_probability', [None] * len(batch_data))
            })
            
            # 添加原始ID（如果存在）
            if 'SK_ID_CURR' in batch_data.columns:
                batch_df['SK_ID_CURR'] = batch_data['SK_ID_CURR'].values
            
            all_results.append(batch_df)
            
            processed_samples += len(batch_data)
            self._log_info(f"已处理: {processed_samples}/{total_samples} 样本")
        
        # 合并所有结果
        final_results = pd.concat(all_results, ignore_index=True)
        
        # 保存结果
        if output_file.endswith('.csv'):
            final_results.to_csv(output_file, index=False)
        elif output_file.endswith('.parquet'):
            final_results.to_parquet(output_file, index=False)
        
        self._log_info(f"批量预测完成，结果已保存: {output_file}")
        
        return {
            'total_samples': total_samples,
            'processed_samples': processed_samples,
            'output_file': output_file,
            'inference_stats': self.inference_stats
        }
    
    def get_model_info(self) -> Dict:
        """获取已加载模型的信息"""
        model_info = {}
        
        for model_name, model_data in self.loaded_models.items():
            metadata = model_data['metadata']
            model_info[model_name] = {
                'type': model_data['type'],
                'version': metadata.get('version', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'feature_count': len(metadata.get('feature_names', [])),
                'performance': metadata.get('performance', {})
            }
        
        return model_info
    
    def get_inference_stats(self) -> Dict:
        """获取推理统计信息"""
        return self.inference_stats.copy()
    
    def _preprocess_input_data(self, input_data: Union[pd.DataFrame, str, Dict]) -> pd.DataFrame:
        """预处理输入数据"""
        if isinstance(input_data, str):
            # 文件路径
            if input_data.endswith('.csv'):
                data = pd.read_csv(input_data)
            elif input_data.endswith('.parquet'):
                data = pd.read_parquet(input_data)
            else:
                raise ValueError(f"不支持的文件格式: {input_data}")
        elif isinstance(input_data, dict):
            # 数据字典
            data = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            # DataFrame
            data = input_data.copy()
        else:
            raise ValueError(f"不支持的输入数据类型: {type(input_data)}")
        
        return data
    
    def _apply_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用特征工程"""
        if self.loaded_feature_builder is None:
            raise ValueError("特征工程器未加载，请先调用load_feature_builder()")
        
        # 应用特征工程
        features_df = self.loaded_feature_builder.transform({'application_test': data})
        
        # 如果有保存的特征名称，确保特征顺序一致
        if self.feature_names is not None:
            # 确保所有需要的特征都存在
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                self._log_warning(f"缺少特征: {missing_features}")
                # 用0填充缺失特征
                for feature in missing_features:
                    features_df[feature] = 0
            
            # 按照训练时的特征顺序排列
            features_df = features_df[self.feature_names]
        
        return features_df
    
    def _predict_with_models(self, 
                           features_df: pd.DataFrame,
                           model_names: List[str] = None,
                           return_probabilities: bool = True) -> Dict:
        """使用模型进行预测"""
        if not self.loaded_models:
            raise ValueError("没有加载任何模型，请先调用load_models()")
        
        if model_names is None:
            model_names = list(self.loaded_models.keys())
        
        predictions = {}
        
        for model_name in model_names:
            if model_name not in self.loaded_models:
                self._log_warning(f"模型未加载: {model_name}")
                continue
            
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            
            try:
                if return_probabilities and hasattr(model, 'predict_proba'):
                    # 返回概率
                    pred_proba = model.predict_proba(features_df)
                    if pred_proba.shape[1] == 2:
                        # 二分类，取正类概率
                        predictions[model_name] = pred_proba[:, 1]
                    else:
                        # 多分类，返回所有概率
                        predictions[model_name] = pred_proba
                else:
                    # 返回预测值
                    predictions[model_name] = model.predict(features_df)
                
            except Exception as e:
                self._log_error(f"模型预测失败: {model_name}, 错误: {e}")
        
        return predictions
    
    def _ensemble_predictions(self, 
                            predictions: Dict,
                            ensemble_method: str = 'average') -> np.ndarray:
        """集成预测结果"""
        if not predictions:
            raise ValueError("没有可用的预测结果")
        
        pred_arrays = list(predictions.values())
        
        if ensemble_method == 'average':
            # 简单平均
            ensemble_pred = np.mean(pred_arrays, axis=0)
        elif ensemble_method == 'weighted':
            # 加权平均（基于模型性能）
            weights = self._get_model_weights(list(predictions.keys()))
            ensemble_pred = np.average(pred_arrays, axis=0, weights=weights)
        elif ensemble_method == 'voting':
            # 投票法（硬投票）
            binary_preds = [pred > 0.5 for pred in pred_arrays]
            ensemble_pred = np.mean(binary_preds, axis=0)
        else:
            raise ValueError(f"不支持的集成方法: {ensemble_method}")
        
        return ensemble_pred
    
    def _get_model_weights(self, model_names: List[str]) -> List[float]:
        """获取模型权重"""
        weights = []
        
        for model_name in model_names:
            metadata = self.loaded_models[model_name]['metadata']
            performance = metadata.get('performance', {})
            
            # 使用CV AUC作为权重
            cv_auc = performance.get('cv_auc', 0.5)
            weights.append(cv_auc)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # 如果没有性能信息，使用等权重
            weights = [1.0 / len(model_names)] * len(model_names)
        
        return weights
    
    def _postprocess_predictions(self, 
                               ensemble_prediction: np.ndarray,
                               original_data: pd.DataFrame,
                               individual_predictions: Dict) -> Dict:
        """后处理预测结果"""
        results = {
            'ensemble_prediction': ensemble_prediction.tolist(),
            'individual_predictions': {
                name: pred.tolist() for name, pred in individual_predictions.items()
            }
        }
        
        # 如果是概率预测，添加概率信息
        if np.all((ensemble_prediction >= 0) & (ensemble_prediction <= 1)):
            results['ensemble_probability'] = ensemble_prediction.tolist()
            results['ensemble_binary'] = (ensemble_prediction > 0.5).astype(int).tolist()
        
        # 添加原始数据的ID（如果存在）
        if 'SK_ID_CURR' in original_data.columns:
            results['ids'] = original_data['SK_ID_CURR'].tolist()
        
        return results
    
    def _discover_models(self, model_dir: str = None) -> List[Dict]:
        """自动发现可用的模型"""
        available_models = self.model_serializer.list_available_models()
        
        model_configs = []
        for metadata in available_models:
            if metadata.get('exists', False):
                model_configs.append({
                    'name': metadata['model_name'],
                    'type': metadata['model_type'],
                    'version': metadata['version']
                })
        
        return model_configs
    
    def _update_inference_stats(self, num_samples: int, inference_time: float):
        """更新推理统计信息"""
        self.inference_stats['total_predictions'] += num_samples
        self.inference_stats['total_time'] += inference_time
        
        if self.inference_stats['total_predictions'] > 0:
            self.inference_stats['avg_time_per_prediction'] = (
                self.inference_stats['total_time'] / self.inference_stats['total_predictions']
            )
    
    def run(self, **kwargs) -> Dict:
        """运行推理管线"""
        self._log_info("=" * 60)
        self._log_info("Inference Pipeline 开始运行")
        self._log_info("=" * 60)
        
        try:
            # 1. 加载模型
            self._log_info("阶段 1: 加载模型")
            models = self.load_models()
            
            # 2. 加载特征工程器
            self._log_info("阶段 2: 加载特征工程器")
            feature_builder = self.load_feature_builder()
            
            # 3. 准备推理
            self._log_info("阶段 3: 推理管线准备完成")
            
            results = {
                'loaded_models': len(models),
                'model_info': self.get_model_info(),
                'pipeline_status': 'ready_for_inference'
            }
            
            self._log_info("=" * 60)
            self._log_info("Inference Pipeline 准备完成")
            self._log_info("=" * 60)
            
            return results
            
        except Exception as e:
            self._log_error(f"Inference Pipeline 初始化失败: {e}")
            raise
