"""
特征变换模块

实现各种特征变换功能，包括缩放、归一化、幂变换等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor


class FeatureTransformers(BaseProcessor):
    """
    特征变换器
    
    管理所有特征变换功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化特征变换器
        
        Args:
            config: 变换配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.transformers = {}
        self.transform_stats = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> 'FeatureTransformers':
        """
        拟合特征变换器
        
        Args:
            data: 数据字典
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        
        # 获取主表
        main_df = data.get('application_train') or data.get('application_test')
        if main_df is not None:
            self._fit_transformers(main_df)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        执行特征变换
        
        Args:
            df: 输入DataFrame
            **kwargs: 其他参数
            
        Returns:
            变换后的DataFrame
        """
        self._validate_fitted()
        return self.apply_transformations(df)
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用所有特征变换
        
        Args:
            df: 输入DataFrame
            
        Returns:
            变换后的DataFrame
        """
        self._log_info("应用特征变换")
        
        # 处理无穷大值和NaN
        df = self._handle_infinite_values(df)
        
        # 应用缩放变换
        df = self._apply_scaling_transforms(df)
        
        # 应用幂变换
        df = self._apply_power_transforms(df)
        
        # 最终清理
        df = self._final_cleanup(df)
        
        return df
    
    def _fit_transformers(self, df: pd.DataFrame):
        """拟合变换器"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除ID列和目标列
        numerical_cols = [col for col in numerical_cols if col not in ['SK_ID_CURR', 'TARGET']]
        
        if not numerical_cols:
            return
        
        numerical_config = self.config.get('numerical_features', {})
        scaling_config = numerical_config.get('scaling', {})
        
        # 拟合缩放器
        if scaling_config.get('scale_all', False):
            method = scaling_config.get('method', 'standard')
            scaler = self._create_scaler(method)
            
            if scaler is not None:
                # 处理无穷大值和NaN
                clean_data = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
                clean_data = clean_data.fillna(clean_data.median())
                
                scaler.fit(clean_data)
                self.transformers['global_scaler'] = {
                    'scaler': scaler,
                    'columns': numerical_cols,
                    'method': method
                }
        
        # 拟合幂变换器
        transformations_config = numerical_config.get('transformations', {})
        power_config = transformations_config.get('power_transform', {})
        
        if power_config.get('enable', False):
            power_cols = power_config.get('columns', [])
            power_cols = [col for col in power_cols if col in numerical_cols]
            
            if power_cols:
                method = power_config.get('method', 'yeo-johnson')
                power_transformer = PowerTransformer(method=method, standardize=True)
                
                # 处理数据
                clean_data = df[power_cols].replace([np.inf, -np.inf], np.nan)
                clean_data = clean_data.fillna(clean_data.median())
                
                try:
                    power_transformer.fit(clean_data)
                    self.transformers['power_transformer'] = {
                        'transformer': power_transformer,
                        'columns': power_cols,
                        'method': method
                    }
                except Exception as e:
                    self._log_warning(f"拟合幂变换器失败: {e}")
    
    def _create_scaler(self, method: str):
        """创建缩放器"""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'quantile':
            return QuantileTransformer(output_distribution='normal')
        else:
            self._log_warning(f"未知的缩放方法: {method}")
            return None
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理无穷大值"""
        # 替换无穷大值为NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 对数值列填充NaN
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if df[col].isna().any():
                # 使用中位数填充
                median_value = df[col].median()
                if pd.isna(median_value):
                    # 如果中位数也是NaN，使用0
                    median_value = 0
                df[col] = df[col].fillna(median_value)
        
        return df
    
    def _apply_scaling_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用缩放变换"""
        if 'global_scaler' not in self.transformers:
            return df
        
        scaler_info = self.transformers['global_scaler']
        scaler = scaler_info['scaler']
        columns = scaler_info['columns']
        
        # 检查列是否存在
        available_cols = [col for col in columns if col in df.columns]
        
        if not available_cols:
            return df
        
        try:
            # 处理数据
            data_to_scale = df[available_cols].replace([np.inf, -np.inf], np.nan)
            data_to_scale = data_to_scale.fillna(data_to_scale.median())
            
            # 应用变换
            scaled_data = scaler.transform(data_to_scale)
            
            # 创建新的列名
            scaled_columns = [f'{col}_scaled' for col in available_cols]
            
            # 添加缩放后的特征
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=scaled_columns,
                index=df.index
            )
            
            df = pd.concat([df, scaled_df], axis=1)
            
            self._log_info(f"应用全局缩放: {len(available_cols)} 个特征")
            
        except Exception as e:
            self._log_warning(f"应用缩放变换失败: {e}")
        
        return df
    
    def _apply_power_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用幂变换"""
        if 'power_transformer' not in self.transformers:
            return df
        
        transformer_info = self.transformers['power_transformer']
        transformer = transformer_info['transformer']
        columns = transformer_info['columns']
        
        # 检查列是否存在
        available_cols = [col for col in columns if col in df.columns]
        
        if not available_cols:
            return df
        
        try:
            # 处理数据
            data_to_transform = df[available_cols].replace([np.inf, -np.inf], np.nan)
            data_to_transform = data_to_transform.fillna(data_to_transform.median())
            
            # 应用变换
            transformed_data = transformer.transform(data_to_transform)
            
            # 创建新的列名
            transformed_columns = [f'{col}_power_transformed' for col in available_cols]
            
            # 添加变换后的特征
            transformed_df = pd.DataFrame(
                transformed_data,
                columns=transformed_columns,
                index=df.index
            )
            
            df = pd.concat([df, transformed_df], axis=1)
            
            self._log_info(f"应用幂变换: {len(available_cols)} 个特征")
            
        except Exception as e:
            self._log_warning(f"应用幂变换失败: {e}")
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """最终清理"""
        # 再次处理无穷大值和NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 对数值列进行最终填充
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if df[col].isna().any():
                # 使用0填充剩余的NaN
                df[col] = df[col].fillna(0)
        
        # 检查数据类型
        for col in df.columns:
            if df[col].dtype == 'object':
                # 确保字符串列没有NaN
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_transform_stats(self) -> Dict[str, Any]:
        """获取变换统计信息"""
        stats = {}
        
        for name, transformer_info in self.transformers.items():
            stats[name] = {
                'type': type(transformer_info.get('transformer', transformer_info.get('scaler'))).__name__,
                'columns_count': len(transformer_info.get('columns', [])),
                'method': transformer_info.get('method', 'unknown')
            }
        
        return stats
