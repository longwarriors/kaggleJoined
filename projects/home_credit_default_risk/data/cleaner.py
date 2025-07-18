"""
数据清洗模块

提供全面的数据清洗功能，包括缺失值处理、异常值处理、
数据类型转换等，确保数据质量。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor
from ..core.utils import reduce_memory_usage


class DataCleaner(BaseProcessor):
    """
    数据清洗器
    
    提供全面的数据清洗功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化数据清洗器
        
        Args:
            config: 清洗配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.cleaning_stats = {}
        self.imputers = {}
        self.encoders = {}
        self.outlier_bounds = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], **kwargs) -> 'DataCleaner':
        """
        拟合数据清洗器
        
        Args:
            data: 数据字典
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        self._fit_imputers()
        self._fit_encoders()
        self._calculate_outlier_bounds()
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, pd.DataFrame] = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        执行数据清洗
        
        Args:
            data: 数据字典，如果不提供则使用fit时的数据
            **kwargs: 其他参数
            
        Returns:
            清洗后的数据字典
        """
        self._validate_fitted()
        
        if data is not None:
            data_to_clean = data
        else:
            data_to_clean = self.data
        
        return self.clean_all_data(data_to_clean)
    
    def clean_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        清洗所有数据
        
        Args:
            data: 原始数据字典
            
        Returns:
            清洗后的数据字典
        """
        self._log_info("开始数据清洗")
        
        cleaned_data = {}
        cleaning_config = self.config.get('cleaning', {})
        
        for name, df in data.items():
            self._log_info(f"清洗数据集: {name}")
            
            # 记录原始统计信息
            original_shape = df.shape
            original_missing = df.isnull().sum().sum()
            
            # 执行清洗步骤
            cleaned_df = df.copy()
            
            # 1. 处理缺失值
            cleaned_df = self._handle_missing_values(cleaned_df, name)
            
            # 2. 处理异常值
            cleaned_df = self._handle_outliers(cleaned_df, name)
            
            # 3. 数据类型转换
            cleaned_df = self._convert_dtypes(cleaned_df, name)
            
            # 4. 处理重复值
            cleaned_df = self._handle_duplicates(cleaned_df, name)
            
            # 5. 内存优化
            if cleaning_config.get('memory_optimization', {}).get('enable', True):
                cleaned_df = reduce_memory_usage(cleaned_df, verbose=False)
            
            # 记录清洗统计信息
            self.cleaning_stats[name] = {
                'original_shape': original_shape,
                'cleaned_shape': cleaned_df.shape,
                'original_missing': original_missing,
                'cleaned_missing': cleaned_df.isnull().sum().sum(),
                'rows_removed': original_shape[0] - cleaned_df.shape[0],
                'columns_removed': original_shape[1] - cleaned_df.shape[1]
            }
            
            cleaned_data[name] = cleaned_df
            
            self._log_info(f"{name} 清洗完成: {original_shape} -> {cleaned_df.shape}")
        
        self._log_info("数据清洗完成")
        return cleaned_data
    
    def _handle_missing_values(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            处理后的DataFrame
        """
        missing_config = self.config.get('cleaning', {}).get('missing_values', {})
        
        # 删除缺失值过多的列
        drop_threshold = missing_config.get('drop_threshold', 0.95)
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratios[missing_ratios > drop_threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self._log_info(f"删除高缺失列 ({len(cols_to_drop)}): {cols_to_drop[:5]}...")
        
        # 填充缺失值
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 数值列填充
        if numerical_cols:
            strategy = missing_config.get('numerical_strategy', 'median')
            if dataset_name in self.imputers and 'numerical' in self.imputers[dataset_name]:
                imputer = self.imputers[dataset_name]['numerical']
                df[numerical_cols] = imputer.transform(df[numerical_cols])
            else:
                # 如果没有预训练的imputer，使用简单填充
                fill_value = missing_config.get('numerical_fill_value', 0)
                if strategy == 'median':
                    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                elif strategy == 'mean':
                    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
                else:
                    df[numerical_cols] = df[numerical_cols].fillna(fill_value)
        
        # 类别列填充
        if categorical_cols:
            strategy = missing_config.get('categorical_strategy', 'mode')
            fill_value = missing_config.get('categorical_fill_value', 'Unknown')
            
            for col in categorical_cols:
                if strategy == 'mode' and not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                else:
                    df[col] = df[col].fillna(fill_value)
        
        # 创建缺失值指示器
        if missing_config.get('create_missing_indicators', False):
            for col in df.columns:
                if df[col].isnull().any():
                    df[f'{col}_was_missing'] = df[col].isnull().astype(int)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            处理后的DataFrame
        """
        outlier_config = self.config.get('cleaning', {}).get('outliers', {})
        treatment = outlier_config.get('treatment', 'clip')
        
        if treatment == 'none':
            return df
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_process = outlier_config.get('columns_to_process', [])
        
        if not cols_to_process:
            cols_to_process = numerical_cols
        else:
            cols_to_process = [col for col in cols_to_process if col in numerical_cols]
        
        for col in cols_to_process:
            if col in self.outlier_bounds:
                lower_bound, upper_bound = self.outlier_bounds[col]
            else:
                # 计算边界
                method = outlier_config.get('detection_method', 'iqr')
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    factor = outlier_config.get('iqr_factor', 1.5)
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    threshold = outlier_config.get('zscore_threshold', 3.0)
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                else:
                    continue
            
            # 应用处理策略
            if treatment == 'clip':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif treatment == 'remove':
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df = df[~outlier_mask]
        
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        转换数据类型
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            转换后的DataFrame
        """
        dtype_config = self.config.get('cleaning', {}).get('dtype_conversion', {})
        
        # 强制转换
        force_conversions = dtype_config.get('force_conversions', {})
        for col, dtype in force_conversions.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self._log_warning(f"无法转换列 {col} 到类型 {dtype}: {e}")
        
        # 自动推断数据类型
        if dtype_config.get('auto_infer', True):
            # 尝试将object类型的数值列转换为数值类型
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # 尝试转换为数值
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # 如果转换成功的比例超过90%，则认为是数值列
                    if numeric_series.notna().sum() / len(df) > 0.9:
                        df[col] = numeric_series
                except:
                    pass
        
        # 处理类别列
        categorical_config = dtype_config.get('categorical_columns', {})
        auto_threshold = categorical_config.get('auto_detect_threshold', 50)
        force_categorical = categorical_config.get('force_categorical', [])
        
        # 自动检测类别列
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            if unique_count <= auto_threshold or col in force_categorical:
                df[col] = df[col].astype('category')
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        处理重复值
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            处理后的DataFrame
        """
        duplicate_config = self.config.get('cleaning', {}).get('duplicates', {})
        
        if not duplicate_config.get('drop_duplicates', False):
            return df
        
        subset = duplicate_config.get('subset', None)
        keep = duplicate_config.get('keep', 'first')
        
        original_len = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = original_len - len(df)
        
        if removed_count > 0:
            self._log_info(f"删除 {removed_count} 行重复数据")
        
        return df
    
    def _fit_imputers(self):
        """拟合缺失值填充器"""
        missing_config = self.config.get('cleaning', {}).get('missing_values', {})
        
        for name, df in self.data.items():
            self.imputers[name] = {}
            
            # 数值列imputer
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                strategy = missing_config.get('numerical_strategy', 'median')
                if strategy in ['mean', 'median', 'most_frequent']:
                    imputer = SimpleImputer(strategy=strategy)
                    imputer.fit(df[numerical_cols])
                    self.imputers[name]['numerical'] = imputer
    
    def _fit_encoders(self):
        """拟合编码器"""
        for name, df in self.data.items():
            self.encoders[name] = {}
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                encoder = LabelEncoder()
                # 只对非空值进行拟合
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    encoder.fit(non_null_values)
                    self.encoders[name][col] = encoder
    
    def _calculate_outlier_bounds(self):
        """计算异常值边界"""
        outlier_config = self.config.get('cleaning', {}).get('outliers', {})
        method = outlier_config.get('detection_method', 'iqr')
        
        for name, df in self.data.items():
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numerical_cols:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    factor = outlier_config.get('iqr_factor', 1.5)
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    threshold = outlier_config.get('zscore_threshold', 3.0)
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                else:
                    continue
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
    
    def get_cleaning_stats(self) -> Dict[str, Dict]:
        """获取清洗统计信息"""
        return self.cleaning_stats.copy()
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """获取清洗摘要"""
        if not self.cleaning_stats:
            return {}
        
        summary = {
            'total_datasets': len(self.cleaning_stats),
            'total_rows_processed': 0,
            'total_rows_removed': 0,
            'total_columns_removed': 0,
            'datasets_summary': {}
        }
        
        for name, stats in self.cleaning_stats.items():
            summary['total_rows_processed'] += stats['original_shape'][0]
            summary['total_rows_removed'] += stats['rows_removed']
            summary['total_columns_removed'] += stats['columns_removed']
            
            summary['datasets_summary'][name] = {
                'shape_change': f"{stats['original_shape']} -> {stats['cleaned_shape']}",
                'missing_reduction': stats['original_missing'] - stats['cleaned_missing'],
                'data_reduction_percent': (stats['rows_removed'] / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
            }
        
        return summary
