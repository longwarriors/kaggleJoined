"""
特征构建模块

实现全面的特征构建功能，包括数值特征处理、类别特征编码、
时间特征、交叉特征、聚合特征和领域知识特征。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor
from .encoders import FeatureEncoders
from .aggregators import FeatureAggregators
from .transformers import FeatureTransformers


class FeatureBuilder(BaseProcessor):
    """
    特征构建器
    
    统一管理所有特征构建功能
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化特征构建器
        
        Args:
            config: 特征工程配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        
        # 初始化子模块
        self.encoders = FeatureEncoders(config, logger)
        self.aggregators = FeatureAggregators(config, logger)
        self.transformers = FeatureTransformers(config, logger)
        
        # 存储拟合的处理器
        self.fitted_processors = {}
        self.feature_stats = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], target: Optional[pd.Series] = None, **kwargs) -> 'FeatureBuilder':
        """
        拟合特征构建器
        
        Args:
            data: 数据字典
            target: 目标变量（用于目标编码等）
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        self.target = target
        
        # 拟合子模块
        self.encoders.fit(data, target)
        self.aggregators.fit(data)
        self.transformers.fit(data)
        
        # 拟合数值特征处理器
        self._fit_numerical_processors()
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """
        执行特征构建
        
        Args:
            data: 数据字典，如果不提供则使用fit时的数据
            **kwargs: 其他参数
            
        Returns:
            构建后的特征DataFrame
        """
        self._validate_fitted()
        
        if data is not None:
            data_to_transform = data
        else:
            data_to_transform = self.data
        
        return self.build_all_features(data_to_transform)
    
    def build_all_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        构建所有特征
        
        Args:
            data: 数据字典
            
        Returns:
            完整的特征DataFrame
        """
        self._log_info("开始构建所有特征")
        
        # 获取主表数据
        main_df = data.get('application_train') or data.get('application_test')
        if main_df is None:
            raise ValueError("未找到主表数据")
        
        # 复制主表作为基础
        features_df = main_df.copy()
        original_features = features_df.shape[1]
        
        # 1. 聚合特征（从辅助表）
        self._log_info("构建聚合特征")
        aggregated_features = self.aggregators.transform(data)
        features_df = features_df.merge(aggregated_features, on='SK_ID_CURR', how='left')
        
        # 2. 数值特征处理
        self._log_info("处理数值特征")
        features_df = self.build_numerical_features(features_df)
        
        # 3. 类别特征编码
        self._log_info("编码类别特征")
        features_df = self.encoders.transform(features_df)
        
        # 4. 时间特征
        self._log_info("构建时间特征")
        features_df = self.build_temporal_features(features_df)
        
        # 5. 领域知识特征
        self._log_info("构建领域知识特征")
        features_df = self.build_domain_features(features_df)
        
        # 6. 交叉特征
        self._log_info("构建交叉特征")
        features_df = self.build_interaction_features(features_df)
        
        # 7. 特征变换
        self._log_info("应用特征变换")
        features_df = self.transformers.transform(features_df)
        
        # 记录特征统计信息
        self.feature_stats = {
            'original_features': original_features,
            'final_features': features_df.shape[1],
            'added_features': features_df.shape[1] - original_features,
            'feature_names': features_df.columns.tolist()
        }
        
        self._log_info(f"特征构建完成: {original_features} -> {features_df.shape[1]} 特征")
        
        return features_df
    
    def build_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建数值特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        numerical_config = self.config.get('numerical_features', {})
        
        # 获取数值列
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除ID列和目标列
        numerical_cols = [col for col in numerical_cols if col not in ['SK_ID_CURR', 'TARGET']]
        
        if not numerical_cols:
            return df
        
        # 标准化/归一化
        scaling_config = numerical_config.get('scaling', {})
        if scaling_config.get('method') and not scaling_config.get('scale_all', False):
            cols_to_scale = scaling_config.get('columns_to_scale', [])
            if cols_to_scale:
                cols_to_scale = [col for col in cols_to_scale if col in numerical_cols]
                if cols_to_scale and 'scaler' in self.fitted_processors:
                    scaled_features = self.fitted_processors['scaler'].transform(df[cols_to_scale])
                    scaled_df = pd.DataFrame(
                        scaled_features, 
                        columns=[f'{col}_scaled' for col in cols_to_scale],
                        index=df.index
                    )
                    df = pd.concat([df, scaled_df], axis=1)
        
        # 分箱
        binning_config = numerical_config.get('binning', {})
        if binning_config.get('enable', False):
            cols_to_bin = binning_config.get('columns_to_bin', [])
            cols_to_bin = [col for col in cols_to_bin if col in numerical_cols]
            
            if cols_to_bin and 'binner' in self.fitted_processors:
                binned_features = self.fitted_processors['binner'].transform(df[cols_to_bin])
                binned_df = pd.DataFrame(
                    binned_features,
                    columns=[f'{col}_binned' for col in cols_to_bin],
                    index=df.index
                )
                df = pd.concat([df, binned_df], axis=1)
        
        # 数学变换
        transformations_config = numerical_config.get('transformations', {})
        
        # 对数变换
        log_config = transformations_config.get('log_transform', {})
        if log_config.get('enable', False):
            log_cols = log_config.get('columns', [])
            log_cols = [col for col in log_cols if col in numerical_cols]
            use_log1p = log_config.get('use_log1p', True)
            
            for col in log_cols:
                if use_log1p:
                    df[f'{col}_log1p'] = np.log1p(df[col].clip(lower=0))
                else:
                    df[f'{col}_log'] = np.log(df[col].clip(lower=1e-8))
        
        # 平方根变换
        sqrt_config = transformations_config.get('sqrt_transform', {})
        if sqrt_config.get('enable', False):
            sqrt_cols = sqrt_config.get('columns', [])
            sqrt_cols = [col for col in sqrt_cols if col in numerical_cols]
            
            for col in sqrt_cols:
                df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
        
        return df
    
    def build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建时间特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加时间特征后的DataFrame
        """
        temporal_config = self.config.get('temporal_features', {})
        date_config = temporal_config.get('date_columns', {})
        
        date_cols = date_config.get('columns', [])
        date_cols = [col for col in date_cols if col in df.columns]
        
        for col in date_cols:
            # 转换为年数（DAYS_*列通常是负值，表示距今天数）
            if col == 'DAYS_BIRTH' and date_config.get('extract_age', True):
                df['AGE_YEARS'] = (-df[col] / 365.25).clip(lower=0, upper=100)
                df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], 
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder', 'Old'])
                df['AGE_GROUP'] = df['AGE_GROUP'].astype(str)
            
            elif col == 'DAYS_EMPLOYED' and date_config.get('extract_employment_duration', True):
                # 处理异常值（365243表示失业）
                employment_years = (-df[col] / 365.25).replace(365243/365.25, np.nan)
                df['EMPLOYMENT_YEARS'] = employment_years.clip(lower=0, upper=50)
                df['IS_EMPLOYED'] = (df[col] != 365243).astype(int)
            
            elif col in ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']:
                df[f'{col}_YEARS'] = (-df[col] / 365.25).clip(lower=0)
        
        # 创建时间差特征
        if date_config.get('create_time_differences', True):
            if 'DAYS_BIRTH' in df.columns and 'DAYS_EMPLOYED' in df.columns:
                # 开始工作时的年龄
                df['AGE_WHEN_STARTED_WORK'] = df['AGE_YEARS'] - df['EMPLOYMENT_YEARS']
                df['AGE_WHEN_STARTED_WORK'] = df['AGE_WHEN_STARTED_WORK'].clip(lower=0)
        
        return df
    
    def build_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建领域知识特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加领域特征后的DataFrame
        """
        domain_config = self.config.get('domain_features', {})
        
        if not domain_config.get('enable', True):
            return df
        
        # 信贷相关特征
        credit_config = domain_config.get('credit_features', {})
        
        if credit_config.get('debt_to_income', True):
            if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
                df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-8)
        
        if credit_config.get('annuity_to_income', True):
            if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
                df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-8)
        
        if credit_config.get('credit_to_annuity', True):
            if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
                df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1e-8)
        
        # 收入稳定性特征
        demographic_config = domain_config.get('demographic_features', {})
        
        if demographic_config.get('income_stability', True):
            if 'NAME_INCOME_TYPE' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
                # 基于收入类型的稳定性评分
                income_stability_map = {
                    'Working': 1.0,
                    'Commercial associate': 0.8,
                    'Pensioner': 0.9,
                    'State servant': 0.95,
                    'Student': 0.3,
                    'Businessman': 0.6,
                    'Maternity leave': 0.4,
                    'Unemployed': 0.1
                }
                df['INCOME_STABILITY_SCORE'] = df['NAME_INCOME_TYPE'].map(income_stability_map).fillna(0.5)
                df['ADJUSTED_INCOME'] = df['AMT_INCOME_TOTAL'] * df['INCOME_STABILITY_SCORE']
        
        # EXT_SOURCE组合特征
        ext_config = domain_config.get('ext_source_features', {})
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        available_ext = [col for col in ext_sources if col in df.columns]
        
        if len(available_ext) >= 2:
            if ext_config.get('mean', True):
                df['EXT_SOURCE_MEAN'] = df[available_ext].mean(axis=1)
            
            if ext_config.get('std', True):
                df['EXT_SOURCE_STD'] = df[available_ext].std(axis=1)
            
            if ext_config.get('max', True):
                df['EXT_SOURCE_MAX'] = df[available_ext].max(axis=1)
            
            if ext_config.get('min', True):
                df['EXT_SOURCE_MIN'] = df[available_ext].min(axis=1)
            
            if ext_config.get('product', True) and len(available_ext) == 3:
                df['EXT_SOURCE_PRODUCT'] = df[available_ext].prod(axis=1)
            
            # 两两组合
            if ext_config.get('pairwise_combinations', True):
                for i in range(len(available_ext)):
                    for j in range(i+1, len(available_ext)):
                        col1, col2 = available_ext[i], available_ext[j]
                        df[f'{col1}_X_{col2}'] = df[col1] * df[col2]
                        df[f'{col1}_DIV_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        return df
    
    def build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建交叉特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加交叉特征后的DataFrame
        """
        interaction_config = self.config.get('interaction_features', {})
        
        if not interaction_config.get('enable', True):
            return df
        
        important_features = interaction_config.get('important_features', [])
        available_features = [col for col in important_features if col in df.columns]
        
        if len(available_features) < 2:
            return df
        
        interaction_types = interaction_config.get('interaction_types', {})
        max_interactions = interaction_config.get('max_interactions', 50)
        
        new_features = {}
        interaction_count = 0
        
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                if interaction_count >= max_interactions:
                    break
                
                col1, col2 = available_features[i], available_features[j]
                
                # 乘积交叉
                if interaction_types.get('multiply', True):
                    new_features[f'{col1}_X_{col2}'] = df[col1] * df[col2]
                    interaction_count += 1
                
                # 除法交叉
                if interaction_types.get('divide', True) and interaction_count < max_interactions:
                    new_features[f'{col1}_DIV_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    interaction_count += 1
                
                # 差值交叉
                if interaction_types.get('subtract', True) and interaction_count < max_interactions:
                    new_features[f'{col1}_MINUS_{col2}'] = df[col1] - df[col2]
                    interaction_count += 1
                
                # 加法交叉
                if interaction_types.get('add', False) and interaction_count < max_interactions:
                    new_features[f'{col1}_PLUS_{col2}'] = df[col1] + df[col2]
                    interaction_count += 1
            
            if interaction_count >= max_interactions:
                break
        
        # 添加新特征
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            # 处理无穷大值和NaN
            new_features_df = new_features_df.replace([np.inf, -np.inf], np.nan)
            df = pd.concat([df, new_features_df], axis=1)
        
        return df
    
    def _fit_numerical_processors(self):
        """拟合数值特征处理器"""
        numerical_config = self.config.get('numerical_features', {})
        
        # 获取主表
        main_df = self.data.get('application_train') or self.data.get('application_test')
        if main_df is None:
            return
        
        numerical_cols = main_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['SK_ID_CURR', 'TARGET']]
        
        # 拟合缩放器
        scaling_config = numerical_config.get('scaling', {})
        method = scaling_config.get('method')
        cols_to_scale = scaling_config.get('columns_to_scale', [])
        cols_to_scale = [col for col in cols_to_scale if col in numerical_cols]
        
        if method and cols_to_scale:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'quantile':
                scaler = QuantileTransformer()
            else:
                scaler = StandardScaler()
            
            scaler.fit(main_df[cols_to_scale])
            self.fitted_processors['scaler'] = scaler
        
        # 拟合分箱器
        binning_config = numerical_config.get('binning', {})
        if binning_config.get('enable', False):
            cols_to_bin = binning_config.get('columns_to_bin', [])
            cols_to_bin = [col for col in cols_to_bin if col in numerical_cols]
            
            if cols_to_bin:
                strategy = binning_config.get('strategy', 'quantile')
                n_bins = binning_config.get('n_bins', 5)
                encode = binning_config.get('encode', 'ordinal')
                
                binner = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
                binner.fit(main_df[cols_to_bin])
                self.fitted_processors['binner'] = binner
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """获取特征统计信息"""
        return self.feature_stats.copy()
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_stats.get('feature_names', [])
