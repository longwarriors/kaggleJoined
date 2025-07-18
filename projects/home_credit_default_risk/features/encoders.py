"""
特征编码模块

实现各种类别特征编码方法，包括独热编码、标签编码、目标编码等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor


class TargetEncoder:
    """目标编码器"""
    
    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1, 
                 noise_level: float = 0.01, cv_folds: int = 5):
        """
        初始化目标编码器
        
        Args:
            smoothing: 平滑参数
            min_samples_leaf: 最小样本数
            noise_level: 噪声水平
            cv_folds: 交叉验证折数
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.cv_folds = cv_folds
        self.target_mean = None
        self.category_means = {}
        
    def fit(self, X: pd.Series, y: pd.Series):
        """
        拟合目标编码器
        
        Args:
            X: 类别特征
            y: 目标变量
        """
        self.target_mean = y.mean()
        
        # 计算每个类别的目标均值
        category_stats = pd.DataFrame({
            'category': X,
            'target': y
        }).groupby('category').agg({
            'target': ['count', 'mean']
        }).reset_index()
        
        category_stats.columns = ['category', 'count', 'mean']
        
        # 应用平滑
        smoothed_means = (
            category_stats['count'] * category_stats['mean'] + 
            self.smoothing * self.target_mean
        ) / (category_stats['count'] + self.smoothing)
        
        self.category_means = dict(zip(category_stats['category'], smoothed_means))
        
    def transform(self, X: pd.Series) -> pd.Series:
        """
        转换类别特征
        
        Args:
            X: 类别特征
            
        Returns:
            编码后的特征
        """
        encoded = X.map(self.category_means).fillna(self.target_mean)
        
        # 添加噪声（防止过拟合）
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, len(encoded))
            encoded += noise
            
        return encoded
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """
        拟合并转换（使用交叉验证避免过拟合）
        
        Args:
            X: 类别特征
            y: 目标变量
            
        Returns:
            编码后的特征
        """
        encoded = np.zeros(len(X))
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            # 在训练集上拟合
            encoder = TargetEncoder(
                smoothing=self.smoothing,
                min_samples_leaf=self.min_samples_leaf,
                noise_level=0  # 交叉验证时不添加噪声
            )
            encoder.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            # 在验证集上转换
            encoded[val_idx] = encoder.transform(X.iloc[val_idx])
        
        # 保存全局编码器（用于测试集）
        self.fit(X, y)
        
        return pd.Series(encoded, index=X.index)


class FeatureEncoders(BaseProcessor):
    """
    特征编码器
    
    管理所有类别特征编码方法
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化特征编码器
        
        Args:
            config: 编码配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.encoders = {}
        self.encoding_stats = {}
        
    def fit(self, data: Dict[str, pd.DataFrame], target: Optional[pd.Series] = None, **kwargs) -> 'FeatureEncoders':
        """
        拟合特征编码器
        
        Args:
            data: 数据字典
            target: 目标变量
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.data = data
        self.target = target
        
        # 获取主表
        main_df = data.get('application_train') or data.get('application_test')
        if main_df is None:
            raise ValueError("未找到主表数据")
        
        self._fit_encoders(main_df, target)
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        执行特征编码
        
        Args:
            df: 输入DataFrame
            **kwargs: 其他参数
            
        Returns:
            编码后的DataFrame
        """
        self._validate_fitted()
        return self.encode_categorical_features(df)
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        编码类别特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            编码后的DataFrame
        """
        categorical_config = self.config.get('categorical_features', {})
        encoding_config = categorical_config.get('encoding', {})
        
        # 获取类别列
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # 排除ID列
        categorical_cols = [col for col in categorical_cols if col != 'SK_ID_CURR']
        
        if not categorical_cols:
            return df
        
        self._log_info(f"编码 {len(categorical_cols)} 个类别特征")
        
        # 处理稀有类别
        preprocessing_config = categorical_config.get('preprocessing', {})
        if preprocessing_config.get('handle_rare_categories', True):
            df = self._handle_rare_categories(df, categorical_cols, preprocessing_config)
        
        # 根据基数选择编码方法
        high_cardinality_threshold = encoding_config.get('high_cardinality_threshold', 50)
        low_cardinality_method = encoding_config.get('low_cardinality_method', 'onehot')
        high_cardinality_method = encoding_config.get('high_cardinality_method', 'target')
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            
            if cardinality <= high_cardinality_threshold:
                # 低基数特征
                if low_cardinality_method == 'onehot':
                    df = self._apply_onehot_encoding(df, col, encoding_config)
                elif low_cardinality_method == 'label':
                    df = self._apply_label_encoding(df, col)
                elif low_cardinality_method == 'target':
                    df = self._apply_target_encoding(df, col, encoding_config)
            else:
                # 高基数特征
                if high_cardinality_method == 'target':
                    df = self._apply_target_encoding(df, col, encoding_config)
                elif high_cardinality_method == 'label':
                    df = self._apply_label_encoding(df, col)
                # 高基数特征通常不使用独热编码
        
        return df
    
    def _fit_encoders(self, df: pd.DataFrame, target: Optional[pd.Series]):
        """拟合编码器"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != 'SK_ID_CURR']
        
        categorical_config = self.config.get('categorical_features', {})
        encoding_config = categorical_config.get('encoding', {})
        
        for col in categorical_cols:
            self.encoders[col] = {}
            
            # 标签编码器
            label_encoder = LabelEncoder()
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                label_encoder.fit(non_null_values.astype(str))
                self.encoders[col]['label'] = label_encoder
            
            # 独热编码器
            onehot_config = encoding_config.get('onehot_encoding', {})
            max_categories = onehot_config.get('max_categories', 20)
            
            if df[col].nunique() <= max_categories:
                onehot_encoder = OneHotEncoder(
                    drop=onehot_config.get('drop_first', True),
                    handle_unknown=onehot_config.get('handle_unknown', 'ignore'),
                    sparse_output=False
                )
                onehot_encoder.fit(df[[col]].fillna('Missing'))
                self.encoders[col]['onehot'] = onehot_encoder
            
            # 目标编码器
            if target is not None:
                target_config = encoding_config.get('target_encoding', {})
                target_encoder = TargetEncoder(
                    smoothing=target_config.get('smoothing', 1.0),
                    min_samples_leaf=target_config.get('min_samples_leaf', 1),
                    noise_level=target_config.get('noise_level', 0.01),
                    cv_folds=target_config.get('cv_folds', 5)
                )
                
                # 只对训练数据拟合
                valid_mask = df[col].notna() & target.notna()
                if valid_mask.sum() > 0:
                    target_encoder.fit(df.loc[valid_mask, col], target.loc[valid_mask])
                    self.encoders[col]['target'] = target_encoder
    
    def _handle_rare_categories(self, df: pd.DataFrame, categorical_cols: List[str], 
                               preprocessing_config: Dict) -> pd.DataFrame:
        """处理稀有类别"""
        rare_threshold = preprocessing_config.get('rare_threshold', 0.01)
        rare_value = preprocessing_config.get('rare_value', 'RARE')
        
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < rare_threshold].index
            
            if len(rare_categories) > 0:
                df[col] = df[col].replace(rare_categories, rare_value)
                self._log_info(f"列 {col}: 将 {len(rare_categories)} 个稀有类别替换为 '{rare_value}'")
        
        return df
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, col: str, encoding_config: Dict) -> pd.DataFrame:
        """应用独热编码"""
        if col not in self.encoders or 'onehot' not in self.encoders[col]:
            return df
        
        encoder = self.encoders[col]['onehot']
        
        # 处理缺失值
        col_data = df[[col]].fillna('Missing')
        
        try:
            encoded_features = encoder.transform(col_data)
            feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0]]
            
            # 创建编码后的DataFrame
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=feature_names,
                index=df.index
            )
            
            # 删除原列并添加编码后的列
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)
            
            self._log_info(f"独热编码 {col}: 生成 {len(feature_names)} 个特征")
            
        except Exception as e:
            self._log_warning(f"独热编码失败 {col}: {e}")
        
        return df
    
    def _apply_label_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """应用标签编码"""
        if col not in self.encoders or 'label' not in self.encoders[col]:
            return df
        
        encoder = self.encoders[col]['label']
        
        try:
            # 处理未见过的类别
            col_data = df[col].astype(str).fillna('Missing')
            
            # 对于未见过的类别，使用-1
            encoded_values = []
            for value in col_data:
                try:
                    encoded_values.append(encoder.transform([value])[0])
                except ValueError:
                    encoded_values.append(-1)  # 未知类别
            
            df[f'{col}_encoded'] = encoded_values
            df = df.drop(columns=[col])
            
            self._log_info(f"标签编码 {col}: 生成 1 个特征")
            
        except Exception as e:
            self._log_warning(f"标签编码失败 {col}: {e}")
        
        return df
    
    def _apply_target_encoding(self, df: pd.DataFrame, col: str, encoding_config: Dict) -> pd.DataFrame:
        """应用目标编码"""
        if col not in self.encoders or 'target' not in self.encoders[col]:
            # 如果没有目标编码器，回退到标签编码
            return self._apply_label_encoding(df, col)
        
        encoder = self.encoders[col]['target']
        
        try:
            col_data = df[col].fillna('Missing')
            encoded_values = encoder.transform(col_data)
            
            df[f'{col}_target_encoded'] = encoded_values
            df = df.drop(columns=[col])
            
            self._log_info(f"目标编码 {col}: 生成 1 个特征")
            
        except Exception as e:
            self._log_warning(f"目标编码失败 {col}: {e}")
            # 回退到标签编码
            return self._apply_label_encoding(df, col)
        
        return df
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """获取编码统计信息"""
        return self.encoding_stats.copy()
