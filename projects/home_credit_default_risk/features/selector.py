"""
特征选择模块

实现各种特征选择方法，包括过滤法、包裹法、嵌入法等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseProcessor


class FeatureSelector(BaseProcessor):
    """
    特征选择器
    
    实现多种特征选择方法
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化特征选择器
        
        Args:
            config: 特征选择配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.selectors = {}
        self.selected_features = []
        self.feature_scores = {}
        self.selection_stats = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'FeatureSelector':
        """
        拟合特征选择器
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.X = X
        self.y = y
        
        # 排除ID列
        feature_cols = [col for col in X.columns if col != 'SK_ID_CURR']
        self.feature_cols = feature_cols
        
        if y is not None:
            self._fit_selectors(X[feature_cols], y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        执行特征选择
        
        Args:
            X: 输入特征数据
            **kwargs: 其他参数
            
        Returns:
            选择后的特征数据
        """
        self._validate_fitted()
        return self.select_features(X)
    
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        选择特征
        
        Args:
            X: 输入特征数据
            
        Returns:
            选择后的特征数据
        """
        selection_config = self.config.get('feature_selection', {})
        
        if not selection_config.get('enable', True):
            return X
        
        self._log_info("开始特征选择")
        
        # 保留ID列
        result_df = X[['SK_ID_CURR']].copy() if 'SK_ID_CURR' in X.columns else pd.DataFrame(index=X.index)
        
        # 获取特征列
        feature_cols = [col for col in X.columns if col != 'SK_ID_CURR']
        
        if not feature_cols:
            return result_df
        
        # 应用各种选择方法
        selected_features_sets = []
        
        # 1. 过滤法
        filter_features = self._apply_filter_methods(X[feature_cols])
        if filter_features:
            selected_features_sets.append(set(filter_features))
        
        # 2. 包裹法
        wrapper_features = self._apply_wrapper_methods(X[feature_cols])
        if wrapper_features:
            selected_features_sets.append(set(wrapper_features))
        
        # 3. 嵌入法
        embedded_features = self._apply_embedded_methods(X[feature_cols])
        if embedded_features:
            selected_features_sets.append(set(embedded_features))
        
        # 合并选择结果
        if selected_features_sets:
            # 取交集（保守策略）或并集（激进策略）
            final_features = set.intersection(*selected_features_sets) if len(selected_features_sets) > 1 else selected_features_sets[0]
            
            # 如果交集太小，使用并集
            if len(final_features) < 50 and len(selected_features_sets) > 1:
                final_features = set.union(*selected_features_sets)
            
            final_features = list(final_features)
        else:
            # 如果没有选择方法，保留所有特征
            final_features = feature_cols
        
        # 确保选择的特征存在于数据中
        final_features = [col for col in final_features if col in X.columns]
        
        if final_features:
            result_df = pd.concat([result_df, X[final_features]], axis=1)
        
        self.selected_features = final_features
        
        # 记录选择统计
        self.selection_stats = {
            'original_features': len(feature_cols),
            'selected_features': len(final_features),
            'reduction_ratio': 1 - len(final_features) / len(feature_cols) if feature_cols else 0,
            'selected_feature_names': final_features
        }
        
        self._log_info(f"特征选择完成: {len(feature_cols)} -> {len(final_features)} 特征")
        
        return result_df
    
    def _fit_selectors(self, X: pd.DataFrame, y: pd.Series):
        """拟合选择器"""
        selection_config = self.config.get('feature_selection', {})
        
        # 处理数据
        X_clean = self._prepare_data_for_selection(X)
        y_clean = y.loc[X_clean.index]
        
        # 拟合过滤法选择器
        self._fit_filter_selectors(X_clean, y_clean, selection_config.get('filter_methods', {}))
        
        # 拟合包裹法选择器
        self._fit_wrapper_selectors(X_clean, y_clean, selection_config.get('wrapper_methods', {}))
        
        # 拟合嵌入法选择器
        self._fit_embedded_selectors(X_clean, y_clean, selection_config.get('embedded_methods', {}))
    
    def _prepare_data_for_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """为特征选择准备数据"""
        # 处理无穷大值和NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # 删除全为NaN的列
        X_clean = X_clean.dropna(axis=1, how='all')
        
        # 填充剩余的NaN
        for col in X_clean.columns:
            if X_clean[col].dtype in ['object', 'category']:
                X_clean[col] = X_clean[col].fillna('Unknown')
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean
    
    def _fit_filter_selectors(self, X: pd.DataFrame, y: pd.Series, filter_config: Dict):
        """拟合过滤法选择器"""
        # 方差阈值
        variance_config = filter_config.get('variance_threshold', {})
        if variance_config.get('enable', True):
            threshold = variance_config.get('threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X.select_dtypes(include=[np.number]))
            self.selectors['variance'] = selector
        
        # 互信息
        mutual_info_config = filter_config.get('mutual_info', {})
        if mutual_info_config.get('enable', True) and y is not None:
            k_best = mutual_info_config.get('k_best', 500)
            k_best = min(k_best, X.shape[1])  # 确保不超过特征数量
            
            selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            selector.fit(X.select_dtypes(include=[np.number]), y)
            self.selectors['mutual_info'] = selector
        
        # 卡方检验（仅适用于非负特征）
        chi2_config = filter_config.get('chi2', {})
        if chi2_config.get('enable', False) and y is not None:
            # 选择非负特征
            non_negative_cols = []
            for col in X.select_dtypes(include=[np.number]).columns:
                if (X[col] >= 0).all():
                    non_negative_cols.append(col)
            
            if non_negative_cols:
                k_best = chi2_config.get('k_best', 100)
                k_best = min(k_best, len(non_negative_cols))
                
                selector = SelectKBest(score_func=chi2, k=k_best)
                selector.fit(X[non_negative_cols], y)
                self.selectors['chi2'] = selector
                self.selectors['chi2_columns'] = non_negative_cols
    
    def _fit_wrapper_selectors(self, X: pd.DataFrame, y: pd.Series, wrapper_config: Dict):
        """拟合包裹法选择器"""
        # 递归特征消除
        rfe_config = wrapper_config.get('rfe', {})
        if rfe_config.get('enable', False) and y is not None:
            n_features = rfe_config.get('n_features_to_select', 100)
            n_features = min(n_features, X.shape[1])
            step = rfe_config.get('step', 10)
            
            # 使用逻辑回归作为基础估计器
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
            
            try:
                selector.fit(X.select_dtypes(include=[np.number]), y)
                self.selectors['rfe'] = selector
            except Exception as e:
                self._log_warning(f"RFE拟合失败: {e}")
    
    def _fit_embedded_selectors(self, X: pd.DataFrame, y: pd.Series, embedded_config: Dict):
        """拟合嵌入法选择器"""
        # L1正则化
        l1_config = embedded_config.get('l1_regularization', {})
        if l1_config.get('enable', True) and y is not None:
            C = l1_config.get('C', 0.01)
            
            estimator = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42, max_iter=1000)
            selector = SelectFromModel(estimator=estimator)
            
            try:
                selector.fit(X.select_dtypes(include=[np.number]), y)
                self.selectors['l1'] = selector
            except Exception as e:
                self._log_warning(f"L1正则化选择器拟合失败: {e}")
        
        # 树模型特征重要性
        tree_config = embedded_config.get('tree_importance', {})
        if tree_config.get('enable', True) and y is not None:
            threshold = tree_config.get('threshold', 0.001)
            
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(estimator=estimator, threshold=threshold)
            
            try:
                selector.fit(X.select_dtypes(include=[np.number]), y)
                self.selectors['tree'] = selector
            except Exception as e:
                self._log_warning(f"树模型选择器拟合失败: {e}")
    
    def _apply_filter_methods(self, X: pd.DataFrame) -> List[str]:
        """应用过滤法"""
        selected_features = []
        
        # 方差阈值
        if 'variance' in self.selectors:
            selector = self.selectors['variance']
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                X_num = X[numerical_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                mask = selector.get_support()
                variance_features = [numerical_cols[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(variance_features)
        
        # 互信息
        if 'mutual_info' in self.selectors and self.y is not None:
            selector = self.selectors['mutual_info']
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                mask = selector.get_support()
                mi_features = [numerical_cols[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(mi_features)
        
        # 卡方检验
        if 'chi2' in self.selectors and self.y is not None:
            selector = self.selectors['chi2']
            chi2_columns = self.selectors.get('chi2_columns', [])
            
            if chi2_columns:
                mask = selector.get_support()
                chi2_features = [chi2_columns[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(chi2_features)
        
        return list(set(selected_features))
    
    def _apply_wrapper_methods(self, X: pd.DataFrame) -> List[str]:
        """应用包裹法"""
        selected_features = []
        
        # RFE
        if 'rfe' in self.selectors:
            selector = self.selectors['rfe']
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                mask = selector.get_support()
                rfe_features = [numerical_cols[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(rfe_features)
        
        return list(set(selected_features))
    
    def _apply_embedded_methods(self, X: pd.DataFrame) -> List[str]:
        """应用嵌入法"""
        selected_features = []
        
        # L1正则化
        if 'l1' in self.selectors:
            selector = self.selectors['l1']
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                mask = selector.get_support()
                l1_features = [numerical_cols[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(l1_features)
        
        # 树模型重要性
        if 'tree' in self.selectors:
            selector = self.selectors['tree']
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                mask = selector.get_support()
                tree_features = [numerical_cols[i] for i in range(len(mask)) if mask[i]]
                selected_features.extend(tree_features)
        
        return list(set(selected_features))
    
    def get_selected_features(self) -> List[str]:
        """获取选择的特征列表"""
        return self.selected_features.copy()
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        return self.selection_stats.copy()
    
    def get_feature_scores(self) -> Dict[str, Any]:
        """获取特征评分"""
        scores = {}
        
        # 互信息评分
        if 'mutual_info' in self.selectors:
            selector = self.selectors['mutual_info']
            numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
            if hasattr(selector, 'scores_'):
                scores['mutual_info'] = dict(zip(numerical_cols, selector.scores_))
        
        # 树模型重要性
        if 'tree' in self.selectors:
            selector = self.selectors['tree']
            if hasattr(selector.estimator_, 'feature_importances_'):
                numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
                scores['tree_importance'] = dict(zip(numerical_cols, selector.estimator_.feature_importances_))
        
        return scores
