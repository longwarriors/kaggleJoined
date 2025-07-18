"""
基线模型模块

实现简单的基线模型，用于建立性能基准。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseModel
from .evaluator import ModelEvaluator


class BaselineModel(BaseModel):
    """
    基线模型类
    
    实现简单模型的训练和评估，建立性能基准
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化基线模型
        
        Args:
            config: 模型配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.baseline_models = {}
        self.baseline_results = {}
        self.best_baseline_model = None
        self.best_baseline_score = 0.0
        
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练基线模型
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 其他参数
            
        Returns:
            训练结果字典
        """
        self._log_info("开始训练基线模型")
        
        # 准备数据
        X_clean, y_clean = self._prepare_data(X, y)
        
        # 获取基线模型配置
        baseline_config = self.config.get('baseline_models', {})
        
        # 训练各个基线模型
        for model_name, model_config in baseline_config.items():
            if model_config.get('enable', True):
                self._log_info(f"训练基线模型: {model_name}")
                
                try:
                    # 创建模型
                    model = self._create_baseline_model(model_name, model_config)
                    
                    # 训练模型
                    model.fit(X_clean, y_clean)
                    
                    # 评估模型
                    scores = self._evaluate_baseline_model(model, X_clean, y_clean, model_name)
                    
                    # 保存模型和结果
                    self.baseline_models[model_name] = model
                    self.baseline_results[model_name] = scores
                    
                    # 更新最佳模型
                    primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'roc_auc')
                    if scores.get(primary_metric, 0) > self.best_baseline_score:
                        self.best_baseline_score = scores[primary_metric]
                        self.best_baseline_model = model_name
                    
                    self._log_info(f"{model_name} 训练完成，{primary_metric}: {scores.get(primary_metric, 0):.4f}")
                    
                except Exception as e:
                    self._log_error(f"训练 {model_name} 失败: {e}")
        
        # 记录训练历史
        self.training_history = {
            'baseline_results': self.baseline_results,
            'best_model': self.best_baseline_model,
            'best_score': self.best_baseline_score
        }
        
        self.is_trained = True
        self._log_info(f"基线模型训练完成，最佳模型: {self.best_baseline_model}")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        使用最佳基线模型进行预测
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        self._validate_trained()
        
        if self.best_baseline_model is None:
            raise ValueError("没有可用的基线模型")
        
        model = self.baseline_models[self.best_baseline_model]
        X_clean = self._prepare_prediction_data(X)
        
        return model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        使用最佳基线模型进行概率预测
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
            
        Returns:
            预测概率
        """
        self._validate_trained()
        
        if self.best_baseline_model is None:
            raise ValueError("没有可用的基线模型")
        
        model = self.baseline_models[self.best_baseline_model]
        X_clean = self._prepare_prediction_data(X)
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_clean)[:, 1]  # 返回正类概率
        else:
            # 如果模型不支持概率预测，返回决策函数结果
            if hasattr(model, 'decision_function'):
                return model.decision_function(X_clean)
            else:
                raise ValueError(f"模型 {self.best_baseline_model} 不支持概率预测")
    
    def _create_baseline_model(self, model_name: str, model_config: Dict):
        """
        创建基线模型
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
            
        Returns:
            模型实例
        """
        params = model_config.get('params', {})
        
        if model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'decision_tree':
            return DecisionTreeClassifier(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"未知的基线模型: {model_name}")
    
    def _evaluate_baseline_model(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """
        评估基线模型
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            model_name: 模型名称
            
        Returns:
            评估结果字典
        """
        # 获取评估配置
        eval_config = self.config.get('evaluation', {})
        cv_config = eval_config.get('cross_validation', {})
        
        # 设置交叉验证
        cv = StratifiedKFold(
            n_splits=cv_config.get('n_splits', 5),
            shuffle=cv_config.get('shuffle', True),
            random_state=cv_config.get('random_state', 42)
        )
        
        # 计算交叉验证分数
        cv_scores = {}
        
        # ROC AUC
        try:
            if hasattr(model, 'predict_proba'):
                auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                cv_scores['roc_auc'] = {
                    'mean': auc_scores.mean(),
                    'std': auc_scores.std(),
                    'scores': auc_scores.tolist()
                }
            else:
                # 对于不支持概率预测的模型，使用决策函数
                auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                cv_scores['roc_auc'] = {
                    'mean': auc_scores.mean(),
                    'std': auc_scores.std(),
                    'scores': auc_scores.tolist()
                }
        except Exception as e:
            self._log_warning(f"计算 {model_name} 的AUC失败: {e}")
            cv_scores['roc_auc'] = {'mean': 0.0, 'std': 0.0, 'scores': []}
        
        # 准确率
        try:
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_scores['accuracy'] = {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores.tolist()
            }
        except Exception as e:
            self._log_warning(f"计算 {model_name} 的准确率失败: {e}")
            cv_scores['accuracy'] = {'mean': 0.0, 'std': 0.0, 'scores': []}
        
        # F1分数
        try:
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_scores['f1'] = {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores.tolist()
            }
        except Exception as e:
            self._log_warning(f"计算 {model_name} 的F1分数失败: {e}")
            cv_scores['f1'] = {'mean': 0.0, 'std': 0.0, 'scores': []}
        
        # 添加主要指标的简化访问
        primary_metric = eval_config.get('primary_metric', 'roc_auc')
        if primary_metric in cv_scores:
            cv_scores[primary_metric] = cv_scores[primary_metric]['mean']
        
        return cv_scores
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            清理后的特征数据和目标变量
        """
        # 移除ID列
        feature_cols = [col for col in X.columns if col != 'SK_ID_CURR']
        X_clean = X[feature_cols].copy()
        
        # 处理无穷大值和NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 填充缺失值
        for col in X_clean.columns:
            if X_clean[col].dtype in ['object', 'category']:
                X_clean[col] = X_clean[col].fillna('Unknown')
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # 确保目标变量对齐
        y_clean = y.loc[X_clean.index]
        
        # 保存特征名称
        self.feature_names = X_clean.columns.tolist()
        
        return X_clean, y_clean
    
    def _prepare_prediction_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        准备预测数据
        
        Args:
            X: 特征数据
            
        Returns:
            清理后的特征数据
        """
        # 选择训练时使用的特征
        if hasattr(self, 'feature_names'):
            available_features = [col for col in self.feature_names if col in X.columns]
            X_clean = X[available_features].copy()
        else:
            feature_cols = [col for col in X.columns if col != 'SK_ID_CURR']
            X_clean = X[feature_cols].copy()
        
        # 处理无穷大值和NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 填充缺失值
        for col in X_clean.columns:
            if X_clean[col].dtype in ['object', 'category']:
                X_clean[col] = X_clean[col].fillna('Unknown')
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean
    
    def get_baseline_results(self) -> Dict[str, Any]:
        """获取基线模型结果"""
        return self.baseline_results.copy()
    
    def get_best_baseline_model(self) -> Tuple[str, Any]:
        """获取最佳基线模型"""
        if self.best_baseline_model is None:
            return None, None
        
        return self.best_baseline_model, self.baseline_models[self.best_baseline_model]
    
    def establish_baseline(self) -> float:
        """
        建立基线性能
        
        Returns:
            基线性能分数
        """
        if not self.is_trained:
            raise ValueError("请先训练基线模型")
        
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'roc_auc')
        baseline_score = self.best_baseline_score
        
        self._log_info(f"基线性能已建立: {primary_metric} = {baseline_score:.4f}")
        self._log_info(f"后续模型需要超过此基线才被认为是改进")
        
        return baseline_score
    
    def compare_with_baseline(self, model_score: float, model_name: str = "新模型") -> Dict[str, Any]:
        """
        与基线模型比较
        
        Args:
            model_score: 新模型的分数
            model_name: 新模型名称
            
        Returns:
            比较结果
        """
        if not self.is_trained:
            raise ValueError("请先训练基线模型")
        
        baseline_score = self.best_baseline_score
        improvement = model_score - baseline_score
        improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
        
        comparison = {
            'baseline_score': baseline_score,
            'new_model_score': model_score,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'is_better': model_score > baseline_score,
            'baseline_model': self.best_baseline_model,
            'new_model': model_name
        }
        
        if comparison['is_better']:
            self._log_info(f"{model_name} 超过基线 {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            self._log_info(f"{model_name} 未超过基线 {improvement:.4f} ({improvement_pct:.2f}%)")
        
        return comparison
