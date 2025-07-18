"""
超参数优化模块

实现各种超参数优化方法，包括网格搜索、随机搜索、贝叶斯优化等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from ..core.base import BaseProcessor
from ..core.utils import get_timestamp


class HyperparameterOptimizer(BaseProcessor):
    """
    超参数优化器
    
    支持多种优化方法和模型
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化超参数优化器
        
        Args:
            config: 优化配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.optimization_results = {}
        self.best_params = {}
        self.best_scores = {}
        
    def fit(self, data: Tuple[pd.DataFrame, pd.Series], **kwargs) -> 'HyperparameterOptimizer':
        """
        拟合优化器（准备数据）
        
        Args:
            data: (X, y) 元组
            **kwargs: 其他参数
            
        Returns:
            self
        """
        X, y = data
        self.X, self.y = self._prepare_data(X, y)
        self.is_fitted = True
        return self
    
    def transform(self, data: Optional[Tuple[pd.DataFrame, pd.Series]] = None, **kwargs) -> Dict:
        """
        执行超参数优化
        
        Args:
            data: 数据，如果不提供则使用fit时的数据
            **kwargs: 其他参数
            
        Returns:
            优化结果字典
        """
        self._validate_fitted()
        
        if data is not None:
            X, y = data
            self.X, self.y = self._prepare_data(X, y)
        
        return self.optimize_all_models()
    
    def optimize_all_models(self) -> Dict:
        """
        优化所有配置的模型
        
        Returns:
            优化结果字典
        """
        tuning_config = self.config.get('hyperparameter_tuning', {})
        
        if not tuning_config.get('enable', True):
            self._log_info("超参数优化已禁用")
            return {}
        
        self._log_info("开始超参数优化")
        
        models_to_tune = tuning_config.get('models_to_tune', [])
        method = tuning_config.get('method', 'optuna')
        
        for model_name in models_to_tune:
            self._log_info(f"优化模型: {model_name}")
            
            try:
                if method == 'optuna':
                    best_params, best_score = self._optimize_with_optuna(model_name, tuning_config)
                elif method == 'grid_search':
                    best_params, best_score = self._optimize_with_grid_search(model_name, tuning_config)
                elif method == 'random_search':
                    best_params, best_score = self._optimize_with_random_search(model_name, tuning_config)
                else:
                    self._log_warning(f"未知的优化方法: {method}")
                    continue
                
                self.best_params[model_name] = best_params
                self.best_scores[model_name] = best_score
                
                self._log_info(f"{model_name} 优化完成，最佳分数: {best_score:.4f}")
                
            except Exception as e:
                self._log_error(f"优化 {model_name} 失败: {e}")
        
        self.optimization_results = {
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'optimization_method': method
        }
        
        self._log_info("超参数优化完成")
        return self.optimization_results
    
    def _optimize_with_optuna(self, model_name: str, tuning_config: Dict) -> Tuple[Dict, float]:
        """使用Optuna进行优化"""
        if not HAS_OPTUNA:
            self._log_warning("Optuna未安装，跳过优化")
            return {}, 0.0
        
        def objective(trial):
            # 根据模型类型定义搜索空间
            params = self._get_optuna_search_space(model_name, trial, tuning_config)
            
            # 交叉验证评估
            score = self._evaluate_params_cv(model_name, params)
            return score
        
        # 创建study
        study = optuna.create_study(direction='maximize')
        
        # 优化
        n_trials = tuning_config.get('n_trials', 100)
        timeout = tuning_config.get('timeout', 3600)
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return study.best_params, study.best_value
    
    def _optimize_with_grid_search(self, model_name: str, tuning_config: Dict) -> Tuple[Dict, float]:
        """使用网格搜索进行优化"""
        # 获取搜索空间
        search_spaces = tuning_config.get('search_spaces', {})
        if model_name not in search_spaces:
            self._log_warning(f"未找到 {model_name} 的搜索空间")
            return {}, 0.0
        
        param_grid = search_spaces[model_name]
        
        # 创建模型
        model = self._create_model_for_search(model_name)
        if model is None:
            return {}, 0.0
        
        # 设置交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=tuning_config.get('n_jobs', -1), verbose=0
        )
        
        grid_search.fit(self.X, self.y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def _optimize_with_random_search(self, model_name: str, tuning_config: Dict) -> Tuple[Dict, float]:
        """使用随机搜索进行优化"""
        # 获取搜索空间
        search_spaces = tuning_config.get('search_spaces', {})
        if model_name not in search_spaces:
            self._log_warning(f"未找到 {model_name} 的搜索空间")
            return {}, 0.0
        
        param_distributions = search_spaces[model_name]
        
        # 创建模型
        model = self._create_model_for_search(model_name)
        if model is None:
            return {}, 0.0
        
        # 设置交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            model, param_distributions, cv=cv, scoring='roc_auc',
            n_iter=tuning_config.get('n_trials', 100),
            n_jobs=tuning_config.get('n_jobs', -1), 
            random_state=42, verbose=0
        )
        
        random_search.fit(self.X, self.y)
        
        return random_search.best_params_, random_search.best_score_
    
    def _get_optuna_search_space(self, model_name: str, trial, tuning_config: Dict) -> Dict:
        """获取Optuna搜索空间"""
        search_spaces = tuning_config.get('search_spaces', {})
        if model_name not in search_spaces:
            return {}
        
        space_config = search_spaces[model_name]
        params = {}
        
        for param_name, param_range in space_config.items():
            if isinstance(param_range, list):
                if all(isinstance(x, (int, float)) for x in param_range):
                    # 数值范围
                    if all(isinstance(x, int) for x in param_range):
                        params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range))
                else:
                    # 类别选择
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
        
        return params
    
    def _create_model_for_search(self, model_name: str):
        """为搜索创建模型实例"""
        if model_name == 'xgboost' and HAS_XGBOOST:
            return xgb.XGBClassifier(random_state=42, verbosity=0)
        elif model_name == 'lightgbm' and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        elif model_name == 'catboost' and HAS_CATBOOST:
            return cb.CatBoostClassifier(random_seed=42, verbose=False)
        else:
            self._log_warning(f"无法创建模型 {model_name}")
            return None
    
    def _evaluate_params_cv(self, model_name: str, params: Dict) -> float:
        """使用交叉验证评估参数"""
        try:
            # 创建模型
            if model_name == 'xgboost' and HAS_XGBOOST:
                model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
            elif model_name == 'lightgbm' and HAS_LIGHTGBM:
                model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
            elif model_name == 'catboost' and HAS_CATBOOST:
                model = cb.CatBoostClassifier(**params, random_seed=42, verbose=False)
            else:
                return 0.0
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            self._log_warning(f"评估参数失败: {e}")
            return 0.0
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """准备数据"""
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
        
        return X_clean, y_clean
    
    def get_best_params(self, model_name: str) -> Optional[Dict]:
        """获取指定模型的最佳参数"""
        return self.best_params.get(model_name)
    
    def get_best_score(self, model_name: str) -> Optional[float]:
        """获取指定模型的最佳分数"""
        return self.best_scores.get(model_name)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.optimization_results:
            return {}
        
        summary = {
            'optimized_models': list(self.best_params.keys()),
            'best_model': None,
            'best_overall_score': 0.0,
            'model_scores': self.best_scores.copy()
        }
        
        # 找到最佳模型
        if self.best_scores:
            best_model = max(self.best_scores.items(), key=lambda x: x[1])
            summary['best_model'] = best_model[0]
            summary['best_overall_score'] = best_model[1]
        
        return summary
    
    def update_model_config(self, model_config: Dict) -> Dict:
        """
        使用优化结果更新模型配置
        
        Args:
            model_config: 原始模型配置
            
        Returns:
            更新后的模型配置
        """
        updated_config = model_config.copy()
        
        for model_name, best_params in self.best_params.items():
            if model_name in updated_config.get('advanced_models', {}):
                # 更新参数
                current_params = updated_config['advanced_models'][model_name].get('params', {})
                current_params.update(best_params)
                updated_config['advanced_models'][model_name]['params'] = current_params
                
                self._log_info(f"已更新 {model_name} 的配置参数")
        
        return updated_config
