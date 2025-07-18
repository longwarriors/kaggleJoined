"""
模型训练模块

实现高级模型的训练，包括XGBoost、LightGBM、CatBoost等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from ..core.base import BaseModel
from ..core.utils import save_object, get_timestamp


class ModelTrainer(BaseModel):
    """
    高级模型训练器
    
    支持XGBoost、LightGBM、CatBoost等模型的训练
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化模型训练器
        
        Args:
            config: 模型配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.trained_models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.best_model_score = 0.0
        
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        训练所有配置的高级模型
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 其他参数
            
        Returns:
            训练结果字典
        """
        self._log_info("开始训练高级模型")
        
        # 准备数据
        X_clean, y_clean = self._prepare_data(X, y)
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y_clean, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_clean
        )
        
        # 获取高级模型配置
        advanced_config = self.config.get('advanced_models', {})
        
        # 训练各个模型
        for model_name, model_config in advanced_config.items():
            if model_config.get('enable', True):
                self._log_info(f"训练模型: {model_name}")
                
                try:
                    # 训练模型
                    model, score = self._train_single_model(
                        model_name, model_config, 
                        X_train, y_train, X_val, y_val
                    )
                    
                    if model is not None:
                        self.trained_models[model_name] = model
                        self.model_scores[model_name] = score
                        
                        # 更新最佳模型
                        if score > self.best_model_score:
                            self.best_model_score = score
                            self.best_model_name = model_name
                        
                        self._log_info(f"{model_name} 训练完成，验证AUC: {score:.4f}")
                    
                except Exception as e:
                    self._log_error(f"训练 {model_name} 失败: {e}")
        
        # 记录训练历史
        self.training_history = {
            'model_scores': self.model_scores,
            'best_model': self.best_model_name,
            'best_score': self.best_model_score,
            'training_data_shape': X_train.shape,
            'validation_data_shape': X_val.shape
        }
        
        self.is_trained = True
        self._log_info(f"高级模型训练完成，最佳模型: {self.best_model_name} (AUC: {self.best_model_score:.4f})")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        使用指定模型进行预测
        
        Args:
            X: 特征数据
            model_name: 模型名称，如果不指定则使用最佳模型
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        self._validate_trained()
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 未训练")
        
        model = self.trained_models[model_name]
        X_clean = self._prepare_prediction_data(X)
        
        return self._predict_with_model(model, X_clean, model_name)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        使用指定模型进行概率预测
        
        Args:
            X: 特征数据
            model_name: 模型名称，如果不指定则使用最佳模型
            **kwargs: 其他参数
            
        Returns:
            预测概率
        """
        self._validate_trained()
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 未训练")
        
        model = self.trained_models[model_name]
        X_clean = self._prepare_prediction_data(X)
        
        return self._predict_proba_with_model(model, X_clean, model_name)
    
    def _train_single_model(self, model_name: str, model_config: Dict,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, float]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练好的模型和验证分数
        """
        params = model_config.get('params', {})
        early_stopping_config = model_config.get('early_stopping', {})
        
        if model_name == 'xgboost':
            return self._train_xgboost(params, early_stopping_config, X_train, y_train, X_val, y_val)
        elif model_name == 'lightgbm':
            return self._train_lightgbm(params, early_stopping_config, X_train, y_train, X_val, y_val)
        elif model_name == 'catboost':
            return self._train_catboost(params, early_stopping_config, X_train, y_train, X_val, y_val)
        else:
            self._log_warning(f"未知的模型类型: {model_name}")
            return None, 0.0
    
    def _train_xgboost(self, params: Dict, early_stopping_config: Dict,
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, float]:
        """训练XGBoost模型"""
        if not HAS_XGBOOST:
            self._log_warning("XGBoost未安装，跳过训练")
            return None, 0.0
        
        try:
            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # 设置验证集
            evallist = [(dtrain, 'train'), (dval, 'eval')]
            
            # 训练参数
            train_params = params.copy()
            num_boost_round = train_params.pop('n_estimators', 1000)
            
            # 早停参数
            early_stopping_rounds = None
            if early_stopping_config.get('enable', True):
                early_stopping_rounds = early_stopping_config.get('rounds', 100)
            
            # 训练模型
            model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evallist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # 验证
            y_pred = model.predict(dval)
            score = roc_auc_score(y_val, y_pred)
            
            return model, score
            
        except Exception as e:
            self._log_error(f"XGBoost训练失败: {e}")
            return None, 0.0
    
    def _train_lightgbm(self, params: Dict, early_stopping_config: Dict,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, float]:
        """训练LightGBM模型"""
        if not HAS_LIGHTGBM:
            self._log_warning("LightGBM未安装，跳过训练")
            return None, 0.0
        
        try:
            # 创建数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 训练参数
            train_params = params.copy()
            num_boost_round = train_params.pop('n_estimators', 1000)
            
            # 早停参数
            callbacks = []
            if early_stopping_config.get('enable', True):
                early_stopping_rounds = early_stopping_config.get('rounds', 100)
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            
            # 训练模型
            model = lgb.train(
                train_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=callbacks,
                verbose_eval=False
            )
            
            # 验证
            y_pred = model.predict(X_val)
            score = roc_auc_score(y_val, y_pred)
            
            return model, score
            
        except Exception as e:
            self._log_error(f"LightGBM训练失败: {e}")
            return None, 0.0
    
    def _train_catboost(self, params: Dict, early_stopping_config: Dict,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, float]:
        """训练CatBoost模型"""
        if not HAS_CATBOOST:
            self._log_warning("CatBoost未安装，跳过训练")
            return None, 0.0
        
        try:
            # 训练参数
            train_params = params.copy()
            
            # 早停参数
            if early_stopping_config.get('enable', True):
                train_params['early_stopping_rounds'] = early_stopping_config.get('rounds', 100)
            
            # 创建模型
            model = cb.CatBoostClassifier(**train_params)
            
            # 训练模型
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            # 验证
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            return model, score
            
        except Exception as e:
            self._log_error(f"CatBoost训练失败: {e}")
            return None, 0.0
    
    def _predict_with_model(self, model, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """使用模型进行预测"""
        try:
            if model_name == 'xgboost':
                dtest = xgb.DMatrix(X)
                predictions = model.predict(dtest)
                return (predictions > 0.5).astype(int)
            elif model_name == 'lightgbm':
                predictions = model.predict(X)
                return (predictions > 0.5).astype(int)
            elif model_name == 'catboost':
                return model.predict(X)
            else:
                raise ValueError(f"未知的模型类型: {model_name}")
        except Exception as e:
            self._log_error(f"模型 {model_name} 预测失败: {e}")
            return np.zeros(len(X))
    
    def _predict_proba_with_model(self, model, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """使用模型进行概率预测"""
        try:
            if model_name == 'xgboost':
                dtest = xgb.DMatrix(X)
                return model.predict(dtest)
            elif model_name == 'lightgbm':
                return model.predict(X)
            elif model_name == 'catboost':
                return model.predict_proba(X)[:, 1]
            else:
                raise ValueError(f"未知的模型类型: {model_name}")
        except Exception as e:
            self._log_error(f"模型 {model_name} 概率预测失败: {e}")
            return np.zeros(len(X))
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
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
        """准备预测数据"""
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
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[pd.Series]:
        """获取特征重要性"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        try:
            if model_name == 'xgboost':
                importance = model.get_score(importance_type='gain')
                return pd.Series(importance).sort_values(ascending=False)
            elif model_name == 'lightgbm':
                importance = model.feature_importance(importance_type='gain')
                feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'feature_{i}' for i in range(len(importance))]
                return pd.Series(importance, index=feature_names).sort_values(ascending=False)
            elif model_name == 'catboost':
                importance = model.get_feature_importance()
                feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'feature_{i}' for i in range(len(importance))]
                return pd.Series(importance, index=feature_names).sort_values(ascending=False)
            else:
                return None
        except Exception as e:
            self._log_error(f"获取 {model_name} 特征重要性失败: {e}")
            return None
    
    def save_models(self, save_dir: str = "outputs/models"):
        """保存训练好的模型"""
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = get_timestamp()
        
        for model_name, model in self.trained_models.items():
            try:
                model_file = save_path / f"{model_name}_{timestamp}.pkl"
                save_object(model, model_file)
                self._log_info(f"模型 {model_name} 已保存到: {model_file}")
            except Exception as e:
                self._log_error(f"保存模型 {model_name} 失败: {e}")
    
    def get_model_scores(self) -> Dict[str, float]:
        """获取所有模型的分数"""
        return self.model_scores.copy()
    
    def get_best_model(self) -> Tuple[str, Any, float]:
        """获取最佳模型"""
        if self.best_model_name is None:
            return None, None, 0.0
        
        return (
            self.best_model_name,
            self.trained_models[self.best_model_name],
            self.best_model_score
        )
