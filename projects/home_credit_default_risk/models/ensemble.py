"""
集成建模模块

实现各种集成学习策略，包括投票法、平均法、堆叠法等。

作者：Augment Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseModel
from .trainers import ModelTrainer


class EnsembleModel(BaseModel):
    """
    集成模型类
    
    实现多种集成学习策略
    """
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        """
        初始化集成模型
        
        Args:
            config: 集成配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        self.base_models = {}
        self.ensemble_models = {}
        self.ensemble_scores = {}
        self.best_ensemble_name = None
        self.best_ensemble_score = 0.0
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
             trained_models: Optional[Dict] = None, **kwargs) -> Dict:
        """
        训练集成模型
        
        Args:
            X: 特征数据
            y: 目标变量
            trained_models: 已训练的基础模型字典
            **kwargs: 其他参数
            
        Returns:
            训练结果字典
        """
        self._log_info("开始训练集成模型")
        
        # 准备数据
        X_clean, y_clean = self._prepare_data(X, y)
        
        # 如果没有提供训练好的模型，先训练基础模型
        if trained_models is None:
            self._log_info("训练基础模型")
            model_trainer = ModelTrainer(self.config, self.logger)
            model_trainer.train(X_clean, y_clean)
            self.base_models = model_trainer.trained_models
        else:
            self.base_models = trained_models
        
        if not self.base_models:
            self._log_error("没有可用的基础模型")
            return {}
        
        # 获取集成配置
        ensemble_config = self.config.get('ensemble_models', {})
        
        # 训练各种集成模型
        if ensemble_config.get('voting', {}).get('enable', True):
            self._train_voting_ensemble(X_clean, y_clean, ensemble_config.get('voting', {}))
        
        if ensemble_config.get('averaging', {}).get('enable', True):
            self._train_averaging_ensemble(X_clean, y_clean, ensemble_config.get('averaging', {}))
        
        if ensemble_config.get('stacking', {}).get('enable', True):
            self._train_stacking_ensemble(X_clean, y_clean, ensemble_config.get('stacking', {}))
        
        # 记录训练历史
        self.training_history = {
            'ensemble_scores': self.ensemble_scores,
            'best_ensemble': self.best_ensemble_name,
            'best_score': self.best_ensemble_score,
            'base_models_count': len(self.base_models)
        }
        
        self.is_trained = True
        self._log_info(f"集成模型训练完成，最佳集成: {self.best_ensemble_name} (AUC: {self.best_ensemble_score:.4f})")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame, ensemble_name: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        使用集成模型进行预测
        
        Args:
            X: 特征数据
            ensemble_name: 集成模型名称，如果不指定则使用最佳集成
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        self._validate_trained()
        
        if ensemble_name is None:
            ensemble_name = self.best_ensemble_name
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"集成模型 {ensemble_name} 未训练")
        
        X_clean = self._prepare_prediction_data(X)
        
        if ensemble_name.startswith('voting'):
            return self.ensemble_models[ensemble_name].predict(X_clean)
        elif ensemble_name.startswith('averaging'):
            return self._predict_averaging(X_clean, ensemble_name)
        elif ensemble_name.startswith('stacking'):
            return self._predict_stacking(X_clean, ensemble_name)
        else:
            raise ValueError(f"未知的集成类型: {ensemble_name}")
    
    def predict_proba(self, X: pd.DataFrame, ensemble_name: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        使用集成模型进行概率预测
        
        Args:
            X: 特征数据
            ensemble_name: 集成模型名称，如果不指定则使用最佳集成
            **kwargs: 其他参数
            
        Returns:
            预测概率
        """
        self._validate_trained()
        
        if ensemble_name is None:
            ensemble_name = self.best_ensemble_name
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"集成模型 {ensemble_name} 未训练")
        
        X_clean = self._prepare_prediction_data(X)
        
        if ensemble_name.startswith('voting'):
            return self.ensemble_models[ensemble_name].predict_proba(X_clean)[:, 1]
        elif ensemble_name.startswith('averaging'):
            return self._predict_proba_averaging(X_clean, ensemble_name)
        elif ensemble_name.startswith('stacking'):
            return self._predict_proba_stacking(X_clean, ensemble_name)
        else:
            raise ValueError(f"未知的集成类型: {ensemble_name}")
    
    def _train_voting_ensemble(self, X: pd.DataFrame, y: pd.Series, voting_config: Dict):
        """训练投票集成"""
        try:
            models_to_use = voting_config.get('models', list(self.base_models.keys()))
            voting_type = voting_config.get('voting_type', 'soft')
            
            # 准备模型列表
            estimators = []
            for model_name in models_to_use:
                if model_name in self.base_models:
                    estimators.append((model_name, self.base_models[model_name]))
            
            if len(estimators) < 2:
                self._log_warning("投票集成需要至少2个模型")
                return
            
            # 创建投票分类器
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting=voting_type
            )
            
            # 训练
            voting_clf.fit(X, y)
            
            # 评估
            y_pred_proba = voting_clf.predict_proba(X)[:, 1]
            score = roc_auc_score(y, y_pred_proba)
            
            ensemble_name = f'voting_{voting_type}'
            self.ensemble_models[ensemble_name] = voting_clf
            self.ensemble_scores[ensemble_name] = score
            
            # 更新最佳集成
            if score > self.best_ensemble_score:
                self.best_ensemble_score = score
                self.best_ensemble_name = ensemble_name
            
            self._log_info(f"投票集成训练完成，AUC: {score:.4f}")
            
        except Exception as e:
            self._log_error(f"投票集成训练失败: {e}")
    
    def _train_averaging_ensemble(self, X: pd.DataFrame, y: pd.Series, averaging_config: Dict):
        """训练平均集成"""
        try:
            models_to_use = averaging_config.get('models', list(self.base_models.keys()))
            method = averaging_config.get('method', 'simple')
            weights = averaging_config.get('weights', None)
            
            # 获取预测
            predictions = {}
            for model_name in models_to_use:
                if model_name in self.base_models:
                    model = self.base_models[model_name]
                    
                    # 使用模型训练器的预测方法
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    elif hasattr(model, 'predict'):
                        pred = model.predict(X)
                    else:
                        # 对于XGBoost等模型
                        try:
                            import xgboost as xgb
                            if isinstance(model, xgb.Booster):
                                dtest = xgb.DMatrix(X)
                                pred = model.predict(dtest)
                            else:
                                pred = model.predict(X)
                        except:
                            continue
                    
                    predictions[model_name] = pred
            
            if len(predictions) < 2:
                self._log_warning("平均集成需要至少2个模型的预测")
                return
            
            # 计算平均预测
            if method == 'simple':
                # 简单平均
                avg_pred = np.mean(list(predictions.values()), axis=0)
            elif method == 'weighted' and weights is not None:
                # 加权平均
                weighted_preds = []
                for i, (model_name, pred) in enumerate(predictions.items()):
                    if i < len(weights):
                        weighted_preds.append(pred * weights[i])
                avg_pred = np.sum(weighted_preds, axis=0)
            else:
                # 默认简单平均
                avg_pred = np.mean(list(predictions.values()), axis=0)
            
            # 评估
            score = roc_auc_score(y, avg_pred)
            
            ensemble_name = f'averaging_{method}'
            self.ensemble_models[ensemble_name] = {
                'models': models_to_use,
                'method': method,
                'weights': weights
            }
            self.ensemble_scores[ensemble_name] = score
            
            # 更新最佳集成
            if score > self.best_ensemble_score:
                self.best_ensemble_score = score
                self.best_ensemble_name = ensemble_name
            
            self._log_info(f"平均集成训练完成，AUC: {score:.4f}")
            
        except Exception as e:
            self._log_error(f"平均集成训练失败: {e}")
    
    def _train_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series, stacking_config: Dict):
        """训练堆叠集成"""
        try:
            base_models = stacking_config.get('base_models', list(self.base_models.keys()))
            meta_model_name = stacking_config.get('meta_model', 'logistic_regression')
            cv_folds = stacking_config.get('cv_folds', 5)
            
            # 生成基础模型的交叉验证预测
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=stacking_config.get('shuffle', True),
                random_state=stacking_config.get('random_state', 42)
            )
            
            # 收集基础模型的预测
            base_predictions = np.zeros((len(X), len(base_models)))
            
            for i, model_name in enumerate(base_models):
                if model_name not in self.base_models:
                    continue
                
                model = self.base_models[model_name]
                
                # 交叉验证预测
                cv_preds = np.zeros(len(X))
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y.iloc[train_idx]
                    
                    # 重新训练模型（在折叠上）
                    fold_model = self._clone_model(model, model_name)
                    fold_model.fit(X_train, y_train)
                    
                    # 预测验证集
                    if hasattr(fold_model, 'predict_proba'):
                        val_pred = fold_model.predict_proba(X_val)[:, 1]
                    else:
                        val_pred = fold_model.predict(X_val)
                    
                    cv_preds[val_idx] = val_pred
                
                base_predictions[:, i] = cv_preds
            
            # 训练元模型
            meta_model = self._create_meta_model(meta_model_name)
            meta_model.fit(base_predictions, y)
            
            # 评估
            meta_pred = meta_model.predict_proba(base_predictions)[:, 1]
            score = roc_auc_score(y, meta_pred)
            
            ensemble_name = f'stacking_{meta_model_name}'
            self.ensemble_models[ensemble_name] = {
                'base_models': base_models,
                'meta_model': meta_model,
                'cv_config': {'n_splits': cv_folds, 'shuffle': True, 'random_state': 42}
            }
            self.ensemble_scores[ensemble_name] = score
            
            # 更新最佳集成
            if score > self.best_ensemble_score:
                self.best_ensemble_score = score
                self.best_ensemble_name = ensemble_name
            
            self._log_info(f"堆叠集成训练完成，AUC: {score:.4f}")
            
        except Exception as e:
            self._log_error(f"堆叠集成训练失败: {e}")
    
    def _predict_averaging(self, X: pd.DataFrame, ensemble_name: str) -> np.ndarray:
        """平均集成预测"""
        ensemble_info = self.ensemble_models[ensemble_name]
        models_to_use = ensemble_info['models']
        method = ensemble_info['method']
        weights = ensemble_info.get('weights')
        
        predictions = []
        for model_name in models_to_use:
            if model_name in self.base_models:
                model = self.base_models[model_name]
                pred = self._get_model_prediction(model, X, model_name)
                predictions.append(pred)
        
        if method == 'simple':
            avg_pred = np.mean(predictions, axis=0)
        elif method == 'weighted' and weights is not None:
            weighted_preds = [pred * weight for pred, weight in zip(predictions, weights)]
            avg_pred = np.sum(weighted_preds, axis=0)
        else:
            avg_pred = np.mean(predictions, axis=0)
        
        return (avg_pred > 0.5).astype(int)
    
    def _predict_proba_averaging(self, X: pd.DataFrame, ensemble_name: str) -> np.ndarray:
        """平均集成概率预测"""
        ensemble_info = self.ensemble_models[ensemble_name]
        models_to_use = ensemble_info['models']
        method = ensemble_info['method']
        weights = ensemble_info.get('weights')
        
        predictions = []
        for model_name in models_to_use:
            if model_name in self.base_models:
                model = self.base_models[model_name]
                pred = self._get_model_prediction_proba(model, X, model_name)
                predictions.append(pred)
        
        if method == 'simple':
            return np.mean(predictions, axis=0)
        elif method == 'weighted' and weights is not None:
            weighted_preds = [pred * weight for pred, weight in zip(predictions, weights)]
            return np.sum(weighted_preds, axis=0)
        else:
            return np.mean(predictions, axis=0)
    
    def _predict_stacking(self, X: pd.DataFrame, ensemble_name: str) -> np.ndarray:
        """堆叠集成预测"""
        ensemble_info = self.ensemble_models[ensemble_name]
        base_models = ensemble_info['base_models']
        meta_model = ensemble_info['meta_model']
        
        # 获取基础模型预测
        base_predictions = np.zeros((len(X), len(base_models)))
        for i, model_name in enumerate(base_models):
            if model_name in self.base_models:
                model = self.base_models[model_name]
                pred = self._get_model_prediction_proba(model, X, model_name)
                base_predictions[:, i] = pred
        
        # 元模型预测
        return meta_model.predict(base_predictions)
    
    def _predict_proba_stacking(self, X: pd.DataFrame, ensemble_name: str) -> np.ndarray:
        """堆叠集成概率预测"""
        ensemble_info = self.ensemble_models[ensemble_name]
        base_models = ensemble_info['base_models']
        meta_model = ensemble_info['meta_model']
        
        # 获取基础模型预测
        base_predictions = np.zeros((len(X), len(base_models)))
        for i, model_name in enumerate(base_models):
            if model_name in self.base_models:
                model = self.base_models[model_name]
                pred = self._get_model_prediction_proba(model, X, model_name)
                base_predictions[:, i] = pred
        
        # 元模型预测
        return meta_model.predict_proba(base_predictions)[:, 1]
    
    def _get_model_prediction(self, model, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """获取模型预测"""
        try:
            if model_name == 'xgboost':
                import xgboost as xgb
                if isinstance(model, xgb.Booster):
                    dtest = xgb.DMatrix(X)
                    return (model.predict(dtest) > 0.5).astype(int)
                else:
                    return model.predict(X)
            else:
                return model.predict(X)
        except:
            return np.zeros(len(X))
    
    def _get_model_prediction_proba(self, model, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """获取模型概率预测"""
        try:
            if model_name == 'xgboost':
                import xgboost as xgb
                if isinstance(model, xgb.Booster):
                    dtest = xgb.DMatrix(X)
                    return model.predict(dtest)
                else:
                    return model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                return model.predict(X)
        except:
            return np.zeros(len(X))
    
    def _clone_model(self, model, model_name: str):
        """克隆模型"""
        # 这里简化处理，实际应该根据模型类型进行深度复制
        return model
    
    def _create_meta_model(self, meta_model_name: str):
        """创建元模型"""
        if meta_model_name == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            return LogisticRegression(random_state=42, max_iter=1000)
    
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
    
    def get_ensemble_scores(self) -> Dict[str, float]:
        """获取所有集成模型的分数"""
        return self.ensemble_scores.copy()
    
    def get_best_ensemble(self) -> Tuple[str, Any, float]:
        """获取最佳集成模型"""
        if self.best_ensemble_name is None:
            return None, None, 0.0
        
        return (
            self.best_ensemble_name,
            self.ensemble_models[self.best_ensemble_name],
            self.best_ensemble_score
        )
