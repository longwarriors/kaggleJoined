"""
DRW 加密市场预测比赛 - 梯度提升模型训练
使用XGBoost和LightGBM训练回归模型，包括超参数调优
"""

import numpy as np
import pandas as pd
import os
import gc
import joblib
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 自定义模块
from data_preprocessing import CryptoDataProcessor, pearson_correlation

class GradientBoostingTrainer:
    """梯度提升模型训练器"""
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型 ('xgboost', 'lightgbm')
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_history = {}
        
        # 模型保存路径
        self.model_dir = os.path.join(CryptoDataProcessor().processed_data_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        if self.model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
        elif self.model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def get_param_grid(self) -> Dict[str, list]:
        """获取超参数搜索网格"""
        if self.model_type == 'xgboost':
            return {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [500, 1000, 1500],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        elif self.model_type == 'lightgbm':
            return {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [500, 1000, 1500],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'num_leaves': [31, 63, 127]
            }
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def create_model(self, params: Optional[Dict[str, Any]] = None):
        """创建模型"""
        if params is None:
            params = self.get_default_params()
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              params: Optional[Dict[str, Any]] = None,
              early_stopping_rounds: int = 100) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            params: 模型参数
            early_stopping_rounds: 早停轮数
            
        Returns:
            训练结果字典
        """
        print(f"开始训练{self.model_type}模型...")
        
        # 创建模型
        self.create_model(params)
        
        # 训练模型
        if self.model_type == 'xgboost':
            # XGBoost新版本API
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        elif self.model_type == 'lightgbm':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
        
        # 预测和评估
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # 计算评估指标
        results = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_pearson': pearson_correlation(y_train, train_pred),
            'val_pearson': pearson_correlation(y_val, val_pred)
        }
        
        self.training_history = results
        
        print(f"训练完成!")
        print(f"验证集RMSE: {results['val_rmse']:.6f}")
        print(f"验证集皮尔逊相关系数: {results['val_pearson']:.6f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             search_type: str = 'random',
                             n_iter: int = 50,
                             cv: int = 3) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            search_type: 搜索类型 ('grid', 'random')
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数
            
        Returns:
            最佳参数字典
        """
        print(f"开始{self.model_type}超参数调优...")
        
        # 创建基础模型
        base_params = self.get_default_params()
        base_params['n_estimators'] = 100  # 减少估计器数量以加速搜索
        self.create_model(base_params)
        
        # 获取参数网格
        param_grid = self.get_param_grid()
        
        # 自定义评分函数（皮尔逊相关系数）
        def pearson_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return pearson_correlation(y, y_pred)
        
        # 选择搜索方法
        if search_type == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                scoring=pearson_scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                scoring=pearson_scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")
        
        # 执行搜索
        search.fit(X_train, y_train)
        
        # 保存最佳参数
        self.best_params = search.best_params_
        
        print(f"超参数调优完成!")
        print(f"最佳CV分数: {search.best_score_:.6f}")
        print(f"最佳参数: {self.best_params}")
        
        # 使用最佳参数重新训练
        best_params = {**base_params, **self.best_params}
        best_params['n_estimators'] = 1000  # 恢复估计器数量
        
        results = self.train(X_train, y_train, X_val, y_val, best_params)
        
        return {
            'best_params': self.best_params,
            'best_cv_score': search.best_score_,
            'training_results': results
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("模型不支持特征重要性")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filename: Optional[str] = None):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if filename is None:
            filename = f'{self.model_type}_model.pkl'
        
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, filepath)
        
        # 保存训练历史和参数
        metadata = {
            'model_type': self.model_type,
            'training_history': self.training_history,
            'best_params': self.best_params
        }
        
        metadata_filepath = filepath.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_filepath)
        
        print(f"模型已保存: {filepath}")
        print(f"元数据已保存: {metadata_filepath}")
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        self.model = joblib.load(filepath)
        
        # 加载元数据
        metadata_filepath = filepath.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_filepath):
            metadata = joblib.load(metadata_filepath)
            self.training_history = metadata.get('training_history', {})
            self.best_params = metadata.get('best_params', None)
        
        print(f"模型已加载: {filepath}")


class EnsembleTrainer:
    """集成模型训练器"""
    
    def __init__(self, models: list = None):
        """
        初始化集成训练器
        
        Args:
            models: 模型列表，默认使用XGBoost和LightGBM
        """
        if models is None:
            self.models = [
                GradientBoostingTrainer('xgboost'),
                GradientBoostingTrainer('lightgbm')
            ]
        else:
            self.models = models
        
        self.weights = None
        self.training_results = {}
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  tune_hyperparams: bool = False) -> Dict[str, Any]:
        """
        训练所有模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            tune_hyperparams: 是否进行超参数调优
            
        Returns:
            训练结果字典
        """
        print("开始训练集成模型...")
        
        results = {}
        predictions = {}
        
        for i, model in enumerate(self.models):
            print(f"\n训练模型 {i+1}/{len(self.models)}: {model.model_type}")
            
            if tune_hyperparams:
                result = model.hyperparameter_tuning(X_train, y_train, X_val, y_val)
            else:
                result = model.train(X_train, y_train, X_val, y_val)
            
            results[model.model_type] = result
            predictions[model.model_type] = model.predict(X_val)
        
        # 计算集成权重（基于验证集性能）
        pearson_scores = [results[model.model_type].get('val_pearson', 0) 
                         if isinstance(results[model.model_type], dict) 
                         else results[model.model_type]['training_results']['val_pearson']
                         for model in self.models]
        
        # 使用softmax计算权重
        pearson_scores = np.array(pearson_scores)
        exp_scores = np.exp(pearson_scores * 10)  # 放大差异
        self.weights = exp_scores / exp_scores.sum()
        
        print(f"\n集成权重: {dict(zip([m.model_type for m in self.models], self.weights))}")
        
        # 计算集成预测
        ensemble_pred = np.zeros(len(y_val))
        for i, model in enumerate(self.models):
            ensemble_pred += self.weights[i] * predictions[model.model_type]
        
        # 评估集成性能
        ensemble_pearson = pearson_correlation(y_val, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        
        print(f"\n集成模型性能:")
        print(f"验证集RMSE: {ensemble_rmse:.6f}")
        print(f"验证集皮尔逊相关系数: {ensemble_pearson:.6f}")
        
        self.training_results = {
            'individual_results': results,
            'ensemble_weights': self.weights,
            'ensemble_pearson': ensemble_pearson,
            'ensemble_rmse': ensemble_rmse
        }
        
        return self.training_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """集成预测"""
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用train_all方法")
        
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        # 加权平均
        ensemble_pred = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def save_ensemble(self, filename: str = 'ensemble_model.pkl'):
        """保存集成模型"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'training_results': self.training_results
        }
        
        model_dir = os.path.join(CryptoDataProcessor().processed_data_dir, 'models')
        filepath = os.path.join(model_dir, filename)
        
        joblib.dump(ensemble_data, filepath)
        print(f"集成模型已保存: {filepath}")


if __name__ == "__main__":
    # 测试梯度提升模型训练
    print("测试梯度提升模型训练...")
    
    # 加载数据
    processor = CryptoDataProcessor(use_downsampled=True)
    train_df = processor.load_data('train')
    
    # 数据预处理
    X, y = processor.prepare_features(train_df)
    X_train, X_val, y_train, y_val = processor.split_data(X, y)
    X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
    
    # 训练单个模型
    xgb_trainer = GradientBoostingTrainer('xgboost')
    xgb_results = xgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 训练集成模型
    ensemble = EnsembleTrainer()
    ensemble_results = ensemble.train_all(X_train_scaled, y_train, X_val_scaled, y_val)
    
    print("模型训练测试完成!")
