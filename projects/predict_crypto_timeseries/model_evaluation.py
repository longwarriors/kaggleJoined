"""
DRW 加密市场预测比赛 - 模型评估与集成
使用皮尔逊相关系数评估模型，实现模型集成策略
"""

import numpy as np
import pandas as pd
import os
import gc
import joblib
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# 自定义模块
from data_preprocessing import CryptoDataProcessor, pearson_correlation
from gradient_boosting_models import GradientBoostingTrainer, EnsembleTrainer
from lstm_model import LSTMTrainer

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.results = {}
        self.predictions = {}
        
        # 结果保存路径
        self.results_dir = os.path.join(CryptoDataProcessor().processed_data_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "model") -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            
        Returns:
            评估指标字典
        """
        # 处理NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            print(f"警告: {model_name} 的预测结果全为NaN")
            return {
                'pearson_correlation': 0.0,
                'spearman_correlation': 0.0,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2_score': -float('inf'),
                'valid_predictions': 0
            }
        
        # 计算各种指标
        metrics = {}
        
        try:
            # 皮尔逊相关系数（主要评估指标）
            pearson_corr, pearson_p = pearsonr(y_true_clean, y_pred_clean)
            metrics['pearson_correlation'] = pearson_corr
            metrics['pearson_p_value'] = pearson_p
        except:
            metrics['pearson_correlation'] = 0.0
            metrics['pearson_p_value'] = 1.0
        
        try:
            # 斯皮尔曼相关系数
            spearman_corr, spearman_p = spearmanr(y_true_clean, y_pred_clean)
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
        except:
            metrics['spearman_correlation'] = 0.0
            metrics['spearman_p_value'] = 1.0
        
        # 回归指标
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
        metrics['r2_score'] = r2_score(y_true_clean, y_pred_clean)
        
        # 其他统计信息
        metrics['valid_predictions'] = len(y_true_clean)
        metrics['prediction_std'] = np.std(y_pred_clean)
        metrics['prediction_mean'] = np.mean(y_pred_clean)
        metrics['target_std'] = np.std(y_true_clean)
        metrics['target_mean'] = np.mean(y_true_clean)
        
        return metrics
    
    def evaluate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """
        评估单个模型
        
        Args:
            model: 训练好的模型
            X_val: 验证特征
            y_val: 验证目标
            model_name: 模型名称
            
        Returns:
            评估结果
        """
        print(f"评估模型: {model_name}")
        
        # 预测
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
        else:
            raise ValueError(f"模型 {model_name} 没有predict方法")
        
        # 计算指标
        metrics = self.calculate_metrics(y_val.values, y_pred, model_name)
        
        # 保存结果
        self.results[model_name] = metrics
        self.predictions[model_name] = y_pred
        
        print(f"{model_name} 皮尔逊相关系数: {metrics['pearson_correlation']:.6f}")
        print(f"{model_name} RMSE: {metrics['rmse']:.6f}")
        
        return metrics
    
    def compare_models(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                      y_val: pd.Series) -> pd.DataFrame:
        """
        比较多个模型
        
        Args:
            models: 模型字典 {name: model}
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            比较结果DataFrame
        """
        print("开始模型比较...")
        
        comparison_results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_val, y_val, name)
            comparison_results.append({
                'model': name,
                **metrics
            })
        
        # 创建比较表
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('pearson_correlation', ascending=False)
        
        print("\n=== 模型比较结果 ===")
        print(comparison_df[['model', 'pearson_correlation', 'rmse', 'mae', 'r2_score']].round(6))
        
        return comparison_df
    
    def plot_predictions(self, y_true: np.ndarray, model_names: List[str] = None,
                        save_path: str = None, sample_size: int = 1000):
        """
        绘制预测结果对比图
        
        Args:
            y_true: 真实值
            model_names: 要绘制的模型名称列表
            save_path: 保存路径
            sample_size: 采样大小（用于可视化）
        """
        if model_names is None:
            model_names = list(self.predictions.keys())
        
        # 采样数据用于可视化
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
        else:
            indices = np.arange(len(y_true))
            y_true_sample = y_true
        
        # 创建子图
        n_models = len(model_names)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        if n_models == 1:
            axes = [axes]
        elif n_models <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.predictions:
                continue
            
            y_pred_sample = self.predictions[model_name][indices]
            
            # 散点图
            axes[i].scatter(y_true_sample, y_pred_sample, alpha=0.5, s=1)
            axes[i].plot([y_true_sample.min(), y_true_sample.max()], 
                        [y_true_sample.min(), y_true_sample.max()], 'r--', lw=2)
            
            # 计算相关系数
            corr = pearson_correlation(y_true_sample, y_pred_sample)
            axes[i].set_title(f'{model_name}\nPearson: {corr:.4f}')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predictions')
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(model_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测对比图已保存: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, model_names: List[str] = None,
                      save_path: str = None):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            model_names: 要绘制的模型名称列表
            save_path: 保存路径
        """
        if model_names is None:
            model_names = list(self.predictions.keys())
        
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.predictions:
                continue
            
            y_pred = self.predictions[model_name]
            residuals = y_true - y_pred
            
            axes[i].scatter(y_pred, residuals, alpha=0.5, s=1)
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_title(f'{model_name} Residuals')
            axes[i].set_xlabel('Predictions')
            axes[i].set_ylabel('Residuals')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"残差图已保存: {save_path}")
        
        plt.show()
    
    def save_results(self, filename: str = 'evaluation_results.pkl'):
        """保存评估结果"""
        results_data = {
            'results': self.results,
            'predictions': self.predictions
        }
        
        filepath = os.path.join(self.results_dir, filename)
        joblib.dump(results_data, filepath)
        print(f"评估结果已保存: {filepath}")
    
    def load_results(self, filename: str = 'evaluation_results.pkl'):
        """加载评估结果"""
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        results_data = joblib.load(filepath)
        self.results = results_data['results']
        self.predictions = results_data['predictions']
        
        print(f"评估结果已加载: {filepath}")


class AdvancedEnsemble:
    """高级集成模型"""
    
    def __init__(self, base_models: Dict[str, Any]):
        """
        初始化高级集成模型
        
        Args:
            base_models: 基础模型字典 {name: model}
        """
        self.base_models = base_models
        self.weights = None
        self.stacking_model = None
        self.ensemble_type = 'weighted_average'
    
    def fit_weighted_average(self, X_val: pd.DataFrame, y_val: pd.Series,
                           weight_method: str = 'pearson'):
        """
        拟合加权平均集成
        
        Args:
            X_val: 验证特征
            y_val: 验证目标
            weight_method: 权重计算方法 ('pearson', 'rmse', 'uniform')
        """
        print(f"拟合加权平均集成 (权重方法: {weight_method})")
        
        # 获取所有模型的预测
        predictions = {}
        scores = {}
        
        for name, model in self.base_models.items():
            pred = model.predict(X_val)
            predictions[name] = pred
            
            if weight_method == 'pearson':
                score = pearson_correlation(y_val.values, pred)
                scores[name] = max(0, score)  # 负相关系数设为0
            elif weight_method == 'rmse':
                rmse = np.sqrt(mean_squared_error(y_val.values, pred))
                scores[name] = 1 / (1 + rmse)  # RMSE越小权重越大
            elif weight_method == 'uniform':
                scores[name] = 1.0
            else:
                raise ValueError(f"不支持的权重方法: {weight_method}")
        
        # 计算权重
        total_score = sum(scores.values())
        if total_score == 0:
            # 如果所有分数都是0，使用均匀权重
            self.weights = {name: 1/len(scores) for name in scores}
        else:
            self.weights = {name: score/total_score for name, score in scores.items()}
        
        print("集成权重:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        self.ensemble_type = 'weighted_average'
    
    def fit_stacking(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    meta_model=None):
        """
        拟合堆叠集成
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            meta_model: 元模型，默认使用线性回归
        """
        print("拟合堆叠集成")
        
        if meta_model is None:
            from sklearn.linear_model import Ridge
            meta_model = Ridge(alpha=1.0)
        
        # 获取基础模型在验证集上的预测作为元特征
        meta_features = []
        for name, model in self.base_models.items():
            pred = model.predict(X_val)
            meta_features.append(pred)
        
        meta_X = np.column_stack(meta_features)
        
        # 训练元模型
        self.stacking_model = meta_model
        self.stacking_model.fit(meta_X, y_val)
        
        self.ensemble_type = 'stacking'
        print("堆叠集成训练完成")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """集成预测"""
        if self.ensemble_type == 'weighted_average':
            if self.weights is None:
                raise ValueError("权重尚未计算，请先调用fit_weighted_average")
            
            # 加权平均
            ensemble_pred = np.zeros(len(X))
            for name, model in self.base_models.items():
                pred = model.predict(X)
                ensemble_pred += self.weights[name] * pred
            
            return ensemble_pred
        
        elif self.ensemble_type == 'stacking':
            if self.stacking_model is None:
                raise ValueError("堆叠模型尚未训练，请先调用fit_stacking")
            
            # 获取基础模型预测作为元特征
            meta_features = []
            for name, model in self.base_models.items():
                pred = model.predict(X)
                meta_features.append(pred)
            
            meta_X = np.column_stack(meta_features)
            
            # 元模型预测
            return self.stacking_model.predict(meta_X)
        
        else:
            raise ValueError(f"不支持的集成类型: {self.ensemble_type}")


if __name__ == "__main__":
    # 测试模型评估
    print("测试模型评估...")
    
    # 加载数据
    processor = CryptoDataProcessor(use_downsampled=True)
    train_df = processor.load_data('train')
    
    # 数据预处理
    X, y = processor.prepare_features(train_df)
    X_train, X_val, y_train, y_val = processor.split_data(X, y)
    X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
    
    # 训练一些简单模型用于测试
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    models = {
        'Ridge': Ridge().fit(X_train_scaled, y_train),
        'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train_scaled, y_train)
    }
    
    # 评估模型
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(models, X_val_scaled, y_val)
    
    # 高级集成
    ensemble = AdvancedEnsemble(models)
    ensemble.fit_weighted_average(X_val_scaled, y_val)
    
    # 评估集成模型
    ensemble_pred = ensemble.predict(X_val_scaled)
    ensemble_metrics = evaluator.calculate_metrics(y_val.values, ensemble_pred, 'Ensemble')
    
    print(f"\n集成模型皮尔逊相关系数: {ensemble_metrics['pearson_correlation']:.6f}")
    
    print("模型评估测试完成!")
