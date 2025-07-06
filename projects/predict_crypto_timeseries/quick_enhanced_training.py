"""
快速增强训练 - 使用30%数据，不进行超参数调优，快速获得更好结果
"""

import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import CryptoDataProcessor, pearson_correlation
from gradient_boosting_models import GradientBoostingTrainer
from model_evaluation import AdvancedEnsemble

def quick_enhanced_training():
    """快速增强训练"""
    print("=== 快速增强训练 (30%数据，默认参数) ===")
    
    # 1. 数据预处理
    print("1. 数据加载和预处理...")
    processor = CryptoDataProcessor(use_downsampled=True, downsample_ratio=0.1)
    
    # 加载训练数据
    train_df = processor.load_data('train')
    print(f"训练数据形状: {train_df.shape}")
    
    # 内存优化
    train_df = processor.memory_optimization(train_df)
    
    # 准备特征
    X, y = processor.prepare_features(train_df, is_train=True)
    
    # 分割数据
    X_train, X_val, y_train, y_val = processor.split_data(X, y, test_size=0.2)
    
    # 特征缩放
    X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val, scaler_type='robust')
    
    # 清理内存
    del train_df, X, y, X_train, X_val
    gc.collect()
    
    print(f"预处理完成: 训练集{X_train_scaled.shape}, 验证集{X_val_scaled.shape}")
    
    # 2. 训练模型
    print("\n2. 训练梯度提升模型...")
    
    models = {}
    results = {}
    
    # XGBoost (使用更好的默认参数)
    print("训练XGBoost...")
    xgb_trainer = GradientBoostingTrainer('xgboost')
    
    # 使用更好的默认参数
    better_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    xgb_results = xgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val, better_params)
    models['XGBoost'] = xgb_trainer
    results['XGBoost'] = xgb_results
    
    # LightGBM (使用更好的默认参数)
    print("训练LightGBM...")
    lgb_trainer = GradientBoostingTrainer('lightgbm')
    
    better_lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 127,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    lgb_results = lgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val, better_lgb_params)
    models['LightGBM'] = lgb_trainer
    results['LightGBM'] = lgb_results
    
    # 3. 集成模型
    print("\n3. 创建集成模型...")
    ensemble = AdvancedEnsemble(models)
    ensemble.fit_weighted_average(X_val_scaled, y_val, weight_method='pearson')
    
    # 评估集成性能
    ensemble_pred = ensemble.predict(X_val_scaled)
    ensemble_pearson = pearson_correlation(y_val.values, ensemble_pred)
    
    print(f"集成模型验证集皮尔逊相关系数: {ensemble_pearson:.6f}")
    
    # 4. 加载测试数据并预测
    print("\n4. 加载测试数据...")
    test_df = processor.load_data('test')
    print(f"测试数据形状: {test_df.shape}")
    
    # 内存优化
    test_df = processor.memory_optimization(test_df)
    
    # 准备测试特征
    X_test, _ = processor.prepare_features(test_df, is_train=False)
    
    # 缩放测试特征
    X_test_scaled = pd.DataFrame(
        processor.scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 清理内存
    del test_df, X_test
    gc.collect()
    
    print(f"测试数据预处理完成: {X_test_scaled.shape}")
    
    # 5. 生成预测
    print("\n5. 生成预测...")
    predictions = ensemble.predict(X_test_scaled)
    
    # 创建提交文件
    test_ids = np.arange(1, len(predictions) + 1)
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'prediction': predictions
    })
    
    # 保存提交文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_submission_30pct_{timestamp}.csv"
    
    # 确保提交目录存在
    submission_dir = os.path.join(processor.PROJECT_ROOT, 'submissions')
    os.makedirs(submission_dir, exist_ok=True)
    
    filepath = os.path.join(submission_dir, filename)
    submission_df.to_csv(filepath, index=False)
    
    print(f"\n=== 快速增强训练完成 ===")
    print(f"XGBoost验证皮尔逊相关系数: {results['XGBoost']['val_pearson']:.6f}")
    print(f"LightGBM验证皮尔逊相关系数: {results['LightGBM']['val_pearson']:.6f}")
    print(f"集成模型验证皮尔逊相关系数: {ensemble_pearson:.6f}")
    print(f"提交文件: {filepath}")
    print(f"预测统计: 均值={predictions.mean():.6f}, 标准差={predictions.std():.6f}")
    
    return filepath, ensemble_pearson, results


if __name__ == "__main__":
    filepath, score, results = quick_enhanced_training()
    print(f"\n✅ 快速增强训练完成!")
    print(f"最终皮尔逊相关系数: {score:.6f}")
    print(f"提交文件: {filepath}")
