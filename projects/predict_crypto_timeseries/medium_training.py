"""
中等规模训练脚本 - 使用30%数据进行更好的训练
"""

import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import CryptoDataProcessor
from gradient_boosting_models import GradientBoostingTrainer
from model_evaluation import AdvancedEnsemble

def medium_scale_training(downsample_ratio: float = 0.3, tune_hyperparams: bool = True):
    """
    中等规模训练
    
    Args:
        downsample_ratio: 数据采样比例
        tune_hyperparams: 是否进行超参数调优
    """
    print(f"=== 中等规模训练 (使用{downsample_ratio*100:.0f}%数据) ===")
    
    # 1. 数据预处理
    print("1. 数据加载和预处理...")
    processor = CryptoDataProcessor(use_downsampled=True, downsample_ratio=downsample_ratio)
    
    # 加载训练数据
    train_df = processor.load_data('train')
    print(f"训练数据形状: {train_df.shape}")
    
    # 如果数据太大，重新采样
    if len(train_df) > 200000:
        print(f"重新采样到{downsample_ratio*100:.0f}%...")
        train_df = train_df.sample(frac=downsample_ratio/0.1, random_state=42)  # 调整采样比例
        print(f"采样后数据形状: {train_df.shape}")
    
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
    
    # XGBoost
    print("训练XGBoost...")
    xgb_trainer = GradientBoostingTrainer('xgboost')
    if tune_hyperparams:
        print("进行XGBoost超参数调优...")
        xgb_results = xgb_trainer.hyperparameter_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val, 
            search_type='random', n_iter=30
        )
    else:
        xgb_results = xgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    models['XGBoost'] = xgb_trainer
    results['XGBoost'] = xgb_results
    
    # LightGBM
    print("训练LightGBM...")
    lgb_trainer = GradientBoostingTrainer('lightgbm')
    if tune_hyperparams:
        print("进行LightGBM超参数调优...")
        lgb_results = lgb_trainer.hyperparameter_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val,
            search_type='random', n_iter=30
        )
    else:
        lgb_results = lgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    models['LightGBM'] = lgb_trainer
    results['LightGBM'] = lgb_results
    
    # 3. 集成模型
    print("\n3. 创建集成模型...")
    ensemble = AdvancedEnsemble(models)
    ensemble.fit_weighted_average(X_val_scaled, y_val, weight_method='pearson')
    
    # 评估集成性能
    ensemble_pred = ensemble.predict(X_val_scaled)
    from data_preprocessing import pearson_correlation
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
    filename = f"medium_submission_{int(downsample_ratio*100)}pct_{timestamp}.csv"
    
    # 确保提交目录存在
    submission_dir = os.path.join(processor.PROJECT_ROOT, 'submissions')
    os.makedirs(submission_dir, exist_ok=True)
    
    filepath = os.path.join(submission_dir, filename)
    submission_df.to_csv(filepath, index=False)
    
    print(f"\n=== 中等规模训练完成 ===")
    print(f"数据规模: {downsample_ratio*100:.0f}%")
    print(f"XGBoost验证皮尔逊相关系数: {results['XGBoost'].get('val_pearson', 'N/A'):.6f}")
    print(f"LightGBM验证皮尔逊相关系数: {results['LightGBM'].get('val_pearson', 'N/A'):.6f}")
    print(f"集成模型验证皮尔逊相关系数: {ensemble_pearson:.6f}")
    print(f"提交文件: {filepath}")
    print(f"预测统计: 均值={predictions.mean():.6f}, 标准差={predictions.std():.6f}")
    
    return filepath, ensemble_pearson


def enhanced_training(downsample_ratio: float = 0.5):
    """
    增强训练 - 使用50%数据和更多优化
    """
    print(f"=== 增强训练 (使用{downsample_ratio*100:.0f}%数据) ===")
    
    # 使用更大的数据集和更多优化
    filepath, score = medium_scale_training(
        downsample_ratio=downsample_ratio, 
        tune_hyperparams=True
    )
    
    return filepath, score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='中等规模训练')
    parser.add_argument('--ratio', type=float, default=0.3, help='数据采样比例')
    parser.add_argument('--tune', action='store_true', help='进行超参数调优')
    parser.add_argument('--enhanced', action='store_true', help='增强训练模式')
    
    args = parser.parse_args()
    
    if args.enhanced:
        filepath, score = enhanced_training(args.ratio)
    else:
        filepath, score = medium_scale_training(args.ratio, args.tune)
    
    print(f"\n✅ 训练完成!")
    print(f"最终皮尔逊相关系数: {score:.6f}")
    print(f"提交文件: {filepath}")
