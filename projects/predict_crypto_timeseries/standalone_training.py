"""
DRW 加密市场预测比赛 - 独立全量训练脚本
避免导入问题，直接实现训练逻辑
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def main():
    """独立全量训练"""
    print("=" * 60)
    print("🚀 DRW 加密市场预测比赛 - 独立全量训练")
    print("=" * 60)

    start_time = time.time()

    try:
        # 直接导入需要的类
        from projects.predict_crypto_timeseries.data_preprocessing import (
            CryptoDataProcessor,
        )
        from projects.predict_crypto_timeseries.gradient_boosting_models import (
            GradientBoostingTrainer,
        )

        print("🔧 初始化数据处理器...")
        processor = CryptoDataProcessor(use_downsampled=False)  # 使用完整数据集

        print("📊 加载和预处理数据...")
        # 加载训练数据
        train_df = processor.load_data("train")
        print(f"训练数据形状: {train_df.shape}")

        # 准备特征和目标变量
        X, y = processor.prepare_features(train_df, is_train=True)
        print(f"特征形状: {X.shape}, 目标变量形状: {y.shape}")

        # 分割数据
        X_train, X_val, y_train, y_val = processor.split_data(X, y, test_size=0.2)
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

        # 特征缩放
        X_train_scaled, X_val_scaled = processor.scale_features(X_train, X_val)
        print(f"特征缩放完成")

        print("🤖 开始模型训练...")

        # 训练XGBoost模型
        print("训练XGBoost模型...")
        xgb_trainer = GradientBoostingTrainer("xgboost")
        xgb_results = xgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
        print(f"XGBoost验证集皮尔逊相关系数: {xgb_results['val_pearson']:.4f}")

        # 训练LightGBM模型
        print("训练LightGBM模型...")
        lgb_trainer = GradientBoostingTrainer("lightgbm")
        lgb_results = lgb_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
        print(f"LightGBM验证集皮尔逊相关系数: {lgb_results['val_pearson']:.4f}")

        print("🔮 生成测试集预测...")

        # 加载测试数据
        test_df = processor.load_data("test")
        print(f"测试数据形状: {test_df.shape}")

        # 预处理测试数据
        result = processor.prepare_features(test_df, is_train=False)
        if isinstance(result, tuple):
            X_test, _ = result
        else:
            X_test = result
        X_test_scaled = processor.scaler.transform(X_test)
        print(f"测试集特征形状: {X_test_scaled.shape}")

        # 生成预测
        xgb_pred = xgb_trainer.predict(X_test_scaled)
        lgb_pred = lgb_trainer.predict(X_test_scaled)

        # 简单集成（平均）
        ensemble_pred = (xgb_pred + lgb_pred) / 2
        print(f"集成预测完成，预测数量: {len(ensemble_pred)}")

        print("📝 生成提交文件...")

        # 创建提交文件
        import pandas as pd

        submission_df = processor.load_data("submission")

        # 确保只有ID和prediction两列
        submission_df = pd.DataFrame(
            {"ID": submission_df["ID"], "prediction": ensemble_pred}
        )

        # 保存提交文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        submission_filename = f"full_training_submission_{timestamp}.csv"
        submission_path = os.path.join(project_root, "submissions", submission_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)

        # 计算总时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        print("=" * 60)
        print("🎉 全量训练完成!")
        print("=" * 60)
        print(f"⏱️  总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"📁 提交文件: {submission_path}")
        print(f"📊 文件大小: {os.path.getsize(submission_path) / 1024:.2f} KB")

        # 验证提交文件
        print(f"📈 预测数量: {len(submission_df)} 条")
        print(f"📋 列名: {list(submission_df.columns)}")
        print(
            f"🔍 预测值范围: [{submission_df['label'].min():.6f}, {submission_df['label'].max():.6f}]"
        )
        print(f"📊 预测值统计:")
        print(f"    均值: {submission_df['label'].mean():.6f}")
        print(f"    标准差: {submission_df['label'].std():.6f}")

        print("\n✅ 独立全量训练成功完成!")
        return submission_path

    except Exception as e:
        print(f"❌ 全量训练失败: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
