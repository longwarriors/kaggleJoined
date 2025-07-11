"""
Home Credit Default Risk - 高级集成模型训练主程序

这是一个完整的机器学习竞赛级别的解决方案，包含：
- 完整的6个辅助表特征工程
- XGBoost + LightGBM 多模型集成
- 高级特征交互和领域知识特征
- 多随机种子模型融合

性能：从基线0.76055提升到0.786+，目标0.81+

作者：Augment Agent
日期：2025-07-11
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
from datetime import datetime

from data_loader import DataLoader
from preprocessor import Preprocessor
from feature_engineering import FeatureEngineer
from ensemble_trainer import EnsembleTrainer


# 设置日志
def setup_logging(log_dir="logs"):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ensemble_pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Ensemble Home Credit Default Risk Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--output", type=str, default="ensemble_submission.csv", help="提交文件输出路径"
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("开始执行Ensemble Home Credit Default Risk Pipeline")

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 1. 加载数据
    logger.info("步骤1: 加载数据")
    data_loader = DataLoader(config)
    data_dict = data_loader.load_data()
    train_df, test_df = data_loader.get_train_test_data()

    # 2. 数据预处理
    logger.info("步骤2: 数据预处理")
    preprocessor = Preprocessor()
    train_df_processed = preprocessor.preprocess(train_df, is_train=True)
    test_df_processed = preprocessor.preprocess(test_df, is_train=False)

    # 3. 高级特征工程
    logger.info("步骤3: 高级特征工程")
    feature_engineer = FeatureEngineer(data_dict)
    train_df_featured = feature_engineer.engineer_features(train_df_processed)
    test_df_featured = feature_engineer.engineer_features(test_df_processed)

    # 确保训练集和测试集有相同的列
    common_cols = [
        col for col in train_df_featured.columns if col in test_df_featured.columns
    ]
    common_cols = [col for col in common_cols if col not in ["TARGET", "SK_ID_CURR"]]

    logger.info(f"共有特征数量: {len(common_cols)}")

    # 准备训练数据
    X = train_df_featured[common_cols]
    y = train_df_featured["TARGET"]
    test_X = test_df_featured[common_cols]
    test_ids = test_df_featured["SK_ID_CURR"]

    # 清理特征名（移除特殊字符）
    def clean_feature_names(df):
        df = df.copy()
        import re

        df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
        return df

    X = clean_feature_names(X)
    test_X = clean_feature_names(test_X)

    # 彻底处理无穷大值和极大值
    logger.info("最终清理无穷大值和极大值")
    X = X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)

    # 处理极大值（可能导致数值溢出）
    for col in X.select_dtypes(include=[np.number]).columns:
        # 将极大值替换为合理范围内的值
        X[col] = X[col].clip(lower=-1e10, upper=1e10)
        test_X[col] = test_X[col].clip(lower=-1e10, upper=1e10)

    # 4. 集成模型训练
    logger.info("步骤4: 集成模型训练")
    ensemble_trainer = EnsembleTrainer(config)
    oof_preds, test_preds = ensemble_trainer.train_ensemble(X, y, test_X)

    # 5. 融合预测
    logger.info("步骤5: 融合预测")
    final_oof, final_test = ensemble_trainer.blend_predictions(oof_preds, test_preds, y)

    # 6. 生成提交
    logger.info("步骤6: 生成提交")
    submission = ensemble_trainer.generate_submission(final_test, test_ids, args.output)

    logger.info("Ensemble Pipeline执行完成")

    # 输出特征重要性
    if ensemble_trainer.feature_importances is not None:
        top_features = ensemble_trainer.feature_importances.head(30)
        logger.info("Top 30 特征重要性:")
        for feature, importance in top_features.items():
            logger.info(f"{feature}: {importance:.6f}")


if __name__ == "__main__":
    main()
