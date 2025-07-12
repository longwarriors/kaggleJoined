"""
Home Credit Default Risk - 优化版主程序

专注于最有效的改进技术：
1. Home Credit专用特征工程
2. 更好的数据清理
3. 优化的集成模型
4. 针对性的超参数调优

目标：将AUC从0.78提升到0.8+

作者：Augment Agent
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score

from data_loader import DataLoader
from preprocessor import Preprocessor
from feature_engineering import FeatureEngineer
from credit_specific_features import CreditSpecificFeatureEngineer
from ensemble_trainer import EnsembleTrainer


def setup_logging():
    """设置日志记录"""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/optimized_pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def clean_and_prepare_data(X, test_X, logger):
    """清理和准备数据"""
    logger.info("开始数据清理和准备")
    
    # 1. 处理非数值列
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            logger.info(f"处理分类列: {col}")
            # 对于分类列，使用标签编码
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()

            # 合并训练和测试数据进行编码
            combined = pd.concat([X[col], test_X[col]], axis=0)
            combined_encoded = le.fit_transform(combined.astype(str))

            X[col] = combined_encoded[:len(X)]
            test_X[col] = combined_encoded[len(X):]
    
    # 2. 处理无穷大值
    X = X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)
    
    # 3. 处理极值（使用更保守的方法）
    for col in X.select_dtypes(include=[np.number]).columns:
        # 计算合理的上下界
        q01 = X[col].quantile(0.005)
        q99 = X[col].quantile(0.995)
        
        if pd.notna(q01) and pd.notna(q99) and q01 != q99:
            X[col] = X[col].clip(lower=q01, upper=q99)
            test_X[col] = test_X[col].clip(lower=q01, upper=q99)
    
    # 4. 智能填充缺失值
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
                # 数值列用中位数填充
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                test_X[col] = test_X[col].fillna(fill_value)
            else:
                # 分类列用众数填充
                fill_value = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 0
                X[col] = X[col].fillna(fill_value)
                test_X[col] = test_X[col].fillna(fill_value)
    
    # 5. 最终检查
    logger.info(f"数据清理完成: {X.shape}")
    logger.info(f"数据类型分布: {X.dtypes.value_counts().to_dict()}")

    # 确保没有问题
    assert not X.isnull().any().any(), "训练集仍有NaN值"
    assert not test_X.isnull().any().any(), "测试集仍有NaN值"

    # 只对数值列检查无穷大值
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        assert not np.isinf(X[numeric_cols].values).any(), "训练集仍有无穷大值"
        assert not np.isinf(test_X[numeric_cols].values).any(), "测试集仍有无穷大值"
    
    return X, test_X


def optimize_model_params():
    """返回优化后的模型参数"""
    return {
        'xgboost_params': {
            'max_depth': 7,
            'learning_rate': 0.02,
            'n_estimators': 3000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
        },
        'lightgbm_params': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.02,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_samples': 25,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 3000,
            'n_jobs': -1,
        }
    }


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始执行优化版Home Credit Default Risk Pipeline")

    try:
        # 加载配置
        with open("config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        
        # 使用优化后的参数
        optimized_params = optimize_model_params()
        config.update(optimized_params)

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

        # 3. 基础特征工程
        logger.info("步骤3: 基础特征工程")
        feature_engineer = FeatureEngineer(data_dict)
        train_df_featured = feature_engineer.engineer_features(train_df_processed)
        test_df_featured = feature_engineer.engineer_features(test_df_processed)

        # 4. Home Credit专用特征工程
        logger.info("步骤4: Home Credit专用特征工程")
        credit_engineer = CreditSpecificFeatureEngineer(data_dict)
        train_df_enhanced = credit_engineer.engineer_all_features(train_df_featured)
        test_df_enhanced = credit_engineer.engineer_all_features(test_df_featured)

        # 5. 准备最终数据
        logger.info("步骤5: 准备最终数据")
        
        # 确保训练集和测试集有相同的列
        common_cols = [
            col for col in train_df_enhanced.columns if col in test_df_enhanced.columns
        ]
        common_cols = [col for col in common_cols if col not in ["TARGET", "SK_ID_CURR"]]

        logger.info(f"最终特征数量: {len(common_cols)}")

        # 准备训练数据
        X = train_df_enhanced[common_cols]
        y = train_df_enhanced["TARGET"]
        test_X = test_df_enhanced[common_cols]
        test_ids = test_df_enhanced["SK_ID_CURR"]

        # 清理特征名
        def clean_feature_names(df):
            df = df.copy()
            import re
            df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
            return df

        X = clean_feature_names(X)
        test_X = clean_feature_names(test_X)

        # 数据清理
        X, test_X = clean_and_prepare_data(X, test_X, logger)

        # 6. 集成模型训练
        logger.info("步骤6: 集成模型训练")
        ensemble_trainer = EnsembleTrainer(config)
        oof_preds, test_preds = ensemble_trainer.train_ensemble(X, y, test_X)

        # 7. 融合预测
        logger.info("步骤7: 融合预测")
        final_oof, final_test = ensemble_trainer.blend_predictions(oof_preds, test_preds, y)

        # 8. 生成提交文件
        logger.info("步骤8: 生成提交文件")
        submission = ensemble_trainer.generate_submission(final_test, test_ids, "optimized_submission.csv")

        # 输出最终结果
        final_auc = roc_auc_score(y, final_oof)
        logger.info(f"最终交叉验证AUC: {final_auc:.6f}")
        
        # 显示特征重要性
        if ensemble_trainer.feature_importances is not None:
            top_features = ensemble_trainer.feature_importances.head(30)
            logger.info("Top 30 重要特征:")
            for feature, importance in top_features.items():
                logger.info(f"  {feature}: {importance:.6f}")

        logger.info("优化版Pipeline执行完成")
        logger.info(f"提交文件已保存到: optimized_submission.csv")
        
        # 与原始结果对比
        if os.path.exists("ensemble_submission.csv"):
            logger.info("与原始结果对比:")
            logger.info(f"优化版AUC: {final_auc:.6f}")
            logger.info(f"改进幅度: {final_auc - 0.78:.6f}")

    except Exception as e:
        logger.error(f"Pipeline执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
